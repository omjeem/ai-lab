/**
 * Chapter 7.2 — Grounded Generation.
 *
 * Every round is a real generation from the tiny causal LM (`tinyCausalLM`,
 * `CausalLMDep`), read one real token at a time via repeated
 * `nextTokenDistribution` calls — there is no `ChatModelDep` wrapper for the
 * local model, only the single-token distribution every other World 1/4
 * chapter already uses, so grounded completions are built the same way here:
 * greedy, real, and stopping the moment the model itself stops. Facts come
 * from the same bundled corpus World 7.1 reads (`public/corpora/retrieval-facts.json`),
 * and `prepare()` guards that each fact's `answer` really appears in its own
 * passage, so nothing scored here is an authored key.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep, CorpusDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { createRng } from './shared';

export type GroundedGenerationMode = 'identify-grounded' | 'pick-passage' | 'curate-context';

export interface GroundedGenerationConfig {
  mode: GroundedGenerationMode;
  corpus: string;
  /** Max real tokens to greedily decode per completion. */
  maxTokens: number;
  /** identify-grounded: which facts become rounds. */
  factIds?: string[];
  /** pick-passage: one round per entry, target fact plus its distractors. */
  rounds?: { targetFactId: string; distractorFactIds: string[] }[];
  /** curate-context: the single target/distractor pair to curate. */
  targetFactId?: string;
  distractorFactId?: string;
}

export interface GroundedFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

interface RawCorpus {
  facts: GroundedFact[];
}

export interface IdentifyRound {
  factId: string;
  query: string;
  /** Which side is the real grounded completion — derived, never authored. */
  groundedSide: 'left' | 'right';
  leftText: string;
  rightText: string;
  pick: 'left' | 'right' | null;
}

export interface PickPassageRound {
  targetFactId: string;
  candidateFactIds: string[];
  query: string;
  chosenFactId: string | null;
  decodedText: string | null;
}

export interface PreparedGroundedData {
  facts: GroundedFact[];
  identifyRounds: IdentifyRound[];
  pickRounds: PickPassageRound[];
}

export interface GroundedGenerationState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: GroundedGenerationMode;
  config: GroundedGenerationConfig;
  facts: GroundedFact[];

  identifyRounds: IdentifyRound[];

  pickRounds: PickPassageRound[];

  targetFactId: string;
  distractorFactId: string;
  order: 'correct-first' | 'distractor-first';
  truncateDistractor: boolean;
  attempts: number;
  solved: boolean;
  solvedAtAttempt: number | null;
  lastDecodedText: string | null;
}

export type GroundedGenerationAction =
  | { type: 'PICK_SIDE'; roundIndex: number; side: 'left' | 'right' }
  | { type: 'CHOOSE_PASSAGE'; roundIndex: number; factId: string }
  | { type: 'TEST_PASSAGE'; roundIndex: number; decodedText: string }
  | { type: 'SET_ORDER'; order: 'correct-first' | 'distractor-first' }
  | { type: 'SET_TRUNCATE'; truncate: boolean }
  | { type: 'TEST_CONTEXT'; decodedText: string }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Deterministic small hash, used only to seed a per-fact shuffle/side pick. */
function seedFor(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) >>> 0;
  return h;
}

function shuffled<T>(items: readonly T[], seed: number): T[] {
  const rng = createRng(seed);
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
  return arr;
}

export function containsAnswer(text: string, answer: string): boolean {
  const normalise = (s: string) => s.replace(/,/g, '').toLowerCase();
  return normalise(text).includes(normalise(answer));
}

export function buildGroundedPrompt(contextText: string | null, query: string): string {
  return contextText ? `Context: ${contextText}\nQuestion: ${query}\nAnswer:` : `Question: ${query}\nAnswer:`;
}

export function buildCuratedContextText(
  target: GroundedFact,
  distractor: GroundedFact,
  order: 'correct-first' | 'distractor-first',
  truncateDistractor: boolean
): string {
  const distractorText = truncateDistractor ? distractor.sentences[0]! : distractor.sentences.join(' ');
  const targetText = target.sentences.join(' ');
  return order === 'correct-first' ? `${targetText} ${distractorText}` : `${distractorText} ${targetText}`;
}

/**
 * Greedy real decode: repeatedly reads the model's actual top-1 next token
 * (topK=1 is enough — we only ever take the argmax) and appends it, stopping
 * on an end-of-turn marker, an empty response, a double newline, or the
 * token budget. Every token is a genuine forward pass, never simulated.
 */
export async function greedyDecode(causalLM: CausalLMDep, prompt: string, maxTokens: number): Promise<string> {
  let text = prompt;
  let out = '';
  for (let i = 0; i < maxTokens; i++) {
    const dist = await causalLM.nextTokenDistribution(text, 1);
    const token = dist.tokens[0];
    if (!token) break;
    if (token.includes('<|im_end|>')) break;
    out += token;
    text += token;
    if (out.includes('\n\n')) break;
  }
  return out;
}

export async function prepare(
  config: GroundedGenerationConfig,
  deps: { corpus: CorpusDep; causalLM: CausalLMDep }
): Promise<PreparedGroundedData> {
  const raw = await deps.corpus.load(config.corpus);
  const parsed = JSON.parse(raw) as RawCorpus;

  for (const fact of parsed.facts) {
    const passage = fact.sentences.join(' ');
    if (!passage.includes(fact.answer)) {
      throw new Error(`Fact "${fact.id}"'s answer "${fact.answer}" does not appear in its own passage`);
    }
  }

  const factById = new Map(parsed.facts.map((f) => [f.id, f]));

  const identifyRounds: IdentifyRound[] = [];
  if (config.mode === 'identify-grounded') {
    for (const factId of config.factIds ?? []) {
      const fact = factById.get(factId);
      if (!fact) continue;
      const passage = fact.sentences.join(' ');
      const groundedPrompt = buildGroundedPrompt(passage, fact.query);
      const ungroundedPrompt = buildGroundedPrompt(null, fact.query);
      const [groundedText, ungroundedText] = await Promise.all([
        greedyDecode(deps.causalLM, groundedPrompt, config.maxTokens),
        greedyDecode(deps.causalLM, ungroundedPrompt, config.maxTokens),
      ]);
      const groundedSide: 'left' | 'right' = createRng(seedFor(factId))() < 0.5 ? 'left' : 'right';
      identifyRounds.push({
        factId,
        query: fact.query,
        groundedSide,
        leftText: groundedSide === 'left' ? groundedText : ungroundedText,
        rightText: groundedSide === 'right' ? groundedText : ungroundedText,
        pick: null,
      });
    }
  }

  const pickRounds: PickPassageRound[] = [];
  if (config.mode === 'pick-passage') {
    for (const roundConfig of config.rounds ?? []) {
      const target = factById.get(roundConfig.targetFactId);
      if (!target) continue;
      const candidateFactIds = shuffled(
        [roundConfig.targetFactId, ...roundConfig.distractorFactIds],
        seedFor(roundConfig.targetFactId)
      );
      pickRounds.push({
        targetFactId: roundConfig.targetFactId,
        candidateFactIds,
        query: target.query,
        chosenFactId: null,
        decodedText: null,
      });
    }
  }

  return { facts: parsed.facts, identifyRounds, pickRounds };
}

export function initState(
  config: GroundedGenerationConfig,
  rules: EngineRules,
  prepared: PreparedGroundedData
): GroundedGenerationState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    facts: prepared.facts,

    identifyRounds: prepared.identifyRounds.map((r) => ({ ...r, pick: null })),
    pickRounds: prepared.pickRounds.map((r) => ({ ...r, chosenFactId: null, decodedText: null })),

    targetFactId: config.targetFactId ?? '',
    distractorFactId: config.distractorFactId ?? '',
    order: 'correct-first',
    truncateDistractor: false,
    attempts: 0,
    solved: false,
    solvedAtAttempt: null,
    lastDecodedText: null,
  };
}

export function applyAction(
  state: GroundedGenerationState,
  action: GroundedGenerationAction
): GroundedGenerationState {
  const bump = (next: Partial<GroundedGenerationState>): GroundedGenerationState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'PICK_SIDE': {
      const round = state.identifyRounds[action.roundIndex];
      if (!round) return state;
      const identifyRounds = [...state.identifyRounds];
      identifyRounds[action.roundIndex] = { ...round, pick: action.side };
      return bump({ identifyRounds });
    }

    case 'CHOOSE_PASSAGE': {
      const round = state.pickRounds[action.roundIndex];
      if (!round || !round.candidateFactIds.includes(action.factId)) return state;
      const pickRounds = [...state.pickRounds];
      pickRounds[action.roundIndex] = { ...round, chosenFactId: action.factId, decodedText: null };
      return bump({ pickRounds });
    }

    case 'TEST_PASSAGE': {
      const round = state.pickRounds[action.roundIndex];
      if (!round || !round.chosenFactId) return state;
      const pickRounds = [...state.pickRounds];
      pickRounds[action.roundIndex] = { ...round, decodedText: action.decodedText };
      return bump({ pickRounds });
    }

    case 'SET_ORDER':
      return bump({ order: action.order });

    case 'SET_TRUNCATE':
      return bump({ truncateDistractor: action.truncate });

    case 'TEST_CONTEXT': {
      const attempts = state.attempts + 1;
      const targetFact = state.facts.find((f) => f.id === state.targetFactId);
      const hit = !!targetFact && containsAnswer(action.decodedText, targetFact.answer);
      const alreadySolved = state.solved;
      return bump({
        attempts,
        lastDecodedText: action.decodedText,
        solved: alreadySolved || hit,
        solvedAtAttempt: alreadySolved ? state.solvedAtAttempt : hit ? attempts : null,
      });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        facts: state.facts,
        identifyRounds: state.identifyRounds,
        pickRounds: state.pickRounds,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: GroundedGenerationState): ScoreResult {
  switch (state.mode) {
    case 'identify-grounded': {
      const total = state.identifyRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'predictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.identifyRounds.filter((r) => r.pick !== null && r.pick === r.groundedSide).length;
      return scoreLevel({
        metric: 'predictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'pick-passage': {
      const total = state.pickRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'groundedAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.pickRounds.filter((r) => {
        if (!r.decodedText) return false;
        const target = state.facts.find((f) => f.id === r.targetFactId);
        return !!target && containsAnswer(r.decodedText, target.answer);
      }).length;
      return scoreLevel({
        metric: 'groundedAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'curate-context': {
      const value = state.solved ? state.solvedAtAttempt! : Infinity;
      return scoreLevel({
        metric: 'attemptsToSolve',
        value,
        rules: state.rules,
        breakdown: { attempts: state.attempts, solved: state.solved ? 1 : 0 },
      });
    }
  }
}
