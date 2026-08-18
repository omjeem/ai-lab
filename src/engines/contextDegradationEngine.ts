/**
 * Chapter 8.2 — Context Length & Degradation.
 *
 * Three real windows onto what happens as a prompt grows: whether a planted
 * fact can still be retrieved (`tinyCausalLM`, real greedy decode, reusing
 * `greedyDecode`/`buildGroundedPrompt`/`containsAnswer` from
 * `groundedGenerationEngine.ts` rather than re-deriving them), whether a real
 * transformer's own attention mass to an early token thins out as more tokens
 * compete for it (`attentionModel`, reusing `attentionRow`/`isSpecialToken`
 * from `attentionGuessEngine.ts`), and whether a fixed context budget can
 * still be spent safely once more than one candidate fact could fill it.
 * Every pass/fail, every attention weight and every real decode below comes
 * from an actual forward pass — nothing here is a fitted or assumed curve.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { AttentionDep, AttentionResult, CausalLMDep, CorpusDep } from './deps';
import { attentionRow, isSpecialToken } from './attentionGuessEngine';
import { buildGroundedPrompt, containsAnswer, greedyDecode } from './groundedGenerationEngine';
import { scoreLevel } from './scoringEngine';
import { normalizeDistribution } from './shared';

export type ContextDegradationMode = 'needle-haystack' | 'attention-dilution' | 'budget-subset';

export interface ContextDegradationConfig {
  mode: ContextDegradationMode;
  corpus?: string;
  maxTokens?: number;
  // needle-haystack
  factId?: string;
  fillerCorpus?: string;
  fillerWordCounts?: number[];
  // attention-dilution
  sentences?: string[];
  targets?: ('min' | 'max')[];
  // budget-subset
  targetFactId?: string;
  distractorFactIds?: string[];
  budget?: number;
}

export interface ContextFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

interface RawCorpus {
  facts: ContextFact[];
}

export interface LengthRound {
  fillerWords: number;
  /** Real greedy-decoded completion at this haystack length. */
  decodedText: string;
  /** Real: does the decode actually contain the planted fact's answer. */
  passed: boolean;
  guess: boolean | null;
}

export interface DilutionRound {
  sentence: string;
  tokens: string[];
  queryIndex: number;
  /** Candidate key indices: every token except the query and special tokens. */
  keyIndices: number[];
  /** Real attention distribution over `keyIndices`, averaged across every layer and head. */
  trueRow: number[];
  target: 'min' | 'max';
  guessIndex: number | null;
}

export interface PreparedContextData {
  lengthRounds: LengthRound[];
  dilutionRounds: DilutionRound[];
  targetFact: ContextFact | null;
  distractorFacts: ContextFact[];
}

export interface ContextDegradationState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: ContextDegradationMode;
  config: ContextDegradationConfig;

  lengthRounds: LengthRound[];

  dilutionRounds: DilutionRound[];

  targetFact: ContextFact | null;
  distractorFacts: ContextFact[];
  selectedDistractorId: string | null;
  order: 'target-first' | 'distractor-first';
  attempts: number;
  solved: boolean;
  solvedAtAttempt: number | null;
  lastDecodedText: string | null;
}

export type ContextDegradationAction =
  | { type: 'GUESS_LENGTH'; roundIndex: number; guess: boolean }
  | { type: 'GUESS_DILUTION'; roundIndex: number; tokenIndex: number }
  | { type: 'SELECT_DISTRACTOR'; factId: string }
  | { type: 'SET_ORDER'; order: 'target-first' | 'distractor-first' }
  | { type: 'TEST_SUBSET'; decodedText: string }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

function findQueryIndex(tokens: readonly string[]): number {
  for (let i = tokens.length - 1; i >= 0; i--) {
    if (!isSpecialToken(tokens[i]!)) return i;
  }
  return tokens.length - 1;
}

/** Real attention row for `queryIndex`, averaged across every real layer and head — not just one layer's. */
function meanAcrossLayers(result: AttentionResult, queryIndex: number): number[] {
  const layerCount = result.attention.length;
  if (layerCount === 0) return [];
  const width = result.tokens.length;
  const sums = new Array<number>(width).fill(0);
  for (let layer = 0; layer < layerCount; layer++) {
    const row = attentionRow(result, layer, queryIndex, 'mean');
    for (let k = 0; k < width; k++) sums[k]! += row[k] ?? 0;
  }
  return sums.map((s) => s / layerCount);
}

function loadCorpusFacts(raw: string): ContextFact[] {
  const parsed = JSON.parse(raw) as RawCorpus;
  for (const fact of parsed.facts) {
    const passage = fact.sentences.join(' ');
    if (!passage.includes(fact.answer)) {
      throw new Error(`Fact "${fact.id}"'s answer "${fact.answer}" does not appear in its own passage`);
    }
  }
  return parsed.facts;
}

export async function prepare(
  config: ContextDegradationConfig,
  deps: { corpus?: CorpusDep; causalLM?: CausalLMDep; attention?: AttentionDep }
): Promise<PreparedContextData> {
  const lengthRounds: LengthRound[] = [];
  const dilutionRounds: DilutionRound[] = [];
  let targetFact: ContextFact | null = null;
  let distractorFacts: ContextFact[] = [];

  if (config.mode === 'needle-haystack') {
    if (!deps.corpus || !deps.causalLM) throw new Error('needle-haystack mode requires corpus and causalLM deps');
    const facts = loadCorpusFacts(await deps.corpus.load(config.corpus ?? ''));
    const fact = facts.find((f) => f.id === config.factId);
    if (!fact) throw new Error(`Fact "${config.factId}" not found in corpus`);

    const fillerText = await deps.corpus.load(config.fillerCorpus ?? '');
    const fillerWords = fillerText.replace(/\s+/g, ' ').trim().split(' ');
    const needle = fact.sentences.join(' ');
    const maxTokens = config.maxTokens ?? 16;

    for (const wordCount of config.fillerWordCounts ?? []) {
      const filler = fillerWords.slice(0, wordCount).join(' ');
      const haystack = wordCount === 0 ? needle : `${needle} ${filler}`;
      const prompt = buildGroundedPrompt(haystack, fact.query);
      const decodedText = await greedyDecode(deps.causalLM, prompt, maxTokens);
      lengthRounds.push({
        fillerWords: wordCount,
        decodedText,
        passed: containsAnswer(decodedText, fact.answer),
        guess: null,
      });
    }
  }

  if (config.mode === 'attention-dilution') {
    if (!deps.attention) throw new Error('attention-dilution mode requires an attention dep');
    const targets = config.targets ?? [];
    const sentences = config.sentences ?? [];
    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i]!;
      const result = await deps.attention.attention(sentence);
      if (result.attention.length === 0) throw new Error(`No attention returned for "${sentence}"`);

      const queryIndex = findQueryIndex(result.tokens);
      const keyIndices = result.tokens
        .map((token, idx) => ({ token, idx }))
        .filter(({ token, idx }) => idx !== queryIndex && !isSpecialToken(token))
        .map(({ idx }) => idx);

      const fullRow = meanAcrossLayers(result, queryIndex);
      const trueRow = normalizeDistribution(keyIndices.map((idx) => fullRow[idx] ?? 0));

      dilutionRounds.push({
        sentence,
        tokens: result.tokens,
        queryIndex,
        keyIndices,
        trueRow,
        target: targets[i] ?? 'min',
        guessIndex: null,
      });
    }
  }

  if (config.mode === 'budget-subset') {
    if (!deps.corpus) throw new Error('budget-subset mode requires a corpus dep');
    const facts = loadCorpusFacts(await deps.corpus.load(config.corpus ?? ''));
    targetFact = facts.find((f) => f.id === config.targetFactId) ?? null;
    distractorFacts = facts.filter((f) => (config.distractorFactIds ?? []).includes(f.id));
  }

  return { lengthRounds, dilutionRounds, targetFact, distractorFacts };
}

export function initState(
  config: ContextDegradationConfig,
  rules: EngineRules,
  prepared: PreparedContextData
): ContextDegradationState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    lengthRounds: prepared.lengthRounds.map((r) => ({ ...r, guess: null })),
    dilutionRounds: prepared.dilutionRounds.map((r) => ({ ...r, guessIndex: null })),
    targetFact: prepared.targetFact,
    distractorFacts: prepared.distractorFacts,
    selectedDistractorId: null,
    order: 'target-first',
    attempts: 0,
    solved: false,
    solvedAtAttempt: null,
    lastDecodedText: null,
  };
}

export function applyAction(
  state: ContextDegradationState,
  action: ContextDegradationAction
): ContextDegradationState {
  const bump = (next: Partial<ContextDegradationState>): ContextDegradationState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'GUESS_LENGTH': {
      const round = state.lengthRounds[action.roundIndex];
      if (!round) return state;
      const lengthRounds = [...state.lengthRounds];
      lengthRounds[action.roundIndex] = { ...round, guess: action.guess };
      return bump({ lengthRounds });
    }

    case 'GUESS_DILUTION': {
      const round = state.dilutionRounds[action.roundIndex];
      if (!round || !round.keyIndices.includes(action.tokenIndex)) return state;
      const dilutionRounds = [...state.dilutionRounds];
      dilutionRounds[action.roundIndex] = { ...round, guessIndex: action.tokenIndex };
      return bump({ dilutionRounds });
    }

    case 'SELECT_DISTRACTOR': {
      if (!state.distractorFacts.some((f) => f.id === action.factId)) return state;
      return bump({ selectedDistractorId: action.factId });
    }

    case 'SET_ORDER':
      return bump({ order: action.order });

    case 'TEST_SUBSET': {
      if (!state.targetFact || !state.selectedDistractorId) return state;
      const attempts = state.attempts + 1;
      const hit = containsAnswer(action.decodedText, state.targetFact.answer);
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
        lengthRounds: state.lengthRounds,
        dilutionRounds: state.dilutionRounds,
        targetFact: state.targetFact,
        distractorFacts: state.distractorFacts,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** The real extreme (min or max) key index a dilution round's target asks for. */
export function realExtremeIndex(round: DilutionRound): number {
  let best = round.keyIndices[0]!;
  let bestWeight = round.trueRow[0] ?? 0;
  for (let i = 1; i < round.keyIndices.length; i++) {
    const weight = round.trueRow[i] ?? 0;
    const better = round.target === 'min' ? weight < bestWeight : weight > bestWeight;
    if (better) {
      best = round.keyIndices[i]!;
      bestWeight = weight;
    }
  }
  return best;
}

export function evaluate(state: ContextDegradationState): ScoreResult {
  switch (state.mode) {
    case 'needle-haystack': {
      const total = state.lengthRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'retrievalPredictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.lengthRounds.filter((r) => r.guess !== null && r.guess === r.passed).length;
      return scoreLevel({
        metric: 'retrievalPredictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'attention-dilution': {
      const total = state.dilutionRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'dilutionGuessAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.dilutionRounds.filter(
        (r) => r.guessIndex !== null && r.guessIndex === realExtremeIndex(r)
      ).length;
      return scoreLevel({
        metric: 'dilutionGuessAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'budget-subset': {
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
