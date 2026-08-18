/**
 * Chapter 8.3 — Calibration & Hallucination.
 *
 * Every round asks the real causal LM (`tinyCausalLM`, `CausalLMDep`) a
 * question with no supporting context at all — a genuine test of what it
 * "knows" versus what it fluently invents. Real confidence is the mean
 * probability of the real generated tokens, excluding the first (the same
 * leading-formatting-token noise 8-1's `contentTokenMeanBits` was built to
 * exclude), read off `nextTokenDistribution`'s top-k the same way every other
 * World 1/4/7/8 chapter already does — never a full-vocab softmax, and never
 * a value the player's own guess can influence. Real correctness comes from
 * `containsAnswer` against the same bundled, answer-verified fact corpora
 * World 7's chapters already trust (`retrieval-facts` plus a new
 * `calibration-easy-facts` set of common-knowledge questions this model
 * actually tends to get right, needed because every `retrieval-facts` answer
 * turned out to be hallucinated by this model when asked with no context —
 * see the chapter's plan notes).
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep, CorpusDep } from './deps';
import { buildGroundedPrompt, containsAnswer } from './groundedGenerationEngine';
import { scoreLevel } from './scoringEngine';

export type CalibrationMode = 'predict-correctness' | 'spot-hallucination' | 'reduce-confidence';

export interface CalibrationConfig {
  mode: CalibrationMode;
  /** Corpus ids to load and merge facts from, by id. */
  factSources: string[];
  /** Max real tokens to greedily decode per question. */
  maxTokens: number;
  /** How many real top-k entries to renormalise confidence over (default 40). */
  topK?: number;
  // predict-correctness
  factIds?: string[];
  // spot-hallucination
  rounds?: { candidateFactIds: string[] }[];
  // reduce-confidence
  targetFactIds?: string[];
  /** Full prompt template with a literal "{query}" placeholder. */
  framingTemplate?: string;
}

export interface CalibrationFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

interface RawCorpus {
  facts: CalibrationFact[];
}

export interface BaselineResult {
  factId: string;
  topic: string;
  query: string;
  decodedText: string;
  /** Real: does the ungrounded decode actually contain the verified answer. */
  correct: boolean;
  /** Real mean probability (0..1) of the generated content tokens. */
  confidence: number;
}

export interface PredictRound extends BaselineResult {
  guess: boolean | null;
}

export interface SpotRound {
  candidates: BaselineResult[];
  /** Real: the wrong candidate with the highest real confidence in this round. */
  targetFactId: string;
  guessFactId: string | null;
}

export interface FramingRound extends BaselineResult {
  framedDecodedText: string | null;
  framedConfidence: number | null;
  guess: 'drop' | 'rise' | null;
}

export interface PreparedCalibrationData {
  predictRounds: PredictRound[];
  spotRounds: SpotRound[];
  framingRounds: FramingRound[];
}

export interface CalibrationState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: CalibrationMode;
  config: CalibrationConfig;

  predictRounds: PredictRound[];
  spotRounds: SpotRound[];
  framingRounds: FramingRound[];
}

export type CalibrationAction =
  | { type: 'GUESS_CORRECTNESS'; roundIndex: number; guess: boolean }
  | { type: 'GUESS_HALLUCINATION'; roundIndex: number; factId: string }
  | { type: 'GUESS_DELTA'; roundIndex: number; guess: 'drop' | 'rise' }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/**
 * Greedy real decode that also records each real chosen token's own
 * probability (renormalised over the top `topK`, same convention every
 * top-k-based chapter in this project already uses) — stops on the same
 * real signals `greedyDecode` does.
 */
async function decodeWithConfidence(
  causalLM: CausalLMDep,
  prompt: string,
  maxTokens: number,
  topK: number
): Promise<{ text: string; tokenProbs: number[] }> {
  let text = prompt;
  let out = '';
  const tokenProbs: number[] = [];
  for (let i = 0; i < maxTokens; i++) {
    const dist = await causalLM.nextTokenDistribution(text, topK);
    const token = dist.tokens[0];
    const prob = dist.probs[0];
    if (!token || prob === undefined) break;
    if (token.includes('<|im_end|>')) break;
    out += token;
    text += token;
    tokenProbs.push(prob);
    if (out.includes('\n\n')) break;
  }
  return { text: out, tokenProbs };
}

/** Mean probability of every content token, excluding the first — a generic leading/formatting token shared across nearly every answer regardless of content (see 8-1's `contentTokenMeanBits`). */
function contentTokenMeanProb(probs: readonly number[]): number {
  if (probs.length === 0) return 0;
  if (probs.length === 1) return probs[0]!;
  const rest = probs.slice(1);
  return rest.reduce((a, b) => a + b, 0) / rest.length;
}

function buildFramedPrompt(template: string, query: string): string {
  return template.replace('{query}', query);
}

async function computeBaseline(
  causalLM: CausalLMDep,
  fact: CalibrationFact,
  maxTokens: number,
  topK: number
): Promise<BaselineResult> {
  const prompt = buildGroundedPrompt(null, fact.query);
  const { text, tokenProbs } = await decodeWithConfidence(causalLM, prompt, maxTokens, topK);
  return {
    factId: fact.id,
    topic: fact.topic,
    query: fact.query,
    decodedText: text,
    correct: containsAnswer(text, fact.answer),
    confidence: contentTokenMeanProb(tokenProbs),
  };
}

export async function prepare(
  config: CalibrationConfig,
  deps: { corpus: CorpusDep; causalLM: CausalLMDep }
): Promise<PreparedCalibrationData> {
  const factMap = new Map<string, CalibrationFact>();
  for (const source of config.factSources) {
    const raw = await deps.corpus.load(source);
    const parsed = JSON.parse(raw) as RawCorpus;
    for (const fact of parsed.facts) {
      const passage = fact.sentences.join(' ');
      if (!passage.includes(fact.answer)) {
        throw new Error(`Fact "${fact.id}"'s answer "${fact.answer}" does not appear in its own passage`);
      }
      factMap.set(fact.id, fact);
    }
  }

  const topK = config.topK ?? 40;

  const predictRounds: PredictRound[] = [];
  if (config.mode === 'predict-correctness') {
    for (const factId of config.factIds ?? []) {
      const fact = factMap.get(factId);
      if (!fact) continue;
      const baseline = await computeBaseline(deps.causalLM, fact, config.maxTokens, topK);
      predictRounds.push({ ...baseline, guess: null });
    }
  }

  const spotRounds: SpotRound[] = [];
  if (config.mode === 'spot-hallucination') {
    for (const roundConfig of config.rounds ?? []) {
      const candidates: BaselineResult[] = [];
      for (const factId of roundConfig.candidateFactIds) {
        const fact = factMap.get(factId);
        if (!fact) continue;
        candidates.push(await computeBaseline(deps.causalLM, fact, config.maxTokens, topK));
      }
      const wrongCandidates = candidates.filter((c) => !c.correct);
      if (wrongCandidates.length === 0) {
        throw new Error('spot-hallucination round has no real wrong candidate to find');
      }
      const target = wrongCandidates.reduce((best, c) => (c.confidence > best.confidence ? c : best));
      spotRounds.push({ candidates, targetFactId: target.factId, guessFactId: null });
    }
  }

  const framingRounds: FramingRound[] = [];
  if (config.mode === 'reduce-confidence') {
    const template = config.framingTemplate ?? '{query}';
    for (const factId of config.targetFactIds ?? []) {
      const fact = factMap.get(factId);
      if (!fact) continue;
      const baseline = await computeBaseline(deps.causalLM, fact, config.maxTokens, topK);
      const framedPrompt = buildFramedPrompt(template, fact.query);
      const { text: framedText, tokenProbs: framedProbs } = await decodeWithConfidence(
        deps.causalLM,
        framedPrompt,
        config.maxTokens,
        topK
      );
      framingRounds.push({
        ...baseline,
        framedDecodedText: framedText,
        framedConfidence: contentTokenMeanProb(framedProbs),
        guess: null,
      });
    }
  }

  return { predictRounds, spotRounds, framingRounds };
}

export function initState(
  config: CalibrationConfig,
  rules: EngineRules,
  prepared: PreparedCalibrationData
): CalibrationState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    predictRounds: prepared.predictRounds.map((r) => ({ ...r, guess: null })),
    spotRounds: prepared.spotRounds.map((r) => ({ ...r, guessFactId: null })),
    framingRounds: prepared.framingRounds.map((r) => ({ ...r, guess: null })),
  };
}

export function applyAction(state: CalibrationState, action: CalibrationAction): CalibrationState {
  const bump = (next: Partial<CalibrationState>): CalibrationState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'GUESS_CORRECTNESS': {
      const round = state.predictRounds[action.roundIndex];
      if (!round) return state;
      const predictRounds = [...state.predictRounds];
      predictRounds[action.roundIndex] = { ...round, guess: action.guess };
      return bump({ predictRounds });
    }

    case 'GUESS_HALLUCINATION': {
      const round = state.spotRounds[action.roundIndex];
      if (!round || !round.candidates.some((c) => c.factId === action.factId)) return state;
      const spotRounds = [...state.spotRounds];
      spotRounds[action.roundIndex] = { ...round, guessFactId: action.factId };
      return bump({ spotRounds });
    }

    case 'GUESS_DELTA': {
      const round = state.framingRounds[action.roundIndex];
      if (!round) return state;
      const framingRounds = [...state.framingRounds];
      framingRounds[action.roundIndex] = { ...round, guess: action.guess };
      return bump({ framingRounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        predictRounds: state.predictRounds,
        spotRounds: state.spotRounds,
        framingRounds: state.framingRounds,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Real: did this round's confidence actually drop once the cautious framing was applied. */
export function realDropped(round: FramingRound): boolean {
  return round.framedConfidence !== null && round.framedConfidence < round.confidence;
}

export function evaluate(state: CalibrationState): ScoreResult {
  switch (state.mode) {
    case 'predict-correctness': {
      const total = state.predictRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'confidenceCorrectnessAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.predictRounds.filter((r) => r.guess !== null && r.guess === r.correct).length;
      return scoreLevel({
        metric: 'confidenceCorrectnessAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'spot-hallucination': {
      const total = state.spotRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'hallucinationSpotAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.spotRounds.filter(
        (r) => r.guessFactId !== null && r.guessFactId === r.targetFactId
      ).length;
      return scoreLevel({
        metric: 'hallucinationSpotAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'reduce-confidence': {
      const total = state.framingRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'confidenceDropPredictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.framingRounds.filter(
        (r) => r.guess !== null && r.framedConfidence !== null && (r.guess === 'drop') === realDropped(r)
      ).length;
      return scoreLevel({
        metric: 'confidenceDropPredictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }
  }
}
