/**
 * Chapter 4.1 — N-grams.
 *
 * The tables are counted from the corpus at runtime rather than shipped
 * precomputed, so changing the corpus really does change the model. Perplexity
 * and coverage are measured on held-out text the tables never saw.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CorpusDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, createRng } from './shared';

export type NgramMode = 'beat-the-model' | 'tune-order' | 'sparsity-wall';

export interface NgramPredictionConfig {
  mode: NgramMode;
  corpus: string;
  n: number;
  nRange?: [number, number];
  rounds?: number;
  candidateCount?: number;
  smoothing: 'laplace' | 'none';
  smoothingAlpha: number;
  smoothingAlphaRange?: [number, number];
  heldOutSplit?: number;
  minCoverage?: number;
}

export interface PreparedNgramData {
  /** Whitespace-and-punctuation tokens, in corpus order. */
  tokens: string[];
  vocabulary: string[];
}

export interface PredictionRound {
  context: string[];
  /** The word that actually follows in held-out text. */
  trueNext: string;
  candidates: string[];
  answer: string | null;
}

export interface NgramTable {
  n: number;
  /** context key → { word → count } */
  counts: Map<string, Map<string, number>>;
  contextTotals: Map<string, number>;
  vocabularySize: number;
}

export interface NgramPredictionState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: NgramMode;
  config: NgramPredictionConfig;
  n: number;
  alpha: number;
  trainTokens: string[];
  heldOutTokens: string[];
  table: NgramTable;
  rounds: PredictionRound[];
}

export type NgramPredictionAction =
  | { type: 'ANSWER'; roundIndex: number; word: string }
  | { type: 'SET_N'; value: number }
  | { type: 'SET_ALPHA'; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Splits into word and punctuation tokens, lower-cased. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z']+|[.,!?;:"]/g) ?? [];
}

export async function prepare(
  config: NgramPredictionConfig,
  deps: { corpus: CorpusDep }
): Promise<PreparedNgramData> {
  const text = await deps.corpus.load(config.corpus);
  const tokens = tokenize(text);
  if (tokens.length === 0) throw new Error(`Corpus "${config.corpus}" produced no tokens`);
  return { tokens, vocabulary: [...new Set(tokens)] };
}

/** Counts an n-gram table from tokens — this is the whole "training" step. */
export function buildTable(tokens: readonly string[], n: number, vocabularySize: number): NgramTable {
  const counts = new Map<string, Map<string, number>>();
  const contextTotals = new Map<string, number>();
  const order = Math.max(1, n);

  for (let i = order - 1; i < tokens.length; i++) {
    const context = tokens.slice(i - order + 1, i).join(' ');
    const word = tokens[i]!;

    let bucket = counts.get(context);
    if (!bucket) {
      bucket = new Map<string, number>();
      counts.set(context, bucket);
    }
    bucket.set(word, (bucket.get(word) ?? 0) + 1);
    contextTotals.set(context, (contextTotals.get(context) ?? 0) + 1);
  }

  return { n: order, counts, contextTotals, vocabularySize };
}

/** Add-alpha smoothed probability of `word` following `context`. */
export function probabilityOf(
  table: NgramTable,
  context: string,
  word: string,
  alpha: number
): number {
  const bucket = table.counts.get(context);
  const observed = bucket?.get(word) ?? 0;
  const total = table.contextTotals.get(context) ?? 0;
  const denominator = total + alpha * table.vocabularySize;
  if (denominator === 0) return 1 / Math.max(table.vocabularySize, 1);
  return (observed + alpha) / denominator;
}

function contextKeyAt(tokens: readonly string[], index: number, n: number): string {
  return tokens.slice(Math.max(0, index - n + 1), index).join(' ');
}

/** Mean perplexity over held-out text — lower means better prediction. */
export function heldOutPerplexity(
  table: NgramTable,
  heldOut: readonly string[],
  alpha: number
): number {
  if (heldOut.length < table.n) return Infinity;
  let logSum = 0;
  let count = 0;
  for (let i = table.n - 1; i < heldOut.length; i++) {
    const p = probabilityOf(table, contextKeyAt(heldOut, i, table.n), heldOut[i]!, alpha);
    logSum += Math.log(Math.max(p, 1e-12));
    count++;
  }
  return count === 0 ? Infinity : Math.exp(-logSum / count);
}

/** Share of held-out contexts the table has ever seen. */
export function contextCoverage(table: NgramTable, heldOut: readonly string[]): number {
  if (heldOut.length < table.n) return 0;
  let seen = 0;
  let total = 0;
  for (let i = table.n - 1; i < heldOut.length; i++) {
    if (table.counts.has(contextKeyAt(heldOut, i, table.n))) seen++;
    total++;
  }
  return total === 0 ? 0 : seen / total;
}

function buildRounds(
  config: NgramPredictionConfig,
  table: NgramTable,
  heldOut: readonly string[],
  vocabulary: readonly string[]
): PredictionRound[] {
  const rounds: PredictionRound[] = [];
  const wanted = config.rounds ?? 0;
  if (wanted === 0 || heldOut.length <= table.n) return rounds;

  const rng = createRng(heldOut.length * 31 + table.n);
  let guard = 0;

  while (rounds.length < wanted && guard++ < wanted * 200) {
    const i = table.n - 1 + Math.floor(rng() * (heldOut.length - table.n));
    const context = heldOut.slice(Math.max(0, i - table.n + 1), i);
    const trueNext = heldOut[i]!;
    // Only ask about contexts the table has actually seen, otherwise the
    // question is unanswerable from the model rather than merely hard.
    if (!table.counts.has(context.join(' '))) continue;

    const candidates = new Set<string>([trueNext]);
    let attempts = 0;
    while (candidates.size < (config.candidateCount ?? 4) && attempts++ < 200) {
      candidates.add(vocabulary[Math.floor(rng() * vocabulary.length)]!);
    }

    const shuffled = [...candidates];
    for (let k = shuffled.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [shuffled[k], shuffled[j]] = [shuffled[j]!, shuffled[k]!];
    }

    rounds.push({ context, trueNext, candidates: shuffled, answer: null });
  }

  return rounds;
}

function rebuild(state: NgramPredictionState): NgramPredictionState {
  const table = buildTable(state.trainTokens, state.n, new Set(state.trainTokens).size);
  return {
    ...state,
    table,
    rounds:
      state.mode === 'beat-the-model'
        ? buildRounds(state.config, table, state.heldOutTokens, [...new Set(state.trainTokens)])
        : state.rounds,
  };
}

export function initState(
  config: NgramPredictionConfig,
  rules: EngineRules,
  prepared: PreparedNgramData
): NgramPredictionState {
  const splitAt = Math.floor(prepared.tokens.length * (1 - (config.heldOutSplit ?? 0.2)));
  const trainTokens = prepared.tokens.slice(0, splitAt);
  const heldOutTokens = prepared.tokens.slice(splitAt);

  const base: NgramPredictionState = {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    n: config.n,
    alpha: config.smoothing === 'none' ? 0 : config.smoothingAlpha,
    trainTokens,
    heldOutTokens,
    table: buildTable(trainTokens, config.n, new Set(trainTokens).size),
    rounds: [],
  };

  return rebuild(base);
}

export function applyAction(
  state: NgramPredictionState,
  action: NgramPredictionAction
): NgramPredictionState {
  switch (action.type) {
    case 'ANSWER': {
      const round = state.rounds[action.roundIndex];
      if (!round || !round.candidates.includes(action.word)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, answer: action.word };
      return { ...state, rounds, status: 'active', actionCount: state.actionCount + 1 };
    }

    case 'SET_N': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.nRange ?? [state.config.n, state.config.n];
      const n = Math.round(clamp(action.value, min, max));
      // Changing the order rebuilds the table from the same corpus.
      return {
        ...rebuild({ ...state, n }),
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_ALPHA': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.smoothingAlphaRange ?? [0, 1];
      return {
        ...state,
        alpha: clamp(action.value, min, max),
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        tokens: [...state.trainTokens, ...state.heldOutTokens],
        vocabulary: [...new Set([...state.trainTokens, ...state.heldOutTokens])],
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Top-k predictions from the table for a context — what the model would say. */
export function topPredictions(
  state: NgramPredictionState,
  context: readonly string[],
  k: number
): { word: string; probability: number }[] {
  const key = context.slice(-(state.n - 1)).join(' ');
  const bucket = state.table.counts.get(key);
  if (!bucket) return [];
  return [...bucket.entries()]
    .map(([word]) => ({ word, probability: probabilityOf(state.table, key, word, state.alpha) }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
}

/**
 * Highest order in range whose held-out coverage still clears the floor.
 *
 * Recounted from the same corpus rather than authored, so editing the corpus
 * moves the wall and the level stays correct.
 */
export function findHighestViableOrder(state: NgramPredictionState): number {
  const [min, max] = state.config.nRange ?? [state.n, state.n];
  const floor = state.config.minCoverage ?? 0.5;
  const vocabularySize = new Set(state.trainTokens).size;

  let best = min;
  for (let n = min; n <= max; n++) {
    const table = buildTable(state.trainTokens, n, vocabularySize);
    if (contextCoverage(table, state.heldOutTokens) >= floor) best = n;
  }
  return best;
}

export function evaluate(state: NgramPredictionState): ScoreResult {
  switch (state.mode) {
    case 'beat-the-model': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'predictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.rounds.filter((r) => r.answer !== null && r.answer === r.trueNext).length;
      return scoreLevel({
        metric: 'predictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'tune-order': {
      const perplexity = heldOutPerplexity(state.table, state.heldOutTokens, state.alpha);
      return scoreLevel({
        metric: 'heldOutPerplexity',
        value: perplexity,
        rules: state.rules,
        breakdown: {
          n: state.n,
          alpha: state.alpha,
          coverage: contextCoverage(state.table, state.heldOutTokens),
          contexts: state.table.counts.size,
        },
      });
    }

    case 'sparsity-wall': {
      const coverage = contextCoverage(state.table, state.heldOutTokens);
      const minCoverage = state.config.minCoverage ?? 0.5;
      const highestViable = findHighestViableOrder(state);

      // The task is locating the wall, not climbing as high as possible. Full
      // credit for landing on the highest order that still generalises, tapering
      // for stopping short, nothing at all past the point counts fall apart.
      const value =
        coverage < minCoverage ? 0 : clamp(1 - (highestViable - state.n) * 0.25, 0, 1);

      return scoreLevel({
        metric: 'sparsityScore',
        value,
        rules: state.rules,
        breakdown: {
          n: state.n,
          coverage,
          minCoverage,
          highestViableOrder: highestViable,
          perplexity: heldOutPerplexity(state.table, state.heldOutTokens, state.alpha),
        },
      });
    }
  }
}
