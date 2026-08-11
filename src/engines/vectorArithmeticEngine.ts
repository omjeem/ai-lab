/**
 * Chapter 1.2 — Vector Arithmetic.
 *
 * The arithmetic is genuinely performed on live embeddings; only the candidate
 * pool is fixed. Ranking is recomputed from cosine similarity every time a term
 * changes, so nothing about the answer is stored in the level JSON beyond which
 * pool to search.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { addVectors, clamp, cosineSimilarity, subtractVectors } from './shared';

export type TermOp = 'add' | 'subtract';
export type VectorArithmeticMode = 'fixed-analogy' | 'free-terms' | 'estimate-similarity';

export interface ConfiguredTerm {
  word: string;
  op: TermOp;
}

export interface VectorArithmeticConfig {
  mode: VectorArithmeticMode;
  terms?: ConfiguredTerm[];
  termCount?: number;
  target?: string;
  candidatePool: string[];
  expectedNearest?: string;
  topK: number;
  maxAttempts?: number;
  rounds?: number;
}

export interface PreparedVectorData {
  vectors: Record<string, number[]>;
}

export interface ActiveTerm {
  word: string | null;
  op: TermOp;
  vector: number[] | null;
}

export interface RankedCandidate {
  word: string;
  similarity: number;
}

export interface EstimateRound {
  word: string;
  actualSimilarity: number;
  estimate: number | null;
}

export interface VectorArithmeticState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: VectorArithmeticMode;
  config: VectorArithmeticConfig;
  vectors: Record<string, number[]>;
  terms: ActiveTerm[];
  resultVector: number[] | null;
  ranked: RankedCandidate[];
  guess: string | null;
  revealed: boolean;
  attempts: number;
  bestTargetRankScore: number;
  rounds: EstimateRound[];
}

export type VectorArithmeticAction =
  | { type: 'SET_TERM'; index: number; word: string; vector: number[]; op?: TermOp }
  | { type: 'SET_OP'; index: number; op: TermOp }
  | { type: 'GUESS'; word: string }
  | { type: 'REVEAL' }
  | { type: 'COMMIT_ATTEMPT' }
  | { type: 'ESTIMATE'; roundIndex: number; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

export async function prepare(
  config: VectorArithmeticConfig,
  deps: { embedder: Embedder }
): Promise<PreparedVectorData> {
  const words = new Set<string>(config.candidatePool);
  for (const term of config.terms ?? []) words.add(term.word);
  if (config.target) words.add(config.target);

  const list = [...words];
  if (list.length === 0) return { vectors: {} };

  const embedded = await deps.embedder.embed(list);
  if (embedded.length !== list.length) {
    throw new Error(`Embedder returned ${embedded.length} vectors for ${list.length} words`);
  }

  const vectors: Record<string, number[]> = {};
  list.forEach((word, i) => {
    vectors[word] = embedded[i]!;
  });
  return { vectors };
}

/** Applies the term chain in order: the first term seeds, the rest add or subtract. */
function computeResult(terms: readonly ActiveTerm[]): number[] | null {
  const usable = terms.filter((t) => t.vector !== null);
  if (usable.length === 0 || usable.length !== terms.length) return null;

  let result = [...usable[0]!.vector!];
  if (usable[0]!.op === 'subtract') result = result.map((v) => -v);

  for (let i = 1; i < usable.length; i++) {
    const term = usable[i]!;
    result = term.op === 'add' ? addVectors(result, term.vector!) : subtractVectors(result, term.vector!);
  }
  return result;
}

function rankCandidates(
  result: number[] | null,
  pool: readonly string[],
  vectors: Record<string, number[]>
): RankedCandidate[] {
  if (result === null) return [];
  return pool
    .filter((word) => vectors[word] !== undefined)
    .map((word) => ({ word, similarity: cosineSimilarity(result, vectors[word]!) }))
    .sort((a, b) => b.similarity - a.similarity);
}

export function initState(
  config: VectorArithmeticConfig,
  rules: EngineRules,
  prepared: PreparedVectorData
): VectorArithmeticState {
  const configured = config.terms ?? [];
  const termCount = config.termCount ?? configured.length;

  const terms: ActiveTerm[] = Array.from({ length: termCount }, (_, i) => {
    const term = configured[i];
    return term
      ? { word: term.word, op: term.op, vector: prepared.vectors[term.word] ?? null }
      : { word: null, op: 'add' as TermOp, vector: null };
  });

  const resultVector = computeResult(terms);
  const ranked = rankCandidates(resultVector, config.candidatePool, prepared.vectors);

  // Estimate rounds walk the pool in order, so each round targets a different
  // real candidate and its real similarity to the computed result.
  const rounds: EstimateRound[] =
    config.mode === 'estimate-similarity' && resultVector !== null
      ? config.candidatePool.slice(0, config.rounds ?? 0).map((word) => ({
          word,
          actualSimilarity: prepared.vectors[word]
            ? cosineSimilarity(resultVector, prepared.vectors[word]!)
            : 0,
          estimate: null,
        }))
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    vectors: prepared.vectors,
    terms,
    resultVector,
    ranked,
    guess: null,
    revealed: false,
    attempts: 0,
    bestTargetRankScore: 0,
    rounds,
  };
}

/** Rank score for the target word: 1 at the top of the pool, 0 at the bottom. */
function targetRankScore(state: VectorArithmeticState): number {
  const target = state.config.target;
  if (!target || state.ranked.length < 2) return 0;
  const index = state.ranked.findIndex((r) => r.word === target);
  if (index === -1) return 0;
  return 1 - index / (state.ranked.length - 1);
}

export function applyAction(
  state: VectorArithmeticState,
  action: VectorArithmeticAction
): VectorArithmeticState {
  const bump = (next: Partial<VectorArithmeticState>): VectorArithmeticState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_TERM':
    case 'SET_OP': {
      if (action.index < 0 || action.index >= state.terms.length) return state;
      const terms = [...state.terms];
      const current = terms[action.index]!;
      terms[action.index] =
        action.type === 'SET_TERM'
          ? { word: action.word, vector: action.vector, op: action.op ?? current.op }
          : { ...current, op: action.op };

      const resultVector = computeResult(terms);
      return bump({
        terms,
        resultVector,
        ranked: rankCandidates(resultVector, state.config.candidatePool, state.vectors),
      });
    }

    case 'GUESS': {
      // Once revealed the answer is visible, so late guesses cannot count.
      if (state.revealed) return state;
      if (!state.config.candidatePool.includes(action.word)) return state;
      return bump({ guess: action.word });
    }

    case 'REVEAL':
      return bump({ revealed: true });

    case 'COMMIT_ATTEMPT': {
      const max = state.config.maxAttempts ?? Infinity;
      if (state.attempts >= max) return state;
      if (state.resultVector === null) return state;
      return bump({
        attempts: state.attempts + 1,
        // Keep the best attempt so experimenting cannot lose earned progress.
        bestTargetRankScore: Math.max(state.bestTargetRankScore, targetRankScore(state)),
      });
    }

    case 'ESTIMATE': {
      if (action.roundIndex < 0 || action.roundIndex >= state.rounds.length) return state;
      if (!Number.isFinite(action.value)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = {
        ...rounds[action.roundIndex]!,
        estimate: clamp(action.value, -1, 1),
      };
      return bump({ rounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, { vectors: state.vectors });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: VectorArithmeticState): ScoreResult {
  switch (state.mode) {
    case 'fixed-analogy': {
      const rank = state.guess === null ? -1 : state.ranked.findIndex((r) => r.word === state.guess) + 1;
      // Top-1 is worth 3, second 2, third 1, anything further nothing.
      const value = rank > 0 ? Math.max(0, 4 - rank) : 0;
      return scoreLevel({
        metric: 'analogyAccuracy',
        value,
        rules: state.rules,
        breakdown: {
          guessRank: rank,
          topSimilarity: state.ranked[0]?.similarity ?? 0,
          poolSize: state.ranked.length,
        },
      });
    }

    case 'free-terms':
      return scoreLevel({
        metric: 'targetRankScore',
        value: state.bestTargetRankScore,
        rules: state.rules,
        breakdown: { attempts: state.attempts, currentScore: targetRankScore(state) },
      });

    case 'estimate-similarity': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({
          metric: 'cosineSimilarityError',
          value: 2,
          rules: state.rules,
          breakdown: { answered: 0, total: 0 },
        });
      }

      let answered = 0;
      let sum = 0;
      for (const round of state.rounds) {
        if (round.estimate === null) {
          // Maximum possible cosine error, so skipping is never a strategy.
          sum += 2;
          continue;
        }
        answered++;
        sum += Math.abs(round.estimate - round.actualSimilarity);
      }

      return scoreLevel({
        metric: 'cosineSimilarityError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }
  }
}
