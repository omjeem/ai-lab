/**
 * Chapter 5.1 — Positional Encoding.
 *
 * The sinusoidal formulas are pure mathematics and stay that way. The final
 * level combines them with real token embeddings, so the encoding is shown as
 * one ingredient of a real pipeline rather than an isolated toy.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, cosineSimilarity, createRng } from './shared';

export type PositionalMode = 'identify-position' | 'tune-distinctness' | 'combine-with-embeddings';

export interface PositionalEncodingConfig {
  mode: PositionalMode;
  dModel: number;
  dModelOptions?: number[];
  maxPosition: number;
  base: number;
  baseRange?: [number, number];
  rounds?: number;
  candidateCount?: number;
  seed: number;
  sentencePairs?: [string, string][];
  scale?: number;
  scaleRange?: [number, number];
}

export interface PositionRound {
  truePosition: number;
  candidates: number[];
  answer: number | null;
  encoding: number[];
}

export interface SentencePairRound {
  a: string;
  b: string;
  /** Real divergence between the two combined representations. */
  trueDivergence: number;
  estimate: number | null;
}

export interface PreparedPositionalData {
  /** Word → real embedding, for the combine level. Empty for the pure-maths ones. */
  vectors: Record<string, number[]>;
}

export interface PositionalEncodingState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: PositionalMode;
  config: PositionalEncodingConfig;
  dModel: number;
  base: number;
  scale: number;
  vectors: Record<string, number[]>;
  rounds: PositionRound[];
  pairs: SentencePairRound[];
}

export type PositionalEncodingAction =
  | { type: 'ANSWER_POSITION'; roundIndex: number; value: number }
  | { type: 'SET_D_MODEL'; value: number }
  | { type: 'SET_BASE'; value: number }
  | { type: 'SET_SCALE'; value: number }
  | { type: 'ESTIMATE_DIVERGENCE'; pairIndex: number; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** The sinusoidal encoding from Attention Is All You Need. */
export function positionalVector(position: number, dModel: number, base: number): number[] {
  const vector = new Array<number>(dModel);
  for (let i = 0; i < dModel; i++) {
    const pair = Math.floor(i / 2);
    const frequency = 1 / Math.pow(base, (2 * pair) / dModel);
    vector[i] = i % 2 === 0 ? Math.sin(position * frequency) : Math.cos(position * frequency);
  }
  return vector;
}

/**
 * How separable positions are on average.
 *
 * Deliberately the mean pairwise similarity rather than the worst case: nearby
 * positions are *supposed* to look alike, so a worst-case measure would sit near
 * zero for every sensible configuration and tell the player nothing.
 */
export function encodingDistinctness(maxPosition: number, dModel: number, base: number): number {
  const vectors = Array.from({ length: maxPosition }, (_, p) => positionalVector(p, dModel, base));
  if (vectors.length < 2) return 1;

  let total = 0;
  let pairs = 0;
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      total += Math.abs(cosineSimilarity(vectors[i]!, vectors[j]!));
      pairs++;
    }
  }
  return clamp(1 - total / pairs, 0, 1);
}

/**
 * Token embeddings with positional encoding added, one vector per position.
 *
 * Deliberately not pooled: summing across positions would cancel the encoding
 * out entirely, since every ordering of the same words visits the same set of
 * positions. Keeping the sequence is what makes word order observable at all.
 */
export function combineWithPositions(
  words: readonly string[],
  vectors: Record<string, number[]>,
  base: number,
  scale: number
): number[][] {
  const present = words.filter((w) => vectors[w] !== undefined);
  if (present.length === 0) return [];

  const dModel = vectors[present[0]!]!.length;
  const out: number[][] = [];

  words.forEach((word, position) => {
    const embedding = vectors[word];
    if (!embedding) return;
    const encoding = positionalVector(position, dModel, base);
    out.push(embedding.map((v, i) => v + scale * encoding[i]!));
  });

  return out;
}

const tokenize = (sentence: string): string[] => sentence.toLowerCase().split(/\s+/).filter(Boolean);

export async function prepare(
  config: PositionalEncodingConfig,
  deps: { embedder?: Embedder } = {}
): Promise<PreparedPositionalData> {
  if (config.mode !== 'combine-with-embeddings' || !deps.embedder) return { vectors: {} };

  const words = new Set<string>();
  for (const [a, b] of config.sentencePairs ?? []) {
    for (const word of [...tokenize(a), ...tokenize(b)]) words.add(word);
  }

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

function buildPairs(
  config: PositionalEncodingConfig,
  vectors: Record<string, number[]>,
  base: number,
  scale: number
): SentencePairRound[] {
  return (config.sentencePairs ?? []).map(([a, b]) => {
    const wordsA = tokenize(a);
    const wordsB = tokenize(b);

    // Measures the chapter's actual claim: a word that moved is a different
    // vector by the time attention sees it. Each word present in both orderings
    // is compared against itself at its two positions, so the divergence is zero
    // when the encoding is scaled out and grows as it is scaled in.
    const moved: number[] = [];
    for (const word of new Set(wordsA)) {
      const embedding = vectors[word];
      if (!embedding) continue;
      const posA = wordsA.indexOf(word);
      const posB = wordsB.indexOf(word);
      if (posB === -1 || posA === posB) continue;

      const dModel = embedding.length;
      const atA = embedding.map((v, i) => v + scale * positionalVector(posA, dModel, base)[i]!);
      const atB = embedding.map((v, i) => v + scale * positionalVector(posB, dModel, base)[i]!);
      moved.push(clamp(1 - cosineSimilarity(atA, atB), 0, 2) / 2);
    }

    return {
      a,
      b,
      trueDivergence: moved.length === 0 ? 0 : clamp(moved.reduce((x, y) => x + y, 0) / moved.length, 0, 1),
      estimate: null,
    };
  });
}

export function initState(
  config: PositionalEncodingConfig,
  rules: EngineRules,
  prepared: PreparedPositionalData = { vectors: {} }
): PositionalEncodingState {
  const rng = createRng(config.seed);

  const rounds: PositionRound[] =
    config.mode === 'identify-position'
      ? Array.from({ length: config.rounds ?? 0 }, () => {
          const truePosition = Math.floor(rng() * config.maxPosition);
          const candidates = new Set<number>([truePosition]);
          let guard = 0;
          while (candidates.size < (config.candidateCount ?? 4) && guard++ < 200) {
            candidates.add(Math.floor(rng() * config.maxPosition));
          }
          const shuffled = [...candidates];
          for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
          }
          return {
            truePosition,
            candidates: shuffled,
            answer: null,
            encoding: positionalVector(truePosition, config.dModel, config.base),
          };
        })
      : [];

  const scale = config.scale ?? 1;

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    dModel: config.dModel,
    base: config.base,
    scale,
    vectors: prepared.vectors,
    rounds,
    pairs:
      config.mode === 'combine-with-embeddings'
        ? buildPairs(config, prepared.vectors, config.base, scale)
        : [],
  };
}

export function applyAction(
  state: PositionalEncodingState,
  action: PositionalEncodingAction
): PositionalEncodingState {
  const bump = (next: Partial<PositionalEncodingState>): PositionalEncodingState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'ANSWER_POSITION': {
      const round = state.rounds[action.roundIndex];
      if (!round || !round.candidates.includes(action.value)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, answer: action.value };
      return bump({ rounds });
    }

    case 'SET_D_MODEL': {
      const options = state.config.dModelOptions;
      if (options && !options.includes(action.value)) return state;
      if (!Number.isInteger(action.value) || action.value < 2) return state;
      return bump({ dModel: action.value });
    }

    case 'SET_BASE': {
      if (!Number.isFinite(action.value) || action.value <= 1) return state;
      const [min, max] = state.config.baseRange ?? [2, 1e6];
      const base = clamp(action.value, min, max);
      // Changing the base changes the combined representations too.
      return bump({
        base,
        pairs:
          state.mode === 'combine-with-embeddings'
            ? buildPairs(state.config, state.vectors, base, state.scale)
            : state.pairs,
      });
    }

    case 'SET_SCALE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.scaleRange ?? [0, 2];
      const scale = clamp(action.value, min, max);
      return bump({
        scale,
        pairs:
          state.mode === 'combine-with-embeddings'
            ? buildPairs(state.config, state.vectors, state.base, scale)
            : state.pairs,
      });
    }

    case 'ESTIMATE_DIVERGENCE': {
      const pair = state.pairs[action.pairIndex];
      if (!pair || !Number.isFinite(action.value)) return state;
      const pairs = [...state.pairs];
      pairs[action.pairIndex] = { ...pair, estimate: clamp(action.value, 0, 1) };
      return bump({ pairs });
    }

    case 'RESET':
      return initState(state.config, state.rules, { vectors: state.vectors });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: PositionalEncodingState): ScoreResult {
  switch (state.mode) {
    case 'identify-position': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'positionIdentificationAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.rounds.filter((r) => r.answer === r.truePosition).length;
      return scoreLevel({
        metric: 'positionIdentificationAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'tune-distinctness': {
      const value = encodingDistinctness(state.config.maxPosition, state.dModel, state.base);
      return scoreLevel({
        metric: 'encodingDistinctness',
        value,
        rules: state.rules,
        breakdown: { dModel: state.dModel, base: state.base, positions: state.config.maxPosition },
      });
    }

    case 'combine-with-embeddings': {
      const total = state.pairs.length;
      if (total === 0) {
        return scoreLevel({ metric: 'divergenceEstimateError', value: 1, rules: state.rules });
      }
      let answered = 0;
      let sum = 0;
      for (const pair of state.pairs) {
        if (pair.estimate === null) {
          sum += 1;
          continue;
        }
        answered++;
        sum += Math.abs(pair.estimate - pair.trueDivergence);
      }
      return scoreLevel({
        metric: 'divergenceEstimateError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total, scale: state.scale },
      });
    }
  }
}
