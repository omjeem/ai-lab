/**
 * Chapter 1.3 — Similarity & Distance.
 *
 * Every "right answer" is derived at runtime from real embeddings: the ranking
 * comes from cosine similarity to the anchor, the outlier from mean pairwise
 * similarity, and the metric-disagreement questions from actually comparing how
 * cosine and Euclidean order the same triple.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { cosineSimilarity, createRng, euclideanDistance, mean, spearmanCorrelation } from './shared';

export type SimilarityMode = 'rank' | 'odd-one-out' | 'free-set';

export interface SimilarityRankConfig {
  mode: SimilarityMode;
  anchor?: string | null;
  words?: string[];
  sets?: string[][];
  metric: 'cosine' | 'euclidean' | 'both';
  minWords?: number;
  maxWords?: number;
  disagreementRounds?: number;
  allowUserWords?: boolean;
}

export interface PreparedSimilarityData {
  vectors: Record<string, number[]>;
}

export interface OddOneOutSet {
  words: string[];
  trueOutlier: string;
  answer: string | null;
}

/**
 * "Does cosine rank x closer to the anchor while Euclidean prefers y?"
 * The answer is computed from the real vectors, not authored in JSON.
 */
export interface DisagreementQuestion {
  anchor: string;
  x: string;
  y: string;
  trueDisagreement: boolean;
  answer: boolean | null;
}

export interface SimilarityRankState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: SimilarityMode;
  config: SimilarityRankConfig;
  vectors: Record<string, number[]>;
  ordering: string[];
  trueOrder: string[];
  sets: OddOneOutSet[];
  words: string[];
  questions: DisagreementQuestion[];
}

export type SimilarityRankAction =
  | { type: 'SET_ORDER'; ordering: string[] }
  | { type: 'MOVE_ITEM'; from: number; to: number }
  | { type: 'ANSWER_ODD'; setIndex: number; word: string }
  | { type: 'ADD_WORD'; word: string; vector: number[] }
  | { type: 'REMOVE_WORD'; word: string }
  | { type: 'ANSWER_DISAGREEMENT'; questionIndex: number; value: boolean }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

export async function prepare(
  config: SimilarityRankConfig,
  deps: { embedder: Embedder }
): Promise<PreparedSimilarityData> {
  const words = new Set<string>(config.words ?? []);
  if (config.anchor) words.add(config.anchor);
  for (const set of config.sets ?? []) for (const w of set) words.add(w);

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

/** The word whose mean similarity to the rest of its set is lowest. */
function findOutlier(words: readonly string[], vectors: Record<string, number[]>): string {
  let worst = words[0]!;
  let worstScore = Infinity;
  for (const word of words) {
    const others = words.filter((w) => w !== word);
    const score = mean(
      others.map((o) => cosineSimilarity(vectors[word] ?? [], vectors[o] ?? []))
    );
    if (score < worstScore) {
      worstScore = score;
      worst = word;
    }
  }
  return worst;
}

/**
 * Builds triples where the two metrics may rank differently, then records what
 * they actually do. Deterministic so a level replays identically.
 */
function buildQuestions(
  words: readonly string[],
  vectors: Record<string, number[]>,
  rounds: number
): DisagreementQuestion[] {
  if (words.length < 3 || rounds <= 0) return [];

  const rng = createRng(words.join('|').length * 7919);
  const seen = new Set<string>();
  const questions: DisagreementQuestion[] = [];
  const maxTries = rounds * 40;

  for (let attempt = 0; attempt < maxTries && questions.length < rounds; attempt++) {
    const pick = () => words[Math.floor(rng() * words.length)]!;
    const anchor = pick();
    const x = pick();
    const y = pick();
    if (anchor === x || anchor === y || x === y) continue;

    const key = [anchor, x, y].join('|');
    if (seen.has(key)) continue;
    seen.add(key);

    const va = vectors[anchor] ?? [];
    const vx = vectors[x] ?? [];
    const vy = vectors[y] ?? [];

    // Cosine prefers whichever has the higher similarity; Euclidean prefers
    // whichever is the shorter distance. They can and do disagree.
    const cosinePrefersX = cosineSimilarity(va, vx) > cosineSimilarity(va, vy);
    const euclideanPrefersX = euclideanDistance(va, vx) < euclideanDistance(va, vy);

    questions.push({
      anchor,
      x,
      y,
      trueDisagreement: cosinePrefersX !== euclideanPrefersX,
      answer: null,
    });
  }

  return questions;
}

export function initState(
  config: SimilarityRankConfig,
  rules: EngineRules,
  prepared: PreparedSimilarityData
): SimilarityRankState {
  const vectors = prepared.vectors;

  const trueOrder =
    config.mode === 'rank' && config.anchor
      ? [...(config.words ?? [])].sort(
          (a, b) =>
            cosineSimilarity(vectors[config.anchor!] ?? [], vectors[b] ?? []) -
            cosineSimilarity(vectors[config.anchor!] ?? [], vectors[a] ?? [])
        )
      : [];

  const sets: OddOneOutSet[] =
    config.mode === 'odd-one-out'
      ? (config.sets ?? []).map((words) => ({
          words: [...words],
          trueOutlier: findOutlier(words, vectors),
          answer: null,
        }))
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    vectors: { ...vectors },
    ordering: [...(config.words ?? [])],
    trueOrder,
    sets,
    words: config.mode === 'free-set' ? [] : [...(config.words ?? [])],
    questions: [],
  };
}

function isPermutation(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  const counts = new Map<string, number>();
  for (const x of a) counts.set(x, (counts.get(x) ?? 0) + 1);
  for (const x of b) {
    const n = counts.get(x);
    if (n === undefined || n === 0) return false;
    counts.set(x, n - 1);
  }
  return true;
}

export function applyAction(
  state: SimilarityRankState,
  action: SimilarityRankAction
): SimilarityRankState {
  const bump = (next: Partial<SimilarityRankState>): SimilarityRankState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_ORDER': {
      // Guard against a dropped or duplicated item corrupting the ranking.
      if (!isPermutation(state.ordering, action.ordering)) return state;
      return bump({ ordering: [...action.ordering] });
    }

    case 'MOVE_ITEM': {
      const { from, to } = action;
      if (from < 0 || from >= state.ordering.length) return state;
      if (to < 0 || to >= state.ordering.length) return state;
      const ordering = [...state.ordering];
      const [moved] = ordering.splice(from, 1);
      ordering.splice(to, 0, moved!);
      return bump({ ordering });
    }

    case 'ANSWER_ODD': {
      const set = state.sets[action.setIndex];
      if (!set || !set.words.includes(action.word)) return state;
      const sets = [...state.sets];
      sets[action.setIndex] = { ...set, answer: action.word };
      return bump({ sets });
    }

    case 'ADD_WORD': {
      const max = state.config.maxWords ?? Infinity;
      if (state.words.length >= max) return state;
      if (state.words.includes(action.word)) return state;

      const words = [...state.words, action.word];
      const vectors = { ...state.vectors, [action.word]: action.vector };
      const min = state.config.minWords ?? 3;
      // Questions only become meaningful once the set is big enough.
      const questions =
        words.length >= min
          ? buildQuestions(words, vectors, state.config.disagreementRounds ?? 0)
          : [];
      return bump({ words, vectors, questions });
    }

    case 'REMOVE_WORD': {
      if (!state.words.includes(action.word)) return state;
      const words = state.words.filter((w) => w !== action.word);
      const min = state.config.minWords ?? 3;
      const questions =
        words.length >= min
          ? buildQuestions(words, state.vectors, state.config.disagreementRounds ?? 0)
          : [];
      return bump({ words, questions });
    }

    case 'ANSWER_DISAGREEMENT': {
      const question = state.questions[action.questionIndex];
      if (!question) return state;
      const questions = [...state.questions];
      questions[action.questionIndex] = { ...question, answer: action.value };
      return bump({ questions });
    }

    case 'RESET':
      return initState(state.config, state.rules, { vectors: state.vectors });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: SimilarityRankState): ScoreResult {
  switch (state.mode) {
    case 'rank': {
      if (state.trueOrder.length < 2) {
        return scoreLevel({ metric: 'rankCorrelation', value: 0, rules: state.rules });
      }
      // Compare the position each word holds in the player's list against the
      // position it holds in the model's real similarity ordering.
      const truePosition = new Map(state.trueOrder.map((w, i) => [w, i]));
      const userRanks = state.ordering.map((_, i) => i);
      const trueRanks = state.ordering.map((w) => truePosition.get(w) ?? 0);

      return scoreLevel({
        metric: 'rankCorrelation',
        value: spearmanCorrelation(userRanks, trueRanks),
        rules: state.rules,
        breakdown: { items: state.ordering.length },
      });
    }

    case 'odd-one-out': {
      const total = state.sets.length;
      if (total === 0) {
        return scoreLevel({ metric: 'oddOneOutAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.sets.filter((s) => s.answer !== null && s.answer === s.trueOutlier).length;
      return scoreLevel({
        metric: 'oddOneOutAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'free-set': {
      const total = state.questions.length;
      if (total === 0) {
        return scoreLevel({
          metric: 'metricDisagreementAccuracy',
          value: 0,
          rules: state.rules,
          breakdown: { answered: 0, total: 0 },
        });
      }
      const correct = state.questions.filter(
        (q) => q.answer !== null && q.answer === q.trueDisagreement
      ).length;
      const answered = state.questions.filter((q) => q.answer !== null).length;
      return scoreLevel({
        metric: 'metricDisagreementAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, answered, total },
      });
    }
  }
}
