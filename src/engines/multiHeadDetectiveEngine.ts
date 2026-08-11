/**
 * Chapter 5.3 — Multi-Head Attention.
 *
 * Head behaviour is not authored anywhere. Each head is classified by measuring
 * its real attention matrix — how much mass sits on the previous token, the next
 * token, itself, or the delimiters, and how spread out the rest is. Change the
 * sentence and the classification changes with it.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { AttentionDep, AttentionResult } from './deps';
import { isSpecialToken } from './attentionGuessEngine';
import { scoreLevel } from './scoringEngine';
import { clamp, entropyBits, mean } from './shared';

export type HeadBehaviour = 'previous-token' | 'next-token' | 'self' | 'delimiter' | 'diffuse';
export type MultiHeadMode = 'find-behaviour' | 'classify-all' | 'hunt-dependency';

export interface MultiHeadDetectiveConfig {
  mode: MultiHeadMode;
  sentence: string;
  layer: number;
  layerRange?: [number, number];
  headCount: number;
  targetBehaviour?: HeadBehaviour;
  behaviours: HeadBehaviour[];
  excludeSpecialTokens: boolean;
  rounds?: number;
  confidenceMargin?: number;
  searchAllLayers?: boolean;
  attempts?: number;
  allowUserSentence?: boolean;
}

export interface HeadProfile {
  layer: number;
  head: number;
  /** Measured share of attention mass by pattern. */
  scores: Record<HeadBehaviour, number>;
  behaviour: HeadBehaviour;
  /** Gap between the top two scores — how clear-cut the classification is. */
  confidence: number;
  answer: HeadBehaviour | null;
}

export interface DependencyAttempt {
  layer: number;
  head: number;
  weight: number;
}

export interface PreparedMultiHeadData {
  result: AttentionResult;
}

export interface MultiHeadDetectiveState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: MultiHeadMode;
  config: MultiHeadDetectiveConfig;
  layer: number;
  tokens: string[];
  result: AttentionResult;
  heads: HeadProfile[];
  /** find-behaviour: which head the player nominated per round. */
  nominations: (number | null)[];
  targetHeads: number[];
  dependency: { fromIndex: number; toIndex: number } | null;
  attempts: DependencyAttempt[];
  bestDependencyWeight: number;
}

export type MultiHeadDetectiveAction =
  | { type: 'NOMINATE_HEAD'; roundIndex: number; head: number }
  | { type: 'LABEL_HEAD'; head: number; behaviour: HeadBehaviour }
  | { type: 'SET_LAYER'; value: number }
  | { type: 'SET_DEPENDENCY'; fromIndex: number; toIndex: number }
  | { type: 'PROBE_HEAD'; layer: number; head: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/**
 * Measures what one real attention head does.
 *
 * Every number below is an average over the head's own attention matrix, so the
 * "correct" label is a property of the model rather than of this file.
 */
export function profileHead(
  result: AttentionResult,
  layer: number,
  head: number
): { scores: Record<HeadBehaviour, number>; behaviour: HeadBehaviour; confidence: number } {
  const matrix = result.attention[layer]?.[head] ?? [];
  const n = result.tokens.length;

  const empty: Record<HeadBehaviour, number> = {
    'previous-token': 0,
    'next-token': 0,
    self: 0,
    delimiter: 0,
    diffuse: 0,
  };
  if (matrix.length === 0 || n === 0) {
    return { scores: empty, behaviour: 'diffuse', confidence: 0 };
  }

  const delimiterIndices = result.tokens
    .map((token, i) => (isSpecialToken(token) ? i : -1))
    .filter((i) => i >= 0);

  const previous: number[] = [];
  const next: number[] = [];
  const self: number[] = [];
  const delimiter: number[] = [];
  const entropies: number[] = [];

  for (let query = 0; query < matrix.length; query++) {
    const row = matrix[query] ?? [];
    if (row.length === 0) continue;
    if (query > 0) previous.push(row[query - 1] ?? 0);
    if (query < row.length - 1) next.push(row[query + 1] ?? 0);
    self.push(row[query] ?? 0);
    delimiter.push(delimiterIndices.reduce((sum, i) => sum + (row[i] ?? 0), 0));
    entropies.push(entropyBits(row));
  }

  // Diffuseness is entropy normalised by the maximum possible for this width.
  const maxEntropy = Math.log2(Math.max(n, 2));
  const scores: Record<HeadBehaviour, number> = {
    'previous-token': mean(previous),
    'next-token': mean(next),
    self: mean(self),
    delimiter: mean(delimiter),
    diffuse: mean(entropies) / maxEntropy,
  };

  const ranked = (Object.entries(scores) as [HeadBehaviour, number][]).sort((a, b) => b[1] - a[1]);
  return {
    scores,
    behaviour: ranked[0]![0],
    confidence: (ranked[0]?.[1] ?? 0) - (ranked[1]?.[1] ?? 0),
  };
}

export async function prepare(
  config: MultiHeadDetectiveConfig,
  deps: { attention: AttentionDep }
): Promise<PreparedMultiHeadData> {
  const result = await deps.attention.attention(config.sentence);
  if (result.attention.length === 0) {
    throw new Error(`No attention returned for "${config.sentence}"`);
  }
  return { result };
}

function profileLayer(
  result: AttentionResult,
  layer: number,
  headCount: number
): HeadProfile[] {
  const available = result.attention[layer]?.length ?? 0;
  return Array.from({ length: Math.min(headCount, available) }, (_, head) => {
    const profile = profileHead(result, layer, head);
    return { layer, head, ...profile, answer: null };
  });
}

export function initState(
  config: MultiHeadDetectiveConfig,
  rules: EngineRules,
  prepared: PreparedMultiHeadData
): MultiHeadDetectiveState {
  const heads = profileLayer(prepared.result, config.layer, config.headCount);

  // Rounds ask for a head exhibiting the target behaviour; only heads that
  // genuinely show it can be correct answers.
  const targetHeads = config.targetBehaviour
    ? heads.filter((h) => h.behaviour === config.targetBehaviour).map((h) => h.head)
    : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    layer: config.layer,
    tokens: prepared.result.tokens,
    result: prepared.result,
    heads,
    nominations: new Array<number | null>(config.rounds ?? 0).fill(null),
    targetHeads,
    dependency: null,
    attempts: [],
    bestDependencyWeight: 0,
  };
}

/** Strongest attention between two tokens across every head searched. */
export function bestDependencyAcross(
  result: AttentionResult,
  fromIndex: number,
  toIndex: number,
  layers: number[]
): { layer: number; head: number; weight: number } {
  let best = { layer: layers[0] ?? 0, head: 0, weight: 0 };
  for (const layer of layers) {
    const heads = result.attention[layer] ?? [];
    for (let head = 0; head < heads.length; head++) {
      const weight = heads[head]?.[fromIndex]?.[toIndex] ?? 0;
      if (weight > best.weight) best = { layer, head, weight };
    }
  }
  return best;
}

export function applyAction(
  state: MultiHeadDetectiveState,
  action: MultiHeadDetectiveAction
): MultiHeadDetectiveState {
  const bump = (next: Partial<MultiHeadDetectiveState>): MultiHeadDetectiveState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'NOMINATE_HEAD': {
      if (action.roundIndex < 0 || action.roundIndex >= state.nominations.length) return state;
      if (!state.heads.some((h) => h.head === action.head)) return state;
      const nominations = [...state.nominations];
      nominations[action.roundIndex] = action.head;
      return bump({ nominations });
    }

    case 'LABEL_HEAD': {
      if (!state.config.behaviours.includes(action.behaviour)) return state;
      const index = state.heads.findIndex((h) => h.head === action.head);
      if (index === -1) return state;
      const heads = [...state.heads];
      heads[index] = { ...heads[index]!, answer: action.behaviour };
      return bump({ heads });
    }

    case 'SET_LAYER': {
      const [min, max] = state.config.layerRange ?? [state.config.layer, state.config.layer];
      if (!Number.isInteger(action.value)) return state;
      const layer = Math.round(clamp(action.value, min, max));
      const heads = profileLayer(state.result, layer, state.config.headCount);
      return bump({
        layer,
        heads,
        targetHeads: state.config.targetBehaviour
          ? heads.filter((h) => h.behaviour === state.config.targetBehaviour).map((h) => h.head)
          : [],
      });
    }

    case 'SET_DEPENDENCY': {
      const n = state.tokens.length;
      if (action.fromIndex < 0 || action.fromIndex >= n) return state;
      if (action.toIndex < 0 || action.toIndex >= n) return state;
      // Changing the pair invalidates previous probes.
      return bump({
        dependency: { fromIndex: action.fromIndex, toIndex: action.toIndex },
        attempts: [],
        bestDependencyWeight: 0,
      });
    }

    case 'PROBE_HEAD': {
      if (!state.dependency) return state;
      const max = state.config.attempts ?? Infinity;
      if (state.attempts.length >= max) return state;

      const weight =
        state.result.attention[action.layer]?.[action.head]?.[state.dependency.fromIndex]?.[
          state.dependency.toIndex
        ] ?? 0;

      return bump({
        attempts: [...state.attempts, { layer: action.layer, head: action.head, weight }],
        bestDependencyWeight: Math.max(state.bestDependencyWeight, weight),
      });
    }

    case 'RESET':
      return initState(state.config, state.rules, { result: state.result });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: MultiHeadDetectiveState): ScoreResult {
  switch (state.mode) {
    case 'find-behaviour': {
      const total = state.nominations.length;
      if (total === 0) {
        return scoreLevel({ metric: 'headMatchAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.nominations.filter(
        (head) => head !== null && state.targetHeads.includes(head)
      ).length;
      return scoreLevel({
        metric: 'headMatchAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total, matchingHeads: state.targetHeads.length },
      });
    }

    case 'classify-all': {
      // Only heads whose classification is clear-cut are graded — a head sitting
      // between two behaviours has no defensible answer to mark against.
      const margin = state.config.confidenceMargin ?? 0;
      const gradable = state.heads.filter((h) => h.confidence >= margin);
      if (gradable.length === 0) {
        return scoreLevel({ metric: 'headClassificationAccuracy', value: 0, rules: state.rules });
      }
      const correct = gradable.filter((h) => h.answer === h.behaviour).length;
      return scoreLevel({
        metric: 'headClassificationAccuracy',
        value: correct / gradable.length,
        rules: state.rules,
        breakdown: { correct, gradable: gradable.length, heads: state.heads.length },
      });
    }

    case 'hunt-dependency': {
      if (!state.dependency) {
        return scoreLevel({ metric: 'dependencyHuntScore', value: 0, rules: state.rules });
      }
      const layers = state.config.searchAllLayers
        ? state.result.attention.map((_, i) => i)
        : [state.layer];
      const best = bestDependencyAcross(
        state.result,
        state.dependency.fromIndex,
        state.dependency.toIndex,
        layers
      );

      // Scored against the strongest link that actually exists, so the level
      // rewards finding it rather than hitting an arbitrary number.
      const value = best.weight === 0 ? 0 : clamp(state.bestDependencyWeight / best.weight, 0, 1);
      return scoreLevel({
        metric: 'dependencyHuntScore',
        value,
        rules: state.rules,
        breakdown: {
          bestFound: state.bestDependencyWeight,
          bestPossible: best.weight,
          bestLayer: best.layer,
          bestHead: best.head,
          probes: state.attempts.length,
        },
      });
    }
  }
}
