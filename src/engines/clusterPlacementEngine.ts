/**
 * Chapter 1.1 — What is a Vector?
 *
 * Three modes over one shared idea: the player manipulates points whose real
 * positions come from live embeddings projected to 2D. The grouping a player
 * is scored against is discovered by clustering those embeddings, and the label
 * truth is found by embedding the candidate labels — so there is no answer key
 * anywhere in the level JSON.
 */
import { PCA } from 'ml-pca';
import type { EngineRules, ScoreResult } from '@/types/game';
import type { Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, cosineSimilarity, kMeans, norm, silhouetteScore } from './shared';

export type ClusterMode = 'place-and-cluster' | 'guess-the-label' | 'drag-vector-magnitude';

export interface ClusterPlacementConfig {
  mode: ClusterMode;
  words: string[];
  clusterCount: number;
  canvas: { width: number; height: number };
  allowUserWords: boolean;
  labels?: string[];
  magnitudeTolerance?: number;
  rounds?: number;
}

export interface ClusterItem {
  id: string;
  word: string;
  /** Real embedding, projected to 2D by PCA. */
  trueX: number;
  trueY: number;
  /** Cluster index discovered from the real vectors, not supplied by config. */
  trueCluster: number;
  /** L2 norm of the real embedding. */
  trueMagnitude: number;
  placedX: number | null;
  placedY: number | null;
  assignedLabel: string | null;
  magnitudeGuess: number | null;
}

export interface PreparedClusterData {
  items: ClusterItem[];
  /** Cluster index → the label whose own embedding sits nearest that cluster. */
  labelTruth: Record<number, string>;
}

export interface ClusterPlacementState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: ClusterMode;
  config: ClusterPlacementConfig;
  items: ClusterItem[];
  labelTruth: Record<number, string>;
}

export type ClusterPlacementAction =
  | { type: 'PLACE'; id: string; x: number; y: number }
  | { type: 'ASSIGN_LABEL'; id: string; label: string }
  | { type: 'GUESS_MAGNITUDE'; id: string; value: number }
  | { type: 'ADD_ITEM'; item: ClusterItem }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Default tolerance band for magnitude guesses when a level omits one. */
const DEFAULT_MAGNITUDE_TOLERANCE = 0.35;

/**
 * Runs the real embedding model, projects to 2D, and derives the grouping and
 * label truth from the resulting geometry.
 */
export async function prepare(
  config: ClusterPlacementConfig,
  deps: { embedder: Embedder }
): Promise<PreparedClusterData> {
  if (config.words.length === 0) return { items: [], labelTruth: {} };

  const vectors = await deps.embedder.embed(config.words);
  if (vectors.length !== config.words.length) {
    throw new Error(
      `Embedder returned ${vectors.length} vectors for ${config.words.length} words`
    );
  }

  const projected = projectTo2D(vectors);
  const k = Math.max(1, Math.min(config.clusterCount, config.words.length));
  // Cluster in the full space rather than the projection — PCA is for display,
  // and grouping should reflect the geometry the model actually produced.
  const { assignments, centroids } = kMeans(vectors, k, { seed: 1 });

  const items: ClusterItem[] = config.words.map((word, i) => ({
    id: `item-${i}-${word}`,
    word,
    trueX: projected[i]![0]!,
    trueY: projected[i]![1]!,
    trueCluster: assignments[i]!,
    trueMagnitude: norm(vectors[i]!),
    placedX: null,
    placedY: null,
    assignedLabel: null,
    magnitudeGuess: null,
  }));

  const labelTruth = await deriveLabelTruth(config, centroids, deps.embedder);
  return { items, labelTruth };
}

/**
 * Assigns each cluster the label whose own embedding is closest to its centroid,
 * resolving greedily by confidence so two clusters cannot claim the same label.
 */
async function deriveLabelTruth(
  config: ClusterPlacementConfig,
  centroids: number[][],
  embedder: Embedder
): Promise<Record<number, string>> {
  const labels = config.labels ?? [];
  if (labels.length === 0 || centroids.length === 0) return {};

  const labelVectors = await embedder.embed(labels);
  const pairs: { cluster: number; label: string; similarity: number }[] = [];
  for (let c = 0; c < centroids.length; c++) {
    for (let l = 0; l < labels.length; l++) {
      pairs.push({
        cluster: c,
        label: labels[l]!,
        similarity: cosineSimilarity(centroids[c]!, labelVectors[l]!),
      });
    }
  }
  pairs.sort((a, b) => b.similarity - a.similarity);

  const truth: Record<number, string> = {};
  const takenLabels = new Set<string>();
  for (const pair of pairs) {
    if (truth[pair.cluster] !== undefined || takenLabels.has(pair.label)) continue;
    truth[pair.cluster] = pair.label;
    takenLabels.add(pair.label);
  }
  return truth;
}

/** PCA down to two components, with graceful handling of degenerate input. */
function projectTo2D(vectors: number[][]): number[][] {
  if (vectors.length === 0) return [];
  if (vectors.length < 3 || vectors[0]!.length < 2) {
    return vectors.map((v) => [v[0] ?? 0, v[1] ?? 0]);
  }
  try {
    const pca = new PCA(vectors, { center: true });
    return pca.predict(vectors, { nComponents: 2 }).to2DArray();
  } catch {
    // A rank-deficient set (identical vectors) makes PCA unhappy; fall back to
    // the first two raw dimensions rather than failing the whole chapter.
    return vectors.map((v) => [v[0] ?? 0, v[1] ?? 0]);
  }
}

export function initState(
  config: ClusterPlacementConfig,
  rules: EngineRules,
  prepared: PreparedClusterData
): ClusterPlacementState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    items: prepared.items.map((item) => ({ ...item })),
    labelTruth: { ...prepared.labelTruth },
  };
}

export function applyAction(
  state: ClusterPlacementState,
  action: ClusterPlacementAction
): ClusterPlacementState {
  switch (action.type) {
    case 'RESET':
      return {
        ...state,
        status: 'idle',
        actionCount: 0,
        items: state.items.map((item) => ({
          ...item,
          placedX: null,
          placedY: null,
          assignedLabel: null,
          magnitudeGuess: null,
        })),
      };

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };

    case 'ADD_ITEM': {
      if (!state.config.allowUserWords) return state;
      if (state.items.some((i) => i.word === action.item.word)) return state;
      return {
        ...state,
        status: 'active',
        actionCount: state.actionCount + 1,
        items: [...state.items, { ...action.item }],
      };
    }

    case 'PLACE':
    case 'ASSIGN_LABEL':
    case 'GUESS_MAGNITUDE': {
      const index = state.items.findIndex((i) => i.id === action.id);
      if (index === -1) return state;

      const item = state.items[index]!;
      let updated: ClusterItem;

      if (action.type === 'PLACE') {
        updated = {
          ...item,
          placedX: clamp(action.x, 0, state.config.canvas.width),
          placedY: clamp(action.y, 0, state.config.canvas.height),
        };
      } else if (action.type === 'ASSIGN_LABEL') {
        // Silently ignore labels outside the configured set so a malformed UI
        // event cannot invent an option the level never offered.
        if (!state.config.labels?.includes(action.label)) return state;
        updated = { ...item, assignedLabel: action.label };
      } else {
        if (!Number.isFinite(action.value) || action.value < 0) return state;
        updated = { ...item, magnitudeGuess: action.value };
      }

      const items = [...state.items];
      items[index] = updated;
      return { ...state, status: 'active', actionCount: state.actionCount + 1, items };
    }
  }
}

export function evaluate(state: ClusterPlacementState): ScoreResult {
  switch (state.mode) {
    case 'place-and-cluster':
      return evaluatePlacement(state);
    case 'guess-the-label':
      return evaluateLabels(state);
    case 'drag-vector-magnitude':
      return evaluateMagnitude(state);
  }
}

function evaluatePlacement(state: ClusterPlacementState): ScoreResult {
  const placed = state.items.filter((i) => i.placedX !== null && i.placedY !== null);
  const total = state.items.length;
  const breakdown = { placed: placed.length, total };

  if (placed.length < 2 || total === 0) {
    return scoreLevel({ metric: 'clusterSeparationScore', value: 0, rules: state.rules, breakdown });
  }

  // How well does the player's own layout separate the model's real clusters?
  const points = placed.map((i) => [i.placedX!, i.placedY!]);
  const labels = placed.map((i) => i.trueCluster);
  const silhouette = silhouetteScore(points, labels);

  // Silhouette runs [-1, 1]; rescale to [0, 1] and pro-rate by how much of the
  // set was actually placed, so a lucky pair cannot outscore a full layout.
  const separation = (silhouette + 1) / 2;
  const value = clamp(separation * (placed.length / total), 0, 1);

  return {
    ...scoreLevel({ metric: 'clusterSeparationScore', value, rules: state.rules, breakdown }),
    breakdown: { ...breakdown, silhouette },
  };
}

function evaluateLabels(state: ClusterPlacementState): ScoreResult {
  const total = state.items.length;
  if (total === 0) {
    return scoreLevel({ metric: 'labelAccuracy', value: 0, rules: state.rules, breakdown: { total: 0 } });
  }

  let correct = 0;
  for (const item of state.items) {
    if (item.assignedLabel !== null && item.assignedLabel === state.labelTruth[item.trueCluster]) {
      correct++;
    }
  }

  return scoreLevel({
    metric: 'labelAccuracy',
    value: correct / total,
    rules: state.rules,
    breakdown: { correct, total },
  });
}

function evaluateMagnitude(state: ClusterPlacementState): ScoreResult {
  const tolerance = state.config.magnitudeTolerance ?? DEFAULT_MAGNITUDE_TOLERANCE;
  const total = state.items.length;
  if (total === 0) {
    return scoreLevel({ metric: 'magnitudeAccuracy', value: 0, rules: state.rules, breakdown: { total: 0 } });
  }

  let answered = 0;
  let sum = 0;
  for (const item of state.items) {
    if (item.magnitudeGuess === null) continue;
    answered++;
    // Relative error, scored against the tolerance band the level defines.
    const relative =
      item.trueMagnitude === 0
        ? Math.abs(item.magnitudeGuess)
        : Math.abs(item.magnitudeGuess - item.trueMagnitude) / item.trueMagnitude;
    sum += clamp(1 - relative / tolerance, 0, 1);
  }

  return scoreLevel({
    metric: 'magnitudeAccuracy',
    // Unanswered rounds count as zero rather than being dropped from the mean.
    value: sum / total,
    rules: state.rules,
    breakdown: { answered, total },
  });
}
