/**
 * Chapter 3.3 — Backpropagation.
 *
 * Every gradient the player is scored against is read out of a real backward
 * pass over the trained TinyNet. The vanishing-gradient level measures the
 * actual ratio between first- and last-layer gradient magnitude, which is why
 * swapping sigmoid for a rectifier visibly fixes it.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { TinyNet, type Activation, type EdgeGradient, type TrainingSample } from '@/models/tinyNetTrainer';
import { scoreLevel } from './scoringEngine';
import { generateToyDataset, type ToyDatasetKind } from './toyDatasets';
import { createRng, mean, spearmanCorrelation } from './shared';

export type BackpropMode = 'predict-sign' | 'rank-magnitude' | 'fix-vanishing';
export type WeightDirection = 'increase' | 'decrease';

export interface BackpropVisualConfig {
  dataset: ToyDatasetKind;
  architecture: number[];
  activation: Activation;
  activationOptions?: Activation[];
  seed: number;
  warmupEpochs: number;
  learningRate: number;
  mode: BackpropMode;
  rounds?: number;
  edgesPerRound?: number;
  batchSize: number;
  targetFirstLayerGradientRatio?: number;
}

export interface SignRound {
  edge: EdgeGradient;
  /** The direction the update rule will actually move this weight. */
  trueDirection: WeightDirection;
  answer: WeightDirection | null;
}

export interface RankRound {
  edges: EdgeGradient[];
  /** Player's ordering, largest magnitude first. */
  ordering: number[];
}

export interface BackpropVisualState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: BackpropMode;
  config: BackpropVisualConfig;
  architecture: number[];
  activation: Activation;
  net: TinyNet;
  samples: TrainingSample[];
  gradients: EdgeGradient[];
  signRounds: SignRound[];
  rankRounds: RankRound[];
}

export type BackpropVisualAction =
  | { type: 'ANSWER_SIGN'; roundIndex: number; value: WeightDirection }
  | { type: 'SET_ORDER'; roundIndex: number; ordering: number[] }
  | { type: 'SET_ACTIVATION'; value: Activation }
  | { type: 'SET_HIDDEN_LAYERS'; units: number[] }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Trains the warmup epochs and reads the real gradients out of the result. */
function buildNet(
  config: BackpropVisualConfig,
  architecture: number[],
  activation: Activation
): { net: TinyNet; samples: TrainingSample[]; gradients: EdgeGradient[] } {
  const net = new TinyNet({ architecture, activation, seed: config.seed });
  const samples: TrainingSample[] = generateToyDataset(config.dataset, 200, 0.06, config.seed).map(
    (p) => ({ x: p.x, label: p.label })
  );

  for (let epoch = 0; epoch < config.warmupEpochs; epoch++) {
    net.trainEpoch(samples, {
      learningRate: config.learningRate,
      batchSize: config.batchSize,
      epoch,
    });
  }

  // Gradients are taken from one real batch, the same one the player inspects.
  const batch = samples.slice(0, config.batchSize);
  return { net, samples, gradients: net.edgeGradients(batch) };
}

export function initState(
  config: BackpropVisualConfig,
  rules: EngineRules
): BackpropVisualState {
  const { net, samples, gradients } = buildNet(config, config.architecture, config.activation);
  const rng = createRng(config.seed * 17 + 3);

  const signRounds: SignRound[] =
    config.mode === 'predict-sign' && gradients.length > 0
      ? Array.from({ length: config.rounds ?? 0 }, () => {
          const edge = gradients[Math.floor(rng() * gradients.length)]!;
          return {
            edge,
            // w ← w − lr·g, so a positive gradient pushes the weight down.
            trueDirection: edge.value < 0 ? 'increase' : 'decrease',
            answer: null,
          };
        })
      : [];

  const rankRounds: RankRound[] =
    config.mode === 'rank-magnitude' && gradients.length > 0
      ? Array.from({ length: config.rounds ?? 0 }, () => {
          const picked: EdgeGradient[] = [];
          const used = new Set<number>();
          const wanted = Math.min(config.edgesPerRound ?? 4, gradients.length);
          let guard = 0;
          while (picked.length < wanted && guard++ < wanted * 50) {
            const i = Math.floor(rng() * gradients.length);
            if (used.has(i)) continue;
            used.add(i);
            picked.push(gradients[i]!);
          }
          return { edges: picked, ordering: picked.map((_, i) => i) };
        })
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    architecture: [...config.architecture],
    activation: config.activation,
    net,
    samples,
    gradients,
    signRounds,
    rankRounds,
  };
}

export function applyAction(
  state: BackpropVisualState,
  action: BackpropVisualAction
): BackpropVisualState {
  const bump = (next: Partial<BackpropVisualState>): BackpropVisualState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'ANSWER_SIGN': {
      const round = state.signRounds[action.roundIndex];
      if (!round) return state;
      const signRounds = [...state.signRounds];
      signRounds[action.roundIndex] = { ...round, answer: action.value };
      return bump({ signRounds });
    }

    case 'SET_ORDER': {
      const round = state.rankRounds[action.roundIndex];
      if (!round) return state;
      // Must be a permutation of the edges on offer.
      const sorted = [...action.ordering].sort((a, b) => a - b);
      const expected = round.edges.map((_, i) => i);
      if (sorted.length !== expected.length || sorted.some((v, i) => v !== expected[i])) return state;

      const rankRounds = [...state.rankRounds];
      rankRounds[action.roundIndex] = { ...round, ordering: [...action.ordering] };
      return bump({ rankRounds });
    }

    case 'SET_ACTIVATION': {
      if (state.config.activationOptions && !state.config.activationOptions.includes(action.value)) {
        return state;
      }
      const rebuilt = buildNet(state.config, state.architecture, action.value);
      return bump({ activation: action.value, ...rebuilt });
    }

    case 'SET_HIDDEN_LAYERS': {
      const inputs = state.config.architecture[0]!;
      const units = action.units.map((u) => Math.max(1, Math.round(u)));
      if (units.length === 0) return state;
      const architecture = [inputs, ...units, 1];
      const rebuilt = buildNet(state.config, architecture, state.activation);
      return bump({ architecture, ...rebuilt });
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Mean absolute gradient in one weight layer. */
export function layerGradientMagnitude(gradients: readonly EdgeGradient[], layer: number): number {
  const values = gradients.filter((g) => g.layer === layer).map((g) => Math.abs(g.value));
  return values.length === 0 ? 0 : mean(values);
}

export function evaluate(state: BackpropVisualState): ScoreResult {
  switch (state.mode) {
    case 'predict-sign': {
      const total = state.signRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'gradientSignAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.signRounds.filter((r) => r.answer === r.trueDirection).length;
      return scoreLevel({
        metric: 'gradientSignAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'rank-magnitude': {
      const rounds = state.rankRounds.filter((r) => r.edges.length > 1);
      if (rounds.length === 0) {
        return scoreLevel({ metric: 'gradientRankCorrelation', value: 0, rules: state.rules });
      }
      const correlations = rounds.map((round) => {
        // Position in the player's ordering versus true magnitude rank.
        const playerRank = round.edges.map((_, i) => round.ordering.indexOf(i));
        const trueMagnitude = round.edges.map((e) => -Math.abs(e.value));
        return spearmanCorrelation(playerRank, trueMagnitude);
      });
      return scoreLevel({
        metric: 'gradientRankCorrelation',
        value: mean(correlations),
        rules: state.rules,
        breakdown: { rounds: rounds.length },
      });
    }

    case 'fix-vanishing': {
      const lastLayer = state.net.weights.length - 1;
      const first = layerGradientMagnitude(state.gradients, 0);
      const last = layerGradientMagnitude(state.gradients, lastLayer);
      // How much of the output layer's gradient signal survives the trip back
      // to the first layer. Collapses towards zero in a deep sigmoid stack.
      const ratio = last === 0 ? 0 : first / last;
      return scoreLevel({
        metric: 'firstLayerGradientRatio',
        value: ratio,
        rules: state.rules,
        breakdown: { firstLayerMagnitude: first, lastLayerMagnitude: last, layers: lastLayer + 1 },
      });
    }
  }
}
