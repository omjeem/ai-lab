/**
 * Chapter 3.4 — Training Dynamics.
 *
 * Every curve shown here is produced by actually running the training it
 * describes. The diagnose level trains one real network per candidate learning
 * rate before the player answers, and the budget level charges real gradient
 * updates against a real budget.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { TinyNet, type Activation, type TrainingSample } from '@/models/tinyNetTrainer';
import { scoreLevel } from './scoringEngine';
import { generateToyDataset, splitDataset, type ToyDatasetKind, type ToyPoint } from './toyDatasets';
import { clamp, createRng } from './shared';

export type TrainingDashboardMode = 'diagnose-curve' | 'tune-run' | 'budget-run';

export interface TrainingDashboardConfig {
  mode: TrainingDashboardMode;
  dataset: ToyDatasetKind;
  architecture: number[];
  architectureRange?: { maxHidden: number; maxUnits: number };
  activation: Activation;
  activationOptions?: Activation[];
  seed: number;
  candidateLearningRates?: number[];
  learningRate?: number;
  learningRateRange?: [number, number];
  batchSize: number;
  batchSizeOptions?: number[];
  epochs: number;
  epochsLocked?: boolean;
  maxEpochs?: number;
  updateBudget?: number;
  trainSplit: number;
}

export interface LossCurve {
  /** Shuffled display order; the answer is which rate produced it. */
  id: string;
  losses: number[];
  trueLearningRate: number;
  answer: number | null;
}

export interface TrainingDashboardState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: TrainingDashboardMode;
  config: TrainingDashboardConfig;
  architecture: number[];
  activation: Activation;
  learningRate: number;
  batchSize: number;
  epochs: number;
  curves: LossCurve[];
  trainSet: ToyPoint[];
  validationSet: ToyPoint[];
  lastRun: {
    finalValidationLoss: number;
    finalValidationAccuracy: number;
    updatesUsed: number;
    lossHistory: number[];
  } | null;
}

export type TrainingDashboardAction =
  | { type: 'MATCH_CURVE'; curveId: string; learningRate: number }
  | { type: 'SET_LEARNING_RATE'; value: number }
  | { type: 'SET_BATCH_SIZE'; value: number }
  | { type: 'SET_EPOCHS'; value: number }
  | { type: 'SET_ACTIVATION'; value: Activation }
  | { type: 'SET_HIDDEN_LAYERS'; units: number[] }
  | { type: 'RUN' }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

const asSamples = (points: readonly ToyPoint[]): TrainingSample[] =>
  points.map((p) => ({ x: p.x, label: p.label }));

/** Runs one real training and returns its per-epoch loss curve. */
function runTraining(
  config: TrainingDashboardConfig,
  architecture: number[],
  activation: Activation,
  train: readonly ToyPoint[],
  validation: readonly ToyPoint[],
  options: { learningRate: number; batchSize: number; epochs: number }
) {
  const net = new TinyNet({ architecture, activation, seed: config.seed });
  const trainSamples = asSamples(train);
  const validationSamples = asSamples(validation);
  const lossHistory: number[] = [];

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    net.trainEpoch(trainSamples, {
      learningRate: options.learningRate,
      batchSize: options.batchSize,
      epoch,
    });
    const loss = net.loss(validationSamples);
    // A diverged run produces a genuinely huge loss; keep it finite so the
    // curve still plots rather than breaking the chart.
    lossHistory.push(Number.isFinite(loss) ? Math.min(loss, 50) : 50);
  }

  const batchesPerEpoch = Math.max(1, Math.ceil(train.length / Math.max(1, options.batchSize)));
  return {
    net,
    lossHistory,
    finalValidationLoss: lossHistory.at(-1) ?? net.loss(validationSamples),
    finalValidationAccuracy: net.accuracy(validationSamples),
    updatesUsed: batchesPerEpoch * options.epochs,
  };
}

export function initState(
  config: TrainingDashboardConfig,
  rules: EngineRules
): TrainingDashboardState {
  const { train, validation } = splitDataset(
    generateToyDataset(config.dataset, 240, 0.06, config.seed),
    config.trainSplit
  );

  // The diagnose level needs the real runs up front — the player is matching
  // curves that were genuinely produced by those learning rates.
  const curves: LossCurve[] = [];
  if (config.mode === 'diagnose-curve' && config.candidateLearningRates) {
    const runs = config.candidateLearningRates.map((rate) => ({
      rate,
      losses: runTraining(config, config.architecture, config.activation, train, validation, {
        learningRate: rate,
        batchSize: config.batchSize,
        epochs: config.epochs,
      }).lossHistory,
    }));

    // Deterministic shuffle so the display order does not give the answer away.
    const rng = createRng(config.seed * 101 + 7);
    const order = runs.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }

    order.forEach((sourceIndex, displayIndex) => {
      const run = runs[sourceIndex]!;
      curves.push({
        id: `curve-${displayIndex}`,
        losses: run.losses,
        trueLearningRate: run.rate,
        answer: null,
      });
    });
  }

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    architecture: [...config.architecture],
    activation: config.activation,
    learningRate: config.learningRate ?? 0.1,
    batchSize: config.batchSize,
    epochs: config.epochs,
    curves,
    trainSet: train,
    validationSet: validation,
    lastRun: null,
  };
}

export function applyAction(
  state: TrainingDashboardState,
  action: TrainingDashboardAction
): TrainingDashboardState {
  const bump = (next: Partial<TrainingDashboardState>): TrainingDashboardState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'MATCH_CURVE': {
      const index = state.curves.findIndex((c) => c.id === action.curveId);
      if (index === -1) return state;
      if (!state.config.candidateLearningRates?.includes(action.learningRate)) return state;
      const curves = [...state.curves];
      curves[index] = { ...curves[index]!, answer: action.learningRate };
      return bump({ curves });
    }

    case 'SET_LEARNING_RATE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.learningRateRange ?? [0, Infinity];
      return bump({ learningRate: clamp(action.value, min, max) });
    }

    case 'SET_BATCH_SIZE': {
      const options = state.config.batchSizeOptions;
      if (options && !options.includes(action.value)) return state;
      if (!Number.isInteger(action.value) || action.value < 1) return state;
      return bump({ batchSize: action.value });
    }

    case 'SET_EPOCHS': {
      if (state.config.epochsLocked) return state;
      if (!Number.isFinite(action.value)) return state;
      const max = state.config.maxEpochs ?? state.config.epochs;
      return bump({ epochs: Math.round(clamp(action.value, 1, max)) });
    }

    case 'SET_ACTIVATION': {
      const options = state.config.activationOptions;
      if (options && !options.includes(action.value)) return state;
      return bump({ activation: action.value, lastRun: null });
    }

    case 'SET_HIDDEN_LAYERS': {
      const limits = state.config.architectureRange;
      if (!limits) return state;
      const units = action.units
        .slice(0, limits.maxHidden)
        .map((u) => Math.round(clamp(u, 1, limits.maxUnits)));
      if (units.length === 0) return state;
      return bump({
        architecture: [state.config.architecture[0]!, ...units, 1],
        lastRun: null,
      });
    }

    case 'RUN': {
      const run = runTraining(
        state.config,
        state.architecture,
        state.activation,
        state.trainSet,
        state.validationSet,
        { learningRate: state.learningRate, batchSize: state.batchSize, epochs: state.epochs }
      );
      return bump({
        lastRun: {
          finalValidationLoss: run.finalValidationLoss,
          finalValidationAccuracy: run.finalValidationAccuracy,
          updatesUsed: run.updatesUsed,
          lossHistory: run.lossHistory,
        },
      });
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Updates the current settings would consume, so the UI can warn before a run. */
export function projectedUpdates(state: TrainingDashboardState): number {
  const batches = Math.max(1, Math.ceil(state.trainSet.length / Math.max(1, state.batchSize)));
  return batches * state.epochs;
}

export function evaluate(state: TrainingDashboardState): ScoreResult {
  if (state.mode === 'diagnose-curve') {
    const total = state.curves.length;
    if (total === 0) {
      return scoreLevel({ metric: 'curveMatchAccuracy', value: 0, rules: state.rules });
    }
    const correct = state.curves.filter((c) => c.answer === c.trueLearningRate).length;
    return scoreLevel({
      metric: 'curveMatchAccuracy',
      value: correct / total,
      rules: state.rules,
      breakdown: { correct, total },
    });
  }

  const run = state.lastRun;
  if (!run) {
    // Nothing trained yet: report a loss too high to pass rather than zero,
    // which would read as a perfect score on an lte metric.
    return scoreLevel({
      metric: 'finalValidationLoss',
      value: Number.POSITIVE_INFINITY,
      rules: state.rules,
      breakdown: { updatesUsed: 0 },
    });
  }

  const budget = state.config.updateBudget;
  // Overspending the compute budget is charged back into the loss, so the
  // level cannot be won by simply training for longer than allowed.
  const penalty =
    budget !== undefined && run.updatesUsed > budget
      ? ((run.updatesUsed - budget) / budget) * 0.5
      : 0;

  return scoreLevel({
    metric: 'finalValidationLoss',
    value: run.finalValidationLoss + penalty,
    rules: state.rules,
    breakdown: {
      rawLoss: run.finalValidationLoss,
      accuracy: run.finalValidationAccuracy,
      updatesUsed: run.updatesUsed,
      budgetPenalty: penalty,
    },
  });
}
