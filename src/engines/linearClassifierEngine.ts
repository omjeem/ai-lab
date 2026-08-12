/**
 * Chapter 2.1 — The Perceptron.
 *
 * A genuine perceptron, trained here rather than illustrated. Weights start
 * where the level says and move only when the rule misclassifies a real sample,
 * which is why XOR visibly refuses to converge no matter how long it runs.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { scoreLevel } from './scoringEngine';
import { clamp, createRng } from './shared';

export type DatasetKind = 'linearly-separable' | 'xor';

export interface LinearClassifierConfig {
  dataset: DatasetKind;
  sampleCount: number;
  seed: number;
  learningRate: number;
  learningRateRange?: [number, number];
  initialWeights: number[];
  initialBias: number;
  maxSteps: number;
  noise: number;
  provesLimitation?: boolean;
}

export interface Sample {
  x: number[];
  label: 1 | -1;
}

export interface HistoryEntry {
  step: number;
  accuracy: number;
  weights: number[];
  bias: number;
}

export interface LinearClassifierState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  config: LinearClassifierConfig;
  samples: Sample[];
  weights: number[];
  bias: number;
  learningRate: number;
  steps: number;
  updates: number;
  cursor: number;
  converged: boolean;
  convergedAtStep: number | null;
  history: HistoryEntry[];
}

export type LinearClassifierAction =
  | { type: 'STEP' }
  | { type: 'RUN'; steps: number }
  | { type: 'SET_LEARNING_RATE'; value: number }
  | { type: 'SET_WEIGHTS'; weights: number[]; bias: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Minimum distance from the planting boundary, so separable data really is. */
const MARGIN = 0.15;

export function generateDataset(
  kind: DatasetKind,
  count: number,
  seed: number,
  noise: number
): Sample[] {
  const rng = createRng(seed);
  const samples: Sample[] = [];
  let guard = 0;

  while (samples.length < count && guard < count * 200) {
    guard++;
    const x0 = rng() * 2 - 1;
    const x1 = rng() * 2 - 1;

    if (kind === 'linearly-separable') {
      // Planted boundary x0 = x1, with a margin so a perceptron can converge.
      if (Math.abs(x0 - x1) < MARGIN) continue;
      samples.push({ x: [x0, x1], label: x0 - x1 > 0 ? 1 : -1 });
    } else {
      if (Math.abs(x0) < MARGIN || Math.abs(x1) < MARGIN) continue;
      samples.push({ x: [x0, x1], label: x0 * x1 > 0 ? 1 : -1 });
    }
  }

  // Label noise is applied after planting, using its own draw, so the same seed
  // yields the same points whether or not noise is on.
  if (noise > 0) {
    const noiseRng = createRng(seed * 31 + 7);
    for (const sample of samples) {
      if (noiseRng() < noise) sample.label = sample.label === 1 ? -1 : 1;
    }
  }

  return samples;
}

/** Which side of the boundary a point falls on. Exported so a canvas can draw
 *  the model's own decision rather than re-implementing it and drifting. */
export function predict(weights: readonly number[], bias: number, x: readonly number[]): 1 | -1 {
  let sum = bias;
  for (let i = 0; i < weights.length; i++) sum += weights[i]! * (x[i] ?? 0);
  return sum > 0 ? 1 : -1;
}

export function accuracyOf(
  weights: readonly number[],
  bias: number,
  samples: readonly Sample[]
): number {
  if (samples.length === 0) return 0;
  const correct = samples.filter((s) => predict(weights, bias, s.x) === s.label).length;
  return correct / samples.length;
}

export function initState(
  config: LinearClassifierConfig,
  rules: EngineRules
): LinearClassifierState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    config,
    samples: generateDataset(config.dataset, config.sampleCount, config.seed, config.noise),
    weights: [...config.initialWeights],
    bias: config.initialBias,
    learningRate: config.learningRate,
    steps: 0,
    updates: 0,
    cursor: 0,
    converged: false,
    convergedAtStep: null,
    history: [],
  };
}

/** One perceptron update against the sample under the cursor. */
function step(state: LinearClassifierState): LinearClassifierState {
  if (state.steps >= state.config.maxSteps || state.samples.length === 0) return state;

  const sample = state.samples[state.cursor]!;
  let weights = state.weights;
  let bias = state.bias;
  let updates = state.updates;

  // The rule fires only on a mistake — this is the whole learning algorithm.
  if (sample.label * (dotWithBias(weights, bias, sample.x)) <= 0) {
    weights = weights.map((w, i) => w + state.learningRate * sample.label * (sample.x[i] ?? 0));
    bias = bias + state.learningRate * sample.label;
    updates++;
  }

  const steps = state.steps + 1;
  const accuracy = accuracyOf(weights, bias, state.samples);
  const converged = state.converged || accuracy === 1;

  return {
    ...state,
    weights,
    bias,
    updates,
    steps,
    cursor: (state.cursor + 1) % state.samples.length,
    converged,
    convergedAtStep: state.convergedAtStep ?? (accuracy === 1 ? steps : null),
    history: [...state.history, { step: steps, accuracy, weights: [...weights], bias }],
    status: 'active',
  };
}

function dotWithBias(weights: readonly number[], bias: number, x: readonly number[]): number {
  let sum = bias;
  for (let i = 0; i < weights.length; i++) sum += weights[i]! * (x[i] ?? 0);
  return sum;
}

export function applyAction(
  state: LinearClassifierState,
  action: LinearClassifierAction
): LinearClassifierState {
  switch (action.type) {
    case 'STEP':
      return { ...step(state), actionCount: state.actionCount + 1 };

    case 'RUN': {
      let next = state;
      const budget = Math.min(action.steps, state.config.maxSteps - state.steps);
      for (let i = 0; i < budget; i++) next = step(next);
      return { ...next, actionCount: state.actionCount + 1 };
    }

    case 'SET_LEARNING_RATE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.learningRateRange ?? [0, Infinity];
      return {
        ...state,
        learningRate: clamp(action.value, min, max),
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_WEIGHTS': {
      if (action.weights.length !== state.weights.length) return state;
      if (!action.weights.every(Number.isFinite) || !Number.isFinite(action.bias)) return state;
      const accuracy = accuracyOf(action.weights, action.bias, state.samples);
      return {
        ...state,
        weights: [...action.weights],
        bias: action.bias,
        converged: state.converged || accuracy === 1,
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: LinearClassifierState): ScoreResult {
  const accuracy = accuracyOf(state.weights, state.bias, state.samples);
  const metric = state.rules.passCriteria.metric;

  if (metric === 'convergenceSteps') {
    // Never converging costs the whole budget, which is the honest answer for
    // data a linear boundary cannot separate.
    const value = state.convergedAtStep ?? state.config.maxSteps;
    return scoreLevel({
      metric,
      value,
      rules: state.rules,
      breakdown: { accuracy, updates: state.updates, steps: state.steps },
    });
  }

  return scoreLevel({
    metric: 'classificationAccuracy',
    value: accuracy,
    rules: state.rules,
    breakdown: { updates: state.updates, steps: state.steps, converged: state.converged ? 1 : 0 },
  });
}
