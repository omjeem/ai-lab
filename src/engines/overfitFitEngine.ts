/**
 * Chapter 2.4 — Overfitting & Regularization.
 *
 * Real ridge regression, solved by Gaussian elimination on the normal
 * equations, refitted every time a control moves. Training and validation loss
 * are measured on genuinely disjoint splits, which is the only reason the
 * chapter's point lands.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { scoreLevel } from './scoringEngine';
import { clamp, createRng, mean } from './shared';

export type TrueFunction = 'cubic' | 'sine';

export interface OverfitFitConfig {
  trueFunction: TrueFunction;
  sampleCount: number;
  noiseSigma: number;
  seed: number;
  trainSplit: number;
  degree: number;
  degreeRange?: [number, number];
  degreeLocked?: boolean;
  lambda: number;
  lambdaRange?: [number, number];
  /**
   * Ceiling on validation loss for gap-scored levels. Without it, flattening the
   * fit into a useless constant would score a near-zero generalisation gap.
   */
  maxValidationLoss?: number;
}

export interface OverfitFitState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  config: OverfitFitConfig;
  degree: number;
  lambda: number;
  trainSet: [number, number][];
  validationSet: [number, number][];
  coefficients: number[];
  trainLoss: number;
  validationLoss: number;
  /** Dense sample of the fitted polynomial, for the plot. */
  curve: [number, number][];
}

export type OverfitFitAction =
  | { type: 'SET_DEGREE'; value: number }
  | { type: 'SET_LAMBDA'; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

const CURVE_RESOLUTION = 80;
const X_MIN = -1;
const X_MAX = 1;

function trueValue(fn: TrueFunction, x: number): number {
  return fn === 'cubic' ? 2 * x ** 3 - x : Math.sin(Math.PI * x);
}

/** Box-Muller, so the noise is genuinely Gaussian rather than uniform. */
function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function generateData(config: OverfitFitConfig): [number, number][] {
  const rng = createRng(config.seed);
  const points: [number, number][] = [];
  for (let i = 0; i < config.sampleCount; i++) {
    const x = X_MIN + (X_MAX - X_MIN) * rng();
    const y = trueValue(config.trueFunction, x) + gaussian(rng) * config.noiseSigma;
    points.push([x, y]);
  }
  return points.sort((a, b) => a[0] - b[0]);
}

/**
 * Ridge regression by the normal equations: (XᵀX + λI)β = Xᵀy.
 * The intercept term is deliberately left unpenalised.
 */
export function fitRidge(
  points: readonly [number, number][],
  degree: number,
  lambda: number
): number[] {
  const width = degree + 1;
  const xtx: number[][] = Array.from({ length: width }, () => new Array<number>(width).fill(0));
  const xty = new Array<number>(width).fill(0);

  for (const [x, y] of points) {
    const powers = new Array<number>(width);
    powers[0] = 1;
    for (let d = 1; d < width; d++) powers[d] = powers[d - 1]! * x;

    for (let i = 0; i < width; i++) {
      xty[i]! += powers[i]! * y;
      for (let j = 0; j < width; j++) xtx[i]![j]! += powers[i]! * powers[j]!;
    }
  }

  for (let i = 1; i < width; i++) xtx[i]![i]! += lambda;
  // A hair of Tikhonov on every diagonal keeps the solve stable when the design
  // matrix is rank-deficient (more coefficients than distinct points).
  for (let i = 0; i < width; i++) xtx[i]![i]! += 1e-9;

  return solveLinearSystem(xtx, xty);
}

/** Gaussian elimination with partial pivoting. */
function solveLinearSystem(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row]![col]!) > Math.abs(m[pivot]![col]!)) pivot = row;
    }
    if (Math.abs(m[pivot]![col]!) < 1e-14) continue;
    [m[col], m[pivot]] = [m[pivot]!, m[col]!];

    const p = m[col]![col]!;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = m[row]![col]! / p;
      if (factor === 0) continue;
      for (let k = col; k <= n; k++) m[row]![k]! -= factor * m[col]![k]!;
    }
  }

  return Array.from({ length: n }, (_, i) => {
    const d = m[i]![i]!;
    return Math.abs(d) < 1e-14 ? 0 : m[i]![n]! / d;
  });
}

function predict(coefficients: readonly number[], x: number): number {
  let y = 0;
  let power = 1;
  for (const c of coefficients) {
    y += c * power;
    power *= x;
  }
  return y;
}

function meanSquaredError(points: readonly [number, number][], coefficients: readonly number[]): number {
  if (points.length === 0) return 0;
  return mean(points.map(([x, y]) => (y - predict(coefficients, x)) ** 2));
}

function refit(state: OverfitFitState): OverfitFitState {
  const coefficients = fitRidge(state.trainSet, state.degree, state.lambda);
  const curve: [number, number][] = Array.from({ length: CURVE_RESOLUTION }, (_, i) => {
    const x = X_MIN + ((X_MAX - X_MIN) * i) / (CURVE_RESOLUTION - 1);
    return [x, predict(coefficients, x)];
  });

  return {
    ...state,
    coefficients,
    trainLoss: meanSquaredError(state.trainSet, coefficients),
    validationLoss: meanSquaredError(state.validationSet, coefficients),
    curve,
  };
}

export function initState(config: OverfitFitConfig, rules: EngineRules): OverfitFitState {
  const points = generateData(config);
  // Deterministic interleaved split: every k-th point is held out, so both sets
  // span the same x range and validation loss stays a fair test.
  const splitEvery = Math.max(2, Math.round(1 / Math.max(1 - config.trainSplit, 1e-6)));
  const trainSet: [number, number][] = [];
  const validationSet: [number, number][] = [];
  points.forEach((point, i) => {
    (i % splitEvery === splitEvery - 1 ? validationSet : trainSet).push(point);
  });

  return refit({
    rules,
    status: 'idle',
    actionCount: 0,
    config,
    degree: config.degree,
    lambda: config.lambda,
    trainSet,
    validationSet: validationSet.length > 0 ? validationSet : trainSet.slice(-1),
    coefficients: [],
    trainLoss: 0,
    validationLoss: 0,
    curve: [],
  });
}

export function applyAction(state: OverfitFitState, action: OverfitFitAction): OverfitFitState {
  switch (action.type) {
    case 'SET_DEGREE': {
      if (state.config.degreeLocked) return state;
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.degreeRange ?? [1, 12];
      const degree = Math.round(clamp(action.value, min, max));
      return refit({ ...state, degree, status: 'active', actionCount: state.actionCount + 1 });
    }

    case 'SET_LAMBDA': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.lambdaRange ?? [0, 1];
      return refit({
        ...state,
        lambda: clamp(action.value, min, max),
        status: 'active',
        actionCount: state.actionCount + 1,
      });
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: OverfitFitState): ScoreResult {
  const gap = Math.abs(state.validationLoss - state.trainLoss);
  const breakdown = {
    trainLoss: state.trainLoss,
    validationLoss: state.validationLoss,
    generalizationGap: gap,
    degree: state.degree,
    lambda: state.lambda,
  };

  const metric = state.rules.passCriteria.metric;
  if (metric !== 'generalizationGap') {
    return scoreLevel({ metric, value: state.validationLoss, rules: state.rules, breakdown });
  }

  // Closing the gap by giving up on the fit is not the lesson. Any validation
  // loss above the ceiling is charged back against the gap, so an over-smoothed
  // constant scores worse than a real fit with a modest gap.
  const ceiling = state.config.maxValidationLoss;
  const penalty =
    ceiling !== undefined && state.validationLoss > ceiling
      ? (state.validationLoss - ceiling) * 10
      : 0;

  return scoreLevel({
    metric,
    value: gap + penalty,
    rules: state.rules,
    breakdown: { ...breakdown, usefulnessPenalty: penalty },
  });
}
