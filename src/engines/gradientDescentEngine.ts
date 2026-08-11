/**
 * Chapter 2.3 — Gradient Descent.
 *
 * Gradients are analytic and exact, and each step uses only the slope under the
 * current point — which is why a bad learning rate really does oscillate and a
 * local basin really does trap you. The optional stretch level swaps the
 * analytic surface for an injected one computed from the live-trained net.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { scoreLevel } from './scoringEngine';
import { clamp, createRng } from './shared';

export type SurfaceKind = 'quadratic' | 'ravine' | 'multi-basin' | 'liveNet';

export interface GradientDescentConfig {
  surface: SurfaceKind;
  surfaceSource?: Record<string, unknown>;
  startPoint: number[];
  learningRate: number;
  learningRateRange?: [number, number];
  momentum?: number;
  momentumRange?: [number, number];
  noiseScale?: number;
  noiseRange?: [number, number];
  seed?: number;
  maxSteps: number;
  convergenceEpsilon: number;
  globalMinimumValue?: number;
}

/** A surface supplied from outside — used for the real live-net loss slice. */
export type SurfaceFn = (x: number, y: number) => number;

export interface TrajectoryPoint {
  step: number;
  x: number;
  y: number;
  loss: number;
  gradientNorm: number;
}

export interface GradientDescentState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  config: GradientDescentConfig;
  position: number[];
  velocity: number[];
  learningRate: number;
  momentum: number;
  noiseScale: number;
  steps: number;
  converged: boolean;
  convergedAtStep: number | null;
  history: TrajectoryPoint[];
  surfaceFn: SurfaceFn | null;
}

export type GradientDescentAction =
  | { type: 'STEP' }
  | { type: 'RUN'; steps: number }
  | { type: 'SET_LEARNING_RATE'; value: number }
  | { type: 'SET_MOMENTUM'; value: number }
  | { type: 'SET_NOISE'; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Analytic surfaces. Each has a matching hand-derived gradient below. */
export function surfaceValue(surface: SurfaceKind, x: number, y: number): number {
  switch (surface) {
    case 'quadratic':
      return x * x + y * y;
    case 'ravine':
      // Steep in y, nearly flat in x — the classic ill-conditioned valley.
      return 0.05 * x * x + 5 * y * y;
    case 'multi-basin': {
      const deep = Math.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2));
      const shallow = Math.exp(-((x - 1.5) ** 2 + (y - 1.5) ** 2));
      return -2 * deep - 1.2 * shallow + 0.02 * (x * x + y * y);
    }
    case 'liveNet':
      // Replaced by the injected surface; flat if none was supplied.
      return 0;
  }
}

export function surfaceGradient(surface: SurfaceKind, x: number, y: number): [number, number] {
  switch (surface) {
    case 'quadratic':
      return [2 * x, 2 * y];
    case 'ravine':
      return [0.1 * x, 10 * y];
    case 'multi-basin': {
      const deep = Math.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2));
      const shallow = Math.exp(-((x - 1.5) ** 2 + (y - 1.5) ** 2));
      return [
        4 * (x + 1.5) * deep + 2.4 * (x - 1.5) * shallow + 0.04 * x,
        4 * (y + 1.5) * deep + 2.4 * (y - 1.5) * shallow + 0.04 * y,
      ];
    }
    case 'liveNet':
      return [0, 0];
  }
}

/** Central differences, used when the surface comes from a live model. */
function numericalGradient(fn: SurfaceFn, x: number, y: number): [number, number] {
  const h = 1e-4;
  return [
    (fn(x + h, y) - fn(x - h, y)) / (2 * h),
    (fn(x, y + h) - fn(x, y - h)) / (2 * h),
  ];
}

function valueAt(state: GradientDescentState, x: number, y: number): number {
  if (state.surfaceFn) return state.surfaceFn(x, y);
  return surfaceValue(state.config.surface, x, y);
}

function gradientAt(state: GradientDescentState, x: number, y: number): [number, number] {
  if (state.surfaceFn) return numericalGradient(state.surfaceFn, x, y);
  return surfaceGradient(state.config.surface, x, y);
}

export function initState(
  config: GradientDescentConfig,
  rules: EngineRules,
  deps: { surfaceFn?: SurfaceFn } = {}
): GradientDescentState {
  const x = config.startPoint[0] ?? 0;
  const y = config.startPoint[1] ?? 0;
  const surfaceFn = deps.surfaceFn ?? null;

  const base: GradientDescentState = {
    rules,
    status: 'idle',
    actionCount: 0,
    config,
    position: [x, y],
    velocity: [0, 0],
    learningRate: config.learningRate,
    momentum: config.momentum ?? 0,
    noiseScale: config.noiseScale ?? 0,
    steps: 0,
    converged: false,
    convergedAtStep: null,
    history: [],
    surfaceFn,
  };

  const [gx, gy] = gradientAt(base, x, y);
  base.history = [
    { step: 0, x, y, loss: valueAt(base, x, y), gradientNorm: Math.hypot(gx, gy) },
  ];
  return base;
}

function step(state: GradientDescentState): GradientDescentState {
  if (state.steps >= state.config.maxSteps) return state;

  const [x, y] = [state.position[0]!, state.position[1]!];
  if (!Number.isFinite(x) || !Number.isFinite(y)) return state;

  const [gx, gy] = gradientAt(state, x, y);

  // Optional gradient noise, seeded so a run replays exactly.
  let nx = 0;
  let ny = 0;
  if (state.noiseScale > 0) {
    const rng = createRng((state.config.seed ?? 1) * 1000 + state.steps);
    nx = (rng() * 2 - 1) * state.noiseScale;
    ny = (rng() * 2 - 1) * state.noiseScale;
  }

  const vx = state.momentum * state.velocity[0]! - state.learningRate * (gx + nx);
  const vy = state.momentum * state.velocity[1]! - state.learningRate * (gy + ny);
  const px = x + vx;
  const py = y + vy;

  const steps = state.steps + 1;
  const gradientNorm = Math.hypot(gx, gy);
  const converged = state.converged || gradientNorm < state.config.convergenceEpsilon;

  return {
    ...state,
    position: [px, py],
    velocity: [vx, vy],
    steps,
    converged,
    convergedAtStep:
      state.convergedAtStep ?? (gradientNorm < state.config.convergenceEpsilon ? steps : null),
    history: [
      ...state.history,
      { step: steps, x: px, y: py, loss: valueAt(state, px, py), gradientNorm },
    ],
    status: 'active',
  };
}

export function applyAction(
  state: GradientDescentState,
  action: GradientDescentAction
): GradientDescentState {
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
      return { ...state, learningRate: clamp(action.value, min, max), actionCount: state.actionCount + 1 };
    }

    case 'SET_MOMENTUM': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.momentumRange ?? [0, 0.99];
      return { ...state, momentum: clamp(action.value, min, max), actionCount: state.actionCount + 1 };
    }

    case 'SET_NOISE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.noiseRange ?? [0, 0];
      return { ...state, noiseScale: clamp(action.value, min, max), actionCount: state.actionCount + 1 };
    }

    case 'RESET':
      return initState(state.config, state.rules, { surfaceFn: state.surfaceFn ?? undefined });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: GradientDescentState): ScoreResult {
  const [x, y] = [state.position[0]!, state.position[1]!];
  const loss = valueAt(state, x, y);
  const metric = state.rules.passCriteria.metric;

  if (metric === 'stepsToConverge') {
    return scoreLevel({
      metric,
      value: state.convergedAtStep ?? state.config.maxSteps,
      rules: state.rules,
      breakdown: { finalLoss: loss, steps: state.steps },
    });
  }

  return scoreLevel({
    metric: 'finalLoss',
    value: loss,
    rules: state.rules,
    breakdown: {
      steps: state.steps,
      x,
      y,
      converged: state.converged ? 1 : 0,
    },
  });
}
