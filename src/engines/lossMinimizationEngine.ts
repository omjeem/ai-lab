/**
 * Chapter 2.2 — Loss Functions.
 *
 * Model-free by design: a loss is a mathematical statement about what counts as
 * wrong, and it exists independently of any trained model. Every number shown is
 * still computed live from the player's current line.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { scoreLevel } from './scoringEngine';
import { clamp, mean } from './shared';

export type LossType = 'mse' | 'mae' | 'crossEntropy' | 'hinge';
export type LossMode = 'minimize' | 'match-loss';

export interface LossScenario {
  id: string;
  prompt: string;
  answer: LossType;
}

export interface LossMinimizationConfig {
  mode: LossMode;
  lossType?: LossType;
  compareWith?: LossType;
  points?: [number, number][];
  initialSlope?: number;
  initialIntercept?: number;
  slopeRange?: [number, number];
  interceptRange?: [number, number];
  options?: LossType[];
  scenarios?: LossScenario[];
}

export interface LossMinimizationState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: LossMode;
  config: LossMinimizationConfig;
  lossType: LossType;
  slope: number;
  intercept: number;
  answers: Record<string, LossType>;
}

export type LossMinimizationAction =
  | { type: 'SET_SLOPE'; value: number }
  | { type: 'SET_INTERCEPT'; value: number }
  | { type: 'SET_LOSS_TYPE'; value: LossType }
  | { type: 'ANSWER_SCENARIO'; scenarioId: string; answer: string }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Loss of a line against a point set, computed for real on every call. */
export function computeLoss(
  lossType: LossType,
  points: readonly [number, number][],
  slope: number,
  intercept: number
): number {
  if (points.length === 0) return 0;

  const residuals = points.map(([x, y]) => y - (slope * x + intercept));

  switch (lossType) {
    case 'mse':
      return mean(residuals.map((r) => r * r));
    case 'mae':
      return mean(residuals.map((r) => Math.abs(r)));
    case 'hinge':
      // Margin loss on the sign of the residual, for the comparison view.
      return mean(residuals.map((r) => Math.max(0, 1 - Math.abs(r))));
    case 'crossEntropy': {
      // Treats the fitted value as a logit and the target sign as the class,
      // so the shape of the penalty is visible next to the regression losses.
      return mean(
        points.map(([x, y]) => {
          const p = 1 / (1 + Math.exp(-(slope * x + intercept)));
          const target = y > 0 ? 1 : 0;
          const eps = 1e-12;
          return -(target * Math.log(p + eps) + (1 - target) * Math.log(1 - p + eps));
        })
      );
    }
  }
}

export function initState(
  config: LossMinimizationConfig,
  rules: EngineRules
): LossMinimizationState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    lossType: config.lossType ?? 'mse',
    slope: config.initialSlope ?? 0,
    intercept: config.initialIntercept ?? 0,
    answers: {},
  };
}

export function applyAction(
  state: LossMinimizationState,
  action: LossMinimizationAction
): LossMinimizationState {
  const bump = (next: Partial<LossMinimizationState>): LossMinimizationState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_SLOPE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.slopeRange ?? [-Infinity, Infinity];
      return bump({ slope: clamp(action.value, min, max) });
    }

    case 'SET_INTERCEPT': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.interceptRange ?? [-Infinity, Infinity];
      return bump({ intercept: clamp(action.value, min, max) });
    }

    case 'SET_LOSS_TYPE':
      return bump({ lossType: action.value });

    case 'ANSWER_SCENARIO': {
      const scenario = state.config.scenarios?.find((s) => s.id === action.scenarioId);
      if (!scenario) return state;
      // Only options the level actually offered can be recorded.
      if (!state.config.options?.includes(action.answer as LossType)) return state;
      return bump({ answers: { ...state.answers, [action.scenarioId]: action.answer as LossType } });
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: LossMinimizationState): ScoreResult {
  if (state.mode === 'match-loss') {
    const scenarios = state.config.scenarios ?? [];
    if (scenarios.length === 0) {
      return scoreLevel({ metric: 'lossChoiceAccuracy', value: 0, rules: state.rules });
    }
    const correct = scenarios.filter((s) => state.answers[s.id] === s.answer).length;
    return scoreLevel({
      metric: 'lossChoiceAccuracy',
      value: correct / scenarios.length,
      rules: state.rules,
      breakdown: { correct, total: scenarios.length },
    });
  }

  const points = state.config.points ?? [];
  const value = computeLoss(state.lossType, points, state.slope, state.intercept);

  // Both regression losses are always reported so the outlier level can show
  // the two answers side by side without a second evaluation pass.
  const breakdown: Record<string, number> = {
    slope: state.slope,
    intercept: state.intercept,
    mse: computeLoss('mse', points, state.slope, state.intercept),
    mae: computeLoss('mae', points, state.slope, state.intercept),
  };
  if (state.config.compareWith) {
    breakdown.comparison = computeLoss(state.config.compareWith, points, state.slope, state.intercept);
  }

  return scoreLevel({ metric: 'lossValue', value, rules: state.rules, breakdown });
}
