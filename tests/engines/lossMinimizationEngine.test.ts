import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  computeLoss,
  type LossMinimizationConfig,
} from '@/engines/lossMinimizationEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-2-classical-ml/2-2-loss-functions.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as LossMinimizationConfig;

describe('computeLoss', () => {
  const points: [number, number][] = [
    [0, 0],
    [1, 1],
    [2, 2],
  ];

  it('returns zero for a perfect fit under both regression losses', () => {
    expect(computeLoss('mse', points, 1, 0)).toBeCloseTo(0);
    expect(computeLoss('mae', points, 1, 0)).toBeCloseTo(0);
  });

  it('squares errors under mse and does not under mae', () => {
    const shifted: [number, number][] = [
      [0, 2],
      [1, 3],
      [2, 4],
    ];
    expect(computeLoss('mse', shifted, 1, 0)).toBeCloseTo(4);
    expect(computeLoss('mae', shifted, 1, 0)).toBeCloseTo(2);
  });

  it('punishes a single large outlier far more under mse', () => {
    const withOutlier: [number, number][] = [
      [0, 0],
      [1, 1],
      [2, 12],
    ];
    const mse = computeLoss('mse', withOutlier, 1, 0);
    const mae = computeLoss('mae', withOutlier, 1, 0);
    expect(mse).toBeGreaterThan(mae * 3);
  });

  it('returns zero for an empty point set rather than NaN', () => {
    expect(computeLoss('mse', [], 1, 0)).toBe(0);
  });
});

describe('lossMinimizationEngine — minimize mode', () => {
  const config = configFor(0);

  it('starts at the configured line', () => {
    const state = initState(config, rulesFor(0));
    expect(state.slope).toBe(config.initialSlope);
    expect(state.intercept).toBe(config.initialIntercept);
    expect(state.status).toBe('idle');
  });

  it('recomputes the real loss as the line moves', () => {
    let state = initState(config, rulesFor(0));
    const before = evaluate(state).value;
    state = applyAction(state, { type: 'SET_SLOPE', value: 2 });
    state = applyAction(state, { type: 'SET_INTERCEPT', value: 1 });
    expect(evaluate(state).value).toBeLessThan(before);
  });

  it('scores a near-optimal fit at three stars', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_SLOPE', value: 1.99 });
    state = applyAction(state, { type: 'SET_INTERCEPT', value: 1.04 });

    const result = evaluate(state);
    expect(result.metric).toBe('lossValue');
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('clamps slope and intercept to their configured ranges', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_SLOPE', value: 999 });
    expect(state.slope).toBe(config.slopeRange![1]);
    state = applyAction(state, { type: 'SET_INTERCEPT', value: -999 });
    expect(state.intercept).toBe(config.interceptRange![0]);
  });

  it('rejects non-finite parameters', () => {
    let state = initState(config, rulesFor(0));
    const before = state.slope;
    state = applyAction(state, { type: 'SET_SLOPE', value: NaN });
    expect(state.slope).toBe(before);
  });

  it('reports both losses in the breakdown when a comparison is configured', () => {
    const state = initState(configFor(1), rulesFor(1));
    const result = evaluate(state);
    expect(result.breakdown.mse).toBeGreaterThan(0);
    expect(result.breakdown.mae).toBeGreaterThan(0);
    expect(result.breakdown.mse).not.toBe(result.breakdown.mae);
  });

  it('resets to the configured start', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_SLOPE', value: 3 });
    state = applyAction(state, { type: 'RESET' });
    expect(state.slope).toBe(config.initialSlope);
  });
});

describe('lossMinimizationEngine — match loss mode', () => {
  const config = configFor(2);

  it('starts with no scenario answered', () => {
    const state = initState(config, rulesFor(2));
    expect(Object.keys(state.answers)).toHaveLength(0);
    expect(evaluate(state).value).toBe(0);
  });

  it('scores all-correct answers at 1', () => {
    let state = initState(config, rulesFor(2));
    for (const scenario of config.scenarios!) {
      state = applyAction(state, {
        type: 'ANSWER_SCENARIO',
        scenarioId: scenario.id,
        answer: scenario.answer,
      });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('lossChoiceAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('gives partial credit', () => {
    let state = initState(config, rulesFor(2));
    const scenarios = config.scenarios!;
    state = applyAction(state, {
      type: 'ANSWER_SCENARIO',
      scenarioId: scenarios[0]!.id,
      answer: scenarios[0]!.answer,
    });
    const wrong = config.options!.find((o) => o !== scenarios[1]!.answer)!;
    state = applyAction(state, { type: 'ANSWER_SCENARIO', scenarioId: scenarios[1]!.id, answer: wrong });

    expect(evaluate(state).value).toBeCloseTo(1 / scenarios.length);
  });

  it('rejects an answer that is not one of the offered options', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, {
      type: 'ANSWER_SCENARIO',
      scenarioId: config.scenarios![0]!.id,
      answer: 'huber',
    });
    expect(state.answers[config.scenarios![0]!.id]).toBeUndefined();
  });

  it('ignores an unknown scenario id', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'ANSWER_SCENARIO', scenarioId: 'nope', answer: 'mse' });
    expect(Object.keys(state.answers)).toHaveLength(0);
  });

  it('lets an answer be changed', () => {
    const scenario = config.scenarios![0]!;
    let state = initState(config, rulesFor(2));
    const wrong = config.options!.find((o) => o !== scenario.answer)!;
    state = applyAction(state, { type: 'ANSWER_SCENARIO', scenarioId: scenario.id, answer: wrong });
    state = applyAction(state, { type: 'ANSWER_SCENARIO', scenarioId: scenario.id, answer: scenario.answer });
    expect(state.answers[scenario.id]).toBe(scenario.answer);
  });
});

describe('lossMinimizationEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as LossMinimizationConfig;
      const state = initState(config, rulesFor(i));
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
