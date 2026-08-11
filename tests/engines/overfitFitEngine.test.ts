import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  fitRidge,
  type OverfitFitConfig,
} from '@/engines/overfitFitEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-2-classical-ml/2-4-overfitting.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as OverfitFitConfig;

describe('fitRidge', () => {
  it('recovers an exact linear relationship', () => {
    const points: [number, number][] = [
      [0, 1],
      [1, 3],
      [2, 5],
    ];
    const coefficients = fitRidge(points, 1, 0);
    expect(coefficients[0]).toBeCloseTo(1, 4);
    expect(coefficients[1]).toBeCloseTo(2, 4);
  });

  it('recovers a quadratic when given enough degree', () => {
    const points: [number, number][] = [
      [-1, 1],
      [0, 0],
      [1, 1],
      [2, 4],
    ];
    const coefficients = fitRidge(points, 2, 0);
    expect(coefficients[2]).toBeCloseTo(1, 3);
  });

  it('shrinks coefficients as the ridge penalty grows', () => {
    const points: [number, number][] = [
      [0, 1],
      [1, 3],
      [2, 5],
      [3, 7],
    ];
    const plain = fitRidge(points, 3, 0);
    const penalised = fitRidge(points, 3, 10);
    const magnitude = (c: number[]) => c.reduce((a, b) => a + Math.abs(b), 0);
    expect(magnitude(penalised)).toBeLessThan(magnitude(plain));
  });

  it('returns a coefficient per degree plus the intercept', () => {
    expect(fitRidge([[0, 0], [1, 1]], 4, 0.1)).toHaveLength(5);
  });

  it('survives fewer points than coefficients thanks to the penalty', () => {
    const coefficients = fitRidge([[0, 1], [1, 2]], 8, 0.01);
    expect(coefficients).toHaveLength(9);
    expect(coefficients.every(Number.isFinite)).toBe(true);
  });
});

describe('overfitFitEngine — data and state', () => {
  const config = configFor(0);

  it('splits data deterministically into train and validation', () => {
    const a = initState(config, rulesFor(0));
    const b = initState(config, rulesFor(0));
    expect(a.trainSet).toEqual(b.trainSet);
    expect(a.validationSet).toEqual(b.validationSet);
    expect(a.trainSet.length + a.validationSet.length).toBe(config.sampleCount);
    expect(a.validationSet.length).toBeGreaterThan(0);
  });

  it('starts at the configured degree and penalty', () => {
    const state = initState(config, rulesFor(0));
    expect(state.degree).toBe(config.degree);
    expect(state.lambda).toBe(config.lambda);
    expect(state.status).toBe('idle');
  });

  it('clamps degree and lambda to their ranges', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_DEGREE', value: 99 });
    expect(state.degree).toBe(config.degreeRange![1]);
    state = applyAction(state, { type: 'SET_LAMBDA', value: -5 });
    expect(state.lambda).toBe(config.lambdaRange![0]);
  });

  it('rounds a fractional degree rather than producing a broken design matrix', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_DEGREE', value: 4.7 });
    expect(Number.isInteger(state.degree)).toBe(true);
  });

  it('refuses to change a locked degree', () => {
    const locked = configFor(1);
    let state = initState(locked, rulesFor(1));
    state = applyAction(state, { type: 'SET_DEGREE', value: 2 });
    expect(state.degree).toBe(locked.degree);
  });

  it('rejects non-finite values', () => {
    let state = initState(config, rulesFor(0));
    const before = state.lambda;
    state = applyAction(state, { type: 'SET_LAMBDA', value: NaN });
    expect(state.lambda).toBe(before);
  });

  it('resets to the configured start', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_DEGREE', value: 9 });
    state = applyAction(state, { type: 'RESET' });
    expect(state.degree).toBe(config.degree);
  });
});

describe('overfitFitEngine — the overfitting story', () => {
  const config = configFor(0);

  it('drives training loss down as degree rises', () => {
    let low = initState(config, rulesFor(0));
    low = applyAction(low, { type: 'SET_DEGREE', value: 1 });
    let high = initState(config, rulesFor(0));
    high = applyAction(high, { type: 'SET_DEGREE', value: 11 });

    expect(evaluate(high).breakdown.trainLoss).toBeLessThan(evaluate(low).breakdown.trainLoss!);
  });

  it('lets validation loss turn back up once capacity outruns the data', () => {
    const losses: number[] = [];
    for (let degree = 1; degree <= 12; degree++) {
      let state = initState(config, rulesFor(0));
      state = applyAction(state, { type: 'SET_DEGREE', value: degree });
      losses.push(evaluate(state).breakdown.validationLoss!);
    }
    const bestIndex = losses.indexOf(Math.min(...losses));
    expect(bestIndex).toBeLessThan(losses.length - 1);
    expect(losses.at(-1)).toBeGreaterThan(losses[bestIndex]!);
  });

  it('recovers a high-degree fit using the ridge penalty alone', () => {
    const locked = configFor(1);
    let unpenalised = initState(locked, rulesFor(1));
    unpenalised = applyAction(unpenalised, { type: 'SET_LAMBDA', value: 0 });

    let penalised = initState(locked, rulesFor(1));
    penalised = applyAction(penalised, { type: 'SET_LAMBDA', value: 0.05 });

    expect(evaluate(penalised).value).toBeLessThan(evaluate(unpenalised).value);
  });

  it('reports both losses and the gap in the breakdown', () => {
    const state = initState(config, rulesFor(0));
    const result = evaluate(state);
    expect(result.breakdown.trainLoss).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.validationLoss).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.generalizationGap).toBeGreaterThanOrEqual(0);
  });

  it('scores the gap metric on the gap level', () => {
    const state = initState(configFor(2), rulesFor(2));
    const result = evaluate(state);
    expect(result.metric).toBe('generalizationGap');
    expect(Number.isFinite(result.value)).toBe(true);
  });

  it('refuses to reward closing the gap by abandoning the fit', () => {
    const gapConfig = configFor(2);
    expect(gapConfig.maxValidationLoss).toBeDefined();

    // Flatten the fit: minimum degree, maximum penalty. The raw gap collapses,
    // but the fit is useless, so the score must not.
    let useless = initState(gapConfig, rulesFor(2));
    useless = applyAction(useless, { type: 'SET_DEGREE', value: 1 });
    useless = applyAction(useless, { type: 'SET_LAMBDA', value: gapConfig.lambdaRange![1] });

    const result = evaluate(useless);
    expect(result.breakdown.validationLoss).toBeGreaterThan(gapConfig.maxValidationLoss!);
    expect(result.breakdown.usefulnessPenalty).toBeGreaterThan(0);
    expect(result.value).toBeGreaterThan(result.breakdown.generalizationGap!);
    expect(result.passed).toBe(false);
  });

  it('leaves a genuinely good fit unpenalised', () => {
    const gapConfig = configFor(2);
    let good = initState(gapConfig, rulesFor(2));
    good = applyAction(good, { type: 'SET_DEGREE', value: 3 });
    good = applyAction(good, { type: 'SET_LAMBDA', value: 0.01 });

    const result = evaluate(good);
    expect(result.breakdown.validationLoss).toBeLessThanOrEqual(gapConfig.maxValidationLoss!);
    expect(result.breakdown.usefulnessPenalty).toBe(0);
  });

  it('exposes a dense curve for plotting the fitted polynomial', () => {
    const state = initState(config, rulesFor(0));
    expect(state.curve.length).toBeGreaterThan(20);
    for (const [x, y] of state.curve) {
      expect(Number.isFinite(x)).toBe(true);
      expect(Number.isFinite(y)).toBe(true);
    }
  });
});

describe('overfitFitEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as OverfitFitConfig;
      const state = initState(config, rulesFor(i));
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
