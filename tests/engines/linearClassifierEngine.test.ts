import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  generateDataset,
  accuracyOf,
  type LinearClassifierConfig,
} from '@/engines/linearClassifierEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-2-classical-ml/2-1-perceptron.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as LinearClassifierConfig;

describe('linearClassifierEngine — dataset generation', () => {
  it('is deterministic for a seed', () => {
    const a = generateDataset('linearly-separable', 30, 5, 0);
    const b = generateDataset('linearly-separable', 30, 5, 0);
    expect(a).toEqual(b);
  });

  it('differs across seeds', () => {
    const a = generateDataset('linearly-separable', 30, 1, 0);
    const b = generateDataset('linearly-separable', 30, 2, 0);
    expect(a).not.toEqual(b);
  });

  it('produces the requested sample count with both classes present', () => {
    const data = generateDataset('linearly-separable', 40, 11, 0);
    expect(data).toHaveLength(40);
    expect(data.some((d) => d.label === 1)).toBe(true);
    expect(data.some((d) => d.label === -1)).toBe(true);
  });

  it('makes a noiseless separable set genuinely separable by the planting rule', () => {
    const data = generateDataset('linearly-separable', 60, 3, 0);
    for (const point of data) {
      expect(Math.sign(point.x[0]! - point.x[1]!)).toBe(point.label);
    }
  });

  it('flips roughly the requested fraction of labels as noise', () => {
    const clean = generateDataset('linearly-separable', 200, 7, 0);
    const noisy = generateDataset('linearly-separable', 200, 7, 0.2);
    const flipped = clean.filter((c, i) => c.label !== noisy[i]!.label).length;
    expect(flipped).toBeGreaterThan(20);
    expect(flipped).toBeLessThan(60);
  });

  it('labels XOR by the product of the coordinate signs', () => {
    const data = generateDataset('xor', 60, 4, 0);
    for (const point of data) {
      expect(Math.sign(point.x[0]! * point.x[1]!)).toBe(point.label);
    }
  });
});

describe('linearClassifierEngine — initState', () => {
  it('starts from the configured weights, untrained', () => {
    const config = configFor(0);
    const state = initState(config, rulesFor(0));

    expect(state.weights).toEqual(config.initialWeights);
    expect(state.bias).toBe(config.initialBias);
    expect(state.updates).toBe(0);
    expect(state.steps).toBe(0);
    expect(state.status).toBe('idle');
    expect(state.samples).toHaveLength(config.sampleCount);
    expect(state.converged).toBe(false);
  });

  it('uses the configured learning rate', () => {
    const config = configFor(1);
    expect(initState(config, rulesFor(1)).learningRate).toBe(config.learningRate);
  });
});

describe('linearClassifierEngine — training', () => {
  const config = configFor(0);

  it('leaves the weights alone on a correctly classified sample', () => {
    // Weights that already classify the planting rule perfectly.
    let state = initState({ ...config, initialWeights: [1, -1], initialBias: 0 }, rulesFor(0));
    const before = [...state.weights];
    state = applyAction(state, { type: 'STEP' });

    expect(state.weights).toEqual(before);
    expect(state.updates).toBe(0);
    expect(state.steps).toBe(1);
  });

  it('moves the weights towards a misclassified sample', () => {
    let state = initState({ ...config, initialWeights: [-1, 1], initialBias: 0 }, rulesFor(0));
    const before = [...state.weights];
    state = applyAction(state, { type: 'STEP' });

    expect(state.weights).not.toEqual(before);
    expect(state.updates).toBe(1);
  });

  it('converges on separable data and records when it did', () => {
    let state = initState({ ...config, initialWeights: [0, 0], initialBias: 0 }, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    expect(state.converged).toBe(true);
    expect(state.convergedAtStep).toBeGreaterThan(0);
    expect(accuracyOf(state.weights, state.bias, state.samples)).toBe(1);
  });

  it('cannot separate XOR no matter how long it runs', () => {
    const xorConfig = configFor(2);
    let state = initState(xorConfig, rulesFor(2));
    state = applyAction(state, { type: 'RUN', steps: xorConfig.maxSteps });

    expect(state.converged).toBe(false);
    expect(accuracyOf(state.weights, state.bias, state.samples)).toBeLessThan(0.85);
  });

  it('stops stepping at maxSteps', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps * 3 });
    expect(state.steps).toBeLessThanOrEqual(config.maxSteps);
  });

  it('cycles through the dataset rather than re-reading one sample', () => {
    let state = initState(config, rulesFor(0));
    for (let i = 0; i < config.sampleCount + 2; i++) {
      state = applyAction(state, { type: 'STEP' });
    }
    expect(state.cursor).toBe((config.sampleCount + 2) % config.sampleCount);
  });

  it('clamps the learning rate to the configured range', () => {
    const ranged = configFor(1);
    let state = initState(ranged, rulesFor(1));

    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 99 });
    expect(state.learningRate).toBe(ranged.learningRateRange![1]);
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: -1 });
    expect(state.learningRate).toBe(ranged.learningRateRange![0]);
  });

  it('rejects a non-finite learning rate', () => {
    let state = initState(configFor(1), rulesFor(1));
    const before = state.learningRate;
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: NaN });
    expect(state.learningRate).toBe(before);
  });

  it('accepts manually set weights and re-evaluates from them', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_WEIGHTS', weights: [1, -1], bias: 0 });
    expect(state.weights).toEqual([1, -1]);
    expect(evaluate(state).value).toBe(1);
  });

  it('ignores a weight vector of the wrong arity', () => {
    let state = initState(config, rulesFor(0));
    const before = [...state.weights];
    state = applyAction(state, { type: 'SET_WEIGHTS', weights: [1], bias: 0 });
    expect(state.weights).toEqual(before);
  });

  it('records history for the training visualisation', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: 10 });
    expect(state.history.length).toBeGreaterThan(0);
    for (const entry of state.history) {
      expect(entry.accuracy).toBeGreaterThanOrEqual(0);
      expect(entry.accuracy).toBeLessThanOrEqual(1);
    }
  });

  it('resets back to the configured start', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: 50 });
    state = applyAction(state, { type: 'RESET' });

    expect(state.weights).toEqual(config.initialWeights);
    expect(state.steps).toBe(0);
    expect(state.updates).toBe(0);
    expect(state.history).toEqual([]);
  });
});

describe('linearClassifierEngine — evaluate', () => {
  it('reports accuracy for accuracy-scored levels', () => {
    const config = configFor(0);
    let state = initState({ ...config, initialWeights: [0, 0], initialBias: 0 }, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const result = evaluate(state);
    expect(result.metric).toBe('classificationAccuracy');
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
    expect(result.xpEarned).toBe(config.sampleCount > 0 ? game.levels[0]!.xpReward : 0);
  });

  it('reports the step count for convergence-scored levels', () => {
    const config = configFor(1);
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const result = evaluate(state);
    expect(result.metric).toBe('convergenceSteps');
    expect(result.value).toBeGreaterThan(0);
    expect(result.breakdown.accuracy).toBeGreaterThan(0.5);
  });

  it('charges the full step budget when it never converges', () => {
    const config = { ...configFor(1), dataset: 'xor' as const };
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const result = evaluate(state);
    expect(result.value).toBe(config.maxSteps);
    expect(result.passed).toBe(false);
  });

  it('scores an untrained model at its starting accuracy', () => {
    const config = configFor(0);
    const state = initState(config, rulesFor(0));
    const result = evaluate(state);
    expect(result.value).toBeCloseTo(accuracyOf(config.initialWeights, config.initialBias, state.samples));
  });
});

describe('linearClassifierEngine — level config coverage', () => {
  it('trains and scores every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as LinearClassifierConfig;
      let state = initState(config, rulesFor(i));
      state = applyAction(state, { type: 'RUN', steps: config.maxSteps });
      const result = evaluate(state);

      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
