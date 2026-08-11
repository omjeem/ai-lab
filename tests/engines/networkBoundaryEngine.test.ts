import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  boundaryGrid,
  hiddenRepresentation,
  type NetworkBoundaryConfig,
} from '@/engines/networkBoundaryEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-3-neural-networks/3-2-layers-forward-pass.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as NetworkBoundaryConfig;

describe('networkBoundaryEngine — setup', () => {
  const config = configFor(0);

  it('builds the configured architecture and splits the data', () => {
    const state = initState(config, rulesFor(0));
    expect(state.architecture).toEqual(config.architecture);
    expect(state.epochsTrained).toBe(0);
    expect(state.trainSet.length + state.validationSet.length).toBe(config.sampleCount);
    expect(state.validationSet.length).toBeGreaterThan(0);
    expect(state.history).toEqual([]);
  });

  it('keeps train and validation disjoint', () => {
    const state = initState(config, rulesFor(0));
    for (const point of state.validationSet) expect(state.trainSet).not.toContain(point);
  });
});

describe('networkBoundaryEngine — training', () => {
  const config = configFor(0);

  it('learns the circles dataset to the target accuracy', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 0.3 });
    state = applyAction(state, { type: 'TRAIN', epochs: config.maxEpochs });

    const result = evaluate(state);
    expect(result.metric).toBe('decisionBoundaryAccuracy');
    expect(result.value).toBeGreaterThan(game.levels[0]!.passCriteria.threshold);
    expect(result.passed).toBe(true);
  });

  it('records one history entry per epoch', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'TRAIN', epochs: 12 });
    expect(state.history).toHaveLength(12);
    expect(state.history.at(-1)!.epoch).toBe(12);
    for (const entry of state.history) {
      expect(Number.isFinite(entry.trainLoss)).toBe(true);
      expect(entry.validationAccuracy).toBeGreaterThanOrEqual(0);
      expect(entry.validationAccuracy).toBeLessThanOrEqual(1);
    }
  });

  it('does not mutate the network held by the previous state', () => {
    const before = initState(configFor(0), rulesFor(0));
    const beforeWeights = JSON.stringify(before.net.weights);
    applyAction(before, { type: 'TRAIN', epochs: 30 });
    expect(JSON.stringify(before.net.weights)).toBe(beforeWeights);
  });

  it('stops at maxEpochs', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'TRAIN', epochs: config.maxEpochs * 3 });
    expect(state.epochsTrained).toBe(config.maxEpochs);
  });

  it('discards training when the architecture changes', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'TRAIN', epochs: 20 });
    state = applyAction(state, { type: 'SET_HIDDEN_LAYERS', units: [6, 6] });

    expect(state.epochsTrained).toBe(0);
    expect(state.history).toEqual([]);
    expect(state.architecture).toEqual([2, 6, 6, 1]);
  });

  it('clamps hidden layers to the configured limits', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_HIDDEN_LAYERS', units: [99, 99, 99, 99, 99] });
    const limits = config.architectureRange!;
    expect(state.architecture.length - 2).toBeLessThanOrEqual(limits.maxHidden);
    for (const units of state.architecture.slice(1, -1)) {
      expect(units).toBeLessThanOrEqual(limits.maxUnits);
    }
  });

  it('refuses an activation the level did not offer', () => {
    let state = initState(configFor(2), rulesFor(2));
    const before = state.activation;
    state = applyAction(state, { type: 'SET_ACTIVATION', value: 'sigmoid' });
    expect(state.activation).toBe(before);
  });

  it('accepts an offered activation and rebuilds', () => {
    let state = initState(configFor(2), rulesFor(2));
    state = applyAction(state, { type: 'TRAIN', epochs: 10 });
    state = applyAction(state, { type: 'SET_ACTIVATION', value: 'relu' });
    expect(state.activation).toBe('relu');
    expect(state.epochsTrained).toBe(0);
  });

  it('clamps the learning rate to its range', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 999 });
    expect(state.learningRate).toBe(config.learningRateRange![1]);
  });

  it('resets to an untrained network', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'TRAIN', epochs: 30 });
    state = applyAction(state, { type: 'RESET' });
    expect(state.epochsTrained).toBe(0);
  });
});

describe('networkBoundaryEngine — efficiency scoring', () => {
  const config = configFor(1);

  it('does not penalise a network inside the parameter budget', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'TRAIN', epochs: config.maxEpochs });

    const result = evaluate(state);
    expect(result.metric).toBe('efficiencyScore');
    expect(result.breakdown.parameters).toBeLessThanOrEqual(config.parameterBudget!);
    expect(result.value).toBeCloseTo(result.breakdown.accuracy!);
  });

  it('scales the score down for an oversized network', () => {
    let small = initState(config, rulesFor(1));
    small = applyAction(small, { type: 'TRAIN', epochs: config.maxEpochs });

    let large = initState(config, rulesFor(1));
    large = applyAction(large, { type: 'SET_HIDDEN_LAYERS', units: [8, 8, 8] });
    large = applyAction(large, { type: 'TRAIN', epochs: config.maxEpochs });

    const largeResult = evaluate(large);
    expect(largeResult.breakdown.parameters).toBeGreaterThan(config.parameterBudget!);
    expect(largeResult.value).toBeLessThan(largeResult.breakdown.accuracy!);
    expect(largeResult.value).toBeLessThan(evaluate(small).value);
  });
});

describe('networkBoundaryEngine — visualisation hooks', () => {
  it('samples the real decision surface on a grid', () => {
    const state = initState(configFor(0), rulesFor(0));
    const grid = boundaryGrid(state, 12);
    expect(grid).toHaveLength(12);
    expect(grid[0]).toHaveLength(12);
    for (const value of grid.flat()) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('exposes hidden-layer activations without the input or output layer', () => {
    const state = initState(configFor(2), rulesFor(2));
    const hidden = hiddenRepresentation(state, [0.3, -0.4]);
    expect(hidden).toHaveLength(configFor(2).architecture.length - 2);
    expect(hidden[0]).toHaveLength(configFor(2).architecture[1]);
  });
});

describe('networkBoundaryEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as NetworkBoundaryConfig;
      let state = initState(config, rulesFor(i));
      state = applyAction(state, { type: 'TRAIN', epochs: 20 });
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
