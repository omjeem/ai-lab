import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  projectedUpdates,
  type TrainingDashboardConfig,
} from '@/engines/trainingDashboardEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-3-neural-networks/3-4-training-dynamics.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as TrainingDashboardConfig;

describe('trainingDashboardEngine — diagnose curve', () => {
  const config = configFor(0);

  it('runs one real training per candidate learning rate', () => {
    const state = initState(config, rulesFor(0));
    expect(state.curves).toHaveLength(config.candidateLearningRates!.length);
    for (const curve of state.curves) {
      expect(curve.losses).toHaveLength(config.epochs);
      expect(config.candidateLearningRates).toContain(curve.trueLearningRate);
      expect(curve.answer).toBeNull();
      expect(curve.losses.every(Number.isFinite)).toBe(true);
    }
  });

  it('produces genuinely different curves for different rates', () => {
    const state = initState(config, rulesFor(0));
    const signatures = state.curves.map((c) => c.losses.join(','));
    expect(new Set(signatures).size).toBe(signatures.length);
  });

  it('leaves a tiny learning rate barely moving and a huge one badly behaved', () => {
    const state = initState(config, rulesFor(0));
    const tiny = state.curves.find((c) => c.trueLearningRate === 0.001)!;
    const huge = state.curves.find((c) => c.trueLearningRate === 10)!;

    const improvement = (c: typeof tiny) => c.losses[0]! - c.losses.at(-1)!;
    expect(Math.abs(improvement(tiny))).toBeLessThan(0.2);
    expect(huge.losses.at(-1)).toBeGreaterThan(Math.min(...state.curves.map((c) => c.losses.at(-1)!)));
  });

  it('does not present the curves in candidate order', () => {
    const state = initState(config, rulesFor(0));
    const presented = state.curves.map((c) => c.trueLearningRate);
    expect(presented).not.toEqual(config.candidateLearningRates);
    expect([...presented].sort()).toEqual([...config.candidateLearningRates!].sort());
  });

  it('scores a full correct match at 1', () => {
    let state = initState(config, rulesFor(0));
    for (const curve of state.curves) {
      state = applyAction(state, {
        type: 'MATCH_CURVE',
        curveId: curve.id,
        learningRate: curve.trueLearningRate,
      });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('curveMatchAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('rejects a learning rate that was not a candidate', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'MATCH_CURVE', curveId: state.curves[0]!.id, learningRate: 0.42 });
    expect(state.curves[0]!.answer).toBeNull();
  });

  it('ignores an unknown curve id', () => {
    const state = initState(config, rulesFor(0));
    expect(applyAction(state, { type: 'MATCH_CURVE', curveId: 'nope', learningRate: 0.05 }).curves).toEqual(
      state.curves
    );
  });
});

describe('trainingDashboardEngine — tune run', () => {
  const config = configFor(1);

  it('fails before anything has been trained', () => {
    const state = initState(config, rulesFor(1));
    const result = evaluate(state);
    expect(result.passed).toBe(false);
    expect(result.breakdown.updatesUsed).toBe(0);
  });

  it('records a real run with a loss history', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'RUN' });

    expect(state.lastRun).not.toBeNull();
    expect(state.lastRun!.lossHistory).toHaveLength(config.epochs);
    expect(state.lastRun!.updatesUsed).toBeGreaterThan(0);
    expect(Number.isFinite(evaluate(state).value)).toBe(true);
  });

  it('accepts only the offered batch sizes', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 8 });
    expect(state.batchSize).toBe(8);
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 7 });
    expect(state.batchSize).toBe(8);
  });

  it('refuses to change a locked epoch count', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_EPOCHS', value: 10 });
    expect(state.epochs).toBe(config.epochs);
  });

  it('uses more updates with a smaller batch size', () => {
    let small = initState(config, rulesFor(1));
    small = applyAction(small, { type: 'SET_BATCH_SIZE', value: 4 });

    let large = initState(config, rulesFor(1));
    large = applyAction(large, { type: 'SET_BATCH_SIZE', value: 64 });

    expect(projectedUpdates(small)).toBeGreaterThan(projectedUpdates(large));
  });

  it('can reach the pass threshold with reasonable settings', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 0.3 });
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 8 });
    state = applyAction(state, { type: 'RUN' });

    expect(evaluate(state).value).toBeLessThan(game.levels[1]!.passCriteria.threshold);
  });
});

describe('trainingDashboardEngine — compute budget', () => {
  const config = configFor(2);

  it('charges overspend back into the score', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 4 });
    state = applyAction(state, { type: 'SET_EPOCHS', value: config.maxEpochs! });
    state = applyAction(state, { type: 'RUN' });

    const result = evaluate(state);
    expect(result.breakdown.updatesUsed).toBeGreaterThan(config.updateBudget!);
    expect(result.breakdown.budgetPenalty).toBeGreaterThan(0);
    expect(result.value).toBeGreaterThan(result.breakdown.rawLoss!);
  });

  it('leaves a run inside the budget unpenalised', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 32 });
    state = applyAction(state, { type: 'SET_EPOCHS', value: 200 });
    state = applyAction(state, { type: 'RUN' });

    const result = evaluate(state);
    expect(result.breakdown.updatesUsed).toBeLessThanOrEqual(config.updateBudget!);
    expect(result.breakdown.budgetPenalty).toBe(0);
    expect(result.value).toBeCloseTo(result.breakdown.rawLoss!);
  });

  it('clamps epochs to the configured maximum', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_EPOCHS', value: 99_999 });
    expect(state.epochs).toBe(config.maxEpochs);
  });

  it('discards a previous run when the architecture changes', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'RUN' });
    state = applyAction(state, { type: 'SET_HIDDEN_LAYERS', units: [6, 6] });
    expect(state.lastRun).toBeNull();
    expect(state.architecture).toEqual([2, 6, 6, 1]);
  });

  it('resets back to the configured start', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_BATCH_SIZE', value: 4 });
    state = applyAction(state, { type: 'RUN' });
    state = applyAction(state, { type: 'RESET' });
    expect(state.lastRun).toBeNull();
    expect(state.batchSize).toBe(config.batchSize);
  });
});

describe('trainingDashboardEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as TrainingDashboardConfig;
      let state = initState(config, rulesFor(i));
      if (config.mode !== 'diagnose-curve') state = applyAction(state, { type: 'RUN' });
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
    }
  });
});
