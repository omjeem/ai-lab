import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  layerGradientMagnitude,
  type BackpropVisualConfig,
} from '@/engines/backpropVisualEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-3-neural-networks/3-3-backpropagation.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as BackpropVisualConfig;

describe('backpropVisualEngine — predict sign', () => {
  const config = configFor(0);

  it('builds one round per configured round from real gradients', () => {
    const state = initState(config, rulesFor(0));
    expect(state.signRounds).toHaveLength(config.rounds!);
    expect(state.gradients.length).toBeGreaterThan(0);
    for (const round of state.signRounds) {
      expect(state.gradients).toContain(round.edge);
      expect(round.answer).toBeNull();
      expect(['increase', 'decrease']).toContain(round.trueDirection);
    }
  });

  it('derives the direction from the real update rule', () => {
    const state = initState(config, rulesFor(0));
    for (const round of state.signRounds) {
      // w ← w − lr·g: a negative gradient increases the weight.
      expect(round.trueDirection).toBe(round.edge.value < 0 ? 'increase' : 'decrease');
    }
  });

  it('is deterministic for a seed', () => {
    const a = initState(config, rulesFor(0));
    const b = initState(config, rulesFor(0));
    expect(a.signRounds.map((r) => r.trueDirection)).toEqual(b.signRounds.map((r) => r.trueDirection));
  });

  it('scores all-correct answers at 1', () => {
    let state = initState(config, rulesFor(0));
    state.signRounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ANSWER_SIGN', roundIndex: i, value: round.trueDirection });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('gradientSignAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores all-wrong answers at 0', () => {
    let state = initState(config, rulesFor(0));
    state.signRounds.forEach((round, i) => {
      state = applyAction(state, {
        type: 'ANSWER_SIGN',
        roundIndex: i,
        value: round.trueDirection === 'increase' ? 'decrease' : 'increase',
      });
    });
    expect(evaluate(state).value).toBe(0);
  });

  it('counts unanswered rounds as wrong', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, {
      type: 'ANSWER_SIGN',
      roundIndex: 0,
      value: state.signRounds[0]!.trueDirection,
    });
    expect(evaluate(state).value).toBeCloseTo(1 / config.rounds!);
  });

  it('ignores an out-of-range round', () => {
    const state = initState(config, rulesFor(0));
    expect(applyAction(state, { type: 'ANSWER_SIGN', roundIndex: 99, value: 'increase' }).signRounds).toEqual(
      state.signRounds
    );
  });
});

describe('backpropVisualEngine — rank magnitude', () => {
  const config = configFor(1);

  it('offers the configured number of distinct edges per round', () => {
    const state = initState(config, rulesFor(1));
    expect(state.rankRounds).toHaveLength(config.rounds!);
    for (const round of state.rankRounds) {
      expect(round.edges).toHaveLength(config.edgesPerRound!);
      const keys = round.edges.map((e) => `${e.layer}:${e.from}:${e.to}`);
      expect(new Set(keys).size).toBe(keys.length);
    }
  });

  it('scores a perfect ordering at 1', () => {
    let state = initState(config, rulesFor(1));
    state.rankRounds.forEach((round, i) => {
      const ordering = round.edges
        .map((edge, index) => ({ index, magnitude: Math.abs(edge.value) }))
        .sort((a, b) => b.magnitude - a.magnitude)
        .map((e) => e.index);
      state = applyAction(state, { type: 'SET_ORDER', roundIndex: i, ordering });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('gradientRankCorrelation');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores a reversed ordering at -1 and fails', () => {
    let state = initState(config, rulesFor(1));
    state.rankRounds.forEach((round, i) => {
      const ordering = round.edges
        .map((edge, index) => ({ index, magnitude: Math.abs(edge.value) }))
        .sort((a, b) => a.magnitude - b.magnitude)
        .map((e) => e.index);
      state = applyAction(state, { type: 'SET_ORDER', roundIndex: i, ordering });
    });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(-1);
    expect(result.passed).toBe(false);
  });

  it('rejects an ordering that is not a permutation', () => {
    let state = initState(config, rulesFor(1));
    const before = [...state.rankRounds[0]!.ordering];
    state = applyAction(state, { type: 'SET_ORDER', roundIndex: 0, ordering: [0, 0, 1] });
    expect(state.rankRounds[0]!.ordering).toEqual(before);
  });
});

describe('backpropVisualEngine — vanishing gradients', () => {
  const config = configFor(2);

  it('measures a genuinely collapsed ratio in the deep sigmoid stack', () => {
    const state = initState(config, rulesFor(2));
    const result = evaluate(state);

    expect(result.metric).toBe('firstLayerGradientRatio');
    expect(result.value).toBeLessThan(config.targetFirstLayerGradientRatio!);
    expect(result.passed).toBe(false);
    expect(result.breakdown.firstLayerMagnitude).toBeLessThan(result.breakdown.lastLayerMagnitude!);
  });

  it('recovers when the activation is swapped for a rectifier', () => {
    const before = initState(config, rulesFor(2));
    const after = applyAction(before, { type: 'SET_ACTIVATION', value: 'relu' });

    expect(evaluate(after).value).toBeGreaterThan(evaluate(before).value);
    expect(evaluate(after).passed).toBe(true);
  });

  it('recovers when the stack is made shallower', () => {
    const before = initState(config, rulesFor(2));
    const after = applyAction(before, { type: 'SET_HIDDEN_LAYERS', units: [6] });

    expect(after.architecture).toEqual([2, 6, 1]);
    expect(evaluate(after).value).toBeGreaterThan(evaluate(before).value);
  });

  it('refuses an activation outside the offered options', () => {
    const state = initState(config, rulesFor(2));
    const next = applyAction(state, { type: 'SET_ACTIVATION', value: 'linear' });
    expect(next.activation).toBe(state.activation);
  });

  it('computes layer magnitude as the mean absolute gradient', () => {
    const state = initState(config, rulesFor(2));
    const layer0 = state.gradients.filter((g) => g.layer === 0);
    const expected = layer0.reduce((a, g) => a + Math.abs(g.value), 0) / layer0.length;
    expect(layerGradientMagnitude(state.gradients, 0)).toBeCloseTo(expected);
  });

  it('returns zero magnitude for a layer with no edges', () => {
    expect(layerGradientMagnitude([], 3)).toBe(0);
  });
});

describe('backpropVisualEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as BackpropVisualConfig;
      const state = initState(config, rulesFor(i));
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
