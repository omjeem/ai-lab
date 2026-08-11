import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  currentOutputs,
  neuronOutput,
  preActivation,
  type NeuronTuningConfig,
} from '@/engines/neuronTuningEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-3-neural-networks/3-1-neurons-activations.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as NeuronTuningConfig;

describe('neuron arithmetic', () => {
  it('computes the weighted sum, bias and non-linearity', () => {
    expect(preActivation([2, 3], 1, [1, 2])).toBe(9);
    expect(neuronOutput([2, 3], 1, 'relu', [1, 2])).toBe(9);
    expect(neuronOutput([2, 3], 1, 'relu', [-1, -2])).toBe(0);
    expect(neuronOutput([1], 0, 'tanh', [0])).toBe(0);
  });
});

describe('neuronTuningEngine — match output', () => {
  const config = configFor(0);

  it('starts at the configured parameters with a fixed probe set', () => {
    const state = initState(config, rulesFor(0));
    expect(state.weights).toEqual(config.weights);
    expect(state.bias).toBe(config.bias);
    expect(state.probes).toHaveLength(config.probePoints);
    expect(state.targetOutputs).toHaveLength(config.probePoints);
    expect(state.status).toBe('idle');
  });

  it('generates the same probes every time', () => {
    expect(initState(config, rulesFor(0)).probes).toEqual(initState(config, rulesFor(0)).probes);
  });

  it('scores an exact parameter match at 1', () => {
    let state = initState(config, rulesFor(0));
    config.targetWeights!.forEach((w, i) => {
      state = applyAction(state, { type: 'SET_WEIGHT', index: i, value: w });
    });
    state = applyAction(state, { type: 'SET_BIAS', value: config.targetBias! });

    const result = evaluate(state);
    expect(result.metric).toBe('outputMatchScore');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
    expect(currentOutputs(state)).toEqual(state.targetOutputs);
  });

  it('scores a badly wrong neuron at zero', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_WEIGHT', index: 0, value: -3 });
    state = applyAction(state, { type: 'SET_WEIGHT', index: 1, value: 3 });
    state = applyAction(state, { type: 'SET_BIAS', value: -3 });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('clamps weights and bias to their configured ranges', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_WEIGHT', index: 0, value: 99 });
    expect(state.weights[0]).toBe(config.weightRange[1]);
    state = applyAction(state, { type: 'SET_BIAS', value: -99 });
    expect(state.bias).toBe(config.biasRange[0]);
  });

  it('ignores an out-of-range weight index and non-finite values', () => {
    const state = initState(config, rulesFor(0));
    expect(applyAction(state, { type: 'SET_WEIGHT', index: 9, value: 1 }).weights).toEqual(state.weights);
    expect(applyAction(state, { type: 'SET_WEIGHT', index: 0, value: NaN }).weights).toEqual(state.weights);
  });

  it('does not mutate the previous state', () => {
    const before = initState(config, rulesFor(0));
    const after = applyAction(before, { type: 'SET_BIAS', value: 1.5 });
    expect(before.bias).toBe(config.bias);
    expect(after.bias).toBe(1.5);
  });
});

describe('neuronTuningEngine — identify activation', () => {
  const config = configFor(1);

  it('generates one trace per round, each from a real activation', () => {
    const state = initState(config, rulesFor(1));
    expect(state.rounds).toHaveLength(config.rounds!);
    for (const round of state.rounds) {
      expect(config.candidates).toContain(round.trueActivation);
      expect(round.trace).toHaveLength(config.probePoints);
      expect(round.answer).toBeNull();
    }
  });

  it('produces a trace that matches recomputing that activation directly', () => {
    const state = initState(config, rulesFor(1));
    const round = state.rounds[0]!;
    const recomputed = state.probes.map((p) =>
      neuronOutput(config.weights, config.bias, round.trueActivation, p)
    );
    expect(round.trace).toEqual(recomputed);
  });

  it('scores all-correct identification at 1', () => {
    let state = initState(config, rulesFor(1));
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ANSWER_ACTIVATION', roundIndex: i, value: round.trueActivation });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('activationIdentificationAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('counts unanswered rounds as wrong', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, {
      type: 'ANSWER_ACTIVATION',
      roundIndex: 0,
      value: state.rounds[0]!.trueActivation,
    });
    expect(evaluate(state).value).toBeCloseTo(1 / config.rounds!);
  });

  it('rejects an activation that was not offered', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'ANSWER_ACTIVATION', roundIndex: 0, value: 'linear' });
    expect(state.rounds[0]!.answer).toBeNull();
  });

  it('ignores an out-of-range round index', () => {
    const state = initState(config, rulesFor(1));
    expect(applyAction(state, { type: 'ANSWER_ACTIVATION', roundIndex: 99, value: 'relu' }).rounds).toEqual(
      state.rounds
    );
  });
});

describe('neuronTuningEngine — dead neuron', () => {
  const config = configFor(2);

  it('starts genuinely dead', () => {
    const state = initState(config, rulesFor(2));
    const result = evaluate(state);
    expect(result.metric).toBe('activeFraction');
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('revives when the bias is raised enough', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_WEIGHT', index: 0, value: 1 });
    state = applyAction(state, { type: 'SET_WEIGHT', index: 1, value: 1 });
    state = applyAction(state, { type: 'SET_BIAS', value: 1 });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(config.minActiveFraction!);
    expect(result.passed).toBe(true);
  });

  it('refuses to let the activation be swapped, which would dodge the lesson', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_ACTIVATION', value: 'leakyRelu' });
    expect(state.activation).toBe('relu');
  });

  it('measures activity on the pre-activation, not the clipped output', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_BIAS', value: 3 });
    const active = state.probes.filter((p) => preActivation(state.weights, state.bias, p) > 0).length;
    expect(evaluate(state).breakdown.active).toBe(active);
  });
});

describe('neuronTuningEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as NeuronTuningConfig;
      const state = initState(config, rulesFor(i));
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
