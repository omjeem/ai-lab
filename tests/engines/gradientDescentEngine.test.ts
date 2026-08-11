import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  surfaceValue,
  surfaceGradient,
  type GradientDescentConfig,
} from '@/engines/gradientDescentEngine';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-2-classical-ml/2-3-gradient-descent.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as GradientDescentConfig;

describe('gradientDescentEngine — surfaces', () => {
  it('puts the quadratic minimum at the origin', () => {
    expect(surfaceValue('quadratic', 0, 0)).toBeCloseTo(0);
    expect(surfaceValue('quadratic', 1, 1)).toBeGreaterThan(0);
  });

  it('computes analytic gradients that vanish at the quadratic minimum', () => {
    const [gx, gy] = surfaceGradient('quadratic', 0, 0);
    expect(gx).toBeCloseTo(0);
    expect(gy).toBeCloseTo(0);
  });

  it('matches its gradient to a numerical derivative', () => {
    for (const surface of ['quadratic', 'ravine', 'multi-basin'] as const) {
      const x = 0.7;
      const y = -1.3;
      const h = 1e-5;
      const [gx, gy] = surfaceGradient(surface, x, y);
      const nx = (surfaceValue(surface, x + h, y) - surfaceValue(surface, x - h, y)) / (2 * h);
      const ny = (surfaceValue(surface, x, y + h) - surfaceValue(surface, x, y - h)) / (2 * h);
      expect(gx).toBeCloseTo(nx, 4);
      expect(gy).toBeCloseTo(ny, 4);
    }
  });

  it('makes the ravine far steeper in one axis than the other', () => {
    const [gx, gy] = surfaceGradient('ravine', 1, 1);
    expect(Math.abs(gy)).toBeGreaterThan(Math.abs(gx) * 10);
  });

  it('gives the multi-basin surface a deeper global minimum than its local one', () => {
    const global = surfaceValue('multi-basin', -1.5, -1.5);
    const local = surfaceValue('multi-basin', 1.5, 1.5);
    expect(global).toBeLessThan(local);
  });
});

describe('gradientDescentEngine — stepping', () => {
  const config = configFor(0);

  it('starts at the configured point with no steps taken', () => {
    const state = initState(config, rulesFor(0));
    expect(state.position).toEqual(config.startPoint);
    expect(state.steps).toBe(0);
    expect(state.converged).toBe(false);
    expect(state.history).toHaveLength(1);
  });

  it('moves downhill on a step', () => {
    let state = initState(config, rulesFor(0));
    const before = surfaceValue('quadratic', state.position[0]!, state.position[1]!);
    state = applyAction(state, { type: 'STEP' });
    expect(surfaceValue('quadratic', state.position[0]!, state.position[1]!)).toBeLessThan(before);
  });

  it('reaches the minimum and records convergence', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    expect(state.converged).toBe(true);
    expect(state.convergedAtStep).toBeGreaterThan(0);
    expect(evaluate(state).value).toBeLessThan(0.01);
  });

  it('diverges with too large a learning rate instead of pretending to converge', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 1.2 });
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const loss = evaluate(state).value;
    expect(state.converged).toBe(false);
    expect(loss > 1 || !Number.isFinite(loss)).toBe(true);
  });

  it('stops at maxSteps', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps * 5 });
    expect(state.steps).toBeLessThanOrEqual(config.maxSteps);
  });

  it('clamps the learning rate and momentum to their ranges', () => {
    const ravine = configFor(1);
    let state = initState(ravine, rulesFor(1));

    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 99 });
    expect(state.learningRate).toBe(ravine.learningRateRange![1]);
    state = applyAction(state, { type: 'SET_MOMENTUM', value: 5 });
    expect(state.momentum).toBe(ravine.momentumRange![1]);
    state = applyAction(state, { type: 'SET_MOMENTUM', value: -1 });
    expect(state.momentum).toBe(ravine.momentumRange![0]);
  });

  it('rejects non-finite hyperparameters', () => {
    let state = initState(config, rulesFor(0));
    const before = state.learningRate;
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: Infinity });
    expect(state.learningRate).toBe(before);
  });

  it('accumulates velocity when momentum is on', () => {
    const ravine = configFor(1);
    let withMomentum = initState(ravine, rulesFor(1));
    withMomentum = applyAction(withMomentum, { type: 'SET_MOMENTUM', value: 0.9 });
    withMomentum = applyAction(withMomentum, { type: 'RUN', steps: 5 });

    let without = initState(ravine, rulesFor(1));
    without = applyAction(without, { type: 'RUN', steps: 5 });

    expect(withMomentum.position).not.toEqual(without.position);
  });

  it('records a history point per step for the trajectory plot', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: 7 });
    expect(state.history).toHaveLength(8);
    for (const point of state.history) {
      expect(Number.isFinite(point.loss)).toBe(true);
    }
  });

  it('resets to the configured start', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RUN', steps: 20 });
    state = applyAction(state, { type: 'RESET' });
    expect(state.position).toEqual(config.startPoint);
    expect(state.steps).toBe(0);
  });
});

describe('gradientDescentEngine — escaping a local minimum', () => {
  const config = configFor(2);

  it('settles in the nearer, shallower basin with plain descent', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(config.globalMinimumValue!);
    expect(result.passed).toBe(false);
  });

  it('can reach the global basin with enough momentum', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 0.1 });
    state = applyAction(state, { type: 'SET_MOMENTUM', value: 0.95 });
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    expect(evaluate(state).value).toBeLessThan(-1);
  });
});

describe('gradientDescentEngine — live-net surface', () => {
  const config = configFor(3);

  it('uses an injected surface function rather than a built-in formula', () => {
    // A planted bowl offset from the origin: only an injected surface can produce it.
    const injected = (x: number, y: number) => (x - 2) ** 2 + (y + 1) ** 2;
    let state = initState(config, rulesFor(3), { surfaceFn: injected });
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    expect(state.position[0]).toBeCloseTo(2, 1);
    expect(state.position[1]).toBeCloseTo(-1, 1);
    expect(evaluate(state).value).toBeLessThan(0.01);
  });

  it('falls back to a flat surface when no live net is supplied, without crashing', () => {
    let state = initState(config, rulesFor(3));
    state = applyAction(state, { type: 'RUN', steps: 10 });
    expect(Number.isFinite(evaluate(state).value)).toBe(true);
  });
});

describe('gradientDescentEngine — evaluate', () => {
  it('reports steps to converge for convergence-scored levels', () => {
    const config = configFor(1);
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 0.09 });
    state = applyAction(state, { type: 'SET_MOMENTUM', value: 0.9 });
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    const result = evaluate(state);
    expect(result.metric).toBe('stepsToConverge');
    expect(result.value).toBeGreaterThan(0);
    expect(result.value).toBeLessThanOrEqual(config.maxSteps);
  });

  it('charges the full step budget when convergence never happens', () => {
    const config = configFor(1);
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: config.learningRateRange![0] });
    state = applyAction(state, { type: 'RUN', steps: config.maxSteps });

    expect(evaluate(state).value).toBe(config.maxSteps);
  });
});

describe('gradientDescentEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as GradientDescentConfig;
      let state = initState(config, rulesFor(i));
      state = applyAction(state, { type: 'RUN', steps: Math.min(config.maxSteps, 50) });
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value) || Number.isNaN(result.value)).toBe(true);
    }
  });
});
