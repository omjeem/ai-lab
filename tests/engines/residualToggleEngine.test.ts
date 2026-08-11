import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  computeTrace,
  type ResidualToggleConfig,
} from '@/engines/residualToggleEngine';
import type { HiddenStateDep, HiddenStateResult } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-5-transformers/5-4-residuals-layernorm.json';

const DIM = 12;

/**
 * Fake model whose hidden states grow with depth, as an unnormalised stack
 * genuinely does — so toggling the plumbing has something real to act on.
 */
function makeHiddenStates(sentence: string, layers: number): HiddenStateResult {
  const tokens = ['[CLS]', ...sentence.toLowerCase().split(/\s+/).filter(Boolean), '[SEP]'];
  const hiddenStates = Array.from({ length: layers }, (_, l) =>
    tokens.map((_, t) =>
      Array.from({ length: DIM }, (_, d) => Math.sin((t + 1) * (d + 1) * 0.3) * (1 + l * 0.6))
    )
  );
  return { tokens, hiddenStates };
}

const fakeHiddenStates: HiddenStateDep = {
  async hiddenStates(sentence: string) {
    return makeHiddenStates(sentence, 7);
  },
};

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as ResidualToggleConfig;

describe('computeTrace', () => {
  const result = makeHiddenStates('the engineer reviewed the schematic', 7);

  it('returns one magnitude per layer', () => {
    const { trace } = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    });
    expect(trace).toHaveLength(7);
    expect(trace.every(Number.isFinite)).toBe(true);
  });

  it('shows magnitude climbing with depth when nothing normalises it', () => {
    const { trace } = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    });
    expect(trace.at(-1)).toBeGreaterThan(trace[0]!);
  });

  it('flattens the trace once layer norm is applied', () => {
    const raw = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    }).trace;
    const normalised = computeTrace(result, {
      residual: true,
      layerNorm: true,
      residualScale: 1,
      normEpsilon: 1e-5,
    }).trace;

    const spread = (t: number[]) => Math.max(...t) / Math.max(Math.min(...t), 1e-9);
    expect(spread(normalised)).toBeLessThan(spread(raw));
  });

  it('reports smaller magnitudes without the residual path', () => {
    const withResidual = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    }).trace;
    const without = computeTrace(result, {
      residual: false,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    }).trace;

    // Only the block's own contribution survives, which is much smaller.
    expect(without.at(-1)).toBeLessThan(withResidual.at(-1)!);
  });

  it('scales the trace with the residual scale', () => {
    const single = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 1,
      normEpsilon: 1e-5,
    }).trace;
    const doubled = computeTrace(result, {
      residual: true,
      layerNorm: false,
      residualScale: 2,
      normEpsilon: 1e-5,
    }).trace;
    expect(doubled[0]).toBeCloseTo(single[0]! * 2);
  });

  it('reports per-layer variance alongside the trace', () => {
    const { variances } = computeTrace(result, {
      residual: true,
      layerNorm: true,
      residualScale: 1,
      normEpsilon: 1e-5,
    });
    expect(variances).toHaveLength(7);
    expect(variances.every((v) => v >= 0)).toBe(true);
  });
});

describe('residualToggleEngine — predict trace', () => {
  const config = configFor(0);

  it('asks about layers spread across the stack', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(0), prepared);

    expect(state.rounds).toHaveLength(config.rounds!);
    const layers = state.rounds.map((r) => r.layer);
    expect(new Set(layers).size).toBe(layers.length);
    for (const round of state.rounds) {
      expect(round.trueValue).toBeGreaterThan(0);
      expect(round.estimate).toBeNull();
    }
  });

  it('rejects a model returning no hidden states', async () => {
    const broken: HiddenStateDep = {
      async hiddenStates() {
        return { tokens: ['a'], hiddenStates: [] };
      },
    };
    await expect(prepare(config, { hiddenStates: broken })).rejects.toThrow();
  });

  it('scores exact estimates at zero error', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(0), prepared);
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ESTIMATE_TRACE', roundIndex: i, value: round.trueValue });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('traceEstimateError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('charges maximum error for unanswered rounds', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(0), prepared);
    expect(evaluate(state).value).toBe(1);
  });

  it('rejects negative estimates', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'ESTIMATE_TRACE', roundIndex: 0, value: -1 });
    expect(state.rounds[0]!.estimate).toBeNull();
  });
});

describe('residualToggleEngine — toggle stability', () => {
  const config = configFor(1);

  it('starts with both mechanisms off, as the level configures', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.residual).toBe(false);
    expect(state.layerNorm).toBe(false);
  });

  it('reaches a stable band once the plumbing is switched on', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(1), prepared);
    const before = evaluate(state).value;

    state = applyAction(state, { type: 'SET_RESIDUAL', value: true });
    state = applyAction(state, { type: 'SET_LAYER_NORM', value: true });

    const result = evaluate(state);
    expect(result.metric).toBe('stabilityScore');
    expect(result.value).toBeGreaterThanOrEqual(before);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('recomputes the trace on every toggle', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const before = initState(config, rulesFor(1), prepared);
    const after = applyAction(before, { type: 'SET_LAYER_NORM', value: true });
    expect(after.trace).not.toEqual(before.trace);
  });

  it('keeps a reference trace with both mechanisms on', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.referenceTrace).toHaveLength(state.trace.length);
  });
});

describe('residualToggleEngine — minimise drift', () => {
  const config = configFor(2);

  it('measures drift between consecutive layers', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(2), prepared);
    const result = evaluate(state);

    expect(result.metric).toBe('activationDrift');
    expect(result.breakdown.drift).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.value)).toBe(true);
  });

  it('clamps the residual scale and epsilon to their ranges', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(2), prepared);

    state = applyAction(state, { type: 'SET_RESIDUAL_SCALE', value: 99 });
    expect(state.residualScale).toBe(config.residualScaleRange![1]);
    state = applyAction(state, { type: 'SET_NORM_EPSILON', value: 99 });
    expect(state.normEpsilon).toBe(config.normEpsilonRange![1]);
  });

  it('rejects a non-positive epsilon, which would divide by zero', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(2), prepared);
    const before = state.normEpsilon;
    state = applyAction(state, { type: 'SET_NORM_EPSILON', value: 0 });
    expect(state.normEpsilon).toBe(before);
  });

  it('charges a penalty when the representation is flattened away', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    let state = initState(config, rulesFor(2), prepared);

    // A huge epsilon divides every activation down towards nothing.
    state = applyAction(state, { type: 'SET_RESIDUAL_SCALE', value: 0 });
    const result = evaluate(state);
    expect(result.breakdown.variance).toBeLessThan(config.minRepresentationVariance!);
    expect(result.breakdown.collapsePenalty).toBeGreaterThan(0);
    expect(result.value).toBeGreaterThan(result.breakdown.drift!);
  });

  it('leaves a healthy configuration unpenalised', async () => {
    const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
    const state = initState(config, rulesFor(2), prepared);
    const result = evaluate(state);
    expect(result.breakdown.collapsePenalty).toBe(0);
    expect(result.passed).toBe(true);
  });
});

describe('residualToggleEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as ResidualToggleConfig;
      const prepared = await prepare(config, { hiddenStates: fakeHiddenStates });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
