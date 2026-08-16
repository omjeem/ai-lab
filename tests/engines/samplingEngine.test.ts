import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  reshape,
  currentReshape,
  type SamplingConfig,
  type PromptDistribution,
} from '@/engines/samplingEngine';
import type { CausalLMDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import { entropyBits } from '@/engines/shared';
import game from '@data/games/world-4-sequence-models/4-3-sampling-strategies.json';

/**
 * Confident prompts get a peaked distribution, open-ended ones a flat one —
 * which is what makes the top-k versus top-p comparison meaningful.
 */
const fakeLM: CausalLMDep = {
  async nextTokenDistribution(prompt: string, topK: number) {
    const confident = /symbol|boiling|fibonacci/i.test(prompt);
    const weights = Array.from({ length: topK }, (_, i) =>
      confident ? 1 / (i + 1) ** 4 : 1 / (i + 1) ** 0.4
    );
    const total = weights.reduce((a, b) => a + b, 0);
    return {
      tokens: Array.from({ length: topK }, (_, i) => `tok${i}`),
      probs: weights.map((w) => w / total),
    };
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as SamplingConfig;

const uniform = (n: number): PromptDistribution => ({
  prompt: 'p',
  tokens: Array.from({ length: n }, (_, i) => `t${i}`),
  probs: Array.from({ length: n }, () => 1 / n),
});

describe('reshape — temperature', () => {
  const peaked: PromptDistribution = {
    prompt: 'p',
    tokens: ['a', 'b', 'c', 'd'],
    probs: [0.7, 0.2, 0.07, 0.03],
  };

  it('leaves the distribution alone at temperature 1 with no truncation', () => {
    const result = reshape(peaked, { temperature: 1, topK: 4, topP: 1 });
    expect(result.probs[0]).toBeCloseTo(0.7, 5);
    expect(result.entropyBits).toBeCloseTo(entropyBits(peaked.probs), 5);
  });

  it('sharpens as temperature falls', () => {
    const cold = reshape(peaked, { temperature: 0.3, topK: 4, topP: 1 });
    const warm = reshape(peaked, { temperature: 1, topK: 4, topP: 1 });
    expect(cold.probs[0]).toBeGreaterThan(warm.probs[0]!);
    expect(cold.entropyBits).toBeLessThan(warm.entropyBits);
  });

  it('flattens as temperature rises', () => {
    const hot = reshape(peaked, { temperature: 5, topK: 4, topP: 1 });
    expect(hot.entropyBits).toBeGreaterThan(entropyBits(peaked.probs));
    expect(hot.entropyBits).toBeLessThanOrEqual(Math.log2(4) + 1e-9);
  });

  it('always renormalises to a valid distribution', () => {
    for (const temperature of [0.1, 0.5, 1, 2, 5]) {
      const result = reshape(peaked, { temperature, topK: 3, topP: 0.8 });
      expect(result.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      expect(result.probs.every((p) => p >= 0)).toBe(true);
    }
  });

  it('handles an empty distribution without throwing', () => {
    const result = reshape({ prompt: '', tokens: [], probs: [] }, { temperature: 1, topK: 5, topP: 1 });
    expect(result.probs).toEqual([]);
    expect(result.entropyBits).toBe(0);
  });
});

describe('reshape — truncation', () => {
  const peaked: PromptDistribution = {
    prompt: 'p',
    tokens: ['a', 'b', 'c', 'd', 'e'],
    probs: [0.5, 0.25, 0.15, 0.07, 0.03],
  };

  it('keeps exactly k tokens under top-k', () => {
    const result = reshape(peaked, { temperature: 1, topK: 2, topP: 1 });
    expect(result.keptCount).toBe(2);
    expect(result.tokens).toEqual(['a', 'b']);
    expect(result.keptMass).toBeCloseTo(0.75);
  });

  it('keeps the smallest prefix reaching the mass threshold under top-p', () => {
    const result = reshape(peaked, { temperature: 1, topK: 5, topP: 0.7 });
    expect(result.keptCount).toBe(2);
    expect(result.keptMass).toBeGreaterThanOrEqual(0.7);
  });

  it('adapts its cut to how confident the distribution is', () => {
    const confident = reshape(peaked, { temperature: 1, topK: 20, topP: 0.9 });
    const flat = reshape(uniform(10), { temperature: 1, topK: 20, topP: 0.9 });
    // The same top-p keeps far fewer tokens when the model is sure.
    expect(confident.keptCount).toBeLessThan(flat.keptCount);
  });

  it('always keeps at least one token', () => {
    const result = reshape(peaked, { temperature: 1, topK: 1, topP: 0.01 });
    expect(result.keptCount).toBe(1);
    expect(result.probs[0]).toBeCloseTo(1);
  });

  it('applies top-k before top-p', () => {
    // top-k=2 caps the pool, so top-p cannot reach further down the tail.
    const result = reshape(peaked, { temperature: 1, topK: 2, topP: 1 });
    expect(result.keptCount).toBe(2);
  });

  it('exposes the full ranked pool for display, not just what was kept', () => {
    const result = reshape(peaked, { temperature: 1, topK: 2, topP: 1 });
    expect(result.rankedTokens).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(result.rankedProbs).toHaveLength(5);
    // Unlike `probs`, the ranked pool is not renormalised over the kept set.
    expect(result.rankedProbs.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(result.rankedProbs[0]).toBeCloseTo(result.keptMass * result.probs[0]!);
  });
});

describe('samplingEngine — entropy target', () => {
  const config = configFor(0);

  it('pulls the real distribution for the configured prompt', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    expect(prepared.distributions).toHaveLength(1);
    expect(prepared.distributions[0]!.prompt).toBe(config.prompt);
    expect(prepared.distributions[0]!.probs).toHaveLength(config.topK);
  });

  it('rejects a model returning mismatched arrays', async () => {
    const broken: CausalLMDep = {
      async nextTokenDistribution() {
        return { tokens: ['a'], probs: [0.5, 0.5] };
      },
    };
    await expect(prepare(config, { causalLM: broken })).rejects.toThrow();
  });

  it('can be tuned onto the target band', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let best = Infinity;
    for (let t = 0.1; t <= 2.5; t += 0.02) {
      let state = initState(config, rulesFor(0), prepared);
      state = applyAction(state, { type: 'SET_TEMPERATURE', value: t });
      best = Math.min(best, evaluate(state).value);
    }
    expect(best).toBeLessThanOrEqual(config.toleranceBits!);
  });

  it('reports the measured entropy alongside the target', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(0), prepared);
    const result = evaluate(state);
    expect(result.metric).toBe('entropyError');
    expect(result.breakdown.target).toBe(config.targetEntropyBits);
    expect(result.breakdown.entropyBits).toBeGreaterThan(0);
  });

  it('clamps temperature to its configured range', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: 99 });
    expect(state.temperature).toBe(config.temperatureRange![1]);
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: -1 });
    expect(state.temperature).toBe(config.temperatureRange![0]);
  });

  it('rejects non-finite control values', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(0), prepared);
    expect(applyAction(state, { type: 'SET_TEMPERATURE', value: NaN }).temperature).toBe(
      state.temperature
    );
  });

  it('exposes the current reshaped view for rendering', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(0), prepared);
    const view = currentReshape(state)!;
    expect(view.tokens.length).toBeGreaterThan(0);
    expect(view.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });
});

describe('samplingEngine — top-k versus top-p', () => {
  const config = configFor(1);

  it('loads one distribution per prompt', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    expect(prepared.distributions).toHaveLength(config.prompts!.length);
  });

  it('scores a setting that behaves on both prompts higher than one that does not', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });

    let balanced = initState(config, rulesFor(1), prepared);
    balanced = applyAction(balanced, { type: 'SET_TOP_P', value: 0.9 });
    balanced = applyAction(balanced, { type: 'SET_TOP_K', value: 50 });

    let lopsided = initState(config, rulesFor(1), prepared);
    lopsided = applyAction(lopsided, { type: 'SET_TOP_P', value: 1 });
    lopsided = applyAction(lopsided, { type: 'SET_TOP_K', value: 1 });

    expect(evaluate(balanced).value).toBeGreaterThan(evaluate(lopsided).value);
  });

  it('can reach the pass threshold', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let best = 0;
    for (let p = 0.5; p <= 1; p += 0.01) {
      let state = initState(config, rulesFor(1), prepared);
      state = applyAction(state, { type: 'SET_TOP_P', value: p });
      state = applyAction(state, { type: 'SET_TOP_K', value: config.topKRange![1] });
      best = Math.max(best, evaluate(state).value);
    }
    expect(best).toBeGreaterThanOrEqual(game.levels[1]!.passCriteria.threshold);
  });

  it('reports the worst-performing prompt in the breakdown', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(1), prepared);
    const result = evaluate(state);
    expect(result.metric).toBe('truncationBalanceScore');
    expect(result.breakdown.worst).toBeLessThanOrEqual(result.value);
  });

  it('clamps top-k to its range and rejects fractional values', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_TOP_K', value: 9999 });
    expect(state.topK).toBe(config.topKRange![1]);
    const before = state.topK;
    state = applyAction(state, { type: 'SET_TOP_K', value: 3.5 });
    expect(state.topK).toBe(before);
  });
});

describe('samplingEngine — per-task tuning', () => {
  const config = configFor(2);

  it('gives every task its own independent settings', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    expect(Object.keys(state.taskSettings)).toHaveLength(config.tasks!.length);

    state = applyAction(state, { type: 'SELECT_TASK', taskId: 'factual' });
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: 0.2 });
    state = applyAction(state, { type: 'SELECT_TASK', taskId: 'creative' });
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: 2 });

    expect(state.taskSettings['factual']!.temperature).toBeCloseTo(0.2);
    expect(state.taskSettings['creative']!.temperature).toBeCloseTo(2);
  });

  it('restores a task settings when it is selected again', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SELECT_TASK', taskId: 'factual' });
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: 0.3 });
    state = applyAction(state, { type: 'SELECT_TASK', taskId: 'code' });
    state = applyAction(state, { type: 'SELECT_TASK', taskId: 'factual' });
    expect(state.temperature).toBeCloseTo(0.3);
  });

  it('ignores an unknown task id', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(2), prepared);
    expect(applyAction(state, { type: 'SELECT_TASK', taskId: 'nope' }).activeTaskId).toBe(
      state.activeTaskId
    );
  });

  it('scores each task against its own entropy target', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);

    // Search each task independently, as a player would.
    for (const task of config.tasks!) {
      state = applyAction(state, { type: 'SELECT_TASK', taskId: task.id });
      let bestT = state.temperature;
      let bestError = Infinity;
      for (let t = 0.1; t <= 2.5; t += 0.02) {
        const trial = applyAction(state, { type: 'SET_TEMPERATURE', value: t });
        const distribution = trial.distributions.find((d) => d.prompt === task.prompt)!;
        const view = reshape(distribution, trial.taskSettings[task.id]!);
        const error = Math.abs(view.entropyBits - task.targetEntropyBits);
        if (error < bestError) {
          bestError = error;
          bestT = t;
        }
      }
      state = applyAction(state, { type: 'SET_TEMPERATURE', value: bestT });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('taskTuningScore');
    expect(result.value).toBeGreaterThanOrEqual(game.levels[2]!.passCriteria.threshold);
  });

  it('accepts a user-supplied distribution only when the level allows it', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let open = initState(config, rulesFor(2), prepared);
    const count = open.distributions.length;
    open = applyAction(open, { type: 'ADD_DISTRIBUTION', distribution: uniform(4) });
    expect(open.distributions).toHaveLength(count + 1);

    const closedConfig = configFor(0);
    const closedPrepared = await prepare(closedConfig, { causalLM: fakeLM });
    let closed = initState(closedConfig, rulesFor(0), closedPrepared);
    const closedCount = closed.distributions.length;
    closed = applyAction(closed, { type: 'ADD_DISTRIBUTION', distribution: uniform(4) });
    expect(closed.distributions).toHaveLength(closedCount);
  });
});

describe('samplingEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as SamplingConfig;
      const prepared = await prepare(config, { causalLM: fakeLM });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
