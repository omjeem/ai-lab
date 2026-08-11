import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ProbabilityWheelConfig,
} from '@/engines/probabilityWheelEngine';
import type { CausalLMDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import { entropyBits } from '@/engines/shared';
import game from '@data/games/world-1-fundamentals/1-5-probability.json';

/** Fake LM returning a sharply peaked distribution, deterministic per prompt. */
const fakeLM: CausalLMDep = {
  async nextTokenDistribution(prompt: string, topK: number) {
    const base = Array.from({ length: topK }, (_, i) => 1 / (i + 1) ** 2);
    const total = base.reduce((a, b) => a + b, 0);
    return {
      tokens: Array.from({ length: topK }, (_, i) => `${prompt.slice(0, 3)}_tok${i}`),
      probs: base.map((p) => p / total),
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as ProbabilityWheelConfig;

describe('probabilityWheelEngine — prepare', () => {
  it('pulls one real distribution per configured prompt', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { causalLM: fakeLM });

    expect(prepared.rounds).toHaveLength(config.prompts.length);
    for (const round of prepared.rounds) {
      expect(round.tokens).toHaveLength(config.topK);
      expect(round.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      expect(round.actualEntropyBits).toBeCloseTo(entropyBits(round.probs));
    }
  });

  it('returns no rounds when the level supplies no prompts', async () => {
    const prepared = await prepare(configFor(2), { causalLM: fakeLM });
    expect(prepared.rounds).toEqual([]);
  });

  it('rejects a model returning mismatched token and probability arrays', async () => {
    const broken: CausalLMDep = {
      async nextTokenDistribution() {
        return { tokens: ['a', 'b'], probs: [1] };
      },
    };
    await expect(prepare(configFor(0), { causalLM: broken })).rejects.toThrow();
  });
});

describe('probabilityWheelEngine — predict top 1', () => {
  const config = configFor(0);

  it('scores every correct pick at full accuracy', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);

    state.rounds.forEach((round, i) => {
      const top = round.tokens[round.probs.indexOf(Math.max(...round.probs))]!;
      state = applyAction(state, { type: 'PICK', roundIndex: i, token: top });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('predictionAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores a wrong pick at zero and fails the level', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);

    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'PICK', roundIndex: i, token: round.tokens.at(-1)! });
    });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('counts unanswered rounds as wrong', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);
    const round = state.rounds[0]!;
    const top = round.tokens[round.probs.indexOf(Math.max(...round.probs))]!;
    state = applyAction(state, { type: 'PICK', roundIndex: 0, token: top });

    expect(evaluate(state).value).toBeCloseTo(1 / state.rounds.length);
  });

  it('ignores a token that is not on the wheel', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'PICK', roundIndex: 0, token: 'not-a-token' });
    expect(state.rounds[0]!.pick).toBeNull();
  });

  it('refuses to add rounds when the level disallows user prompts', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);
    const count = state.rounds.length;
    state = applyAction(state, {
      type: 'ADD_ROUND',
      round: { prompt: 'mine', tokens: ['a'], probs: [1], actualEntropyBits: 0 },
    });
    expect(state.rounds).toHaveLength(count);
  });
});

describe('probabilityWheelEngine — allocate mass', () => {
  const config = configFor(1);

  it('starts with an empty allocation per round', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(1), prepared);
    for (const round of state.rounds) {
      expect(round.allocation).toHaveLength(round.tokens.length);
      expect(round.allocation.every((v) => v === 0)).toBe(true);
    }
  });

  it('scores a perfectly matched allocation at zero error', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(1), prepared);

    state.rounds.forEach((round, i) => {
      round.probs.forEach((p, tokenIndex) => {
        state = applyAction(state, {
          type: 'ALLOCATE',
          roundIndex: i,
          tokenIndex,
          value: p * config.budget!,
        });
      });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('calibrationError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('penalises a flat allocation against a peaked distribution', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(1), prepared);

    state.rounds.forEach((round, i) => {
      round.tokens.forEach((_, tokenIndex) => {
        state = applyAction(state, { type: 'ALLOCATE', roundIndex: i, tokenIndex, value: 10 });
      });
    });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(0.2);
    expect(result.passed).toBe(false);
  });

  it('charges maximum error for a round with nothing allocated', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(1), prepared);
    expect(evaluate(state).value).toBe(1);
  });

  it('caps total allocation at the configured budget', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, tokenIndex: 0, value: 999 });
    expect(state.rounds[0]!.allocation[0]).toBe(config.budget);
  });

  it('rejects negative and non-finite allocations', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, tokenIndex: 0, value: -5 });
    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, tokenIndex: 1, value: NaN });
    expect(state.rounds[0]!.allocation[0]).toBe(0);
    expect(state.rounds[0]!.allocation[1]).toBe(0);
  });

  it('ignores an out-of-range token index', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(1), prepared);
    const next = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, tokenIndex: 99, value: 10 });
    expect(next.rounds[0]!.allocation).toEqual(state.rounds[0]!.allocation);
  });
});

describe('probabilityWheelEngine — estimate entropy', () => {
  const config = configFor(2);

  it('accepts user-added rounds when the level allows own prompts', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    const round = await fakeLM.nextTokenDistribution('my own prompt', config.topK);

    state = applyAction(state, {
      type: 'ADD_ROUND',
      round: { prompt: 'my own prompt', ...round, actualEntropyBits: entropyBits(round.probs) },
    });
    expect(state.rounds).toHaveLength(1);
  });

  it('scores exact entropy estimates at zero error', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);

    for (let i = 0; i < config.rounds!; i++) {
      const dist = await fakeLM.nextTokenDistribution(`prompt ${i}`, config.topK);
      state = applyAction(state, {
        type: 'ADD_ROUND',
        round: { prompt: `prompt ${i}`, ...dist, actualEntropyBits: entropyBits(dist.probs) },
      });
    }
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ESTIMATE_ENTROPY', roundIndex: i, value: round.actualEntropyBits });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('entropyEstimateError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('charges the full entropy range for rounds never played', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(2), prepared);
    const result = evaluate(state);
    expect(result.value).toBe(config.maxEntropyBits);
    expect(result.passed).toBe(false);
  });

  it('will not accept more rounds than the level configures', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    const dist = await fakeLM.nextTokenDistribution('p', config.topK);

    for (let i = 0; i < config.rounds! + 3; i++) {
      state = applyAction(state, {
        type: 'ADD_ROUND',
        round: { prompt: `p${i}`, ...dist, actualEntropyBits: entropyBits(dist.probs) },
      });
    }
    expect(state.rounds).toHaveLength(config.rounds!);
  });

  it('clamps estimates into the configured entropy range', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    const dist = await fakeLM.nextTokenDistribution('p', config.topK);
    state = applyAction(state, {
      type: 'ADD_ROUND',
      round: { prompt: 'p', ...dist, actualEntropyBits: entropyBits(dist.probs) },
    });

    state = applyAction(state, { type: 'ESTIMATE_ENTROPY', roundIndex: 0, value: 99 });
    expect(state.rounds[0]!.entropyEstimate).toBe(config.maxEntropyBits);
    state = applyAction(state, { type: 'ESTIMATE_ENTROPY', roundIndex: 0, value: -3 });
    expect(state.rounds[0]!.entropyEstimate).toBe(0);
  });
});

describe('probabilityWheelEngine — spin', () => {
  it('samples from the real distribution and lands on a token from the wheel', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);

    state = applyAction(state, { type: 'SPIN', roundIndex: 0, random: 0.001 });
    // The lowest random draw must land on the most likely token.
    const round = state.rounds[0]!;
    expect(round.spinResult).toBe(round.tokens[round.probs.indexOf(Math.max(...round.probs))]);

    state = applyAction(state, { type: 'SPIN', roundIndex: 0, random: 0.999 });
    expect(round.tokens).toContain(state.rounds[0]!.spinResult);
  });

  it('does not change the recorded pick', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'PICK', roundIndex: 0, token: state.rounds[0]!.tokens[1]! });
    state = applyAction(state, { type: 'SPIN', roundIndex: 0, random: 0.5 });
    expect(state.rounds[0]!.pick).toBe(state.rounds[0]!.tokens[1]);
  });
});

describe('probabilityWheelEngine — level config coverage', () => {
  it('handles every shipped level end to end', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as ProbabilityWheelConfig;
      const prepared = await prepare(config, { causalLM: fakeLM });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
