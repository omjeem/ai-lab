import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  orderingAccuracy,
  buildRound,
  type TransformerAssemblyConfig,
} from '@/engines/transformerAssemblyEngine';
import type { CausalLMDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-5-transformers/5-5-full-transformer.json';

const fakeLM: CausalLMDep = {
  async nextTokenDistribution(prompt: string, topK: number) {
    const weights = Array.from({ length: topK }, (_, i) => 1 / (i + 1) ** 2);
    const total = weights.reduce((a, b) => a + b, 0);
    return {
      tokens: Array.from({ length: topK }, (_, i) => `${prompt.slice(0, 2)}_t${i}`),
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
const configFor = (i: number) =>
  game.levels[i]!.engineConfig as unknown as TransformerAssemblyConfig;

describe('orderingAccuracy', () => {
  const correct = ['a', 'b', 'c', 'd'];

  it('scores a perfect order at 1 and a reversal at 0', () => {
    expect(orderingAccuracy(correct, correct)).toBe(1);
    expect(orderingAccuracy([...correct].reverse(), correct)).toBe(0);
  });

  it('gives partial credit for a single swap', () => {
    const value = orderingAccuracy(['b', 'a', 'c', 'd'], correct);
    expect(value).toBeGreaterThan(0.5);
    expect(value).toBeLessThan(1);
  });

  it('ignores stages that are not part of the pipeline', () => {
    expect(orderingAccuracy(['a', 'zzz', 'b'], correct)).toBe(1);
  });

  it('handles a single stage', () => {
    expect(orderingAccuracy(['a'], ['a'])).toBe(1);
  });
});

describe('transformerAssemblyEngine — order stages', () => {
  const config = configFor(0);

  it('presents the stages shuffled, not solved', () => {
    const state = initState(config, rulesFor(0));
    expect(state.arrangement).toHaveLength(config.stages.length);
    expect(state.arrangement).not.toEqual(state.correctOrder);
    expect([...state.arrangement].sort()).toEqual([...state.correctOrder].sort());
  });

  it('is deterministic for the configured seed', () => {
    expect(initState(config, rulesFor(0)).arrangement).toEqual(
      initState(config, rulesFor(0)).arrangement
    );
  });

  it('scores the correct order at 1', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_ARRANGEMENT', stages: [...state.correctOrder] });

    const result = evaluate(state);
    expect(result.metric).toBe('assemblyAccuracy');
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
  });

  it('gives partial credit when partial credit is allowed', () => {
    let state = initState(config, rulesFor(0));
    const nearly = [...state.correctOrder];
    [nearly[0], nearly[1]] = [nearly[1]!, nearly[0]!];
    state = applyAction(state, { type: 'SET_ARRANGEMENT', stages: nearly });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(0.9);
    expect(result.value).toBeLessThan(1);
  });

  it('moves a single stage without losing any', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'MOVE_STAGE', from: 0, to: 5 });
    expect([...state.arrangement].sort()).toEqual([...state.correctOrder].sort());
  });

  it('ignores out-of-range moves', () => {
    const state = initState(config, rulesFor(0));
    expect(applyAction(state, { type: 'MOVE_STAGE', from: -1, to: 2 }).arrangement).toEqual(
      state.arrangement
    );
    expect(applyAction(state, { type: 'MOVE_STAGE', from: 0, to: 99 }).arrangement).toEqual(
      state.arrangement
    );
  });

  it('rejects an arrangement that is not a rearrangement of the same stages', () => {
    const state = initState(config, rulesFor(0));
    const next = applyAction(state, { type: 'SET_ARRANGEMENT', stages: ['embed', 'embed'] });
    expect(next.arrangement).toEqual(state.arrangement);
  });
});

describe('transformerAssemblyEngine — fill gaps', () => {
  const config = configFor(1);

  it('starts with the removed stages missing and in the tray', () => {
    const state = initState(config, rulesFor(1));
    expect(state.tray.sort()).toEqual([...config.removedStages!].sort());
    for (const stage of config.removedStages!) {
      expect(state.arrangement).not.toContain(stage);
    }
  });

  it('scores a fully restored pipeline at 1', () => {
    let state = initState(config, rulesFor(1));
    // Insert each missing stage back at its correct index.
    for (const stage of [...state.tray]) {
      const position = state.correctOrder.indexOf(stage);
      const insertAt = state.arrangement.findIndex(
        (s) => state.correctOrder.indexOf(s) > position
      );
      state = applyAction(state, {
        type: 'INSERT_STAGE',
        stage,
        position: insertAt === -1 ? state.arrangement.length : insertAt,
      });
    }

    expect(state.tray).toHaveLength(0);
    const result = evaluate(state);
    expect(result.metric).toBe('pipelineCompleteness');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('weights the critical stages more heavily', () => {
    const criticalMissing = initState(config, rulesFor(1));
    // softmax is critical; layer-norm-2 is not.
    let withoutCritical = criticalMissing;
    let withoutOptional = criticalMissing;

    for (const stage of [...criticalMissing.tray]) {
      if (stage !== 'softmax') {
        withoutCritical = applyAction(withoutCritical, {
          type: 'INSERT_STAGE',
          stage,
          position: withoutCritical.arrangement.length,
        });
      }
      if (stage !== 'layer-norm-2') {
        withoutOptional = applyAction(withoutOptional, {
          type: 'INSERT_STAGE',
          stage,
          position: withoutOptional.arrangement.length,
        });
      }
    }

    expect(evaluate(withoutCritical).breakdown.completeness).toBeLessThan(
      evaluate(withoutOptional).breakdown.completeness!
    );
  });

  it('refuses to insert a stage that is not in the tray', () => {
    let state = initState(config, rulesFor(1));
    const before = state.arrangement.length;
    state = applyAction(state, { type: 'INSERT_STAGE', stage: 'attention', position: 0 });
    expect(state.arrangement).toHaveLength(before);
  });

  it('sends a removed stage back to the tray', () => {
    let state = initState(config, rulesFor(1));
    const trayBefore = state.tray.length;
    state = applyAction(state, { type: 'REMOVE_STAGE', position: 0 });
    expect(state.tray).toHaveLength(trayBefore + 1);
  });

  it('penalises restoring stages in the wrong order', () => {
    let state = initState(config, rulesFor(1));
    for (const stage of [...state.tray]) {
      state = applyAction(state, { type: 'INSERT_STAGE', stage, position: 0 });
    }
    expect(evaluate(state).value).toBeLessThan(1);
  });
});

describe('transformerAssemblyEngine — run end to end', () => {
  const config = configFor(2);

  it('pulls a real distribution for the prompt', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    expect(prepared.rounds).toHaveLength(1);
    expect(prepared.rounds[0]!.distribution.tokens).toHaveLength(config.topK!);
  });

  it('shuffles the choices so the ranking is not given away', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(2), prepared);
    expect(state.rounds[0]!.choices.sort()).toEqual(
      [...prepared.rounds[0]!.distribution.tokens].sort()
    );
  });

  it('rejects a model returning mismatched arrays', async () => {
    const broken: CausalLMDep = {
      async nextTokenDistribution() {
        return { tokens: ['a'], probs: [0.5, 0.5] };
      },
    };
    await expect(prepare(config, { causalLM: broken })).rejects.toThrow();
  });

  it('scores a correct top-token call, gated by the pipeline being right', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);

    // Fill the remaining rounds with the same prompt so the level is complete.
    for (let i = state.rounds.length; i < config.rounds!; i++) {
      state = applyAction(state, {
        type: 'ADD_ROUND',
        round: buildRound(`prompt ${i}`, await fakeLM.nextTokenDistribution(`prompt ${i}`, 5), i),
      });
    }

    state.rounds.forEach((round, i) => {
      const probs = round.distribution.probs;
      let best = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k]! > probs[best]!) best = k;
      state = applyAction(state, {
        type: 'ANSWER_TOP_TOKEN',
        roundIndex: i,
        token: round.distribution.tokens[best]!,
      });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('endToEndScore');
    expect(result.breakdown.pipeline).toBe(1);
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('drops the score when the pipeline is scrambled', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, {
      type: 'SET_ARRANGEMENT',
      stages: [...state.correctOrder].reverse(),
    });

    const round = state.rounds[0]!;
    const probs = round.distribution.probs;
    let best = 0;
    for (let k = 1; k < probs.length; k++) if (probs[k]! > probs[best]!) best = k;
    state = applyAction(state, {
      type: 'ANSWER_TOP_TOKEN',
      roundIndex: 0,
      token: round.distribution.tokens[best]!,
    });

    expect(evaluate(state).value).toBe(0);
  });

  it('counts unplayed rounds against the score', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(0);
  });

  it('rejects a token that was not offered', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'ANSWER_TOP_TOKEN', roundIndex: 0, token: 'nope' });
    expect(state.rounds[0]!.answer).toBeNull();
  });

  it('stops accepting rounds past the configured count', async () => {
    const prepared = await prepare(config, { causalLM: fakeLM });
    let state = initState(config, rulesFor(2), prepared);
    for (let i = 0; i < config.rounds! + 4; i++) {
      state = applyAction(state, {
        type: 'ADD_ROUND',
        round: buildRound('extra', await fakeLM.nextTokenDistribution('extra', 5), i),
      });
    }
    expect(state.rounds.length).toBeLessThanOrEqual(config.rounds!);
  });
});

describe('transformerAssemblyEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as TransformerAssemblyConfig;
      const prepared = await prepare(config, { causalLM: fakeLM });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
