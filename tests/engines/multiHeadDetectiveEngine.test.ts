import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  profileHead,
  bestDependencyAcross,
  type MultiHeadDetectiveConfig,
} from '@/engines/multiHeadDetectiveEngine';
import type { AttentionDep, AttentionResult } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import { normalizeDistribution } from '@/engines/shared';
import game from '@data/games/world-5-transformers/5-3-multi-head-attention.json';

const LAYERS = 6;
const HEADS = 12;

/**
 * Fake model whose heads have planted, measurable specialities: head 0 looks at
 * the previous token, head 1 at the next, head 2 at itself, head 3 at the
 * delimiters, and the rest spread out. The engine has to discover this by
 * measuring, not by being told.
 */
function makeAttention(sentence: string): AttentionResult {
  const tokens = ['[CLS]', ...sentence.toLowerCase().split(/\s+/).filter(Boolean), '[SEP]'];
  const delimiters = tokens.map((t, i) => (t.startsWith('[') ? i : -1)).filter((i) => i >= 0);

  const attention = Array.from({ length: LAYERS }, () =>
    Array.from({ length: HEADS }, (_, head) =>
      tokens.map((_, query) =>
        normalizeDistribution(
          tokens.map((_, key) => {
            switch (head) {
              case 0:
                return key === query - 1 ? 20 : 0.05;
              case 1:
                return key === query + 1 ? 20 : 0.05;
              case 2:
                return key === query ? 20 : 0.05;
              case 3:
                return delimiters.includes(key) ? 20 : 0.05;
              default:
                return 1;
            }
          })
        )
      )
    )
  );

  return { tokens, attention };
}

const fakeAttention: AttentionDep = {
  async attention(sentence: string) {
    return makeAttention(sentence);
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
  game.levels[i]!.engineConfig as unknown as MultiHeadDetectiveConfig;

describe('profileHead', () => {
  const result = makeAttention('the quiet librarian returned the overdue book');

  it('identifies a previous-token head from its real matrix', () => {
    expect(profileHead(result, 0, 0).behaviour).toBe('previous-token');
  });

  it('identifies next-token, self and delimiter heads', () => {
    expect(profileHead(result, 0, 1).behaviour).toBe('next-token');
    expect(profileHead(result, 0, 2).behaviour).toBe('self');
    expect(profileHead(result, 0, 3).behaviour).toBe('delimiter');
  });

  it('calls an unfocused head diffuse', () => {
    expect(profileHead(result, 0, 7).behaviour).toBe('diffuse');
  });

  it('reports a score for every behaviour', () => {
    const { scores } = profileHead(result, 0, 0);
    for (const key of ['previous-token', 'next-token', 'self', 'delimiter', 'diffuse'] as const) {
      expect(Number.isFinite(scores[key])).toBe(true);
    }
  });

  it('gives a clear-cut head high confidence and a uniform one low', () => {
    expect(profileHead(result, 0, 0).confidence).toBeGreaterThan(
      profileHead(result, 0, 7).confidence
    );
  });

  it('degrades gracefully for a head that does not exist', () => {
    const profile = profileHead(result, 99, 99);
    expect(profile.behaviour).toBe('diffuse');
    expect(profile.confidence).toBe(0);
  });
});

describe('multiHeadDetectiveEngine — find behaviour', () => {
  const config = configFor(0);

  it('profiles the configured number of heads', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.heads).toHaveLength(config.headCount);
    expect(state.nominations).toHaveLength(config.rounds!);
  });

  it('rejects a model returning no attention', async () => {
    const broken: AttentionDep = {
      async attention() {
        return { tokens: ['a'], attention: [] };
      },
    };
    await expect(prepare(config, { attention: broken })).rejects.toThrow();
  });

  it('finds the heads that genuinely show the target behaviour', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.targetHeads).toContain(0);
  });

  it('scores nominating a matching head at 1', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    for (let i = 0; i < config.rounds!; i++) {
      state = applyAction(state, { type: 'NOMINATE_HEAD', roundIndex: i, head: state.targetHeads[0]! });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('headMatchAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores nominating a non-matching head at 0', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    const wrong = state.heads.find((h) => !state.targetHeads.includes(h.head))!.head;
    for (let i = 0; i < config.rounds!; i++) {
      state = applyAction(state, { type: 'NOMINATE_HEAD', roundIndex: i, head: wrong });
    }
    expect(evaluate(state).value).toBe(0);
  });

  it('ignores a head index that does not exist', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'NOMINATE_HEAD', roundIndex: 0, head: 999 });
    expect(state.nominations[0]).toBeNull();
  });

  it('re-profiles when the layer changes', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_LAYER', value: 5 });
    expect(state.layer).toBe(5);
    expect(state.heads.every((h) => h.layer === 5)).toBe(true);
  });

  it('clamps the layer to its range', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_LAYER', value: 99 });
    expect(state.layer).toBe(config.layerRange![1]);
  });

  it('discards nominations when the layer changes, rather than re-grading them against the new layer', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    for (let i = 0; i < config.rounds!; i++) {
      state = applyAction(state, { type: 'NOMINATE_HEAD', roundIndex: i, head: state.targetHeads[0]! });
    }
    expect(evaluate(state).value).toBeCloseTo(1);

    // Only looking at another layer must not silently flip an already-correct
    // score — the old nominations are meaningless against a different layer's
    // heads, so they should be cleared, not re-graded.
    state = applyAction(state, { type: 'SET_LAYER', value: state.layer === 0 ? 1 : 0 });
    expect(state.nominations.every((n) => n === null)).toBe(true);
    expect(evaluate(state).value).toBe(0);
  });
});

describe('multiHeadDetectiveEngine — classify all', () => {
  const config = configFor(1);

  it('scores every correct label at 1', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);
    for (const head of state.heads) {
      state = applyAction(state, { type: 'LABEL_HEAD', head: head.head, behaviour: head.behaviour });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('headClassificationAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('only grades heads whose classification is clear-cut', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(1), prepared);
    const result = evaluate(state);
    expect(result.breakdown.gradable).toBeLessThanOrEqual(result.breakdown.heads!);
    expect(result.breakdown.gradable).toBeGreaterThan(0);
  });

  it('rejects a behaviour outside the offered set', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);
    // @ts-expect-error deliberately invalid behaviour
    state = applyAction(state, { type: 'LABEL_HEAD', head: 0, behaviour: 'not-a-behaviour' });
    expect(state.heads[0]!.answer).toBeNull();
  });

  it('scores wrong labels at 0', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);
    for (const head of state.heads) {
      const wrong = config.behaviours.find((b) => b !== head.behaviour)!;
      state = applyAction(state, { type: 'LABEL_HEAD', head: head.head, behaviour: wrong });
    }
    expect(evaluate(state).value).toBe(0);
  });
});

describe('multiHeadDetectiveEngine — hunt dependency', () => {
  const config = configFor(2);

  it('scores zero before a token pair is chosen', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(0);
  });

  it('scores against the strongest link that actually exists', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 3, toIndex: 2 });

    const layers = prepared.result.attention.map((_, i) => i);
    const best = bestDependencyAcross(prepared.result, 3, 2, layers);
    state = applyAction(state, { type: 'PROBE_HEAD', layer: best.layer, head: best.head });

    const result = evaluate(state);
    expect(result.metric).toBe('dependencyHuntScore');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('gives partial credit for finding a weaker head', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 3, toIndex: 2 });
    // Head 7 is diffuse, so it carries far less of this particular link.
    state = applyAction(state, { type: 'PROBE_HEAD', layer: 0, head: 7 });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(0);
    expect(result.value).toBeLessThan(1);
  });

  it('discards probes when the token pair changes', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 3, toIndex: 2 });
    state = applyAction(state, { type: 'PROBE_HEAD', layer: 0, head: 0 });
    expect(state.attempts).toHaveLength(1);

    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 4, toIndex: 1 });
    expect(state.attempts).toHaveLength(0);
    expect(state.bestDependencyWeight).toBe(0);
  });

  it('rejects an out-of-range token index', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 999, toIndex: 0 });
    expect(state.dependency).toBeNull();
  });

  it('stops accepting probes past the attempt limit', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_DEPENDENCY', fromIndex: 3, toIndex: 2 });
    for (let i = 0; i < config.attempts! + 4; i++) {
      state = applyAction(state, { type: 'PROBE_HEAD', layer: 0, head: i % 12 });
    }
    expect(state.attempts).toHaveLength(config.attempts!);
  });

  it('searches every layer when configured to', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const all = bestDependencyAcross(prepared.result, 3, 2, [0, 1, 2, 3, 4, 5]);
    const one = bestDependencyAcross(prepared.result, 3, 2, [0]);
    expect(all.weight).toBeGreaterThanOrEqual(one.weight);
  });
});

describe('multiHeadDetectiveEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as MultiHeadDetectiveConfig;
      const prepared = await prepare(config, { attention: fakeAttention });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
