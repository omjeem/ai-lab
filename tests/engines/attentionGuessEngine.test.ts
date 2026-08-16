import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  attentionRow,
  buildRound,
  editDistance,
  isSpecialToken,
  type AttentionGuessConfig,
} from '@/engines/attentionGuessEngine';
import type { AttentionDep, AttentionResult } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import { normalizeDistribution } from '@/engines/shared';
import game from '@data/games/world-5-transformers/5-2-self-attention.json';

const LAYERS = 6;
const HEADS = 12;

/**
 * Fake transformer with a planted, checkable pattern: the token "it" attends
 * most strongly to whichever noun the sentence mentions first, unless the word
 * "large" appears, in which case the second noun wins. That gives the flip level
 * something real to discover.
 */
function makeAttention(sentence: string): AttentionResult {
  const tokens = ['[CLS]', ...sentence.toLowerCase().split(/\s+/).filter(Boolean), '[SEP]'];
  const nouns = ['trophy', 'suitcase', 'doctor', 'nurse', 'cat', 'mat', 'glass', 'water'];
  const nounIndices = tokens
    .map((t, i) => (nouns.includes(t) ? i : -1))
    .filter((i) => i >= 0);
  const favoured = /small/.test(sentence.toLowerCase()) ? nounIndices[1] : nounIndices[0];

  const attention = Array.from({ length: LAYERS }, (_, layer) =>
    Array.from({ length: HEADS }, (_, head) =>
      tokens.map((_, query) => {
        const raw = tokens.map((_, key) => {
          if (key === favoured && tokens[query] === 'it') return 8;
          if (key === query - 1) return 2 + head * 0.1;
          if (key === query) return 1;
          return 0.2 + layer * 0.01;
        });
        return normalizeDistribution(raw);
      })
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as AttentionGuessConfig;

describe('attention helpers', () => {
  it('recognises the special tokens a model adds', () => {
    expect(isSpecialToken('[CLS]')).toBe(true);
    expect(isSpecialToken('[SEP]')).toBe(true);
    expect(isSpecialToken('cat')).toBe(false);
  });

  it('averages heads into one row that still sums to one', () => {
    const result = makeAttention('the cat sat on the mat');
    const row = attentionRow(result, 2, 1, 'mean');
    expect(row).toHaveLength(result.tokens.length);
    expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it('takes the per-key maximum under max aggregation', () => {
    const result = makeAttention('the cat sat on the mat');
    const meanRow = attentionRow(result, 2, 1, 'mean');
    const maxRow = attentionRow(result, 2, 1, 'max');
    for (let i = 0; i < meanRow.length; i++) {
      expect(maxRow[i]).toBeGreaterThanOrEqual(meanRow[i]! - 1e-12);
    }
  });

  it('returns an empty row for a layer that does not exist', () => {
    expect(attentionRow(makeAttention('a b'), 99, 0, 'mean')).toEqual([]);
  });

  it('measures edit distance', () => {
    expect(editDistance('abc', 'abc')).toBe(0);
    expect(editDistance('abc', 'abd')).toBe(1);
    expect(editDistance('', 'abc')).toBe(3);
    expect(editDistance('kitten', 'sitting')).toBe(3);
  });
});

describe('buildRound', () => {
  const config = configFor(0);

  it('drops the query and the special tokens from the candidate keys', () => {
    const result = makeAttention('the cat sat on the mat because it was warm');
    const round = buildRound(config, result, 4);

    expect(round.keyIndices).not.toContain(round.queryIndex);
    for (const index of round.keyIndices) {
      expect(isSpecialToken(result.tokens[index]!)).toBe(false);
    }
  });

  it('picks the pronoun as the query position', () => {
    const result = makeAttention('the cat sat on the mat because it was warm');
    const round = buildRound(config, result, 4);
    expect(result.tokens[round.queryIndex]).toBe('it');
  });

  it('renormalises the row over the surviving keys', () => {
    const result = makeAttention('the trophy did not fit in the suitcase because it was big');
    const round = buildRound(config, result, 4);
    expect(round.trueRow).toHaveLength(round.keyIndices.length);
    expect(round.trueRow.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });
});

describe('attentionGuessEngine — guess argmax', () => {
  const config = configFor(0);

  it('builds one round per sentence', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.rounds).toHaveLength(config.sentences.length);
    expect(state.rounds.every((r) => r.guessIndex === null)).toBe(true);
  });

  it('rejects a model that returns no attention', async () => {
    const broken: AttentionDep = {
      async attention() {
        return { tokens: ['a'], attention: [] };
      },
    };
    await expect(prepare(config, { attention: broken })).rejects.toThrow();
  });

  it('scores correct argmax guesses at 1', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);

    state.rounds.forEach((round, i) => {
      let best = 0;
      for (let k = 1; k < round.trueRow.length; k++) {
        if (round.trueRow[k]! > round.trueRow[best]!) best = k;
      }
      state = applyAction(state, { type: 'GUESS', roundIndex: i, keyIndex: round.keyIndices[best]! });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('attentionGuessAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores a wrong guess at zero', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);

    state.rounds.forEach((round, i) => {
      let worst = 0;
      for (let k = 1; k < round.trueRow.length; k++) {
        if (round.trueRow[k]! < round.trueRow[worst]!) worst = k;
      }
      state = applyAction(state, { type: 'GUESS', roundIndex: i, keyIndex: round.keyIndices[worst]! });
    });

    expect(evaluate(state).value).toBe(0);
  });

  it('rejects a key that is not a candidate', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', roundIndex: 0, keyIndex: 999 });
    expect(state.rounds[0]!.guessIndex).toBeNull();
  });

  it('clamps the layer to its range', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_LAYER', value: 99 });
    expect(state.layer).toBe(config.layerRange![1]);
    state = applyAction(state, { type: 'SET_LAYER', value: -5 });
    expect(state.layer).toBe(config.layerRange![0]);
  });

  it('recomputes every round for the newly selected layer, not just the label', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);

    const newLayer = config.layerRange![1]!;
    state = applyAction(state, { type: 'SET_LAYER', value: newLayer });

    // What SET_LAYER produced should be indistinguishable from building the
    // round fresh at that layer — not left over from whatever layer the
    // level opened on.
    const expected = buildRound(config, prepared.results[0]!, newLayer);
    expect(state.rounds[0]!.trueRow).toEqual(expected.trueRow);
  });

  it('keeps an existing guess when the layer changes under it', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);

    const guessedKey = state.rounds[0]!.keyIndices[0]!;
    state = applyAction(state, { type: 'GUESS', roundIndex: 0, keyIndex: guessedKey });
    state = applyAction(state, { type: 'SET_LAYER', value: config.layerRange![1]! });

    expect(state.rounds[0]!.guessIndex).toBe(guessedKey);
  });

  it('RESET rebuilds from the real prepared attention, not an empty tensor', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', roundIndex: 0, keyIndex: state.rounds[0]!.keyIndices[0]! });

    state = applyAction(state, { type: 'RESET' });

    expect(state.rounds[0]!.trueRow.length).toBeGreaterThan(0);
    expect(state.rounds[0]!.trueRow.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(state.rounds[0]!.guessIndex).toBeNull();
  });
});

describe('attentionGuessEngine — draw the distribution', () => {
  const config = configFor(1);

  it('scores a perfectly matched allocation at zero error', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);

    state.rounds.forEach((round, roundIndex) => {
      round.trueRow.forEach((weight, slot) => {
        state = applyAction(state, {
          type: 'ALLOCATE',
          roundIndex,
          keyIndex: round.keyIndices[slot]!,
          value: weight * config.budget!,
        });
      });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('attentionDistributionError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('charges maximum error for a round with nothing allocated', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(1), prepared);
    expect(evaluate(state).value).toBe(1);
  });

  it('penalises a flat allocation against a peaked row', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);

    state.rounds.forEach((round, roundIndex) => {
      for (const keyIndex of round.keyIndices) {
        state = applyAction(state, { type: 'ALLOCATE', roundIndex, keyIndex, value: 10 });
      }
    });

    expect(evaluate(state).value).toBeGreaterThan(0.1);
  });

  it('rejects negative allocations and unknown keys', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);
    const key = state.rounds[0]!.keyIndices[0]!;

    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, keyIndex: key, value: -5 });
    expect(state.rounds[0]!.allocation[0]).toBe(0);
    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, keyIndex: 999, value: 5 });
    expect(state.rounds[0]!.allocation.every((v) => v === 0)).toBe(true);
  });

  it('caps an allocation at the budget', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(1), prepared);
    const key = state.rounds[0]!.keyIndices[0]!;
    state = applyAction(state, { type: 'ALLOCATE', roundIndex: 0, keyIndex: key, value: 9999 });
    expect(state.rounds[0]!.allocation[0]).toBe(config.budget);
  });
});

describe('attentionGuessEngine — flip the reference', () => {
  const config = configFor(2);

  it('records a baseline winner and two candidates', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(2), prepared);

    expect(state.baseline).not.toBeNull();
    expect(state.baseline!.winnerIndex).toBeGreaterThanOrEqual(0);
    expect(state.baseline!.candidates).toHaveLength(2);
  });

  it('scores zero before any edit is attempted', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(0);
  });

  it('rewards an edit that genuinely flips which token wins', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);

    // "small" flips the planted preference to the second noun.
    const edited = 'the trophy did not fit in the suitcase because it was too small';
    state = applyAction(state, {
      type: 'SUBMIT_EDIT',
      sentence: edited,
      attention: await fakeAttention.attention(edited),
    });

    const result = evaluate(state);
    expect(state.attempts[0]!.flipped).toBe(true);
    expect(result.metric).toBe('referenceFlipScore');
    expect(result.passed).toBe(true);
  });

  it('scores nothing for an edit that changes no winner', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);

    const edited = 'the trophy did not fit in the suitcase because it was too heavy';
    state = applyAction(state, {
      type: 'SUBMIT_EDIT',
      sentence: edited,
      attention: await fakeAttention.attention(edited),
    });

    expect(state.attempts[0]!.flipped).toBe(false);
    expect(evaluate(state).value).toBe(0);
  });

  it('rewards a smaller edit more than a sprawling one', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });

    const small = 'the trophy did not fit in the suitcase because it was too small';
    const sprawling = 'a completely different small sentence about the trophy and the suitcase and it';

    let a = initState(config, rulesFor(2), prepared);
    a = applyAction(a, { type: 'SUBMIT_EDIT', sentence: small, attention: await fakeAttention.attention(small) });

    let b = initState(config, rulesFor(2), prepared);
    b = applyAction(b, {
      type: 'SUBMIT_EDIT',
      sentence: sprawling,
      attention: await fakeAttention.attention(sprawling),
    });

    if (b.attempts[0]!.flipped) {
      expect(evaluate(a).value).toBeGreaterThan(evaluate(b).value);
    }
    expect(a.attempts[0]!.editDistance).toBeLessThan(b.attempts[0]!.editDistance);
  });

  it('keeps the best attempt and stops at the attempt limit', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);

    const good = 'the trophy did not fit in the suitcase because it was too small';
    state = applyAction(state, { type: 'SUBMIT_EDIT', sentence: good, attention: await fakeAttention.attention(good) });
    const best = evaluate(state).value;

    for (let i = 0; i < config.attempts! + 3; i++) {
      const bad = `the trophy did not fit in the suitcase because it was heavy ${i}`;
      state = applyAction(state, { type: 'SUBMIT_EDIT', sentence: bad, attention: await fakeAttention.attention(bad) });
    }

    expect(state.attempts).toHaveLength(config.attempts!);
    expect(evaluate(state).value).toBe(best);
  });

  it('is not fooled by an edit that only shifts the winning token to a new index', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);

    // Inserting a word before the winner shifts every later token's index —
    // the same word ("trophy") still wins, just one slot further along.
    // Comparing by raw index would call this a flip; it is not one.
    const shifted = 'well the trophy did not fit in the suitcase because it was too large';
    state = applyAction(state, {
      type: 'SUBMIT_EDIT',
      sentence: shifted,
      attention: await fakeAttention.attention(shifted),
    });

    expect(state.attempts[0]!.winnerToken).toBe('trophy');
    expect(state.attempts[0]!.flipped).toBe(false);
  });

  it('recomputes the baseline together with round 0 when the layer changes', async () => {
    const prepared = await prepare(config, { attention: fakeAttention });
    let state = initState(config, rulesFor(2), prepared);

    const newLayer = config.layerRange![1]!;
    state = applyAction(state, { type: 'SET_LAYER', value: newLayer });

    const expectedRound = buildRound(config, prepared.results[0]!, newLayer);
    let best = 0;
    for (let i = 1; i < expectedRound.trueRow.length; i++) {
      if (expectedRound.trueRow[i]! > expectedRound.trueRow[best]!) best = i;
    }
    const expectedWinnerToken = expectedRound.tokens[expectedRound.keyIndices[best]!];

    expect(state.baseline!.winnerToken).toBe(expectedWinnerToken);
  });
});

describe('attentionGuessEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as AttentionGuessConfig;
      const prepared = await prepare(config, { attention: fakeAttention });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
