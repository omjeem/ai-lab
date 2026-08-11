import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type TokenMergeConfig,
} from '@/engines/tokenMergeEngine';
import type { TokenizerDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-1-fundamentals/1-4-tokenization.json';

/**
 * Fake BPE tokenizer with a small, real merge table. "lower" must merge as
 * l+o -> lo, w+e -> we, lo+we -> lowe, lowe+r -> lower, in that rank order.
 */
const MERGES = new Map<string, number>([
  ['l o', 0],
  ['w e', 1],
  ['lo we', 2],
  ['lowe r', 3],
  ['n e', 4],
  ['ne w', 5],
  ['e s', 6],
  ['es t', 7],
  ['new est', 8],
  ['t o', 9],
  ['k e', 10],
  ['to ke', 11],
]);

const fakeTokenizer: TokenizerDep = {
  async tokenize(text: string) {
    // Deterministic segmentation: chunks of three characters.
    const out: string[] = [];
    for (let i = 0; i < text.length; i += 3) out.push(text.slice(i, i + 3));
    return out.length > 0 ? out : [''];
  },
  async encode(text) {
    return (await this.tokenize(text)).map((t) => t.length);
  },
  async decode(ids) {
    return ids.join('');
  },
  async mergeRanks() {
    return MERGES;
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as TokenMergeConfig;

describe('tokenMergeEngine — count tokens', () => {
  const config = configFor(0);

  it('reads the real token count for every sample', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    expect(prepared.samples).toHaveLength(config.samples!.length);
    for (const sample of prepared.samples) {
      expect(sample.count).toBe(sample.tokens.length);
      expect(sample.count).toBeGreaterThan(0);
    }
  });

  it('starts with no guesses recorded', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.samples.every((s) => s.guess === null)).toBe(true);
    expect(state.status).toBe('idle');
  });

  it('scores every exact guess at full accuracy', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(0), prepared);
    state.samples.forEach((s, i) => {
      state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: i, value: s.count });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('tokenCountAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('accepts guesses inside the configured tolerance', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(0), prepared);
    state.samples.forEach((s, i) => {
      state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: i, value: s.count + config.tolerance! });
    });
    expect(evaluate(state).value).toBeCloseTo(1);
  });

  it('rejects guesses outside the tolerance', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(0), prepared);
    state.samples.forEach((s, i) => {
      state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: i, value: s.count + config.tolerance! + 1 });
    });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('counts unanswered samples as wrong', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: 0, value: state.samples[0]!.count });
    expect(evaluate(state).value).toBeCloseTo(1 / state.samples.length);
  });

  it('ignores negative, fractional and out-of-range guesses', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: 0, value: -3 });
    expect(state.samples[0]!.guess).toBeNull();
    state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: 0, value: 2.5 });
    expect(state.samples[0]!.guess).toBeNull();
    state = applyAction(state, { type: 'GUESS_COUNT', sampleIndex: 99, value: 3 });
    expect(state.samples.every((s) => s.guess === null)).toBe(true);
  });
});

describe('tokenMergeEngine — merge puzzle', () => {
  const config = { ...configFor(1), words: ['lower'] };

  it('derives the real merge order from the tokenizer merge ranks', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    const puzzle = prepared.puzzles[0]!;

    expect(puzzle.symbols).toEqual(['l', 'o', 'w', 'e', 'r']);
    expect(puzzle.merges.map((m) => `${m.left} ${m.right}`)).toEqual([
      'l o',
      'w e',
      'lo we',
      'lowe r',
    ]);
  });

  it('respects maxMergesPerWord', async () => {
    const prepared = await prepare({ ...config, maxMergesPerWord: 2 }, { tokenizer: fakeTokenizer });
    expect(prepared.puzzles[0]!.merges).toHaveLength(2);
  });

  it('accepts the correct merge and advances the symbol list', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);

    state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position: 0 });
    expect(state.puzzles[0]!.symbols).toEqual(['lo', 'w', 'e', 'r']);
    expect(state.puzzles[0]!.correctSteps).toBe(1);
    expect(state.puzzles[0]!.attemptedSteps).toBe(1);
  });

  it('counts a wrong merge as attempted but not correct, and still applies it', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);

    // "o w" is not the lowest-ranked pair available; "l o" is.
    state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position: 1 });
    expect(state.puzzles[0]!.correctSteps).toBe(0);
    expect(state.puzzles[0]!.attemptedSteps).toBe(1);
    expect(state.puzzles[0]!.symbols).toEqual(['l', 'ow', 'e', 'r']);
  });

  it('scores a fully correct puzzle at 1', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);

    // l+o, then w+e (now at position 1), then lo+we, then lowe+r.
    for (const position of [0, 1, 0, 0]) {
      state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('mergeOrderAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
    expect(state.puzzles[0]!.done).toBe(true);
  });

  it('refuses merges once the puzzle is finished', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);
    for (const position of [0, 1, 0, 0]) {
      state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position });
    }
    const before = state.puzzles[0]!.attemptedSteps;
    state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position: 0 });
    expect(state.puzzles[0]!.attemptedSteps).toBe(before);
  });

  it('ignores a position with no right-hand neighbour', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);
    const last = state.puzzles[0]!.symbols.length - 1;
    state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position: last });
    expect(state.puzzles[0]!.attemptedSteps).toBe(0);
  });

  it('scores zero before any merges are made', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    expect(evaluate(initState(config, rulesFor(1), prepared)).value).toBe(0);
  });

  it('resets a puzzle back to characters', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'APPLY_MERGE', puzzleIndex: 0, position: 0 });
    state = applyAction(state, { type: 'RESET' });
    expect(state.puzzles[0]!.symbols).toEqual(['l', 'o', 'w', 'e', 'r']);
    expect(state.puzzles[0]!.correctSteps).toBe(0);
  });
});

describe('tokenMergeEngine — break the tokenizer', () => {
  const config = configFor(2);

  it('scores an efficient string near zero and a shattered one near one', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let efficient = initState(config, rulesFor(2), prepared);
    efficient = applyAction(efficient, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: 'a'.repeat(20), tokenCount: 5, charCount: 20 },
    });
    expect(evaluate(efficient).value).toBe(0);

    let shattered = initState(config, rulesFor(2), prepared);
    shattered = applyAction(shattered, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: '🧬'.repeat(10), tokenCount: 20, charCount: 20 },
    });
    expect(evaluate(shattered).value).toBe(1);
    expect(evaluate(shattered).stars).toBe(3);
  });

  it('keeps the best attempt across submissions', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(2), prepared);

    state = applyAction(state, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: 'x'.repeat(20), tokenCount: 15, charCount: 20 },
    });
    const best = evaluate(state).value;

    state = applyAction(state, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: 'y'.repeat(20), tokenCount: 5, charCount: 20 },
    });
    expect(evaluate(state).value).toBe(best);
  });

  it('rejects attempts outside the configured length bounds', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(2), prepared);

    state = applyAction(state, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: 'ab', tokenCount: 2, charCount: 2 },
    });
    state = applyAction(state, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: 'z'.repeat(200), tokenCount: 100, charCount: 200 },
    });
    expect(state.attempts).toHaveLength(0);
  });

  it('stops accepting attempts past the configured limit', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(2), prepared);

    for (let i = 0; i < config.attempts! + 3; i++) {
      state = applyAction(state, {
        type: 'SUBMIT_ATTEMPT',
        attempt: { text: `sample-text-${i}`, tokenCount: 6, charCount: 14 },
      });
    }
    expect(state.attempts).toHaveLength(config.attempts!);
  });

  it('ignores an attempt with a zero character count', async () => {
    const prepared = await prepare(config, { tokenizer: fakeTokenizer });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, {
      type: 'SUBMIT_ATTEMPT',
      attempt: { text: '', tokenCount: 0, charCount: 0 },
    });
    expect(state.attempts).toHaveLength(0);
    expect(evaluate(state).value).toBe(0);
  });
});

describe('tokenMergeEngine — level config coverage', () => {
  it('handles every shipped level end to end', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as TokenMergeConfig;
      const prepared = await prepare(config, { tokenizer: fakeTokenizer });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
