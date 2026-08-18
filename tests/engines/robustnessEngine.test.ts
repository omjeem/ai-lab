import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type RobustnessConfig,
} from '@/engines/robustnessEngine';
import type { CausalLMDep, CorpusDep, TokenDistribution } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

const PROMPTS = {
  instruction: 'RULE: answer in one word.',
  query: 'Q?',
  attacks: [
    { id: 'atk-good', label: 'Good Attack', text: 'ATTACKTEXT-GOOD' },
    { id: 'atk-bad', label: 'Bad Attack', text: 'ATTACKTEXT-BAD' },
    { id: 'atk-other', label: 'Other Attack', text: 'ATTACKTEXT-OTHER' },
  ],
  defenses: [
    { id: 'def-good', label: 'Good Defense', text: 'DEFENSETEXT-GOOD' },
    { id: 'def-bad', label: 'Bad Defense', text: 'DEFENSETEXT-BAD' },
  ],
};

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') return JSON.stringify({ ...PROMPTS, attacks: [], defenses: [] });
    return JSON.stringify(PROMPTS);
  },
};

interface Script {
  match: (prompt: string) => boolean;
  tokens: string[];
}

/** Emits a scripted answer one real-shaped token at a time, keyed by whichever
 * script's `match` first fires on the live prompt — mirrors calling a real
 * CausalLMDep repeatedly with a growing prompt (same convention as
 * tests/engines/calibrationEngine.test.ts's `scriptedCausalLM`). */
function scriptedCausalLM(scripts: Script[]): CausalLMDep {
  return {
    async nextTokenDistribution(prompt: string): Promise<TokenDistribution> {
      const script = scripts.find((s) => s.match(prompt));
      if (!script) return { tokens: [], probs: [] };
      const markerIndex = prompt.lastIndexOf('Answer:') + 'Answer:'.length;
      const already = prompt.slice(markerIndex);
      let cumulative = '';
      for (const token of script.tokens) {
        if (cumulative === already) return { tokens: [token], probs: [1] };
        cumulative += token;
      }
      return { tokens: [], probs: [] };
    },
  };
}

// find-attack: ATTACKTEXT-GOOD really breaks the one-word rule; ATTACKTEXT-BAD
// and ATTACKTEXT-OTHER don't.
const findAttackCausalLM = scriptedCausalLM([
  { match: (p) => p.includes('ATTACKTEXT-GOOD'), tokens: [' many', ' words', ' here', '\n\n'] },
  { match: (p) => p.includes('ATTACKTEXT-BAD'), tokens: [' Word', '\n\n'] },
  { match: (p) => p.includes('ATTACKTEXT-OTHER'), tokens: [' Word', '\n\n'] },
]);

// test-defense: against the fixed attack (ATTACKTEXT-GOOD), DEFENSETEXT-GOOD
// really resists; DEFENSETEXT-BAD doesn't.
const testDefenseCausalLM = scriptedCausalLM([
  { match: (p) => p.includes('DEFENSETEXT-GOOD'), tokens: [' Word', '\n\n'] },
  { match: (p) => p.includes('DEFENSETEXT-BAD'), tokens: [' many', ' words', '\n\n'] },
]);

// defense-transfer: the fixed defense (DEFENSETEXT-GOOD) resists atk-bad but
// fails against atk-other.
const transferCausalLM = scriptedCausalLM([
  { match: (p) => p.includes('ATTACKTEXT-BAD'), tokens: [' Word', '\n\n'] },
  { match: (p) => p.includes('ATTACKTEXT-OTHER'), tokens: [' fails', ' here', '\n\n'] },
]);

const attemptsRules: EngineRules = {
  passCriteria: { metric: 'attemptsToSolve', threshold: 4, comparator: 'lte' },
  starsRules: [
    { threshold: 4, stars: 1 },
    { threshold: 2, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const transferRules: EngineRules = {
  passCriteria: { metric: 'transferPredictionAccuracy', threshold: 0.3, comparator: 'gte' },
  starsRules: [
    { threshold: 0.3, stars: 1 },
    { threshold: 0.6, stars: 2 },
    { threshold: 0.9, stars: 3 },
  ],
  xpReward: 50,
};

describe('robustnessEngine — find-attack', () => {
  const config: RobustnessConfig = {
    mode: 'find-attack',
    corpus: 'prompts',
    maxTokens: 10,
    expectedWordCount: 1,
    attackIds: ['atk-bad', 'atk-good', 'atk-other'],
  };

  it('skips an attack id that does not exist in the corpus rather than fabricating one', async () => {
    const badConfig: RobustnessConfig = { ...config, corpus: 'bad-corpus' };
    const prepared = await prepare(badConfig, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    expect(prepared.attackRounds).toHaveLength(0);
  });

  it('computes real violation per attack from a real decode, never authored', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    expect(prepared.attackRounds).toHaveLength(3);
    const good = prepared.attackRounds.find((r) => r.id === 'atk-good')!;
    const bad = prepared.attackRounds.find((r) => r.id === 'atk-bad')!;
    expect(good.violates).toBe(true);
    expect(bad.violates).toBe(false);
  });

  it('scores attemptsToSolve, only counting attempts up to the real solve', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    let state = initState(config, attemptsRules, prepared);
    state = applyAction(state, { type: 'TEST_ATTACK', id: 'atk-bad' }); // fails
    expect(state.solved).toBe(false);
    state = applyAction(state, { type: 'TEST_ATTACK', id: 'atk-good' }); // real success
    expect(state.solved).toBe(true);
    expect(state.solvedAtAttempt).toBe(2);

    const result = evaluate(state);
    expect(result.metric).toBe('attemptsToSolve');
    expect(result.value).toBe(2);
  });

  it('does not count re-testing an already-tested attack as a new attempt', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    let state = initState(config, attemptsRules, prepared);
    state = applyAction(state, { type: 'TEST_ATTACK', id: 'atk-bad' });
    state = applyAction(state, { type: 'TEST_ATTACK', id: 'atk-bad' });
    expect(state.attempts).toBe(1);
  });

  it('scores the unsolved case as failing', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    const state = initState(config, attemptsRules, prepared);
    expect(evaluate(state).passed).toBe(false);
  });
});

describe('robustnessEngine — test-defense', () => {
  const config: RobustnessConfig = {
    mode: 'test-defense',
    corpus: 'prompts',
    maxTokens: 10,
    expectedWordCount: 1,
    fixedAttackId: 'atk-good',
    defenseIds: ['def-bad', 'def-good'],
  };

  it('computes real resistance per defense against the fixed real attack', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: testDefenseCausalLM });
    const good = prepared.defenseRounds.find((r) => r.id === 'def-good')!;
    const bad = prepared.defenseRounds.find((r) => r.id === 'def-bad')!;
    expect(good.resists).toBe(true);
    expect(bad.resists).toBe(false);
  });

  it('throws when the fixed attack id does not exist in the corpus', async () => {
    const badConfig: RobustnessConfig = { ...config, fixedAttackId: 'nope' };
    await expect(prepare(badConfig, { corpus: fakeCorpus, causalLM: testDefenseCausalLM })).rejects.toThrow();
  });

  it('scores attemptsToSolve for finding a real working defense', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: testDefenseCausalLM });
    let state = initState(config, attemptsRules, prepared);
    state = applyAction(state, { type: 'TEST_DEFENSE', id: 'def-bad' });
    state = applyAction(state, { type: 'TEST_DEFENSE', id: 'def-good' });
    expect(evaluate(state).value).toBe(2);
  });
});

describe('robustnessEngine — defense-transfer', () => {
  const config: RobustnessConfig = {
    mode: 'defense-transfer',
    corpus: 'prompts',
    maxTokens: 10,
    expectedWordCount: 1,
    fixedDefenseId: 'def-good',
    transferAttackIds: ['atk-bad', 'atk-other'],
  };

  it('computes real resistance of the fixed defense against each transfer attack', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: transferCausalLM });
    expect(prepared.transferRounds).toHaveLength(2);
    const resistsBad = prepared.transferRounds.find((r) => r.id === 'atk-bad')!;
    const failsOther = prepared.transferRounds.find((r) => r.id === 'atk-other')!;
    expect(resistsBad.resists).toBe(true);
    expect(failsOther.resists).toBe(false);
  });

  it('scores real prediction accuracy against generalisation, not the player\'s hope', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: transferCausalLM });
    let state = initState(config, transferRules, prepared);
    const badIdx = state.transferRounds.findIndex((r) => r.id === 'atk-bad');
    const otherIdx = state.transferRounds.findIndex((r) => r.id === 'atk-other');
    state = applyAction(state, { type: 'GUESS_TRANSFER', roundIndex: badIdx, guess: true }); // correct
    state = applyAction(state, { type: 'GUESS_TRANSFER', roundIndex: otherIdx, guess: true }); // wrong: real is false

    const result = evaluate(state);
    expect(result.metric).toBe('transferPredictionAccuracy');
    expect(result.value).toBe(0.5);
  });
});

describe('robustnessEngine — reset never re-runs the model', () => {
  it('RESET restores initial state from already-prepared data', async () => {
    let calls = 0;
    const countingCausalLM: CausalLMDep = {
      async nextTokenDistribution(prompt, topK) {
        calls++;
        return findAttackCausalLM.nextTokenDistribution(prompt, topK);
      },
    };
    const config: RobustnessConfig = {
      mode: 'find-attack',
      corpus: 'prompts',
      maxTokens: 10,
      expectedWordCount: 1,
      attackIds: ['atk-good'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: countingCausalLM });
    const callsAfterPrepare = calls;

    let state = initState(config, attemptsRules, prepared);
    state = applyAction(state, { type: 'TEST_ATTACK', id: 'atk-good' });
    state = applyAction(state, { type: 'RESET' });

    expect(calls).toBe(callsAfterPrepare);
    expect(state.attackRounds.every((r) => !r.tested)).toBe(true);
    expect(state.attempts).toBe(0);
  });
});

describe('robustnessEngine — submit', () => {
  it('marks the run complete', async () => {
    const config: RobustnessConfig = {
      mode: 'find-attack',
      corpus: 'prompts',
      maxTokens: 10,
      expectedWordCount: 1,
      attackIds: ['atk-good'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: findAttackCausalLM });
    let state = initState(config, attemptsRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
