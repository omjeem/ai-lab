import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ContextDegradationConfig,
} from '@/engines/contextDegradationEngine';
import type { AttentionDep, AttentionResult, CausalLMDep, CorpusDep, TokenDistribution } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

const FACT = {
  id: 'venus',
  topic: 'Venus',
  sentences: ['A day on Venus takes 243 Earth days.'],
  query: 'How many Earth days?',
  answer: '243',
};
const DISTRACTOR_SAFE = {
  id: 'burj',
  topic: 'Burj Khalifa',
  sentences: ['The Burj Khalifa is 828 meters tall.'],
  query: 'How tall?',
  answer: '828',
};
const DISTRACTOR_RISKY = {
  id: 'statue',
  topic: 'Statue of Liberty',
  sentences: ['The statue is 93 meters tall.'],
  query: 'How tall?',
  answer: '93',
};

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') {
      return JSON.stringify({ facts: [{ ...FACT, answer: 'nowhere-in-passage' }] });
    }
    if (id === 'austen-sample') {
      return Array.from({ length: 2000 }, (_, i) => `filler${i}`).join(' ');
    }
    return JSON.stringify({ facts: [FACT, DISTRACTOR_SAFE, DISTRACTOR_RISKY] });
  },
};

/** Emits the real answer only once the prompt's own filler length is below a fixed real budget — mimics genuine length-driven degradation. */
function makeLengthSensitiveCausalLM(failAboveWords: number): CausalLMDep {
  return {
    async nextTokenDistribution(prompt): Promise<TokenDistribution> {
      const fillerCount = (prompt.match(/filler\d+/g) ?? []).length;
      const target = fillerCount > failAboveWords ? '?? \n\n' : '243\n\n';
      const markerIndex = prompt.lastIndexOf('Answer:') + 'Answer:'.length;
      const already = prompt.slice(markerIndex);
      if (already.length >= target.length) return { tokens: [], probs: [] };
      return { tokens: [target[already.length]!], probs: [1] };
    },
  };
}

/** One layer, one head; every query row is uniform filler except the real last-token query, which carries the real test weights. */
function makeAttention(tokens: string[], lastQueryWeights: number[]): AttentionResult {
  const n = tokens.length;
  const filler = new Array(n).fill(1 / n);
  const head = Array.from({ length: n }, (_, q) => (q === n - 1 ? lastQueryWeights : filler));
  return { tokens, attention: [[head]] };
}

const lengthRules: EngineRules = {
  passCriteria: { metric: 'retrievalPredictionAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const dilutionRules: EngineRules = {
  passCriteria: { metric: 'dilutionGuessAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 45,
};

const subsetRules: EngineRules = {
  passCriteria: { metric: 'attemptsToSolve', threshold: 3, comparator: 'lte' },
  starsRules: [
    { threshold: 3, stars: 1 },
    { threshold: 2, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 50,
};

describe('contextDegradationEngine — needle-haystack', () => {
  const config: ContextDegradationConfig = {
    mode: 'needle-haystack',
    corpus: 'retrieval-facts',
    factId: 'venus',
    fillerCorpus: 'austen-sample',
    fillerWordCounts: [0, 100, 1500],
    maxTokens: 10,
  };
  const causalLM = makeLengthSensitiveCausalLM(500);

  it('rejects a fact whose answer never appears in its own passage', async () => {
    const badConfig: ContextDegradationConfig = { ...config, corpus: 'bad-corpus' };
    await expect(prepare(badConfig, { corpus: fakeCorpus, causalLM })).rejects.toThrow();
  });

  it('computes real pass/fail per haystack length from a real decode, never authored', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM });
    expect(prepared.lengthRounds).toHaveLength(3);
    expect(prepared.lengthRounds[0]!.passed).toBe(true); // 0 filler words
    expect(prepared.lengthRounds[1]!.passed).toBe(true); // 100 filler words
    expect(prepared.lengthRounds[2]!.passed).toBe(false); // 1500 > 500 threshold
  });

  it('scores accuracy of the prediction against the real pass/fail outcome', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM });
    let state = initState(config, lengthRules, prepared);
    state = applyAction(state, { type: 'GUESS_LENGTH', roundIndex: 0, guess: true });
    state = applyAction(state, { type: 'GUESS_LENGTH', roundIndex: 1, guess: true });
    state = applyAction(state, { type: 'GUESS_LENGTH', roundIndex: 2, guess: true }); // wrong: real is false

    const result = evaluate(state);
    expect(result.metric).toBe('retrievalPredictionAccuracy');
    expect(result.value).toBeCloseTo(2 / 3);
  });

  it('does not count an unanswered round as correct', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM });
    const state = initState(config, lengthRules, prepared);
    expect(evaluate(state).value).toBe(0);
  });
});

describe('contextDegradationEngine — attention-dilution', () => {
  const config: ContextDegradationConfig = {
    mode: 'attention-dilution',
    sentences: ['a b c', 'a b c d'],
    targets: ['min', 'max'],
  };

  const attention: AttentionDep = {
    async attention(sentence: string) {
      const tokens = sentence.split(' ');
      // Last token queries all prior tokens; weights favour the first token.
      const row = tokens.map((_, i) => (i === tokens.length - 1 ? 0 : tokens.length - i));
      return makeAttention(tokens, row);
    },
  };

  it('builds one round per sentence with a real attention row over its own tokens', async () => {
    const prepared = await prepare(config, { attention });
    expect(prepared.dilutionRounds).toHaveLength(2);
    expect(prepared.dilutionRounds[0]!.tokens).toEqual(['a', 'b', 'c']);
    expect(prepared.dilutionRounds[0]!.queryIndex).toBe(2);
    expect(prepared.dilutionRounds[0]!.keyIndices).toEqual([0, 1]);
  });

  it('scores a correct min guess and an incorrect max guess', async () => {
    const prepared = await prepare(config, { attention });
    let state = initState(config, dilutionRules, prepared);
    // Round 0 target 'min': weights over keys [0,1] are [3,2] (descending from length) -> min is key 1 ("b").
    state = applyAction(state, { type: 'GUESS_DILUTION', roundIndex: 0, tokenIndex: 1 });
    // Round 1 target 'max': keys [0,1,2] weights [4,3,2] -> max is key 0 ("a"). Guess wrong (key 2).
    state = applyAction(state, { type: 'GUESS_DILUTION', roundIndex: 1, tokenIndex: 2 });

    const result = evaluate(state);
    expect(result.metric).toBe('dilutionGuessAccuracy');
    expect(result.value).toBe(0.5);
  });

  it('rejects a guess index outside the round\'s real candidate keys', async () => {
    const prepared = await prepare(config, { attention });
    let state = initState(config, dilutionRules, prepared);
    // queryIndex itself (2) is not a valid key to guess.
    state = applyAction(state, { type: 'GUESS_DILUTION', roundIndex: 0, tokenIndex: 2 });
    expect(state.dilutionRounds[0]!.guessIndex).toBeNull();
  });
});

describe('contextDegradationEngine — budget-subset', () => {
  const config: ContextDegradationConfig = {
    mode: 'budget-subset',
    corpus: 'retrieval-facts',
    targetFactId: 'venus',
    distractorFactIds: ['burj', 'statue'],
    budget: 2,
    maxTokens: 10,
  };

  it('prepares the target fact and every candidate distractor', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    expect(prepared.targetFact?.id).toBe('venus');
    expect(prepared.distractorFacts.map((f) => f.id).sort()).toEqual(['burj', 'statue']);
  });

  it('requires selecting a distractor and an order before a test can be recorded', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, subsetRules, prepared);
    state = applyAction(state, { type: 'TEST_SUBSET', decodedText: '243' });
    // No distractor selected yet — the attempt must not be recorded.
    expect(state.attempts).toBe(0);

    state = applyAction(state, { type: 'SELECT_DISTRACTOR', factId: 'burj' });
    state = applyAction(state, { type: 'TEST_SUBSET', decodedText: '243' });
    expect(state.attempts).toBe(1);
    expect(state.solved).toBe(true);
  });

  it('records the attempt number at which the real decode first succeeds', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, subsetRules, prepared);
    state = applyAction(state, { type: 'SELECT_DISTRACTOR', factId: 'statue' });
    state = applyAction(state, { type: 'TEST_SUBSET', decodedText: '93 meters, not what was asked' });
    expect(state.solved).toBe(false);
    state = applyAction(state, { type: 'SET_ORDER', order: 'distractor-first' });
    state = applyAction(state, { type: 'TEST_SUBSET', decodedText: 'It takes 243 Earth days' });
    expect(state.solved).toBe(true);
    expect(state.solvedAtAttempt).toBe(2);

    const result = evaluate(state);
    expect(result.metric).toBe('attemptsToSolve');
    expect(result.value).toBe(2);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(2);
  });

  it('scores the worst case (never solved) as failing', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, subsetRules, prepared);
    const result = evaluate(state);
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
  });
});

describe('contextDegradationEngine — reset never re-runs the model', () => {
  it('RESET restores initial state from already-prepared data', async () => {
    let calls = 0;
    const countingCausalLM: CausalLMDep = {
      async nextTokenDistribution(prompt) {
        calls++;
        return makeLengthSensitiveCausalLM(500).nextTokenDistribution(prompt, 1);
      },
    };
    const config: ContextDegradationConfig = {
      mode: 'needle-haystack',
      corpus: 'retrieval-facts',
      factId: 'venus',
      fillerCorpus: 'austen-sample',
      fillerWordCounts: [0],
      maxTokens: 5,
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: countingCausalLM });
    const callsAfterPrepare = calls;

    let state = initState(config, lengthRules, prepared);
    state = applyAction(state, { type: 'GUESS_LENGTH', roundIndex: 0, guess: true });
    state = applyAction(state, { type: 'RESET' });

    expect(calls).toBe(callsAfterPrepare);
    expect(state.lengthRounds.every((r) => r.guess === null)).toBe(true);
  });
});

describe('contextDegradationEngine — submit', () => {
  it('marks the run complete', async () => {
    const config: ContextDegradationConfig = {
      mode: 'needle-haystack',
      corpus: 'retrieval-facts',
      factId: 'venus',
      fillerCorpus: 'austen-sample',
      fillerWordCounts: [0],
      maxTokens: 5,
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeLengthSensitiveCausalLM(500) });
    let state = initState(config, lengthRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
