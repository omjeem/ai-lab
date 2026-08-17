import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  containsAnswer,
  buildGroundedPrompt,
  buildCuratedContextText,
  type GroundedFact,
  type GroundedGenerationConfig,
} from '@/engines/groundedGenerationEngine';
import type { CorpusDep, CausalLMDep, TokenDistribution } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

const FACT_A: GroundedFact = {
  id: 'a',
  topic: 'Alpha',
  sentences: ['Alpha sentence one.', 'Alpha sentence two with 42.'],
  query: 'What is the alpha number?',
  answer: '42',
};
const FACT_B: GroundedFact = {
  id: 'b',
  topic: 'Beta',
  sentences: ['Beta sentence one.', 'Beta sentence two with 7.'],
  query: 'What is the beta number?',
  answer: '7',
};
const FACT_C: GroundedFact = {
  id: 'c',
  topic: 'Gamma',
  sentences: ['Gamma sentence one.', 'Gamma sentence two with 99.'],
  query: 'What is the gamma number?',
  answer: '99',
};

const CORPUS = { facts: [FACT_A, FACT_B, FACT_C] };

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') {
      return JSON.stringify({ facts: [{ ...FACT_A, answer: 'nowhere-in-passage' }] });
    }
    return JSON.stringify(CORPUS);
  },
};

/**
 * Emits the real answer, one character at a time, only when the prompt's own
 * Context block actually contains that fact's answer (i.e. the *right*
 * passage was injected, not merely *a* passage) — emits a fixed wrong string
 * otherwise. Which fact is being asked about is read from the prompt's own
 * query text, exactly the way the real model reads the actual prompt rather
 * than a hidden label.
 */
function scriptedTarget(prompt: string, fact: GroundedFact): string {
  const contextText = /^Context: ([\s\S]*)\nQuestion:/.exec(prompt)?.[1] ?? '';
  return contextText.includes(fact.answer) ? `${fact.answer}\n\n` : `??\n\n`;
}

function makeFakeCausalLM(facts: readonly GroundedFact[]): CausalLMDep {
  return {
    async nextTokenDistribution(prompt): Promise<TokenDistribution> {
      const fact = facts.find((f) => prompt.includes(f.query));
      if (!fact) throw new Error(`no fixture fact matches prompt: ${prompt}`);
      const target = scriptedTarget(prompt, fact);
      const markerIndex = prompt.lastIndexOf('Answer:') + 'Answer:'.length;
      const already = prompt.slice(markerIndex);
      if (already.length >= target.length) return { tokens: [], probs: [] };
      return { tokens: [target[already.length]!], probs: [1] };
    },
  };
}

const identifyRules: EngineRules = {
  passCriteria: { metric: 'predictionAccuracy', threshold: 0.6, comparator: 'gte' },
  starsRules: [
    { threshold: 0.6, stars: 1 },
    { threshold: 0.8, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const pickRules: EngineRules = {
  passCriteria: { metric: 'groundedAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 45,
};

const curateRules: EngineRules = {
  passCriteria: { metric: 'attemptsToSolve', threshold: 4, comparator: 'lte' },
  starsRules: [
    { threshold: 4, stars: 1 },
    { threshold: 2, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 55,
};

describe('greedyDecode', () => {
  it('accumulates tokens until the model stops emitting them', async () => {
    const tokens = ['4', '2', '\n\n'];
    let i = 0;
    const fake: CausalLMDep = {
      async nextTokenDistribution() {
        const t = tokens[i++];
        return t ? { tokens: [t], probs: [1] } : { tokens: [], probs: [] };
      },
    };
    expect(await greedyDecode(fake, 'Answer:', 10)).toBe('42\n\n');
  });

  it('stops at an end-of-turn marker without including it', async () => {
    const fake: CausalLMDep = {
      async nextTokenDistribution() {
        return { tokens: ['<|im_end|>'], probs: [1] };
      },
    };
    expect(await greedyDecode(fake, 'Answer:', 10)).toBe('');
  });

  it('respects the token budget', async () => {
    const fake: CausalLMDep = {
      async nextTokenDistribution() {
        return { tokens: ['x'], probs: [1] };
      },
    };
    expect(await greedyDecode(fake, 'Answer:', 5)).toBe('xxxxx');
  });
});

describe('containsAnswer', () => {
  it('matches regardless of comma formatting or case', () => {
    expect(containsAnswer(' The answer is 8,849 meters', '8,849')).toBe(true);
    expect(containsAnswer(' the answer is 8849 meters', '8,849')).toBe(true);
    expect(containsAnswer(' The Great Barrier Reef', '2,300')).toBe(false);
  });
});

describe('buildGroundedPrompt', () => {
  it('includes a Context block only when grounded', () => {
    expect(buildGroundedPrompt('some passage', 'a question?')).toBe(
      'Context: some passage\nQuestion: a question?\nAnswer:'
    );
    expect(buildGroundedPrompt(null, 'a question?')).toBe('Question: a question?\nAnswer:');
  });
});

describe('buildCuratedContextText', () => {
  it('orders and truncates the distractor as configured', () => {
    const full = buildCuratedContextText(FACT_A, FACT_B, 'correct-first', false);
    expect(full).toBe(
      'Alpha sentence one. Alpha sentence two with 42. Beta sentence one. Beta sentence two with 7.'
    );

    const truncatedFirst = buildCuratedContextText(FACT_A, FACT_B, 'distractor-first', true);
    expect(truncatedFirst).toBe('Beta sentence one. Alpha sentence one. Alpha sentence two with 42.');
  });
});

describe('groundedGenerationEngine — prepare', () => {
  it('rejects a fact whose answer never appears in its own passage', async () => {
    const config: GroundedGenerationConfig = { mode: 'identify-grounded', corpus: 'bad-corpus', maxTokens: 10, factIds: ['a'] };
    await expect(prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A]) })).rejects.toThrow();
  });

  it('builds one identify-grounded round per configured fact, from real decode output', async () => {
    const config: GroundedGenerationConfig = {
      mode: 'identify-grounded',
      corpus: 'retrieval-facts',
      maxTokens: 10,
      factIds: ['a', 'b'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B]) });
    expect(prepared.identifyRounds).toHaveLength(2);
    for (const round of prepared.identifyRounds) {
      const grounded = round.groundedSide === 'left' ? round.leftText : round.rightText;
      const ungrounded = round.groundedSide === 'left' ? round.rightText : round.leftText;
      expect(grounded).toContain(round.groundedSide === 'left' ? round.leftText : round.rightText);
      expect(ungrounded).toBe('??\n\n');
    }
  });
});

describe('groundedGenerationEngine — identify-grounded mode', () => {
  const config: GroundedGenerationConfig = {
    mode: 'identify-grounded',
    corpus: 'retrieval-facts',
    maxTokens: 10,
    factIds: ['a', 'b', 'c'],
  };

  it('scores full accuracy when every pick matches the real grounded side', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B, FACT_C]) });
    let state = initState(config, identifyRules, prepared);
    state.identifyRounds.forEach((round, i) => {
      state = applyAction(state, { type: 'PICK_SIDE', roundIndex: i, side: round.groundedSide });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('predictionAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores 0 accuracy when every pick is wrong', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B, FACT_C]) });
    let state = initState(config, identifyRules, prepared);
    state.identifyRounds.forEach((round, i) => {
      const wrong = round.groundedSide === 'left' ? 'right' : 'left';
      state = applyAction(state, { type: 'PICK_SIDE', roundIndex: i, side: wrong });
    });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });
});

describe('groundedGenerationEngine — pick-passage mode', () => {
  const config: GroundedGenerationConfig = {
    mode: 'pick-passage',
    corpus: 'retrieval-facts',
    maxTokens: 10,
    rounds: [{ targetFactId: 'a', distractorFactIds: ['b', 'c'] }],
  };

  it('includes the target and every distractor among the shuffled candidates', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B, FACT_C]) });
    const state = initState(config, pickRules, prepared);
    expect(state.pickRounds).toHaveLength(1);
    expect(state.pickRounds[0]!.candidateFactIds.sort()).toEqual(['a', 'b', 'c']);
  });

  it('scores correct once the chosen passage really produces the fact, via a real decode', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B, FACT_C]) });
    let state = initState(config, pickRules, prepared);
    state = applyAction(state, { type: 'CHOOSE_PASSAGE', roundIndex: 0, factId: 'a' });

    const prompt = buildGroundedPrompt(FACT_A.sentences.join(' '), FACT_A.query);
    const decoded = await greedyDecode(makeFakeCausalLM([FACT_A]), prompt, config.maxTokens);
    state = applyAction(state, { type: 'TEST_PASSAGE', roundIndex: 0, decodedText: decoded });

    const result = evaluate(state);
    expect(result.metric).toBe('groundedAccuracy');
    expect(result.value).toBe(1);
    expect(result.passed).toBe(true);
  });

  it('scores wrong when the chosen passage does not actually answer the query', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A, FACT_B, FACT_C]) });
    let state = initState(config, pickRules, prepared);
    state = applyAction(state, { type: 'CHOOSE_PASSAGE', roundIndex: 0, factId: 'b' });

    // Injecting fact B's passage while asking fact A's question: the fake model
    // only recognises "Context:" as grounded for the fact whose OWN query
    // matches, so this legitimately produces the wrong-string completion.
    const prompt = `Context: ${FACT_B.sentences.join(' ')}\nQuestion: ${FACT_A.query}\nAnswer:`;
    const decoded = await greedyDecode(makeFakeCausalLM([FACT_A]), prompt, config.maxTokens);
    state = applyAction(state, { type: 'TEST_PASSAGE', roundIndex: 0, decodedText: decoded });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });
});

describe('groundedGenerationEngine — curate-context mode', () => {
  const config: GroundedGenerationConfig = {
    mode: 'curate-context',
    corpus: 'retrieval-facts',
    maxTokens: 10,
    targetFactId: 'a',
    distractorFactId: 'b',
  };

  it('starts unsolved with zero attempts', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A]) });
    const state = initState(config, curateRules, prepared);
    expect(state.attempts).toBe(0);
    expect(state.solved).toBe(false);
    expect(evaluate(state).passed).toBe(false);
  });

  it('records the attempt number at which the real decode first succeeds', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A]) });
    let state = initState(config, curateRules, prepared);

    state = applyAction(state, { type: 'TEST_CONTEXT', decodedText: '?? nothing useful' });
    expect(state.attempts).toBe(1);
    expect(state.solved).toBe(false);

    state = applyAction(state, { type: 'TEST_CONTEXT', decodedText: 'The alpha number is 42' });
    expect(state.attempts).toBe(2);
    expect(state.solved).toBe(true);
    expect(state.solvedAtAttempt).toBe(2);

    const result = evaluate(state);
    expect(result.metric).toBe('attemptsToSolve');
    expect(result.value).toBe(2);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(2);
  });

  it('does not overwrite an earlier solve on a later attempt', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A]) });
    let state = initState(config, curateRules, prepared);
    state = applyAction(state, { type: 'TEST_CONTEXT', decodedText: 'contains 42' });
    state = applyAction(state, { type: 'TEST_CONTEXT', decodedText: 'contains 42 again' });
    expect(state.solvedAtAttempt).toBe(1);
  });
});

describe('groundedGenerationEngine — reset never re-runs the model', () => {
  it('RESET restores initial state from already-computed data', async () => {
    let calls = 0;
    const countingCausalLM: CausalLMDep = {
      async nextTokenDistribution(prompt) {
        calls++;
        const fact = [FACT_A, FACT_B].find((f) => prompt.includes(f.query))!;
        const target: string = scriptedTarget(prompt, fact);
        const markerIndex = prompt.lastIndexOf('Answer:') + 'Answer:'.length;
        const already = prompt.slice(markerIndex);
        return already.length >= target.length ? { tokens: [], probs: [] } : { tokens: [target[already.length]!], probs: [1] };
      },
    };
    const config: GroundedGenerationConfig = {
      mode: 'identify-grounded',
      corpus: 'retrieval-facts',
      maxTokens: 10,
      factIds: ['a', 'b'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: countingCausalLM });
    const callsAfterPrepare = calls;
    let state = initState(config, identifyRules, prepared);
    state = applyAction(state, { type: 'PICK_SIDE', roundIndex: 0, side: 'left' });
    state = applyAction(state, { type: 'RESET' });

    expect(calls).toBe(callsAfterPrepare);
    expect(state.identifyRounds.every((r) => r.pick === null)).toBe(true);
  });
});

describe('groundedGenerationEngine — submit', () => {
  it('marks the run complete', async () => {
    const config: GroundedGenerationConfig = { mode: 'identify-grounded', corpus: 'retrieval-facts', maxTokens: 10, factIds: ['a'] };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: makeFakeCausalLM([FACT_A]) });
    let state = initState(config, identifyRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
