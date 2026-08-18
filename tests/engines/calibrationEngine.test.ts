import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  realDropped,
  type CalibrationConfig,
} from '@/engines/calibrationEngine';
import type { CausalLMDep, CorpusDep, TokenDistribution } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

const FACT_RIGHT = {
  id: 'right-fact',
  topic: 'Right Fact',
  sentences: ['The real answer is forty-two.'],
  query: 'What is the real answer?',
  answer: 'forty-two',
};
const FACT_WRONG_LOW = {
  id: 'wrong-low',
  topic: 'Wrong Low Confidence',
  sentences: ['The real answer is nine.'],
  query: 'How many are there?',
  answer: 'nine',
};
const FACT_WRONG_HIGH = {
  id: 'wrong-high',
  topic: 'Wrong High Confidence',
  sentences: ['The real answer is ten.'],
  query: 'What is the count?',
  answer: 'ten',
};
const FACT_BAD_ANSWER = {
  id: 'bad-answer',
  topic: 'Bad',
  sentences: ['Nothing relevant here.'],
  query: 'irrelevant',
  answer: 'nowhere-in-passage',
};

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') return JSON.stringify({ facts: [FACT_BAD_ANSWER] });
    return JSON.stringify({ facts: [FACT_RIGHT, FACT_WRONG_LOW, FACT_WRONG_HIGH] });
  },
};

interface Script {
  match: (prompt: string) => boolean;
  tokens: string[];
  probs: number[];
}

/** Emits a scripted, deterministic answer one real-shaped token at a time, keyed by whichever script's `match` first fires on the live prompt — mirrors how a real CausalLMDep is called repeatedly with a growing prompt. */
function scriptedCausalLM(scripts: Script[]): CausalLMDep {
  return {
    async nextTokenDistribution(prompt: string): Promise<TokenDistribution> {
      const script = scripts.find((s) => s.match(prompt));
      if (!script) return { tokens: [], probs: [] };
      const markerIndex = prompt.lastIndexOf('Answer:') + 'Answer:'.length;
      const already = prompt.slice(markerIndex);
      let cumulative = '';
      for (let i = 0; i < script.tokens.length; i++) {
        if (cumulative === already) return { tokens: [script.tokens[i]!], probs: [script.probs[i]!] };
        cumulative += script.tokens[i];
      }
      return { tokens: [], probs: [] };
    },
  };
}

// Baseline (ungrounded) scripts: right-fact answers correctly at 0.9 mean
// confidence; wrong-low answers incorrectly at low confidence; wrong-high
// answers incorrectly but at the highest confidence of the three — the
// "dangerous hallucination" case spot-hallucination mode must find.
const baselineCausalLM = scriptedCausalLM([
  { match: (p) => p.includes('real answer?'), tokens: [' forty-two', '\n\n'], probs: [0.9, 0.9] },
  { match: (p) => p.includes('How many are there?'), tokens: [' seven', '\n\n'], probs: [0.2, 0.2] },
  { match: (p) => p.includes('What is the count?'), tokens: [' fifty', '\n\n'], probs: [0.95, 0.95] },
]);

const predictRules: EngineRules = {
  passCriteria: { metric: 'confidenceCorrectnessAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const spotRules: EngineRules = {
  passCriteria: { metric: 'hallucinationSpotAccuracy', threshold: 0.3, comparator: 'gte' },
  starsRules: [
    { threshold: 0.3, stars: 1 },
    { threshold: 0.6, stars: 2 },
    { threshold: 0.9, stars: 3 },
  ],
  xpReward: 45,
};

const dropRules: EngineRules = {
  passCriteria: { metric: 'confidenceDropPredictionAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.65, stars: 2 },
    { threshold: 0.9, stars: 3 },
  ],
  xpReward: 50,
};

describe('calibrationEngine — predict-correctness', () => {
  const config: CalibrationConfig = {
    mode: 'predict-correctness',
    factSources: ['facts'],
    maxTokens: 5,
    factIds: ['right-fact', 'wrong-low', 'wrong-high'],
  };

  it('rejects a fact whose answer never appears in its own passage', async () => {
    const badConfig: CalibrationConfig = { ...config, factSources: ['bad-corpus'], factIds: ['bad-answer'] };
    await expect(prepare(badConfig, { corpus: fakeCorpus, causalLM: baselineCausalLM })).rejects.toThrow();
  });

  it('computes real correctness and confidence per fact from real decodes, never authored', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    expect(prepared.predictRounds).toHaveLength(3);
    const right = prepared.predictRounds.find((r) => r.factId === 'right-fact')!;
    const wrongLow = prepared.predictRounds.find((r) => r.factId === 'wrong-low')!;
    const wrongHigh = prepared.predictRounds.find((r) => r.factId === 'wrong-high')!;
    expect(right.correct).toBe(true);
    expect(wrongLow.correct).toBe(false);
    expect(wrongHigh.correct).toBe(false);
    // Confidence excludes the leading token (formatting-token noise, per 8-1's lesson)
    // and averages the rest — here both real tokens share one value, so it's unchanged.
    expect(right.confidence).toBeCloseTo(0.9);
    expect(wrongLow.confidence).toBeCloseTo(0.2);
    expect(wrongHigh.confidence).toBeCloseTo(0.95);
  });

  it('scores prediction accuracy against real correctness, not the player\'s intuition', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    let state = initState(config, predictRules, prepared);
    const rightIdx = state.predictRounds.findIndex((r) => r.factId === 'right-fact');
    const wrongLowIdx = state.predictRounds.findIndex((r) => r.factId === 'wrong-low');
    const wrongHighIdx = state.predictRounds.findIndex((r) => r.factId === 'wrong-high');

    state = applyAction(state, { type: 'GUESS_CORRECTNESS', roundIndex: rightIdx, guess: true }); // correct
    state = applyAction(state, { type: 'GUESS_CORRECTNESS', roundIndex: wrongLowIdx, guess: false }); // correct
    state = applyAction(state, { type: 'GUESS_CORRECTNESS', roundIndex: wrongHighIdx, guess: true }); // wrong: real is false, high confidence trap

    const result = evaluate(state);
    expect(result.metric).toBe('confidenceCorrectnessAccuracy');
    expect(result.value).toBeCloseTo(2 / 3);
  });

  it('does not count an unanswered round as correct', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    const state = initState(config, predictRules, prepared);
    expect(evaluate(state).value).toBe(0);
  });
});

describe('calibrationEngine — spot-hallucination', () => {
  const config: CalibrationConfig = {
    mode: 'spot-hallucination',
    factSources: ['facts'],
    maxTokens: 5,
    rounds: [{ candidateFactIds: ['right-fact', 'wrong-low', 'wrong-high'] }],
  };

  it('finds the real wrong-and-most-confident candidate per round, never an authored key', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    expect(prepared.spotRounds).toHaveLength(1);
    // wrong-high (0.95) beats wrong-low (0.2) among the WRONG candidates —
    // right-fact's 0.9 doesn't count, it's actually correct.
    expect(prepared.spotRounds[0]!.targetFactId).toBe('wrong-high');
  });

  it('throws if a configured round has no real wrong candidate to find', async () => {
    const allRightCausalLM = scriptedCausalLM([
      { match: () => true, tokens: [' forty-two', '\n\n'], probs: [0.9, 0.9] },
    ]);
    const onlyRight: CalibrationConfig = { ...config, rounds: [{ candidateFactIds: ['right-fact'] }] };
    await expect(prepare(onlyRight, { corpus: fakeCorpus, causalLM: allRightCausalLM })).rejects.toThrow();
  });

  it('scores accuracy of guessing the real dangerous hallucination', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    let state = initState(config, spotRules, prepared);
    state = applyAction(state, { type: 'GUESS_HALLUCINATION', roundIndex: 0, factId: 'wrong-low' }); // wrong guess
    let result = evaluate(state);
    expect(result.metric).toBe('hallucinationSpotAccuracy');
    expect(result.value).toBe(0);

    state = applyAction(state, { type: 'GUESS_HALLUCINATION', roundIndex: 0, factId: 'wrong-high' }); // corrected
    result = evaluate(state);
    expect(result.value).toBe(1);
  });

  it('rejects a guess for a factId outside that round\'s real candidates', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    let state = initState(config, spotRules, prepared);
    state = applyAction(state, { type: 'GUESS_HALLUCINATION', roundIndex: 0, factId: 'not-a-candidate' });
    expect(state.spotRounds[0]!.guessFactId).toBeNull();
  });
});

describe('calibrationEngine — reduce-confidence', () => {
  const framingTemplate = 'Only answer if certain. Question: {query}\nAnswer:';

  // Framed prompts route through the SAME scripted lookup — 'wrong-low'
  // becomes more confident when framed (a real "defense backfires" case),
  // 'wrong-high' becomes less confident (the defense working) — both real,
  // divergent outcomes the level's prediction task must actually resolve.
  const framingCausalLM = scriptedCausalLM([
    { match: (p) => p.includes('Only answer') && p.includes('How many are there?'), tokens: [' nine', '\n\n'], probs: [0.8, 0.8] },
    { match: (p) => p.includes('Only answer') && p.includes('What is the count?'), tokens: [' ten', '\n\n'], probs: [0.1, 0.1] },
    { match: (p) => p.includes('How many are there?'), tokens: [' nine', '\n\n'], probs: [0.2, 0.2] },
    { match: (p) => p.includes('What is the count?'), tokens: [' ten', '\n\n'], probs: [0.95, 0.95] },
  ]);

  const config: CalibrationConfig = {
    mode: 'reduce-confidence',
    factSources: ['facts'],
    maxTokens: 5,
    targetFactIds: ['wrong-low', 'wrong-high'],
    framingTemplate,
  };

  it('computes real baseline and framed confidence from two real decodes per fact', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: framingCausalLM });
    expect(prepared.framingRounds).toHaveLength(2);
    const low = prepared.framingRounds.find((r) => r.factId === 'wrong-low')!;
    const high = prepared.framingRounds.find((r) => r.factId === 'wrong-high')!;
    expect(low.confidence).toBeCloseTo(0.2);
    expect(low.framedConfidence).toBeCloseTo(0.8);
    expect(high.confidence).toBeCloseTo(0.95);
    expect(high.framedConfidence).toBeCloseTo(0.1);
    expect(realDropped(low)).toBe(false); // confidence rose — a real backfire
    expect(realDropped(high)).toBe(true); // confidence dropped — the defense worked
  });

  it('scores accuracy of predicting the real drop/rise direction', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: framingCausalLM });
    let state = initState(config, dropRules, prepared);
    const lowIdx = state.framingRounds.findIndex((r) => r.factId === 'wrong-low');
    const highIdx = state.framingRounds.findIndex((r) => r.factId === 'wrong-high');

    state = applyAction(state, { type: 'GUESS_DELTA', roundIndex: lowIdx, guess: 'drop' }); // wrong: it rose
    state = applyAction(state, { type: 'GUESS_DELTA', roundIndex: highIdx, guess: 'drop' }); // correct

    const result = evaluate(state);
    expect(result.metric).toBe('confidenceDropPredictionAccuracy');
    expect(result.value).toBe(0.5);
  });
});

describe('calibrationEngine — reset never re-runs the model', () => {
  it('RESET restores initial state from already-prepared data', async () => {
    let calls = 0;
    const countingCausalLM: CausalLMDep = {
      async nextTokenDistribution(prompt, topK) {
        calls++;
        return baselineCausalLM.nextTokenDistribution(prompt, topK);
      },
    };
    const config: CalibrationConfig = {
      mode: 'predict-correctness',
      factSources: ['facts'],
      maxTokens: 5,
      factIds: ['right-fact'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: countingCausalLM });
    const callsAfterPrepare = calls;

    let state = initState(config, predictRules, prepared);
    state = applyAction(state, { type: 'GUESS_CORRECTNESS', roundIndex: 0, guess: true });
    state = applyAction(state, { type: 'RESET' });

    expect(calls).toBe(callsAfterPrepare);
    expect(state.predictRounds.every((r) => r.guess === null)).toBe(true);
  });
});

describe('calibrationEngine — submit', () => {
  it('marks the run complete', async () => {
    const config: CalibrationConfig = {
      mode: 'predict-correctness',
      factSources: ['facts'],
      maxTokens: 5,
      factIds: ['right-fact'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, causalLM: baselineCausalLM });
    let state = initState(config, predictRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
