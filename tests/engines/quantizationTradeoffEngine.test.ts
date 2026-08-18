import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type QuantizationTradeoffConfig,
} from '@/engines/quantizationTradeoffEngine';
import type { CorpusDep, PrecisionModelDep, PrecisionRunResult } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

interface FakeRun {
  token: string;
  prob: number;
  inferenceTimeMs?: number;
}

function makeFakePrecisionModel(opts: {
  label: string;
  sizeMB: number;
  loadMs?: number;
  runs?: Record<string, FakeRun>;
  surprisals?: Record<string, number[]>;
}): PrecisionModelDep {
  let loaded = false;
  return {
    label: opts.label,
    sizeMB: opts.sizeMB,
    async ensureLoaded(): Promise<number> {
      if (loaded) return 0;
      loaded = true;
      return opts.loadMs ?? 10;
    },
    async run(prompt: string, topK: number): Promise<PrecisionRunResult> {
      const found = opts.runs?.[prompt];
      const token = found?.token ?? ' unknown';
      const prob = found?.prob ?? 1;
      return {
        topK: { tokens: [token].slice(0, Math.max(1, topK)), probs: [prob] },
        inferenceTimeMs: found?.inferenceTimeMs ?? 5,
      };
    },
    async continuationSurprisal(prompt: string, continuation: string): Promise<number[]> {
      return opts.surprisals?.[`${prompt}|${continuation}`] ?? [0, 0];
    },
  };
}

const FACT_REEF = {
  id: 'reef',
  topic: 'Great Barrier Reef',
  sentences: ['The reef stretches 2,300 kilometers.'],
  query: 'How many kilometers?',
  answer: '2,300',
};
const FACT_VENUS = {
  id: 'venus',
  topic: 'Venus',
  sentences: ['A day on Venus takes 243 Earth days.'],
  query: 'How many Earth days?',
  answer: '243',
};

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') {
      return JSON.stringify({ facts: [{ ...FACT_REEF, answer: 'nowhere-in-passage' }] });
    }
    return JSON.stringify({ facts: [FACT_REEF, FACT_VENUS] });
  },
};

const outputRules: EngineRules = {
  passCriteria: { metric: 'divergenceAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

// Thresholds sit strictly between this level's four achievable fractions
// (0, 1/3, 2/3, 1) rather than on any of them — the exact star-band bug
// already found and fixed once in 7-2 (2/3 ≈ 0.667 failing a 0.67 threshold).
const speedRules: EngineRules = {
  passCriteria: { metric: 'tradeoffPredictionAccuracy', threshold: 0.3, comparator: 'gte' },
  starsRules: [
    { threshold: 0.3, stars: 1 },
    { threshold: 0.5, stars: 2 },
    { threshold: 0.9, stars: 3 },
  ],
  xpReward: 45,
};

const confidenceRules: EngineRules = {
  passCriteria: { metric: 'confidencePredictionAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 50,
};

describe('quantizationTradeoffEngine — compare-outputs', () => {
  const config: QuantizationTradeoffConfig = {
    mode: 'compare-outputs',
    prompts: ['prompt A', 'prompt B'],
    topK: 3,
  };

  const reference = makeFakePrecisionModel({
    label: 'fp32',
    sizeMB: 500,
    runs: {
      'prompt A': { token: ' Paris', prob: 0.9 },
      'prompt B': { token: ' cold', prob: 0.8 },
    },
  });
  const quantized = makeFakePrecisionModel({
    label: 'q8',
    sizeMB: 130,
    runs: {
      'prompt A': { token: ' Paris', prob: 0.7 },
      'prompt B': { token: ' hot', prob: 0.6 },
    },
  });

  it('computes real divergence per prompt from the two injected models, never authored', async () => {
    const prepared = await prepare(config, { reference, quantized });
    expect(prepared.divergenceRounds).toHaveLength(2);
    expect(prepared.divergenceRounds[0]!.diverges).toBe(false); // Paris === Paris
    expect(prepared.divergenceRounds[1]!.diverges).toBe(true); // cold !== hot
  });

  it('scores full accuracy when every guess matches the real computed divergence', async () => {
    const prepared = await prepare(config, { reference, quantized });
    let state = initState(config, outputRules, prepared);
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 0, guess: false });
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 1, guess: true });

    const result = evaluate(state);
    expect(result.metric).toBe('divergenceAccuracy');
    expect(result.value).toBe(1);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('scores 0 when every guess is wrong', async () => {
    const prepared = await prepare(config, { reference, quantized });
    let state = initState(config, outputRules, prepared);
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 0, guess: true });
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 1, guess: false });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('does not count an unanswered round as correct', async () => {
    const prepared = await prepare(config, { reference, quantized });
    let state = initState(config, outputRules, prepared);
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 0, guess: false });

    const result = evaluate(state);
    expect(result.value).toBe(0.5);
  });
});

describe('quantizationTradeoffEngine — compare-speed', () => {
  const config: QuantizationTradeoffConfig = { mode: 'compare-speed', prompts: ['probe prompt'] };

  // Built fresh per test: `ensureLoaded()`'s `loaded` flag is stateful (real
  // loads are idempotent), so a shared fixture across tests would let the
  // second test silently inherit the first's "already loaded" timing.
  function makeVariants() {
    const reference = makeFakePrecisionModel({
      label: 'fp32',
      sizeMB: 500,
      loadMs: 900,
      runs: { 'probe prompt': { token: ' x', prob: 1, inferenceTimeMs: 40 } },
    });
    const quantized = makeFakePrecisionModel({
      label: 'q8',
      sizeMB: 130,
      loadMs: 300,
      runs: { 'probe prompt': { token: ' x', prob: 1, inferenceTimeMs: 20 } },
    });
    return { reference, quantized };
  }

  it('measures real size, load time and inference time for both variants during prepare', async () => {
    const { reference, quantized } = makeVariants();
    const prepared = await prepare(config, { reference, quantized });
    expect(prepared.speedMeasurement).toEqual({
      referenceSizeMB: 500,
      quantizedSizeMB: 130,
      referenceLoadMs: 900,
      quantizedLoadMs: 300,
      referenceInferenceMs: 40,
      quantizedInferenceMs: 20,
    });
  });

  it('scores 1/3 predictions correct when only one guess matches the real measurement', async () => {
    const { reference, quantized } = makeVariants();
    const prepared = await prepare(config, { reference, quantized });
    let state = initState(config, speedRules, prepared);
    // Correct: quantized IS smaller.
    state = applyAction(state, { type: 'PREDICT_SPEED', field: 'smallerIsQuantized', value: true });
    // Wrong: quantized loads faster in this fixture (300 < 900), guessed false.
    state = applyAction(state, { type: 'PREDICT_SPEED', field: 'fasterLoadIsQuantized', value: false });
    // Wrong: quantized is faster per-inference here too (20 < 40), guessed false.
    state = applyAction(state, { type: 'PREDICT_SPEED', field: 'fasterInferenceIsQuantized', value: false });

    const result = evaluate(state);
    expect(result.metric).toBe('tradeoffPredictionAccuracy');
    expect(result.value).toBeCloseTo(1 / 3);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(1);
  });

  it('scores 0 until every prediction has been made', async () => {
    const { reference, quantized } = makeVariants();
    const prepared = await prepare(config, { reference, quantized });
    const state = initState(config, speedRules, prepared);
    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });
});

describe('quantizationTradeoffEngine — compare-confidence', () => {
  const config: QuantizationTradeoffConfig = {
    mode: 'compare-confidence',
    corpus: 'retrieval-facts',
    factIds: ['reef', 'venus'],
  };

  const reference = makeFakePrecisionModel({
    label: 'fp32',
    sizeMB: 500,
    surprisals: {
      'Context: The reef stretches 2,300 kilometers.\nQuestion: How many kilometers?\nAnswer:| 2,300': [2.6, 0.1, 0.05],
      'Context: A day on Venus takes 243 Earth days.\nQuestion: How many Earth days?\nAnswer:| 243': [2.1, 0.1, 0.02],
    },
  });
  const quantized = makeFakePrecisionModel({
    label: 'q8',
    sizeMB: 130,
    surprisals: {
      'Context: The reef stretches 2,300 kilometers.\nQuestion: How many kilometers?\nAnswer:| 2,300': [1.2, 0.3, 0.1],
      'Context: A day on Venus takes 243 Earth days.\nQuestion: How many Earth days?\nAnswer:| 243': [0.4, 0.05, 0.5],
    },
  });

  it('rejects a fact whose answer never appears in its own passage', async () => {
    const badConfig: QuantizationTradeoffConfig = { mode: 'compare-confidence', corpus: 'bad-corpus', factIds: ['reef'] };
    await expect(prepare(badConfig, { corpus: fakeCorpus, reference, quantized })).rejects.toThrow();
  });

  it('computes real content-token-mean surprisal per fact, excluding the leading formatting token', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, reference, quantized });
    expect(prepared.confidenceRounds).toHaveLength(2);
    const reefRound = prepared.confidenceRounds.find((r) => r.factId === 'reef')!;
    // Content mean excludes bits[0] (2.6 / 1.2): mean(0.1, 0.05) = 0.075, mean(0.3, 0.1) = 0.2.
    expect(reefRound.referenceMeanBits).toBeCloseTo(0.075);
    expect(reefRound.quantizedMeanBits).toBeCloseTo(0.2);
    expect(reefRound.referenceMoreConfident).toBe(true);

    const venusRound = prepared.confidenceRounds.find((r) => r.factId === 'venus')!;
    // fp32 mean(0.1, 0.02) = 0.06; q8 mean(0.05, 0.5) = 0.275 -> reference still more confident.
    expect(venusRound.referenceMoreConfident).toBe(true);
  });

  it('scores full accuracy when every pick matches which precision is really more confident', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, reference, quantized });
    let state = initState(config, confidenceRules, prepared);
    state.confidenceRounds.forEach((round, i) => {
      state = applyAction(state, { type: 'PICK_CONFIDENCE', roundIndex: i, pick: round.referenceMoreConfident ? 'reference' : 'quantized' });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('confidencePredictionAccuracy');
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
  });

  it('scores 0 when every pick is wrong', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, reference, quantized });
    let state = initState(config, confidenceRules, prepared);
    state.confidenceRounds.forEach((round, i) => {
      state = applyAction(state, { type: 'PICK_CONFIDENCE', roundIndex: i, pick: round.referenceMoreConfident ? 'quantized' : 'reference' });
    });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });
});

describe('quantizationTradeoffEngine — reset never re-runs the model', () => {
  it('RESET restores initial state from already-prepared data', async () => {
    let calls = 0;
    const countingModel: PrecisionModelDep = {
      label: 'fp32',
      sizeMB: 500,
      async ensureLoaded() {
        return 0;
      },
      async run(_prompt, topK) {
        calls++;
        return { topK: { tokens: [' x'].slice(0, topK), probs: [1] }, inferenceTimeMs: 1 };
      },
      async continuationSurprisal() {
        return [0];
      },
    };
    const config: QuantizationTradeoffConfig = { mode: 'compare-outputs', prompts: ['a'], topK: 1 };
    const prepared = await prepare(config, { reference: countingModel, quantized: countingModel });
    const callsAfterPrepare = calls;

    let state = initState(config, outputRules, prepared);
    state = applyAction(state, { type: 'PICK_DIVERGENCE', roundIndex: 0, guess: true });
    state = applyAction(state, { type: 'RESET' });

    expect(calls).toBe(callsAfterPrepare);
    expect(state.divergenceRounds.every((r) => r.pick === null)).toBe(true);
  });
});

describe('quantizationTradeoffEngine — submit', () => {
  it('marks the run complete', async () => {
    const config: QuantizationTradeoffConfig = { mode: 'compare-outputs', prompts: ['a'], topK: 1 };
    const reference = makeFakePrecisionModel({ label: 'fp32', sizeMB: 500 });
    const quantized = makeFakePrecisionModel({ label: 'q8', sizeMB: 130 });
    const prepared = await prepare(config, { reference, quantized });
    let state = initState(config, outputRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
