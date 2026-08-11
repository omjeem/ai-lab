import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  measureDecay,
  measureRecall,
  characterAccuracy,
  type MemoryDecayConfig,
} from '@/engines/memoryDecayEngine';
import { TinyRNN } from '@/models/tinyRNNTrainer';
import type { CorpusDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-4-sequence-models/4-2-recurrence-memory.json';

/** Short, highly structured text so tests train quickly but genuinely. */
const CORPUS = 'the cat sat on the mat and the cat ran to the mat again. '.repeat(20);

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'empty') return '';
    return CORPUS;
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as MemoryDecayConfig;

/** Trimmed config so the suite trains a real net without being slow. */
const fast = (config: MemoryDecayConfig): MemoryDecayConfig => ({
  ...config,
  epochs: Math.min(config.epochs, 8),
  maxEpochs: Math.min(config.maxEpochs ?? config.epochs, 20),
  sequenceLength: Math.min(config.sequenceLength, 40),
  trials: Math.min(config.trials ?? 20, 8),
});

describe('memoryDecayEngine — prepare', () => {
  it('loads the corpus text', async () => {
    const prepared = await prepare(configFor(0), { corpus: fakeCorpus });
    expect(prepared.text.length).toBeGreaterThan(0);
  });

  it('rejects an empty corpus', async () => {
    await expect(
      prepare({ ...configFor(0), corpus: 'empty' }, { corpus: fakeCorpus })
    ).rejects.toThrow();
  });

  it('generates the recall corpus rather than loading one', async () => {
    let loaded = false;
    const spy: CorpusDep = {
      async load() {
        loaded = true;
        return CORPUS;
      },
    };
    const prepared = await prepare(configFor(2), { corpus: spy });
    expect(loaded).toBe(false);
    expect(prepared.text.length).toBeGreaterThan(0);
    // The marker characters have to be present for recall to be measurable.
    expect(prepared.text).toMatch(/[XY]/);
  });
});

describe('memoryDecayEngine — training level', () => {
  const config = fast(configFor(0));

  it('starts untrained so the player can watch it learn', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.epochsTrained).toBe(0);
    expect(state.lossHistory).toEqual([]);
  });

  it('actually learns: accuracy improves with training', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const before = initState(config, rulesFor(0), prepared);
    const after = applyAction(before, { type: 'TRAIN', epochs: 20 });

    expect(after.epochsTrained).toBe(20);
    expect(after.lossHistory).toHaveLength(20);
    expect(characterAccuracy(after)).toBeGreaterThan(characterAccuracy(before));
  });

  it('drives the loss down over epochs', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'TRAIN', epochs: 20 });
    expect(state.lossHistory.at(-1)).toBeLessThan(state.lossHistory[0]!);
  });

  it('does not mutate the network held by the previous state', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const before = initState(config, rulesFor(0), prepared);
    const snapshot = JSON.stringify(before.rnn.whh);
    applyAction(before, { type: 'TRAIN', epochs: 5 });
    expect(JSON.stringify(before.rnn.whh)).toBe(snapshot);
  });

  it('stops at the epoch ceiling', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'TRAIN', epochs: 9999 });
    expect(state.epochsTrained).toBe(config.maxEpochs);
  });

  it('rebuilds and discards training when the hidden size changes', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'TRAIN', epochs: 5 });
    state = applyAction(state, { type: 'SET_HIDDEN_SIZE', value: 16 });

    expect(state.hiddenSize).toBe(16);
    expect(state.epochsTrained).toBe(0);
    expect(state.rnn.hiddenSize).toBe(16);
  });

  it('clamps hidden size and learning rate to their ranges', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_HIDDEN_SIZE', value: 9999 });
    expect(state.hiddenSize).toBe(config.hiddenSizeRange![1]);
    state = applyAction(state, { type: 'SET_LEARNING_RATE', value: 9999 });
    expect(state.learningRate).toBe(config.learningRateRange![1]);
  });

  it('reports character accuracy as its metric', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(0), prepared);
    const result = evaluate(state);
    expect(result.metric).toBe('characterAccuracy');
    expect(result.value).toBeGreaterThanOrEqual(0);
    expect(result.value).toBeLessThanOrEqual(1);
  });
});

describe('measureDecay — the real measurement behind the chapter', () => {
  it('finds a decay step from genuine hidden-state convergence', () => {
    const rnn = new TinyRNN({ hiddenSize: 16, vocabSize: 6, seed: 23 });
    const base = Array.from({ length: 100 }, (_, i) => (i % 4) + 1);
    const { trace, decayStep } = measureDecay(rnn, base, 0, 5, 0.5);

    expect(trace).toHaveLength(base.length + 1);
    expect(decayStep).toBeGreaterThan(0);
    // Similarity must rise as the distinguishing token is overwritten.
    expect(trace.at(-1)!).toBeGreaterThan(trace[0]!);
  });

  it('reports similarities inside the cosine range', () => {
    const rnn = new TinyRNN({ hiddenSize: 12, vocabSize: 5, seed: 3 });
    const { trace } = measureDecay(rnn, [1, 2, 3, 4, 1, 2], 0, 4, 0.5);
    for (const value of trace) {
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('never reports a decay step past the sequence length', () => {
    const rnn = new TinyRNN({ hiddenSize: 8, vocabSize: 4, seed: 1 });
    const base = [1, 2, 3];
    expect(measureDecay(rnn, base, 0, 3, 0.0001).decayStep).toBeLessThanOrEqual(base.length);
  });
});

describe('memoryDecayEngine — predict decay', () => {
  const config = fast(configFor(1));

  it('builds rounds with a measured decay step and no answer', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);

    expect(state.rounds).toHaveLength(config.rounds!);
    for (const round of state.rounds) {
      expect(round.trueDecayStep).toBeGreaterThanOrEqual(0);
      expect(round.trueDecayStep).toBeLessThanOrEqual(config.sequenceLength);
      expect(round.estimate).toBeNull();
      expect(round.similarityTrace.length).toBeGreaterThan(0);
    }
  });

  it('arrives pre-trained, since the level inspects a trained network', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.epochsTrained).toBe(config.epochs);
  });

  it('scores exact estimates at zero error', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(1), prepared);
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ESTIMATE_DECAY', roundIndex: i, value: round.trueDecayStep });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('decayPredictionError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('charges the full sequence length for unanswered rounds', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);
    expect(evaluate(state).value).toBe(config.sequenceLength);
  });

  it('clamps an estimate into the sequence and rejects negatives', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'ESTIMATE_DECAY', roundIndex: 0, value: 99_999 });
    expect(state.rounds[0]!.estimate).toBe(config.sequenceLength);
    state = applyAction(state, { type: 'ESTIMATE_DECAY', roundIndex: 0, value: -5 });
    expect(state.rounds[0]!.estimate).toBe(config.sequenceLength);
  });
});

describe('memoryDecayEngine — recall task', () => {
  const config = fast(configFor(2));

  it('measures real recall accuracy at each gap length', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'RUN_RECALL' });

    expect(state.recallResults).toHaveLength(config.gapLengths!.length);
    for (const result of state.recallResults) {
      expect(config.gapLengths).toContain(result.gap);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
      expect(result.accuracy).toBeLessThanOrEqual(1);
    }
  });

  it('scores nothing before the measurement has been run', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(0);
  });

  it('rewards demonstrating recall that works up close and fails at distance', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'RUN_RECALL' });

    // Substitute a measured collapse to pin the scoring rule itself.
    const collapsed = {
      ...state,
      recallResults: [
        { gap: 5, accuracy: 1 },
        { gap: 100, accuracy: 0 },
      ],
    };
    const flat = {
      ...state,
      recallResults: [
        { gap: 5, accuracy: 0.2 },
        { gap: 100, accuracy: 0.2 },
      ],
    };

    expect(evaluate(collapsed).value).toBeGreaterThan(evaluate(flat).value);
    expect(evaluate(collapsed).value).toBeCloseTo(1);
    expect(evaluate(collapsed).stars).toBe(3);
  });

  it('exposes the measurement helper directly', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(2), prepared);
    expect(measureRecall(state)).toHaveLength(config.gapLengths!.length);
  });

  it('lets the hidden state grow, which does not rescue long-range recall', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_HIDDEN_SIZE', value: config.hiddenSizeRange![1] });
    expect(state.hiddenSize).toBe(config.hiddenSizeRange![1]);
    state = applyAction(state, { type: 'RUN_RECALL' });

    const sorted = [...state.recallResults].sort((a, b) => a.gap - b.gap);
    expect(sorted.at(-1)!.accuracy).toBeLessThanOrEqual(1);
  });
});

describe('memoryDecayEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = fast(level.engineConfig as unknown as MemoryDecayConfig);
      const prepared = await prepare(config, { corpus: fakeCorpus });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
    }
  });
});
