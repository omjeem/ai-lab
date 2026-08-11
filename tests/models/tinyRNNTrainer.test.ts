import { describe, it, expect } from 'vitest';
import { TinyRNN, buildCharVocab, encodeChars } from '@/models/tinyRNNTrainer';
import { cosineSimilarity } from '@/engines/shared';

describe('character vocabulary', () => {
  it('keeps the most frequent characters within the limit', () => {
    const vocab = buildCharVocab('aaaabbbcc', 3);
    expect(vocab.chars).toHaveLength(3);
    expect(vocab.chars).toContain('a');
    expect(vocab.chars).toContain('b');
  });

  it('encodes text to indices inside the vocabulary', () => {
    const vocab = buildCharVocab('abcabc', 8);
    const encoded = encodeChars('abc', vocab);
    expect(encoded).toHaveLength(3);
    for (const index of encoded) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(vocab.chars.length);
    }
  });

  it('maps unseen characters to the catch-all slot instead of failing', () => {
    const vocab = buildCharVocab('abc', 4);
    const encoded = encodeChars('z', vocab);
    expect(encoded[0]).toBe(vocab.chars.length - 1);
  });
});

describe('TinyRNN — structure', () => {
  it('shapes every matrix from hidden and vocab size', () => {
    const rnn = new TinyRNN({ hiddenSize: 8, vocabSize: 5, seed: 1 });
    expect(rnn.wxh).toHaveLength(8);
    expect(rnn.wxh[0]).toHaveLength(5);
    expect(rnn.whh).toHaveLength(8);
    expect(rnn.whh[0]).toHaveLength(8);
    expect(rnn.why).toHaveLength(5);
    expect(rnn.why[0]).toHaveLength(8);
  });

  it('is deterministic for a seed', () => {
    const a = new TinyRNN({ hiddenSize: 6, vocabSize: 4, seed: 3 });
    const b = new TinyRNN({ hiddenSize: 6, vocabSize: 4, seed: 3 });
    expect(a.whh).toEqual(b.whh);
  });

  it('keeps hidden values inside the tanh range', () => {
    const rnn = new TinyRNN({ hiddenSize: 6, vocabSize: 4, seed: 3 });
    const { hiddenStates } = rnn.forwardSequence([0, 1, 2, 3, 0, 1]);
    for (const value of hiddenStates.flat()) {
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('starts from a zero hidden state and records one per step', () => {
    const rnn = new TinyRNN({ hiddenSize: 5, vocabSize: 4, seed: 1 });
    const { hiddenStates, probabilities } = rnn.forwardSequence([0, 1, 2]);
    expect(hiddenStates).toHaveLength(4);
    expect(hiddenStates[0]!.every((v) => v === 0)).toBe(true);
    expect(probabilities).toHaveLength(3);
  });

  it('emits a proper distribution at each step', () => {
    const rnn = new TinyRNN({ hiddenSize: 5, vocabSize: 6, seed: 1 });
    for (const p of rnn.forwardSequence([0, 1, 2]).probabilities) {
      expect(p).toHaveLength(6);
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      expect(p.every((v) => v >= 0)).toBe(true);
    }
  });
});

describe('TinyRNN — learning', () => {
  /** A perfectly predictable cycle: the model should master it. */
  const cycle = (length: number, period: number) =>
    Array.from({ length }, (_, i) => i % period);

  it('learns a repeating pattern to near-perfect accuracy', () => {
    const rnn = new TinyRNN({ hiddenSize: 24, vocabSize: 4, seed: 5 });
    const sequence = cycle(60, 4);
    const inputs = sequence.slice(0, -1);
    const targets = sequence.slice(1);

    const before = rnn.accuracy(inputs, targets);
    for (let epoch = 0; epoch < 600; epoch++) {
      rnn.trainSequence(inputs, targets, 0.5);
    }
    const after = rnn.accuracy(inputs, targets);

    expect(after).toBeGreaterThan(before);
    expect(after).toBeGreaterThan(0.9);
  });

  it('drives loss down on real text', () => {
    const text = 'the cat sat on the mat and the cat sat again and again on that mat ';
    const vocab = buildCharVocab(text, 32);
    const encoded = encodeChars(text.repeat(3), vocab);
    const rnn = new TinyRNN({ hiddenSize: 32, vocabSize: vocab.chars.length, seed: 19 });

    const inputs = encoded.slice(0, -1);
    const targets = encoded.slice(1);
    const before = rnn.trainSequence(inputs, targets, 0);

    for (let epoch = 0; epoch < 250; epoch++) rnn.trainSequence(inputs, targets, 0.5);
    const after = rnn.trainSequence(inputs, targets, 0);

    expect(after).toBeLessThan(before);
  });

  it('matches its BPTT gradients against numerical estimates', () => {
    const rnn = new TinyRNN({ hiddenSize: 4, vocabSize: 3, seed: 11 });
    const inputs = [0, 1, 2, 1];
    const targets = [1, 2, 1, 0];
    const h = 1e-5;

    const lossOf = () => {
      const { probabilities } = rnn.forwardSequence(inputs);
      let loss = 0;
      for (let t = 0; t < targets.length; t++) {
        loss -= Math.log(Math.max(probabilities[t]![targets[t]!]!, 1e-12));
      }
      return loss / targets.length;
    };

    // One update at a known rate reveals the gradient the trainer applied. The
    // trainer scales by learningRate/T, so dividing the step by the rate
    // recovers the per-step mean gradient that `lossOf` measures.
    const rate = 1e-4;
    const original = rnn.whh.map((r) => [...r]);
    rnn.trainSequence(inputs, targets, rate);
    const applied = rnn.whh.map((row, i) => row.map((v, j) => (original[i]![j]! - v) / rate));

    // Restore and estimate the same gradients numerically.
    rnn.whh = original.map((r) => [...r]);
    for (let i = 0; i < rnn.hiddenSize; i++) {
      for (let j = 0; j < rnn.hiddenSize; j++) {
        const value = rnn.whh[i]![j]!;
        rnn.whh[i]![j] = value + h;
        const up = lossOf();
        rnn.whh[i]![j] = value - h;
        const down = lossOf();
        rnn.whh[i]![j] = value;
        expect(applied[i]![j]).toBeCloseTo((up - down) / (2 * h), 3);
      }
    }
  });

  it('returns zero loss for an empty sequence', () => {
    const rnn = new TinyRNN({ hiddenSize: 4, vocabSize: 3, seed: 1 });
    expect(rnn.trainSequence([], [], 0.1)).toBe(0);
  });

  it('leaves weights untouched at a zero learning rate', () => {
    const rnn = new TinyRNN({ hiddenSize: 4, vocabSize: 3, seed: 1 });
    const before = JSON.stringify(rnn.whh);
    rnn.trainSequence([0, 1, 2], [1, 2, 0], 0);
    expect(JSON.stringify(rnn.whh)).toBe(before);
  });

  it('clones without sharing arrays', () => {
    const rnn = new TinyRNN({ hiddenSize: 4, vocabSize: 3, seed: 1 });
    const copy = rnn.clone();
    copy.whh[0]![0] = 99;
    expect(rnn.whh[0]![0]).not.toBe(99);
  });
});

describe('TinyRNN — the memory limit World 4.2 is about', () => {
  it('overwrites its hidden state, so early inputs stop being recoverable', () => {
    const rnn = new TinyRNN({ hiddenSize: 16, vocabSize: 6, seed: 23 });

    // Two sequences that differ only in their very first token.
    const tail = Array.from({ length: 120 }, (_, i) => (i % 5) + 1);
    const withMarker = rnn.forwardSequence([0, ...tail]).hiddenStates;
    const withoutMarker = rnn.forwardSequence([1, ...tail]).hiddenStates;

    const similarityAt = (step: number) =>
      cosineSimilarity(withMarker[step]!, withoutMarker[step]!);

    // Right after the difference the states diverge; far downstream they converge,
    // which is precisely the information loss the chapter measures.
    expect(similarityAt(1)).toBeLessThan(similarityAt(100));
    expect(similarityAt(100)).toBeGreaterThan(0.99);
  });

  it('produces a monotonically recovering similarity trace over distance', () => {
    const rnn = new TinyRNN({ hiddenSize: 16, vocabSize: 6, seed: 23 });
    const tail = Array.from({ length: 80 }, (_, i) => (i % 4) + 1);
    const a = rnn.forwardSequence([0, ...tail]).hiddenStates;
    const b = rnn.forwardSequence([2, ...tail]).hiddenStates;

    const early = cosineSimilarity(a[2]!, b[2]!);
    const middle = cosineSimilarity(a[20]!, b[20]!);
    const late = cosineSimilarity(a[70]!, b[70]!);

    expect(middle).toBeGreaterThan(early);
    expect(late).toBeGreaterThanOrEqual(middle - 1e-6);
  });
});
