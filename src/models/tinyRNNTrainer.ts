/**
 * A real character-level recurrent network, trained in the browser.
 *
 * World 4.2 turns on this being genuine: the hidden-state decay the chapter
 * shows is measured from these actual activations as they are overwritten step
 * by step, not an animation of what decay would look like. Backpropagation
 * through time is implemented in full below, with gradient clipping, because a
 * scripted approximation would make the chapter a lie.
 */
import { createRng, softmax } from '@/engines/shared';

export interface TinyRNNConfig {
  hiddenSize: number;
  vocabSize: number;
  seed: number;
}

export interface SequenceForward {
  /** Hidden state after each step; index 0 is the zero state before any input. */
  hiddenStates: number[][];
  /** Next-token distribution at each step. */
  probabilities: number[][];
}

/** Gradients above this are rescaled — recurrent nets explode without it. */
const CLIP = 5;

export class TinyRNN {
  readonly hiddenSize: number;
  readonly vocabSize: number;

  /** input → hidden, [hidden][vocab] */
  wxh: number[][];
  /** hidden → hidden, [hidden][hidden] */
  whh: number[][];
  /** hidden → output, [vocab][hidden] */
  why: number[][];
  bh: number[];
  by: number[];

  constructor(config: TinyRNNConfig) {
    this.hiddenSize = config.hiddenSize;
    this.vocabSize = config.vocabSize;
    this.wxh = [];
    this.whh = [];
    this.why = [];
    this.bh = [];
    this.by = [];
    this.reset(config.seed);
  }

  reset(seed: number): void {
    const rng = createRng(seed);
    const scale = 0.08;
    const matrix = (rows: number, cols: number) =>
      Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => (rng() * 2 - 1) * scale)
      );

    this.wxh = matrix(this.hiddenSize, this.vocabSize);
    this.whh = matrix(this.hiddenSize, this.hiddenSize);
    this.why = matrix(this.vocabSize, this.hiddenSize);
    this.bh = new Array<number>(this.hiddenSize).fill(0);
    this.by = new Array<number>(this.vocabSize).fill(0);
  }

  /** One recurrent step: h' = tanh(Wxh·x + Whh·h + bh). */
  step(inputIndex: number, previous: readonly number[]): number[] {
    const next = new Array<number>(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      // The input is one-hot, so its contribution is a single column read.
      let sum = this.bh[i]! + (this.wxh[i]![inputIndex] ?? 0);
      const row = this.whh[i]!;
      for (let j = 0; j < this.hiddenSize; j++) sum += row[j]! * previous[j]!;
      next[i] = Math.tanh(sum);
    }
    return next;
  }

  logits(hidden: readonly number[]): number[] {
    const out = new Array<number>(this.vocabSize);
    for (let v = 0; v < this.vocabSize; v++) {
      let sum = this.by[v]!;
      const row = this.why[v]!;
      for (let i = 0; i < this.hiddenSize; i++) sum += row[i]! * hidden[i]!;
      out[v] = sum;
    }
    return out;
  }

  /** Runs a whole sequence, keeping every hidden state for inspection. */
  forwardSequence(inputs: readonly number[]): SequenceForward {
    const hiddenStates: number[][] = [new Array<number>(this.hiddenSize).fill(0)];
    const probabilities: number[][] = [];

    for (let t = 0; t < inputs.length; t++) {
      const hidden = this.step(inputs[t]!, hiddenStates[t]!);
      hiddenStates.push(hidden);
      probabilities.push(softmax(this.logits(hidden)));
    }

    return { hiddenStates, probabilities };
  }

  /** Next-character distribution after consuming a sequence. */
  predictNext(inputs: readonly number[]): number[] {
    const { probabilities } = this.forwardSequence(inputs);
    return probabilities.at(-1) ?? softmax(new Array<number>(this.vocabSize).fill(0));
  }

  /**
   * Backpropagation through time over one sequence, followed by an SGD update.
   * Returns the mean cross-entropy loss.
   */
  trainSequence(
    inputs: readonly number[],
    targets: readonly number[],
    learningRate: number
  ): number {
    const T = Math.min(inputs.length, targets.length);
    if (T === 0) return 0;

    const { hiddenStates, probabilities } = this.forwardSequence(inputs.slice(0, T));

    const dWxh = this.wxh.map((r) => r.map(() => 0));
    const dWhh = this.whh.map((r) => r.map(() => 0));
    const dWhy = this.why.map((r) => r.map(() => 0));
    const dbh = new Array<number>(this.hiddenSize).fill(0);
    const dby = new Array<number>(this.vocabSize).fill(0);
    let dhNext = new Array<number>(this.hiddenSize).fill(0);

    let loss = 0;

    for (let t = T - 1; t >= 0; t--) {
      const p = probabilities[t]!;
      const target = targets[t]!;
      loss -= Math.log(Math.max(p[target] ?? 1e-12, 1e-12));

      // Softmax + cross-entropy: dy = p − onehot(target).
      const dy = [...p];
      dy[target] = (dy[target] ?? 0) - 1;

      const hidden = hiddenStates[t + 1]!;
      const previous = hiddenStates[t]!;

      for (let v = 0; v < this.vocabSize; v++) {
        const d = dy[v]!;
        if (d === 0) continue;
        dby[v]! += d;
        const row = dWhy[v]!;
        for (let i = 0; i < this.hiddenSize; i++) row[i]! += d * hidden[i]!;
      }

      // dh = Whyᵀ·dy + carried-back gradient, then through tanh'.
      const dh = new Array<number>(this.hiddenSize).fill(0);
      for (let i = 0; i < this.hiddenSize; i++) {
        let sum = dhNext[i]!;
        for (let v = 0; v < this.vocabSize; v++) sum += this.why[v]![i]! * dy[v]!;
        dh[i] = sum;
      }

      const dRaw = new Array<number>(this.hiddenSize);
      for (let i = 0; i < this.hiddenSize; i++) {
        dRaw[i] = dh[i]! * (1 - hidden[i]! * hidden[i]!);
        dbh[i]! += dRaw[i]!;
        dWxh[i]![inputs[t]!]! += dRaw[i]!;
        const row = dWhh[i]!;
        for (let j = 0; j < this.hiddenSize; j++) row[j]! += dRaw[i]! * previous[j]!;
      }

      const carried = new Array<number>(this.hiddenSize).fill(0);
      for (let j = 0; j < this.hiddenSize; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenSize; i++) sum += this.whh[i]![j]! * dRaw[i]!;
        carried[j] = sum;
      }
      dhNext = carried;
    }

    const clip = (v: number) => Math.max(-CLIP, Math.min(CLIP, v));
    const scale = learningRate / T;

    for (let i = 0; i < this.hiddenSize; i++) {
      for (let v = 0; v < this.vocabSize; v++) this.wxh[i]![v]! -= scale * clip(dWxh[i]![v]!);
      for (let j = 0; j < this.hiddenSize; j++) this.whh[i]![j]! -= scale * clip(dWhh[i]![j]!);
      this.bh[i]! -= scale * clip(dbh[i]!);
    }
    for (let v = 0; v < this.vocabSize; v++) {
      for (let i = 0; i < this.hiddenSize; i++) this.why[v]![i]! -= scale * clip(dWhy[v]![i]!);
      this.by[v]! -= scale * clip(dby[v]!);
    }

    return loss / T;
  }

  /** Fraction of positions where the argmax matches the target. */
  accuracy(inputs: readonly number[], targets: readonly number[]): number {
    const T = Math.min(inputs.length, targets.length);
    if (T === 0) return 0;

    const { probabilities } = this.forwardSequence(inputs.slice(0, T));
    let correct = 0;
    for (let t = 0; t < T; t++) {
      const p = probabilities[t]!;
      let best = 0;
      for (let v = 1; v < p.length; v++) if (p[v]! > p[best]!) best = v;
      if (best === targets[t]) correct++;
    }
    return correct / T;
  }

  clone(): TinyRNN {
    const copy = new TinyRNN({ hiddenSize: this.hiddenSize, vocabSize: this.vocabSize, seed: 1 });
    copy.wxh = this.wxh.map((r) => [...r]);
    copy.whh = this.whh.map((r) => [...r]);
    copy.why = this.why.map((r) => [...r]);
    copy.bh = [...this.bh];
    copy.by = [...this.by];
    return copy;
  }
}

/* ────────────────────────────────────────────────────────────
   Character vocabulary
   ──────────────────────────────────────────────────────────── */

export interface CharVocab {
  chars: string[];
  indexOf: Map<string, number>;
}

/** Keeps the most frequent characters, mapping the rest to a shared slot. */
export function buildCharVocab(text: string, limit: number): CharVocab {
  const counts = new Map<string, number>();
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1);

  const chars = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, Math.max(1, limit - 1))
    .map(([ch]) => ch);
  // Final slot is the catch-all for anything that did not make the cut.
  chars.push(' ');

  return { chars, indexOf: new Map(chars.map((ch, i) => [ch, i])) };
}

export function encodeChars(text: string, vocab: CharVocab): number[] {
  const unknown = vocab.chars.length - 1;
  return [...text].map((ch) => vocab.indexOf.get(ch) ?? unknown);
}
