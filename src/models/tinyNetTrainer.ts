/**
 * A real feedforward network, hand-rolled and trained in the browser.
 *
 * No framework: every forward pass, every gradient and every weight update
 * below is the actual arithmetic, which is what lets Worlds 2 and 3 expose
 * genuine per-edge gradients and genuine loss curves rather than animations of
 * what those would look like.
 *
 * Deliberately dependency-free and DOM-free so engine tests can train it
 * directly in milliseconds without downloading anything.
 */
import { createRng } from '@/engines/shared';

export type Activation = 'relu' | 'tanh' | 'sigmoid' | 'leakyRelu' | 'gelu' | 'linear';

const LEAKY_SLOPE = 0.01;

export function activate(kind: Activation, x: number): number {
  switch (kind) {
    case 'relu':
      return Math.max(0, x);
    case 'leakyRelu':
      return x > 0 ? x : LEAKY_SLOPE * x;
    case 'tanh':
      return Math.tanh(x);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'gelu':
      // tanh approximation, the one used in practice.
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    case 'linear':
      return x;
  }
}

/** Derivative with respect to the pre-activation. */
export function activateDerivative(kind: Activation, x: number): number {
  switch (kind) {
    case 'relu':
      return x > 0 ? 1 : 0;
    case 'leakyRelu':
      return x > 0 ? 1 : LEAKY_SLOPE;
    case 'tanh': {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    }
    case 'gelu': {
      const c = Math.sqrt(2 / Math.PI);
      const inner = c * (x + 0.044715 * x ** 3);
      const t = Math.tanh(inner);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
    }
    case 'linear':
      return 1;
  }
}

export interface TinyNetConfig {
  /** Units per layer, input first, output last, e.g. [2, 4, 1]. */
  architecture: number[];
  activation: Activation;
  seed: number;
}

export interface ForwardPass {
  /** Per-layer activations, index 0 being the input. */
  activations: number[][];
  /** Per-layer pre-activation sums, index 0 unused. */
  preActivations: number[][];
  output: number;
}

export interface EdgeGradient {
  layer: number;
  from: number;
  to: number;
  value: number;
}

export interface GradientResult {
  loss: number;
  /** [layer][to][from] */
  weightGradients: number[][][];
  biasGradients: number[][];
}

export interface TrainingSample {
  x: [number, number] | number[];
  label: 0 | 1;
}

const EPSILON = 1e-12;

export class TinyNet {
  readonly architecture: number[];
  readonly activation: Activation;
  /** weights[layer][to][from] */
  weights: number[][][] = [];
  biases: number[][] = [];

  constructor(config: TinyNetConfig) {
    this.architecture = [...config.architecture];
    this.activation = config.activation;
    this.reset(config.seed);
  }

  /** Re-initialises with He/Xavier scaling, deterministic for a seed. */
  reset(seed: number): void {
    const rng = createRng(seed);
    this.weights = [];
    this.biases = [];

    for (let l = 0; l < this.architecture.length - 1; l++) {
      const fanIn = this.architecture[l]!;
      const fanOut = this.architecture[l + 1]!;
      // He for rectifiers, Xavier for saturating activations.
      const scale =
        this.activation === 'relu' || this.activation === 'leakyRelu'
          ? Math.sqrt(2 / fanIn)
          : Math.sqrt(1 / fanIn);

      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      for (let to = 0; to < fanOut; to++) {
        const row: number[] = [];
        for (let from = 0; from < fanIn; from++) {
          row.push((rng() * 2 - 1) * scale);
        }
        layerWeights.push(row);
        layerBiases.push(0);
      }
      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }
  }

  get parameterCount(): number {
    let n = 0;
    for (let l = 0; l < this.weights.length; l++) {
      n += this.weights[l]!.length * this.weights[l]![0]!.length + this.biases[l]!.length;
    }
    return n;
  }

  forward(input: readonly number[]): ForwardPass {
    const activations: number[][] = [[...input]];
    const preActivations: number[][] = [[]];

    for (let l = 0; l < this.weights.length; l++) {
      const previous = activations[l]!;
      const isOutput = l === this.weights.length - 1;
      const pre: number[] = [];
      const act: number[] = [];

      for (let to = 0; to < this.weights[l]!.length; to++) {
        let sum = this.biases[l]![to]!;
        const row = this.weights[l]![to]!;
        for (let from = 0; from < row.length; from++) sum += row[from]! * previous[from]!;
        pre.push(sum);
        // The output layer is always a sigmoid, so the result reads as a probability.
        act.push(isOutput ? activate('sigmoid', sum) : activate(this.activation, sum));
      }

      preActivations.push(pre);
      activations.push(act);
    }

    return { activations, preActivations, output: activations.at(-1)![0]! };
  }

  predict(input: readonly number[]): number {
    return this.forward(input).output;
  }

  /**
   * Backpropagation over a batch, returning the real per-parameter gradients
   * of binary cross-entropy without applying them.
   */
  computeGradients(samples: readonly TrainingSample[]): GradientResult {
    const weightGradients = this.weights.map((layer) => layer.map((row) => row.map(() => 0)));
    const biasGradients = this.biases.map((layer) => layer.map(() => 0));
    if (samples.length === 0) return { loss: 0, weightGradients, biasGradients };

    let totalLoss = 0;

    for (const sample of samples) {
      const { activations, preActivations, output } = this.forward(sample.x);
      const target = sample.label;
      totalLoss -= target * Math.log(output + EPSILON) + (1 - target) * Math.log(1 - output + EPSILON);

      // Sigmoid output with cross-entropy collapses to (prediction - target).
      let delta: number[] = [output - target];

      for (let l = this.weights.length - 1; l >= 0; l--) {
        const previous = activations[l]!;
        for (let to = 0; to < this.weights[l]!.length; to++) {
          const d = delta[to]!;
          biasGradients[l]![to]! += d;
          const row = weightGradients[l]![to]!;
          for (let from = 0; from < row.length; from++) row[from]! += d * previous[from]!;
        }

        if (l === 0) break;

        // Propagate into the layer below, through its activation derivative.
        const nextDelta = new Array<number>(this.architecture[l]!).fill(0);
        for (let from = 0; from < nextDelta.length; from++) {
          let sum = 0;
          for (let to = 0; to < this.weights[l]!.length; to++) {
            sum += this.weights[l]![to]![from]! * delta[to]!;
          }
          nextDelta[from] = sum * activateDerivative(this.activation, preActivations[l]![from]!);
        }
        delta = nextDelta;
      }
    }

    const n = samples.length;
    for (let l = 0; l < weightGradients.length; l++) {
      for (let to = 0; to < weightGradients[l]!.length; to++) {
        const row = weightGradients[l]![to]!;
        for (let from = 0; from < row.length; from++) row[from]! /= n;
        biasGradients[l]![to]! /= n;
      }
    }

    return { loss: totalLoss / n, weightGradients, biasGradients };
  }

  /** One gradient-descent update on a batch. Returns the batch loss. */
  trainBatch(samples: readonly TrainingSample[], learningRate: number): number {
    const { loss, weightGradients, biasGradients } = this.computeGradients(samples);
    for (let l = 0; l < this.weights.length; l++) {
      for (let to = 0; to < this.weights[l]!.length; to++) {
        const row = this.weights[l]![to]!;
        for (let from = 0; from < row.length; from++) {
          row[from]! -= learningRate * weightGradients[l]![to]![from]!;
        }
        this.biases[l]![to]! -= learningRate * biasGradients[l]![to]!;
      }
    }
    return loss;
  }

  /**
   * One pass over the data in mini-batches. Shuffling is seeded by the epoch so
   * a run is reproducible while still varying batch composition between epochs.
   */
  trainEpoch(
    samples: readonly TrainingSample[],
    options: { learningRate: number; batchSize: number; epoch?: number }
  ): number {
    if (samples.length === 0) return 0;

    const rng = createRng((options.epoch ?? 0) * 7919 + 13);
    const order = samples.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }

    const size = Math.max(1, Math.min(options.batchSize, samples.length));
    let total = 0;
    let batches = 0;
    for (let start = 0; start < order.length; start += size) {
      const batch = order.slice(start, start + size).map((i) => samples[i]!);
      total += this.trainBatch(batch, options.learningRate);
      batches++;
    }
    return batches === 0 ? 0 : total / batches;
  }

  /** Mean cross-entropy over a set, without touching the weights. */
  loss(samples: readonly TrainingSample[]): number {
    if (samples.length === 0) return 0;
    let total = 0;
    for (const sample of samples) {
      const p = this.predict(sample.x);
      total -= sample.label * Math.log(p + EPSILON) + (1 - sample.label) * Math.log(1 - p + EPSILON);
    }
    return total / samples.length;
  }

  accuracy(samples: readonly TrainingSample[]): number {
    if (samples.length === 0) return 0;
    const correct = samples.filter((s) => (this.predict(s.x) >= 0.5 ? 1 : 0) === s.label).length;
    return correct / samples.length;
  }

  /** Flattens the most recent gradients into per-edge records for the visualiser. */
  edgeGradients(samples: readonly TrainingSample[]): EdgeGradient[] {
    const { weightGradients } = this.computeGradients(samples);
    const edges: EdgeGradient[] = [];
    for (let l = 0; l < weightGradients.length; l++) {
      for (let to = 0; to < weightGradients[l]!.length; to++) {
        for (let from = 0; from < weightGradients[l]![to]!.length; from++) {
          edges.push({ layer: l, from, to, value: weightGradients[l]![to]![from]! });
        }
      }
    }
    return edges;
  }

  clone(): TinyNet {
    const copy = new TinyNet({
      architecture: this.architecture,
      activation: this.activation,
      seed: 1,
    });
    copy.weights = this.weights.map((l) => l.map((r) => [...r]));
    copy.biases = this.biases.map((l) => [...l]);
    return copy;
  }
}
