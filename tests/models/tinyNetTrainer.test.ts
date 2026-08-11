import { describe, it, expect } from 'vitest';
import {
  TinyNet,
  activate,
  activateDerivative,
  type TrainingSample,
} from '@/models/tinyNetTrainer';
import { generateToyDataset, splitDataset } from '@/engines/toyDatasets';

describe('activation functions', () => {
  it('implements the standard shapes', () => {
    expect(activate('relu', -2)).toBe(0);
    expect(activate('relu', 3)).toBe(3);
    expect(activate('leakyRelu', -2)).toBeCloseTo(-0.02);
    expect(activate('tanh', 0)).toBe(0);
    expect(activate('sigmoid', 0)).toBeCloseTo(0.5);
    expect(activate('linear', 4.2)).toBe(4.2);
    expect(activate('gelu', 0)).toBeCloseTo(0);
  });

  it('saturates sigmoid and tanh at the extremes', () => {
    expect(activate('sigmoid', 40)).toBeCloseTo(1);
    expect(activate('sigmoid', -40)).toBeCloseTo(0);
    expect(activate('tanh', 40)).toBeCloseTo(1);
  });

  it('matches its derivatives numerically', () => {
    const h = 1e-6;
    for (const kind of ['relu', 'tanh', 'sigmoid', 'leakyRelu', 'gelu', 'linear'] as const) {
      for (const x of [-1.7, -0.3, 0.4, 2.1]) {
        const numeric = (activate(kind, x + h) - activate(kind, x - h)) / (2 * h);
        expect(activateDerivative(kind, x)).toBeCloseTo(numeric, 3);
      }
    }
  });

  it('gives sigmoid a vanishing derivative far from zero, which is the whole problem', () => {
    expect(activateDerivative('sigmoid', 10)).toBeLessThan(1e-3);
    expect(activateDerivative('tanh', 10)).toBeLessThan(1e-3);
    expect(activateDerivative('relu', 10)).toBe(1);
  });
});

describe('TinyNet — construction', () => {
  it('builds one weight matrix and bias vector per layer gap', () => {
    const net = new TinyNet({ architecture: [2, 4, 3, 1], activation: 'tanh', seed: 1 });
    expect(net.weights).toHaveLength(3);
    expect(net.weights[0]).toHaveLength(4);
    expect(net.weights[0]![0]).toHaveLength(2);
    expect(net.biases[2]).toHaveLength(1);
  });

  it('counts parameters correctly', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'tanh', seed: 1 });
    // (2*2 + 2) + (2*1 + 1)
    expect(net.parameterCount).toBe(9);
  });

  it('is deterministic for a seed and different across seeds', () => {
    const a = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 7 });
    const b = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 7 });
    const c = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 8 });
    expect(a.weights).toEqual(b.weights);
    expect(a.weights).not.toEqual(c.weights);
  });

  it('starts biases at zero', () => {
    const net = new TinyNet({ architecture: [2, 3, 1], activation: 'relu', seed: 2 });
    expect(net.biases.flat().every((b) => b === 0)).toBe(true);
  });
});

describe('TinyNet — forward pass', () => {
  it('emits a probability from the output sigmoid', () => {
    const net = new TinyNet({ architecture: [2, 4, 1], activation: 'tanh', seed: 3 });
    const out = net.predict([0.5, -0.2]);
    expect(out).toBeGreaterThan(0);
    expect(out).toBeLessThan(1);
  });

  it('exposes activations for every layer including the input', () => {
    const net = new TinyNet({ architecture: [2, 4, 3, 1], activation: 'tanh', seed: 3 });
    const pass = net.forward([1, -1]);
    expect(pass.activations).toHaveLength(4);
    expect(pass.activations[0]).toEqual([1, -1]);
    expect(pass.activations[1]).toHaveLength(4);
    expect(pass.activations[3]).toHaveLength(1);
  });

  it('computes a known two-layer network by hand', () => {
    const net = new TinyNet({ architecture: [2, 1, 1], activation: 'linear', seed: 1 });
    net.weights = [[[2, 3]], [[4]]];
    net.biases = [[1], [0]];
    // hidden = 2*1 + 3*2 + 1 = 9 (linear); output = sigmoid(4*9) ≈ 1
    const pass = net.forward([1, 2]);
    expect(pass.activations[1]![0]).toBeCloseTo(9);
    expect(pass.output).toBeCloseTo(1 / (1 + Math.exp(-36)));
  });
});

describe('TinyNet — gradients', () => {
  it('matches numerically estimated gradients', () => {
    const net = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 5 });
    const samples: TrainingSample[] = [
      { x: [0.4, -0.7], label: 1 },
      { x: [-0.9, 0.2], label: 0 },
      { x: [0.1, 0.8], label: 1 },
    ];

    const { weightGradients } = net.computeGradients(samples);
    const h = 1e-5;

    for (let l = 0; l < net.weights.length; l++) {
      for (let to = 0; to < net.weights[l]!.length; to++) {
        for (let from = 0; from < net.weights[l]![to]!.length; from++) {
          const original = net.weights[l]![to]![from]!;

          net.weights[l]![to]![from] = original + h;
          const up = net.loss(samples);
          net.weights[l]![to]![from] = original - h;
          const down = net.loss(samples);
          net.weights[l]![to]![from] = original;

          expect(weightGradients[l]![to]![from]).toBeCloseTo((up - down) / (2 * h), 4);
        }
      }
    }
  });

  it('matches numerically estimated bias gradients', () => {
    const net = new TinyNet({ architecture: [2, 3, 1], activation: 'relu', seed: 9 });
    const samples: TrainingSample[] = [
      { x: [0.3, 0.6], label: 1 },
      { x: [-0.4, -0.2], label: 0 },
    ];
    const { biasGradients } = net.computeGradients(samples);
    const h = 1e-5;

    for (let l = 0; l < net.biases.length; l++) {
      for (let to = 0; to < net.biases[l]!.length; to++) {
        const original = net.biases[l]![to]!;
        net.biases[l]![to] = original + h;
        const up = net.loss(samples);
        net.biases[l]![to] = original - h;
        const down = net.loss(samples);
        net.biases[l]![to] = original;
        expect(biasGradients[l]![to]).toBeCloseTo((up - down) / (2 * h), 4);
      }
    }
  });

  it('returns zero gradients and zero loss for an empty batch', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'tanh', seed: 1 });
    const result = net.computeGradients([]);
    expect(result.loss).toBe(0);
    expect(result.weightGradients.flat(2).every((g) => g === 0)).toBe(true);
  });

  it('flattens gradients into one record per edge', () => {
    const net = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 1 });
    const edges = net.edgeGradients([{ x: [0.2, 0.3], label: 1 }]);
    expect(edges).toHaveLength(2 * 3 + 3 * 1);
    for (const edge of edges) expect(Number.isFinite(edge.value)).toBe(true);
  });

  it('shows gradient signal shrinking through a deep sigmoid stack', () => {
    const deep = new TinyNet({
      architecture: [2, 6, 6, 6, 6, 1],
      activation: 'sigmoid',
      seed: 31,
    });
    const samples = generateToyDataset('spiral', 60, 0.05, 31).map((p) => ({
      x: p.x,
      label: p.label,
    }));

    const edges = deep.edgeGradients(samples);
    const magnitudeAt = (layer: number) => {
      const values = edges.filter((e) => e.layer === layer).map((e) => Math.abs(e.value));
      return values.reduce((a, b) => a + b, 0) / values.length;
    };

    expect(magnitudeAt(0)).toBeLessThan(magnitudeAt(4));
  });
});

describe('TinyNet — training', () => {
  it('reduces loss on a learnable problem', () => {
    const net = new TinyNet({ architecture: [2, 6, 1], activation: 'tanh', seed: 13 });
    const samples = generateToyDataset('circles', 200, 0.08, 13).map((p) => ({
      x: p.x,
      label: p.label,
    }));

    const before = net.loss(samples);
    for (let epoch = 0; epoch < 200; epoch++) {
      net.trainEpoch(samples, { learningRate: 0.3, batchSize: 16, epoch });
    }
    expect(net.loss(samples)).toBeLessThan(before);
  });

  it('learns circles to high accuracy', () => {
    const net = new TinyNet({ architecture: [2, 8, 1], activation: 'tanh', seed: 13 });
    const { train, validation } = splitDataset(generateToyDataset('circles', 240, 0.06, 13), 0.7);
    const samples = train.map((p) => ({ x: p.x, label: p.label }));

    for (let epoch = 0; epoch < 400; epoch++) {
      net.trainEpoch(samples, { learningRate: 0.3, batchSize: 16, epoch });
    }
    expect(net.accuracy(validation.map((p) => ({ x: p.x, label: p.label })))).toBeGreaterThan(0.9);
  });

  it('learns XOR, which the perceptron could not', () => {
    const net = new TinyNet({ architecture: [2, 4, 1], activation: 'tanh', seed: 7 });
    const samples = generateToyDataset('xor', 200, 0.05, 7).map((p) => ({ x: p.x, label: p.label }));

    for (let epoch = 0; epoch < 500; epoch++) {
      net.trainEpoch(samples, { learningRate: 0.3, batchSize: 16, epoch });
    }
    expect(net.accuracy(samples)).toBeGreaterThan(0.9);
  });

  it('does one update per batch', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'tanh', seed: 1 });
    const before = JSON.stringify(net.weights);
    net.trainBatch([{ x: [0.5, 0.5], label: 1 }], 0.5);
    expect(JSON.stringify(net.weights)).not.toBe(before);
  });

  it('leaves weights alone at a zero learning rate', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'tanh', seed: 1 });
    const before = JSON.stringify(net.weights);
    net.trainEpoch([{ x: [0.5, 0.5], label: 1 }], { learningRate: 0, batchSize: 1 });
    expect(JSON.stringify(net.weights)).toBe(before);
  });

  it('handles a batch size larger than the dataset', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'tanh', seed: 1 });
    const loss = net.trainEpoch([{ x: [0.1, 0.2], label: 0 }], { learningRate: 0.1, batchSize: 999 });
    expect(Number.isFinite(loss)).toBe(true);
  });

  it('trains reproducibly from the same seed', () => {
    const samples = generateToyDataset('xor', 80, 0.05, 3).map((p) => ({ x: p.x, label: p.label }));
    const run = () => {
      const net = new TinyNet({ architecture: [2, 4, 1], activation: 'tanh', seed: 21 });
      for (let epoch = 0; epoch < 40; epoch++) {
        net.trainEpoch(samples, { learningRate: 0.2, batchSize: 8, epoch });
      }
      return net.loss(samples);
    };
    expect(run()).toBe(run());
  });

  it('clones without sharing weight arrays', () => {
    const net = new TinyNet({ architecture: [2, 3, 1], activation: 'tanh', seed: 4 });
    const copy = net.clone();
    expect(copy.weights).toEqual(net.weights);
    copy.weights[0]![0]![0] = 999;
    expect(net.weights[0]![0]![0]).not.toBe(999);
  });

  it('reports accuracy at the 0.5 decision boundary', () => {
    const net = new TinyNet({ architecture: [2, 2, 1], activation: 'linear', seed: 1 });
    net.weights = [[[0, 0], [0, 0]], [[0, 0]]];
    net.biases = [[0, 0], [0]];
    // Every output is exactly 0.5, which counts as class 1.
    expect(net.accuracy([{ x: [1, 1], label: 1 }])).toBe(1);
    expect(net.accuracy([{ x: [1, 1], label: 0 }])).toBe(0);
  });
});

describe('toy datasets', () => {
  it('are deterministic for a seed', () => {
    expect(generateToyDataset('spiral', 50, 0.05, 9)).toEqual(generateToyDataset('spiral', 50, 0.05, 9));
  });

  it('produce the requested count with both classes', () => {
    for (const kind of ['circles', 'xor', 'spiral', 'gaussian'] as const) {
      const data = generateToyDataset(kind, 100, 0.05, 4);
      expect(data).toHaveLength(100);
      expect(data.some((p) => p.label === 1)).toBe(true);
      expect(data.some((p) => p.label === 0)).toBe(true);
    }
  });

  it('makes circles genuinely non-linearly-separable', () => {
    const data = generateToyDataset('circles', 200, 0.05, 4);
    const inner = data.filter((p) => p.label === 1);
    const outer = data.filter((p) => p.label === 0);
    const meanRadius = (pts: typeof data) =>
      pts.reduce((a, p) => a + Math.hypot(p.x[0], p.x[1]), 0) / pts.length;
    expect(meanRadius(inner)).toBeLessThan(meanRadius(outer));
  });

  it('splits into disjoint train and validation sets', () => {
    const data = generateToyDataset('xor', 100, 0.05, 2);
    const { train, validation } = splitDataset(data, 0.7);
    expect(train.length + validation.length).toBe(100);
    expect(validation.length).toBeGreaterThan(0);
    for (const point of validation) expect(train).not.toContain(point);
  });

  it('never returns an empty validation set', () => {
    const { validation } = splitDataset(generateToyDataset('xor', 4, 0, 1), 0.99);
    expect(validation.length).toBeGreaterThan(0);
  });
});
