import { describe, it, expect } from 'vitest';
import {
  createRng,
  clamp,
  dot,
  norm,
  addVectors,
  subtractVectors,
  scaleVector,
  cosineSimilarity,
  euclideanDistance,
  mean,
  standardDeviation,
  softmax,
  entropyBits,
  totalVariationDistance,
  normalizeDistribution,
  ranksOf,
  spearmanCorrelation,
  kMeans,
  silhouetteScore,
  evaluateArithmetic,
} from '@/engines/shared';

describe('createRng', () => {
  it('is deterministic for a given seed', () => {
    const a = createRng(42);
    const b = createRng(42);
    const seqA = [a(), a(), a(), a()];
    const seqB = [b(), b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });

  it('produces different sequences for different seeds', () => {
    const a = createRng(1);
    const b = createRng(2);
    expect(a()).not.toEqual(b());
  });

  it('stays inside [0, 1)', () => {
    const rng = createRng(7);
    for (let i = 0; i < 500; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('clamp', () => {
  it('bounds values on both sides and passes through the middle', () => {
    expect(clamp(5, 0, 1)).toBe(1);
    expect(clamp(-5, 0, 1)).toBe(0);
    expect(clamp(0.5, 0, 1)).toBe(0.5);
  });
});

describe('vector maths', () => {
  it('computes dot products and norms', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(norm([3, 4])).toBe(5);
    expect(norm([0, 0])).toBe(0);
  });

  it('adds, subtracts and scales', () => {
    expect(addVectors([1, 2], [3, 4])).toEqual([4, 6]);
    expect(subtractVectors([5, 5], [1, 2])).toEqual([4, 3]);
    expect(scaleVector([1, -2], 3)).toEqual([3, -6]);
  });

  it('throws on mismatched lengths rather than silently truncating', () => {
    expect(() => dot([1, 2], [1])).toThrow();
    expect(() => addVectors([1, 2], [1])).toThrow();
  });

  it('computes cosine similarity with the expected extremes', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it('ignores magnitude in cosine but not in euclidean', () => {
    expect(cosineSimilarity([1, 1], [10, 10])).toBeCloseTo(1);
    expect(euclideanDistance([1, 1], [10, 10])).toBeCloseTo(Math.sqrt(162));
  });

  it('returns 0 cosine against a zero vector instead of NaN', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });
});

describe('descriptive statistics', () => {
  it('computes mean and standard deviation', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(standardDeviation([2, 2, 2])).toBe(0);
    expect(standardDeviation([1, 3])).toBeCloseTo(1);
  });

  it('returns 0 for empty input rather than NaN', () => {
    expect(mean([])).toBe(0);
    expect(standardDeviation([])).toBe(0);
  });
});

describe('softmax', () => {
  it('produces a distribution summing to one', () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[2]).toBeGreaterThan(p[1]!);
  });

  it('is numerically stable for large logits', () => {
    const p = softmax([1000, 1001, 1002]);
    expect(p.every((v) => Number.isFinite(v))).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it('sharpens as temperature falls and flattens as it rises', () => {
    const sharp = softmax([1, 2, 3], 0.2);
    const flat = softmax([1, 2, 3], 5);
    expect(Math.max(...sharp)).toBeGreaterThan(Math.max(...flat));
  });

  it('treats temperature at or below zero as greedy', () => {
    const p = softmax([1, 5, 2], 0);
    expect(p[1]).toBeCloseTo(1);
    expect(p[0]).toBeCloseTo(0);
  });
});

describe('entropy and distribution distance', () => {
  it('reports maximum entropy for a uniform distribution', () => {
    expect(entropyBits([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2);
    expect(entropyBits([1, 0, 0, 0])).toBeCloseTo(0);
  });

  it('ignores zero-probability terms rather than producing NaN', () => {
    expect(entropyBits([0.5, 0.5, 0])).toBeCloseTo(1);
  });

  it('computes total variation distance between 0 and 1', () => {
    expect(totalVariationDistance([1, 0], [1, 0])).toBeCloseTo(0);
    expect(totalVariationDistance([1, 0], [0, 1])).toBeCloseTo(1);
    expect(totalVariationDistance([0.5, 0.5], [0.75, 0.25])).toBeCloseTo(0.25);
  });

  it('normalises arbitrary non-negative weights', () => {
    expect(normalizeDistribution([2, 2])).toEqual([0.5, 0.5]);
    // An all-zero allocation falls back to uniform instead of dividing by zero.
    expect(normalizeDistribution([0, 0, 0, 0])).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});

describe('rank correlation', () => {
  it('ranks values, averaging ties', () => {
    expect(ranksOf([10, 20, 30])).toEqual([1, 2, 3]);
    expect(ranksOf([10, 10, 30])).toEqual([1.5, 1.5, 3]);
  });

  it('scores identical orderings at 1 and reversed at -1', () => {
    expect(spearmanCorrelation([1, 2, 3, 4], [1, 2, 3, 4])).toBeCloseTo(1);
    expect(spearmanCorrelation([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1);
  });

  it('scores a single swap between the extremes', () => {
    const r = spearmanCorrelation([1, 2, 3, 4], [2, 1, 3, 4]);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
  });

  it('returns 0 when a series has no variance', () => {
    expect(spearmanCorrelation([1, 1, 1], [1, 2, 3])).toBe(0);
  });
});

describe('kMeans', () => {
  it('recovers two well-separated groups', () => {
    const points = [
      [0, 0],
      [0.1, 0.1],
      [-0.1, 0.05],
      [10, 10],
      [10.1, 9.9],
      [9.9, 10.2],
    ];
    const { assignments, centroids } = kMeans(points, 2, { seed: 1 });
    expect(centroids).toHaveLength(2);
    expect(assignments[0]).toBe(assignments[1]);
    expect(assignments[0]).toBe(assignments[2]);
    expect(assignments[3]).toBe(assignments[4]);
    expect(assignments[0]).not.toBe(assignments[3]);
  });

  it('is deterministic for a fixed seed', () => {
    const points = [
      [1, 1],
      [1.2, 0.8],
      [8, 8],
      [8.2, 7.9],
    ];
    const a = kMeans(points, 2, { seed: 5 });
    const b = kMeans(points, 2, { seed: 5 });
    expect(a.assignments).toEqual(b.assignments);
  });

  it('handles k larger than the point count without crashing', () => {
    const { assignments } = kMeans([[1, 1], [2, 2]], 5, { seed: 1 });
    expect(assignments).toHaveLength(2);
  });
});

describe('silhouetteScore', () => {
  it('is near 1 for tight, well-separated clusters', () => {
    const points = [
      [0, 0],
      [0.1, 0],
      [10, 10],
      [10.1, 10],
    ];
    expect(silhouetteScore(points, [0, 0, 1, 1])).toBeGreaterThan(0.9);
  });

  it('is low when the labelling cuts across the real structure', () => {
    const points = [
      [0, 0],
      [0.1, 0],
      [10, 10],
      [10.1, 10],
    ];
    expect(silhouetteScore(points, [0, 1, 0, 1])).toBeLessThan(0);
  });

  it('returns 0 when every point shares one label', () => {
    expect(silhouetteScore([[0, 0], [1, 1]], [0, 0])).toBe(0);
  });
});

describe('evaluateArithmetic', () => {
  it('evaluates addition, subtraction, multiplication and division', () => {
    expect(evaluateArithmetic('3 + 4')).toBe(7);
    expect(evaluateArithmetic('10 - 2')).toBe(8);
    expect(evaluateArithmetic('6 * 7')).toBe(42);
    expect(evaluateArithmetic('20 / 4')).toBe(5);
  });

  it('respects operator precedence and parentheses', () => {
    expect(evaluateArithmetic('2 + 3 * 4')).toBe(14);
    expect(evaluateArithmetic('(2 + 3) * 4')).toBe(20);
  });

  it('handles decimals and negative results', () => {
    expect(evaluateArithmetic('1.5 + 2.5')).toBe(4);
    expect(evaluateArithmetic('3 - 10')).toBe(-7);
  });

  it('returns null for division by zero rather than Infinity/NaN', () => {
    expect(evaluateArithmetic('5 / 0')).toBeNull();
  });

  it('returns null for anything that is not a plain arithmetic expression', () => {
    expect(evaluateArithmetic('alert(1)')).toBeNull();
    expect(evaluateArithmetic('3 + ')).toBeNull();
    expect(evaluateArithmetic('')).toBeNull();
    expect(evaluateArithmetic('process.exit()')).toBeNull();
    expect(evaluateArithmetic('3 + 4; alert(1)')).toBeNull();
  });
});
