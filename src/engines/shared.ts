/**
 * Pure maths shared across every engine.
 *
 * Nothing here touches React, the DOM, or any model wrapper — engines stay
 * testable offline because all of their arithmetic lives in this file and in
 * their own reducers.
 */

/** Deterministic PRNG (mulberry32). Seeded so a level replays identically. */
export function createRng(seed: number): () => number {
  let a = (seed >>> 0) || 1;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function assertSameLength(a: readonly number[], b: readonly number[]): void {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
}

export function dot(a: readonly number[], b: readonly number[]): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

export function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

export function addVectors(a: readonly number[], b: readonly number[]): number[] {
  assertSameLength(a, b);
  return a.map((v, i) => v + b[i]!);
}

export function subtractVectors(a: readonly number[], b: readonly number[]): number[] {
  assertSameLength(a, b);
  return a.map((v, i) => v - b[i]!);
}

export function scaleVector(a: readonly number[], s: number): number[] {
  return a.map((v) => v * s);
}

/** Cosine similarity in [-1, 1]. Returns 0 against a zero vector. */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return clamp(dot(a, b) / (na * nb), -1, 1);
}

export function euclideanDistance(a: readonly number[], b: readonly number[]): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function standardDeviation(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

/**
 * Numerically stable softmax. Temperature at or below zero collapses to a
 * one-hot on the argmax, which is what greedy decoding means.
 */
export function softmax(logits: readonly number[], temperature = 1): number[] {
  if (logits.length === 0) return [];
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[best]!) best = i;
    return logits.map((_, i) => (i === best ? 1 : 0));
  }
  const scaled = logits.map((l) => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

/** Shannon entropy in bits. Zero-probability terms contribute nothing. */
export function entropyBits(probs: readonly number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** Total variation distance in [0, 1] — half the L1 distance. */
export function totalVariationDistance(p: readonly number[], q: readonly number[]): number {
  assertSameLength(p, q);
  let sum = 0;
  for (let i = 0; i < p.length; i++) sum += Math.abs(p[i]! - q[i]!);
  return sum / 2;
}

/** Turns non-negative weights into a distribution; all-zero falls back to uniform. */
export function normalizeDistribution(weights: readonly number[]): number[] {
  if (weights.length === 0) return [];
  const safe = weights.map((w) => (Number.isFinite(w) && w > 0 ? w : 0));
  const total = safe.reduce((a, b) => a + b, 0);
  if (total === 0) return safe.map(() => 1 / safe.length);
  return safe.map((w) => w / total);
}

/** Competition ranks, 1-based, with ties averaged. */
export function ranksOf(values: readonly number[]): number[] {
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Array<number>(values.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j + 1 < indexed.length && indexed[j + 1]!.v === indexed[i]!.v) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k]!.i] = avg;
    i = j + 1;
  }
  return ranks;
}

/** Spearman rank correlation in [-1, 1]. Returns 0 if either series is constant. */
export function spearmanCorrelation(a: readonly number[], b: readonly number[]): number {
  assertSameLength(a, b);
  if (a.length < 2) return 0;

  const ra = ranksOf(a);
  const rb = ranksOf(b);
  const ma = mean(ra);
  const mb = mean(rb);

  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < ra.length; i++) {
    const x = ra[i]! - ma;
    const y = rb[i]! - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  if (da === 0 || db === 0) return 0;
  return clamp(num / Math.sqrt(da * db), -1, 1);
}

export interface KMeansResult {
  assignments: number[];
  centroids: number[][];
}

/**
 * k-means with deterministic k-means++ seeding.
 *
 * Used to discover the real group structure of live embeddings, so the
 * "correct" grouping a player is scored against comes from the model rather
 * than from an answer key in the JSON.
 */
export function kMeans(
  points: readonly number[][],
  k: number,
  opts: { seed?: number; maxIterations?: number } = {}
): KMeansResult {
  const { seed = 1, maxIterations = 50 } = opts;
  const n = points.length;
  if (n === 0) return { assignments: [], centroids: [] };

  const effectiveK = Math.max(1, Math.min(k, n));
  const rng = createRng(seed);
  const dim = points[0]!.length;

  // k-means++ seeding: first centre at random, the rest biased towards distance.
  const centroids: number[][] = [[...points[Math.floor(rng() * n)]!]];
  while (centroids.length < effectiveK) {
    const d2 = points.map((p) => {
      const nearest = Math.min(...centroids.map((c) => euclideanDistance(p, c)));
      return nearest * nearest;
    });
    const total = d2.reduce((a, b) => a + b, 0);
    if (total === 0) {
      centroids.push([...points[centroids.length % n]!]);
      continue;
    }
    let r = rng() * total;
    let chosen = 0;
    for (let i = 0; i < d2.length; i++) {
      r -= d2[i]!;
      if (r <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push([...points[chosen]!]);
  }

  let assignments = new Array<number>(n).fill(0);
  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const d = euclideanDistance(points[i]!, centroids[c]!);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }

    for (let c = 0; c < centroids.length; c++) {
      const members = points.filter((_, i) => assignments[i] === c);
      if (members.length === 0) continue;
      const next = new Array<number>(dim).fill(0);
      for (const m of members) for (let d = 0; d < dim; d++) next[d]! += m[d]!;
      centroids[c] = next.map((v) => v / members.length);
    }

    if (!changed) break;
  }

  return { assignments, centroids };
}

/**
 * Mean silhouette coefficient in [-1, 1] for a labelled point set.
 *
 * Measures how much better each point fits its own cluster than the nearest
 * other one, which is exactly the question "did you group these sensibly?".
 * Returns 0 when there is only one cluster, where the measure is undefined.
 */
export function silhouetteScore(points: readonly number[][], labels: readonly number[]): number {
  assertSameLength(points.map(() => 0), labels);
  const distinct = new Set(labels);
  if (distinct.size < 2 || points.length < 2) return 0;

  const scores: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const own = labels[i]!;
    const sameCluster: number[] = [];
    const otherClusters = new Map<number, number[]>();

    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = euclideanDistance(points[i]!, points[j]!);
      if (labels[j] === own) {
        sameCluster.push(d);
      } else {
        const bucket = otherClusters.get(labels[j]!) ?? [];
        bucket.push(d);
        otherClusters.set(labels[j]!, bucket);
      }
    }

    // A lone point in its cluster is neither well nor badly placed.
    if (sameCluster.length === 0 || otherClusters.size === 0) {
      scores.push(0);
      continue;
    }

    const a = mean(sameCluster);
    const b = Math.min(...[...otherClusters.values()].map((ds) => mean(ds)));
    const denom = Math.max(a, b);
    scores.push(denom === 0 ? 0 : (b - a) / denom);
  }

  return mean(scores);
}
