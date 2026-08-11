/**
 * Two-dimensional toy datasets for Worlds 2 and 3.
 *
 * Deterministic from a seed so a level replays identically, and generated
 * rather than shipped so the geometry is exact.
 */
import { createRng } from './shared';

export type ToyDatasetKind = 'circles' | 'xor' | 'spiral' | 'gaussian';

export interface ToyPoint {
  /** Coordinates, roughly in [-1, 1]. */
  x: [number, number];
  /** Binary class, matching the sigmoid output of the tiny net. */
  label: 0 | 1;
}

export function generateToyDataset(
  kind: ToyDatasetKind,
  count: number,
  noise: number,
  seed: number
): ToyPoint[] {
  const rng = createRng(seed);
  const jitter = () => (rng() * 2 - 1) * noise;
  const points: ToyPoint[] = [];

  switch (kind) {
    case 'circles': {
      // Concentric rings: no straight line separates them.
      for (let i = 0; i < count; i++) {
        const inner = i % 2 === 0;
        const radius = (inner ? 0.3 : 0.85) + jitter();
        const angle = rng() * Math.PI * 2;
        points.push({
          x: [radius * Math.cos(angle) + jitter(), radius * Math.sin(angle) + jitter()],
          label: inner ? 1 : 0,
        });
      }
      break;
    }

    case 'xor': {
      for (let i = 0; i < count; i++) {
        const x0 = rng() * 2 - 1;
        const x1 = rng() * 2 - 1;
        points.push({
          x: [x0 + jitter(), x1 + jitter()],
          label: x0 * x1 > 0 ? 1 : 0,
        });
      }
      break;
    }

    case 'spiral': {
      // Two interleaved arms — the classic depth test.
      for (let i = 0; i < count; i++) {
        const arm = i % 2;
        const t = (i / count) * 3.5 + 0.3;
        const angle = t * 2 + arm * Math.PI;
        points.push({
          x: [(t / 4) * Math.cos(angle) + jitter(), (t / 4) * Math.sin(angle) + jitter()],
          label: arm === 0 ? 1 : 0,
        });
      }
      break;
    }

    case 'gaussian': {
      for (let i = 0; i < count; i++) {
        const positive = i % 2 === 0;
        const cx = positive ? 0.5 : -0.5;
        const cy = positive ? 0.5 : -0.5;
        points.push({
          x: [cx + (rng() * 2 - 1) * 0.35 + jitter(), cy + (rng() * 2 - 1) * 0.35 + jitter()],
          label: positive ? 1 : 0,
        });
      }
      break;
    }
  }

  return points;
}

export interface SplitDataset {
  train: ToyPoint[];
  validation: ToyPoint[];
}

/** Deterministic interleaved split, so both sets cover the same region. */
export function splitDataset(points: readonly ToyPoint[], trainSplit: number): SplitDataset {
  const holdOutEvery = Math.max(2, Math.round(1 / Math.max(1 - trainSplit, 1e-6)));
  const train: ToyPoint[] = [];
  const validation: ToyPoint[] = [];

  points.forEach((point, i) => {
    (i % holdOutEvery === holdOutEvery - 1 ? validation : train).push(point);
  });

  // Never hand back an empty validation set — every metric here depends on it.
  if (validation.length === 0 && train.length > 1) validation.push(train.pop()!);
  return { train, validation };
}
