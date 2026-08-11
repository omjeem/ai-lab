import { describe, it, expect, beforeEach } from 'vitest';
import { softmaxRow, topK, toNestedArray, evictModel, loadOnce, evictAllModels } from '@/models/transformersRuntime';
import { getLoadState, resetLoadStates } from '@/models/modelCache';

describe('softmaxRow', () => {
  it('produces a distribution summing to one', () => {
    const p = softmaxRow([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[2]).toBeGreaterThan(p[0]!);
  });

  it('is stable for large logits', () => {
    const p = softmaxRow([900, 901, 902]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it('returns an empty array for empty input', () => {
    expect(softmaxRow([])).toEqual([]);
  });
});

describe('topK', () => {
  const probs = [0.1, 0.5, 0.05, 0.35];
  const decode = (id: number) => `tok${id}`;

  it('returns the k most likely entries, highest first', () => {
    const ranked = topK(probs, 2, decode);
    expect(ranked.map((r) => r.id)).toEqual([1, 3]);
    expect(ranked[0]!.token).toBe('tok1');
    expect(ranked[0]!.probability).toBeCloseTo(0.5);
  });

  it('never returns more entries than exist', () => {
    expect(topK(probs, 99, decode)).toHaveLength(4);
  });

  it('always returns at least one entry', () => {
    expect(topK(probs, 0, decode)).toHaveLength(1);
  });
});

describe('toNestedArray', () => {
  it('prefers a tolist method, as tensors expose', () => {
    expect(toNestedArray({ tolist: () => [[1, 2]] })).toEqual([[1, 2]]);
  });

  it('falls back to typed-array data', () => {
    expect(toNestedArray({ data: new Float32Array([1, 2, 3]) })).toEqual([1, 2, 3]);
  });

  it('returns an empty array for anything unrecognisable', () => {
    expect(toNestedArray(null)).toEqual([]);
    expect(toNestedArray({ nope: true })).toEqual([]);
  });
});

describe('loadOnce', () => {
  beforeEach(() => {
    resetLoadStates();
    evictAllModels();
  });

  it('runs the factory once and reuses the result', async () => {
    let calls = 0;
    const factory = async () => {
      calls++;
      return { value: calls };
    };

    const a = await loadOnce('k', { modelId: 'm', backend: 'wasm' }, factory);
    const b = await loadOnce('k', { modelId: 'm', backend: 'wasm' }, factory);

    expect(calls).toBe(1);
    expect(a).toBe(b);
  });

  it('marks the model ready once the factory resolves', async () => {
    await loadOnce('k', { modelId: 'm', backend: 'wasm' }, async () => ({}));
    expect(getLoadState('m').status).toBe('ready');
  });

  it('records the failure and lets a retry genuinely try again', async () => {
    let attempts = 0;
    const flaky = async () => {
      attempts++;
      if (attempts === 1) throw new Error('network died');
      return { ok: true };
    };

    await expect(loadOnce('k', { modelId: 'm', backend: 'wasm' }, flaky)).rejects.toThrow(
      'network died'
    );
    expect(getLoadState('m').status).toBe('error');
    expect(getLoadState('m').error).toBe('network died');

    // The rejected promise must not be cached, or retry could never succeed.
    const result = await loadOnce('k', { modelId: 'm', backend: 'wasm' }, flaky);
    expect(result).toEqual({ ok: true });
    expect(attempts).toBe(2);
    expect(getLoadState('m').status).toBe('ready');
  });

  it('passes the requested backend through to the factory', async () => {
    let seen = '';
    await loadOnce('k', { modelId: 'm', backend: 'wasm' }, async ({ backend }) => {
      seen = backend;
      return {};
    });
    expect(seen).toBe('wasm');
  });

  it('forwards progress from the factory to the load state', async () => {
    await loadOnce('k', { modelId: 'm', backend: 'wasm' }, async ({ onProgress }) => {
      onProgress({ progress: 50, file: 'weights.onnx' });
      return {};
    });
    // Ready overwrites the fraction, but the file reported along the way sticks.
    expect(getLoadState('m').file).toBe('weights.onnx');
  });

  it('re-creates the model after an explicit eviction', async () => {
    let calls = 0;
    const factory = async () => ({ id: ++calls });

    await loadOnce('k', { modelId: 'm', backend: 'wasm' }, factory);
    evictModel('k');
    await loadOnce('k', { modelId: 'm', backend: 'wasm' }, factory);

    expect(calls).toBe(2);
  });
});
