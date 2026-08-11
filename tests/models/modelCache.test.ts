import { describe, it, expect, beforeEach } from 'vitest';
import {
  subscribeToModelLoads,
  getLoadState,
  markDownloading,
  markReady,
  markError,
  clearError,
  reportProgress,
  resetLoadStates,
  type ModelLoadState,
} from '@/models/modelCache';

describe('model load state', () => {
  beforeEach(() => resetLoadStates());

  it('starts idle for a model nothing has touched', () => {
    const state = getLoadState('some/model');
    expect(state.status).toBe('idle');
    expect(state.progress).toBeNull();
    expect(state.error).toBeNull();
  });

  it('moves through downloading to ready', () => {
    markDownloading('m');
    expect(getLoadState('m').status).toBe('downloading');
    markReady('m');
    expect(getLoadState('m').status).toBe('ready');
    expect(getLoadState('m').progress).toBe(1);
  });

  it('records an error message from a thrown Error', () => {
    markError('m', new Error('network died'));
    const state = getLoadState('m');
    expect(state.status).toBe('error');
    expect(state.error).toBe('network died');
  });

  it('records an error from a non-Error throw', () => {
    markError('m', 'plain string');
    expect(getLoadState('m').error).toBe('plain string');
  });

  it('clears an error so a retry can start clean', () => {
    markError('m', new Error('boom'));
    clearError('m');
    const state = getLoadState('m');
    expect(state.status).toBe('idle');
    expect(state.error).toBeNull();
  });

  it('notifies subscribers of every transition', () => {
    const seen: ModelLoadState[] = [];
    const unsubscribe = subscribeToModelLoads((state) => seen.push(state));

    markDownloading('m');
    markReady('m');
    unsubscribe();
    markError('m', new Error('after unsubscribe'));

    expect(seen.map((s) => s.status)).toEqual(['downloading', 'ready']);
  });

  it('replays known state to a late subscriber', () => {
    markDownloading('early');
    const seen: string[] = [];
    subscribeToModelLoads((state) => seen.push(state.modelId));
    expect(seen).toContain('early');
  });
});

describe('progress reporting', () => {
  beforeEach(() => resetLoadStates());

  it('reads a 0-100 progress value, as transformers.js reports it', () => {
    reportProgress('m', { progress: 42, file: 'model.onnx' });
    const state = getLoadState('m');
    expect(state.progress).toBeCloseTo(0.42);
    expect(state.file).toBe('model.onnx');
  });

  it('reads a 0-1 progress value, as WebLLM reports it', () => {
    reportProgress('m', { progress: 0.42 });
    expect(getLoadState('m').progress).toBeCloseTo(0.42);
  });

  it('derives progress from loaded and total when no fraction is given', () => {
    reportProgress('m', { loaded: 5_000_000, total: 20_000_000 });
    const state = getLoadState('m');
    expect(state.progress).toBeCloseTo(0.25);
    expect(state.loadedMB).toBe(5);
    expect(state.totalMB).toBe(20);
  });

  it('reports null progress when the source gives nothing to work with', () => {
    reportProgress('m', { file: 'config.json' });
    expect(getLoadState('m').progress).toBeNull();
  });

  it('ignores a zero total rather than dividing by it', () => {
    reportProgress('m', { loaded: 10, total: 0 });
    const state = getLoadState('m');
    expect(state.progress).toBeNull();
    expect(state.totalMB).toBeNull();
  });

  it('clamps progress into the unit range', () => {
    reportProgress('m', { progress: 150 });
    expect(getLoadState('m').progress).toBe(1);
    reportProgress('m', { progress: -0.5 });
    expect(getLoadState('m').progress).toBe(0);
  });

  it('marks the model as downloading while progress arrives', () => {
    reportProgress('m', { progress: 10 });
    expect(getLoadState('m').status).toBe('downloading');
  });
});
