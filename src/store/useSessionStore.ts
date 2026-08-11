'use client';

/**
 * Per-run state for the chapter currently being played.
 *
 * Deliberately not persisted: a run is the live thing on screen, and resuming
 * mid-level would mean restoring model state that no longer exists. What
 * survives a reload is the durable result, which lives in the progress store.
 */
import { create } from 'zustand';
import type { ScoreResult } from '@/types/game';

export type RunPhase = 'concept' | 'loading' | 'playing' | 'scored' | 'failed';

interface SessionState {
  chapterId: string | null;
  levelIndex: number;
  phase: RunPhase;
  attempts: number;
  startedAt: number | null;
  lastResult: ScoreResult | null;
  /** Set when a model-backed chapter could not start. */
  loadError: string | null;

  beginChapter: (chapterId: string) => void;
  setPhase: (phase: RunPhase) => void;
  goToLevel: (index: number) => void;
  registerAttempt: () => void;
  recordResult: (result: ScoreResult) => void;
  setLoadError: (message: string | null) => void;
  endChapter: () => void;
}

export const useSessionStore = create<SessionState>()((set) => ({
  chapterId: null,
  levelIndex: 0,
  phase: 'concept',
  attempts: 0,
  startedAt: null,
  lastResult: null,
  loadError: null,

  beginChapter: (chapterId) =>
    set({
      chapterId,
      levelIndex: 0,
      phase: 'concept',
      attempts: 0,
      startedAt: Date.now(),
      lastResult: null,
      loadError: null,
    }),

  setPhase: (phase) => set({ phase }),

  goToLevel: (levelIndex) =>
    // A new level is a fresh run: attempts and the previous score do not carry.
    set({ levelIndex, phase: 'concept', attempts: 0, lastResult: null, loadError: null }),

  registerAttempt: () => set((state) => ({ attempts: state.attempts + 1 })),

  recordResult: (lastResult) =>
    set({ lastResult, phase: lastResult.passed ? 'scored' : 'failed' }),

  setLoadError: (loadError) =>
    set({ loadError, phase: loadError === null ? 'loading' : 'failed' }),

  endChapter: () =>
    set({ chapterId: null, levelIndex: 0, phase: 'concept', attempts: 0, startedAt: null }),
}));
