'use client';

/**
 * Connects the HUD's "Try again" to the canvas.
 *
 * `ChapterShell` owns that button and all it does is put the shell back into
 * `playing`. A canvas that has revealed its answers — or an engine whose board
 * has been played to a dead end, like a merge puzzle with every pair already
 * merged — has no way to know a retry was asked for, so the player lands on a
 * finished board with the answers still on it and nothing to do but hit the
 * canvas's own Reset. This closes that gap.
 *
 * Only the transition out of a finished run counts. `playing` is also the
 * ordinary state during a run, so firing on the value rather than the change
 * would reset the board mid-play.
 */
import { useEffect, useRef } from 'react';
import { useSessionStore } from '@/store/useSessionStore';

export function useRetrySignal(onRetry: () => void): void {
  const phase = useSessionStore((s) => s.phase);
  const previous = useRef(phase);

  useEffect(() => {
    const was = previous.current;
    previous.current = phase;
    if (phase === 'playing' && (was === 'scored' || was === 'failed')) onRetry();
  }, [phase, onRetry]);
}
