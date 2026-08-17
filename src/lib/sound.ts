/**
 * Short synthesized feedback tones — Web Audio API oscillators, no sample
 * files, so the offline bundle gains zero bytes (Section 6, "sound design").
 *
 * Every export here is unconditional: it always plays if called. Callers are
 * responsible for checking `useGameProgressStore`'s `soundEnabled` first —
 * kept out of this module so it stays a plain, storeless audio primitive.
 */

let sharedContext: AudioContext | null = null;

/** Lazily creates one `AudioContext` for the whole session, resuming it if a
 *  browser has auto-suspended it before the first user gesture. */
function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const AudioContextCtor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextCtor) return null;

  if (!sharedContext) sharedContext = new AudioContextCtor();
  if (sharedContext.state === 'suspended') void sharedContext.resume();
  return sharedContext;
}

interface Note {
  freq: number;
  /** Seconds from the sequence's own start, not wall-clock time. */
  at: number;
  duration: number;
  gain?: number;
}

/** Plays a short sequence of sine-wave notes with a click-free envelope. */
function playNotes(notes: Note[]): void {
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  for (const { freq, at, duration, gain = 0.1 } of notes) {
    const oscillator = ctx.createOscillator();
    const envelope = ctx.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.value = freq;
    oscillator.connect(envelope);
    envelope.connect(ctx.destination);

    const start = now + at;
    const end = start + duration;
    envelope.gain.setValueAtTime(0, start);
    envelope.gain.linearRampToValueAtTime(gain, start + 0.012);
    envelope.gain.exponentialRampToValueAtTime(0.0001, end);

    oscillator.start(start);
    oscillator.stop(end + 0.02);
  }
}

/** A two-note rising blip for a passing level submission. */
export function playCorrectTone(): void {
  playNotes([
    { freq: 660, at: 0, duration: 0.09 },
    { freq: 880, at: 0.07, duration: 0.12 },
  ]);
}

/** A three-note ascending chime for finishing a chapter — still a blip, not a fanfare. */
export function playChapterCompleteTone(): void {
  playNotes([
    { freq: 523.25, at: 0, duration: 0.1 },
    { freq: 659.25, at: 0.09, duration: 0.1 },
    { freq: 783.99, at: 0.18, duration: 0.2, gain: 0.13 },
  ]);
}
