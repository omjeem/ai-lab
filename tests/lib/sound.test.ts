import { describe, it, expect, afterEach, vi } from 'vitest';

/**
 * `sound.ts` keeps one `AudioContext` in module state, so each test imports a
 * fresh copy of the module (`vi.resetModules`) rather than sharing state
 * across cases.
 */

class FakeOscillator {
  type = '';
  frequency = { value: 0 };
  connect = vi.fn();
  start = vi.fn();
  stop = vi.fn();
}

class FakeGainNode {
  gain = {
    setValueAtTime: vi.fn(),
    linearRampToValueAtTime: vi.fn(),
    exponentialRampToValueAtTime: vi.fn(),
  };
  connect = vi.fn();
}

class FakeAudioContext {
  currentTime = 0;
  state: 'running' | 'suspended' = 'running';
  destination = {};
  resume = vi.fn();
  createOscillator = vi.fn(() => new FakeOscillator());
  createGain = vi.fn(() => new FakeGainNode());
}

/** Stubs `window.AudioContext` with a tracked subclass and returns the instances it creates. */
function stubTrackedAudioContext(base: typeof FakeAudioContext = FakeAudioContext) {
  const created: FakeAudioContext[] = [];
  class TrackedContext extends base {
    constructor() {
      super();
      created.push(this);
    }
  }
  vi.stubGlobal('window', { AudioContext: TrackedContext });
  return created;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe('sound — no AudioContext available', () => {
  it('does not throw when the browser has no Web Audio support', async () => {
    vi.stubGlobal('window', {});
    const { playCorrectTone, playChapterCompleteTone } = await import('@/lib/sound');
    expect(() => playCorrectTone()).not.toThrow();
    expect(() => playChapterCompleteTone()).not.toThrow();
  });
});

describe('sound — playCorrectTone', () => {
  it('creates exactly two oscillators, each started and stopped once', async () => {
    const created = stubTrackedAudioContext();
    const { playCorrectTone } = await import('@/lib/sound');
    playCorrectTone();

    expect(created).toHaveLength(1);
    expect(created[0].createOscillator).toHaveBeenCalledTimes(2);
    const oscillators = created[0].createOscillator.mock.results.map((r) => r.value as FakeOscillator);
    for (const osc of oscillators) {
      expect(osc.start).toHaveBeenCalledTimes(1);
      expect(osc.stop).toHaveBeenCalledTimes(1);
    }
    // A rising blip: the second note is pitched above the first.
    expect(oscillators[1].frequency.value).toBeGreaterThan(oscillators[0].frequency.value);
  });

  it('reuses one AudioContext across repeated calls rather than creating a new one each time', async () => {
    const created = stubTrackedAudioContext();
    const { playCorrectTone } = await import('@/lib/sound');
    playCorrectTone();
    playCorrectTone();

    expect(created).toHaveLength(1);
    expect(created[0].createOscillator).toHaveBeenCalledTimes(4);
  });

  it('resumes a suspended context rather than leaving it silent', async () => {
    class SuspendedContext extends FakeAudioContext {
      override state: 'running' | 'suspended' = 'suspended';
    }
    const created = stubTrackedAudioContext(SuspendedContext);

    const { playCorrectTone } = await import('@/lib/sound');
    playCorrectTone();

    expect(created[0].resume).toHaveBeenCalledTimes(1);
  });
});

describe('sound — playChapterCompleteTone', () => {
  it('creates a three-note strictly ascending chime', async () => {
    const created = stubTrackedAudioContext();
    const { playChapterCompleteTone } = await import('@/lib/sound');
    playChapterCompleteTone();

    expect(created[0].createOscillator).toHaveBeenCalledTimes(3);
    const oscillators = created[0].createOscillator.mock.results.map((r) => r.value as FakeOscillator);
    expect(oscillators[0].frequency.value).toBeLessThan(oscillators[1].frequency.value);
    expect(oscillators[1].frequency.value).toBeLessThan(oscillators[2].frequency.value);
  });
});

describe('sound — envelope', () => {
  it('ramps gain up then down, never jumping straight to full volume', async () => {
    const created = stubTrackedAudioContext();
    const { playCorrectTone } = await import('@/lib/sound');
    playCorrectTone();

    const gains = created[0].createGain.mock.results.map((r) => r.value as FakeGainNode);
    expect(gains).toHaveLength(2);
    for (const gain of gains) {
      expect(gain.gain.setValueAtTime).toHaveBeenCalledWith(0, expect.any(Number));
      expect(gain.gain.linearRampToValueAtTime).toHaveBeenCalled();
      expect(gain.gain.exponentialRampToValueAtTime).toHaveBeenCalled();
    }
  });
});
