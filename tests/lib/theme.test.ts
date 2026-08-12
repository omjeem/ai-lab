import { describe, it, expect, afterEach, vi } from 'vitest';
import {
  DEFAULT_THEME,
  THEME_INIT_SCRIPT,
  THEME_STORAGE_KEY,
  isTheme,
  readStoredTheme,
  resolveInitialTheme,
  storeTheme,
  systemTheme,
} from '@/lib/theme';

afterEach(() => vi.unstubAllGlobals());

/** A window with a storage that can be told to misbehave, as private modes do. */
function stubWindow({
  stored,
  prefersLight = false,
  throwOnStorage = false,
  matchMedia = true,
}: {
  stored?: string | null;
  prefersLight?: boolean;
  throwOnStorage?: boolean;
  matchMedia?: boolean;
} = {}) {
  const setItem = vi.fn();
  vi.stubGlobal('window', {
    localStorage: {
      getItem: () => {
        if (throwOnStorage) throw new Error('denied');
        return stored ?? null;
      },
      setItem: (key: string, value: string) => {
        if (throwOnStorage) throw new Error('denied');
        setItem(key, value);
      },
    },
    matchMedia: matchMedia ? () => ({ matches: prefersLight }) : undefined,
  });
  return { setItem };
}

describe('theme — isTheme', () => {
  it('accepts only the two themes', () => {
    expect(isTheme('dark')).toBe(true);
    expect(isTheme('light')).toBe(true);
    expect(isTheme('solarized')).toBe(false);
    expect(isTheme(null)).toBe(false);
    expect(isTheme(undefined)).toBe(false);
  });
});

describe('theme — systemTheme', () => {
  it('follows the OS preference', () => {
    stubWindow({ prefersLight: true });
    expect(systemTheme()).toBe('light');

    stubWindow({ prefersLight: false });
    expect(systemTheme()).toBe('dark');
  });

  it('falls back to the default when matchMedia is missing', () => {
    stubWindow({ matchMedia: false });
    expect(systemTheme()).toBe(DEFAULT_THEME);
  });
});

describe('theme — readStoredTheme', () => {
  it('returns a stored choice', () => {
    stubWindow({ stored: 'light' });
    expect(readStoredTheme()).toBe('light');
  });

  it('returns null when nothing is stored', () => {
    stubWindow({ stored: null });
    expect(readStoredTheme()).toBeNull();
  });

  it('ignores a value that is not a theme', () => {
    stubWindow({ stored: 'chartreuse' });
    expect(readStoredTheme()).toBeNull();
  });

  it('survives storage being blocked rather than throwing', () => {
    stubWindow({ throwOnStorage: true });
    expect(readStoredTheme()).toBeNull();
  });
});

describe('theme — storeTheme', () => {
  it('writes under the shared key, so the inline script reads the same value', () => {
    const { setItem } = stubWindow();
    storeTheme('light');
    expect(setItem).toHaveBeenCalledWith(THEME_STORAGE_KEY, 'light');
  });

  it('swallows a blocked write — a session-only theme beats a crash', () => {
    stubWindow({ throwOnStorage: true });
    expect(() => storeTheme('dark')).not.toThrow();
  });
});

describe('theme — resolveInitialTheme', () => {
  it('prefers the explicit choice over the system', () => {
    stubWindow({ stored: 'dark', prefersLight: true });
    expect(resolveInitialTheme()).toBe('dark');
  });

  it('falls back to the system when no choice was made', () => {
    stubWindow({ stored: null, prefersLight: true });
    expect(resolveInitialTheme()).toBe('light');
  });
});

describe('theme — THEME_INIT_SCRIPT', () => {
  it('reads the same storage key the toggle writes', () => {
    expect(THEME_INIT_SCRIPT).toContain(JSON.stringify(THEME_STORAGE_KEY));
  });

  it('is wrapped so a storage failure cannot break the page', () => {
    expect(THEME_INIT_SCRIPT).toContain('try{');
    expect(THEME_INIT_SCRIPT).toContain('catch');
  });

  it('applies the same theme the module would, for every starting state', () => {
    const run = (stored: string | null, prefersLight: boolean) => {
      const html: Record<string, string> = {};
      const fakeWindow = {
        localStorage: { getItem: () => stored },
        matchMedia: () => ({ matches: prefersLight }),
      };
      const fakeDocument = {
        documentElement: {
          setAttribute: (name: string, value: string) => {
            html[name] = value;
          },
        },
      };
      new Function('window', 'document', 'localStorage', THEME_INIT_SCRIPT)(
        fakeWindow,
        fakeDocument,
        fakeWindow.localStorage
      );
      return html['data-theme'];
    };

    for (const [stored, prefersLight, expected] of [
      ['light', false, 'light'],
      ['dark', true, 'dark'],
      [null, true, 'light'],
      [null, false, 'dark'],
      ['nonsense', true, 'light'],
    ] as const) {
      expect(run(stored, prefersLight)).toBe(expected);

      // The module has to agree, or the first paint and the toggle disagree.
      stubWindow({ stored, prefersLight });
      expect(resolveInitialTheme()).toBe(expected);
    }
  });
});
