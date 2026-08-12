/**
 * Light/dark theme selection.
 *
 * The choice lives in `data-theme` on <html>, which is what the token blocks in
 * globals.css key off. It is written twice: once by a blocking script in <head>
 * so the very first paint is already correct, and again by the toggle.
 *
 * localStorage rather than the persisted progress store — that store hydrates
 * from IndexedDB asynchronously, which is a guaranteed flash of the wrong theme.
 */

export type Theme = 'dark' | 'light';

export const THEME_STORAGE_KEY = 'ai-lab-theme';

/** The instrument was designed on near-black; that stays the default. */
export const DEFAULT_THEME: Theme = 'dark';

export function isTheme(value: unknown): value is Theme {
  return value === 'dark' || value === 'light';
}

/** What the operating system asks for, when the player has not chosen. */
export function systemTheme(): Theme {
  if (typeof window === 'undefined' || !window.matchMedia) return DEFAULT_THEME;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/**
 * The player's explicit choice, or null if they have not made one.
 *
 * Reading localStorage throws in some privacy modes, and a theme preference is
 * never worth taking the app down for.
 */
export function readStoredTheme(): Theme | null {
  if (typeof window === 'undefined') return null;
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(stored) ? stored : null;
  } catch {
    return null;
  }
}

export function storeTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // A session-only theme is better than a crash.
  }
}

export function resolveInitialTheme(): Theme {
  return readStoredTheme() ?? systemTheme();
}

/** Reads back what the pre-paint script actually applied. */
export function currentTheme(): Theme {
  if (typeof document === 'undefined') return DEFAULT_THEME;
  const applied = document.documentElement.dataset.theme;
  return isTheme(applied) ? applied : resolveInitialTheme();
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

/**
 * Runs blocking in <head>, before any bundle exists.
 *
 * Deliberately dependency-free and wrapped in try/catch: if it throws, the page
 * renders unthemed, so it falls back to the dark default explicitly.
 */
export const THEME_INIT_SCRIPT = `(function(){try{var s=localStorage.getItem(${JSON.stringify(
  THEME_STORAGE_KEY
)});var t=(s==='light'||s==='dark')?s:(window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme',${JSON.stringify(
  DEFAULT_THEME
)});}})();`;
