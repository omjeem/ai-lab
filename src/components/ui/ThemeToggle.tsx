'use client';

/**
 * Light/dark switch for the status rail.
 *
 * The icon shows the theme you would switch *to*, which is the convention every
 * other app in the player's dock uses.
 */
import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import {
  applyTheme,
  currentTheme,
  readStoredTheme,
  storeTheme,
  systemTheme,
  type Theme,
} from '@/lib/theme';

export function ThemeToggle() {
  /**
   * Null until mounted. The server cannot know which theme the pre-paint script
   * chose, so rendering an icon during SSR would be a hydration mismatch — the
   * button holds its space and fills in on the client.
   */
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => setTheme(currentTheme()), []);

  // Follow the system while the player has not chosen for themselves.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const query = window.matchMedia('(prefers-color-scheme: light)');

    const onChange = () => {
      if (readStoredTheme() !== null) return;
      const next = systemTheme();
      applyTheme(next);
      setTheme(next);
    };

    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  const next: Theme = theme === 'light' ? 'dark' : 'light';

  return (
    <button
      type="button"
      onClick={() => {
        applyTheme(next);
        storeTheme(next);
        setTheme(next);
      }}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="p-1.5 text-muted transition-colors hover:text-primary"
    >
      {theme === null ? (
        <span className="block h-[14px] w-[14px]" aria-hidden />
      ) : theme === 'light' ? (
        <Moon size={14} strokeWidth={1.75} />
      ) : (
        <Sun size={14} strokeWidth={1.75} />
      )}
    </button>
  );
}
