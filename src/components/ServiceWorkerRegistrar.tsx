'use client';

import { useEffect } from 'react';

/**
 * Registers the offline shell worker.
 *
 * Only in production: in development the worker would serve stale bundles and
 * make every change look like it did not apply.
 */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (process.env.NODE_ENV !== 'production') return;
    if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return;

    const register = () => {
      void navigator.serviceWorker.register('/sw.js').catch(() => {
        // No offline shell is a degraded experience, not a broken one.
      });
    };

    if (document.readyState === 'complete') register();
    else window.addEventListener('load', register, { once: true });
  }, []);

  return null;
}
