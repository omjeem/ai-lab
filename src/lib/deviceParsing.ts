/**
 * Server-side device/browser/OS parsing from the user-agent header.
 *
 * Uses Next.js's own `userAgentFromString` (re-exported from `next/server`,
 * backed by the bundled `ua-parser-js` — no extra dependency, no client-side
 * fingerprinting script).
 */
import { userAgentFromString } from 'next/server';
import type { RequestEnrichment } from '@/types/user';

export function parseDevice(userAgent: string | null): RequestEnrichment['device'] {
  if (!userAgent) return null;

  const { browser, os, device, isBot } = userAgentFromString(userAgent);
  return {
    browser: browser.name ?? null,
    browserVersion: browser.version ?? null,
    os: os.name ?? null,
    osVersion: os.version ?? null,
    // ua-parser-js leaves `device.type` undefined for a plain desktop browser.
    deviceType: device.type ?? 'desktop',
    isBot,
  };
}
