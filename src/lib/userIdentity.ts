'use client';

/**
 * Client identity and the context the onboarding POST carries.
 *
 * The id is generated here and never issued by the server, so a player exists
 * locally the moment they type a name — including with no connection at all.
 */
import type { CreateUserRequest } from '@/types/user';

export function generateUserId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `usr-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
}

/** Chromium-only Navigator extras with no standard TS DOM typing. */
interface NavigatorWithExtras extends Navigator {
  deviceMemory?: number;
  connection?: { effectiveType?: string };
}

/**
 * Client-observable context the server cannot read from headers.
 *
 * Deliberately limited to what the onboarding disclosure covers: screen,
 * viewport, and device signals ordinary analytics collect — timezone,
 * language(s), platform, touch/memory/CPU/network/color-scheme. No canvas or
 * font fingerprinting. Every read stands alone so one unsupported API (most of
 * the Chromium-only ones below are absent in Firefox/Safari) never drops the
 * rest of the object.
 */
export function collectClientContext(): CreateUserRequest['client'] {
  if (typeof window === 'undefined') return undefined;

  const nav = navigator as NavigatorWithExtras;

  return {
    timezoneOffsetMinutes: -new Date().getTimezoneOffset(),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    screenWidth: window.screen?.width,
    screenHeight: window.screen?.height,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio,
    language: navigator.language,
    languages: navigator.languages ? Array.from(navigator.languages) : undefined,
    platform: navigator.platform || undefined,
    touchSupport: navigator.maxTouchPoints > 0,
    deviceMemoryGb: nav.deviceMemory,
    cpuCores: navigator.hardwareConcurrency,
    networkType: nav.connection?.effectiveType,
    colorScheme: window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  };
}

/**
 * Registers the player with the backend.
 *
 * Failure is not fatal: the id already exists locally, so play continues and the
 * profile is created on a later launch.
 */
export async function registerUser(userId: string, name: string): Promise<boolean> {
  try {
    const response = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId, name, client: collectClientContext() }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
