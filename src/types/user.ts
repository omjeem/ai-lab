import { z } from 'zod';

/** What the client sends at onboarding. Everything else is captured server-side. */
export const createUserRequestSchema = z.object({
  userId: z.string().min(8).max(128),
  name: z.string().min(1).max(80),
  /** Client-observable context the server cannot derive from headers. */
  client: z
    .object({
      timezoneOffsetMinutes: z.number().int().min(-900).max(900).optional(),
      timezone: z.string().max(64).optional(),
      screenWidth: z.number().int().positive().max(20000).optional(),
      screenHeight: z.number().int().positive().max(20000).optional(),
      viewportWidth: z.number().int().positive().max(20000).optional(),
      viewportHeight: z.number().int().positive().max(20000).optional(),
      devicePixelRatio: z.number().positive().max(10).optional(),
      language: z.string().max(35).optional(),
      /** `navigator.languages`, most-preferred first. */
      languages: z.array(z.string().max(35)).max(10).optional(),
      platform: z.string().max(64).optional(),
      touchSupport: z.boolean().optional(),
      /** `navigator.deviceMemory` — Chromium-only, absent everywhere else. */
      deviceMemoryGb: z.number().positive().max(1024).optional(),
      cpuCores: z.number().int().positive().max(256).optional(),
      /** `navigator.connection.effectiveType` — Chromium-only. */
      networkType: z.string().max(20).optional(),
      colorScheme: z.enum(['light', 'dark']).optional(),
    })
    .optional(),
});
export type CreateUserRequest = z.infer<typeof createUserRequestSchema>;

/** Enrichment derived server-side from the request. */
export interface RequestEnrichment {
  ip: string | null;
  userAgent: string | null;
  referrer: string | null;
  acceptLanguage: string | null;
  /** The `Host` header — which domain the request came in on. */
  host: string | null;
  geo: {
    country: string | null;
    region: string | null;
    city: string | null;
    /** Which lookup produced this, or null when enrichment is not configured. */
    source: string | null;
    /** Only present from the free Vercel edge headers, not the paid lookup. */
    latitude?: string | null;
    longitude?: string | null;
    timezone?: string | null;
  } | null;
  /** Parsed server-side from the user-agent header — no client fingerprinting JS involved. */
  device: {
    browser: string | null;
    browserVersion: string | null;
    os: string | null;
    osVersion: string | null;
    deviceType: string | null;
    isBot: boolean;
  } | null;
}

export interface UserDocument {
  userId: string;
  name: string;
  firstSeen: Date;
  lastSeen: Date;
  /** Referrer and derived source at first registration only — never overwritten later. */
  firstReferrer: string | null;
  firstSource: string | null;
  client: CreateUserRequest['client'];
  enrichment: RequestEnrichment;
}

/** Shape the admin dashboard reads — one row per user, plus their journey's edges. */
export interface AdminUserRow {
  userId: string;
  name: string;
  firstSeen: string;
  lastSeen: string;
  country: string | null;
  city: string | null;
  ip: string | null;
  userAgent: string | null;
  browser: string | null;
  browserVersion: string | null;
  os: string | null;
  osVersion: string | null;
  deviceType: string | null;
  eventCount: number;
  latitude: string | null;
  longitude: string | null;
  geoTimezone: string | null;
  host: string | null;
  source: string | null;
  /** Path and title of the earliest `page_viewed` event on record for this user. */
  landingPage: string | null;
  landingTitle: string | null;
  /** Path of the most recent `page_viewed` event on record. */
  lastPage: string | null;
  platform: string | null;
  touchSupport: boolean | null;
  deviceMemoryGb: number | null;
  cpuCores: number | null;
  networkType: string | null;
  colorScheme: string | null;
  /** `navigator.languages`, joined for display. */
  languages: string | null;
  acceptLanguage: string | null;
  timezone: string | null;
  screen: string | null;
  viewport: string | null;
  pixelRatio: number | null;
}
