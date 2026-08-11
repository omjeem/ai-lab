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
  geo: {
    country: string | null;
    region: string | null;
    city: string | null;
    /** Which lookup produced this, or null when enrichment is not configured. */
    source: string | null;
  } | null;
}

export interface UserDocument {
  userId: string;
  name: string;
  firstSeen: Date;
  lastSeen: Date;
  client: CreateUserRequest['client'];
  enrichment: RequestEnrichment;
}

/** Shape the admin dashboard reads. */
export interface AdminUserRow {
  userId: string;
  name: string;
  firstSeen: string;
  lastSeen: string;
  country: string | null;
  city: string | null;
  ip: string | null;
  userAgent: string | null;
  eventCount: number;
}
