import { z } from 'zod';

/**
 * Free-form product feedback, offered on the chapter-completion screen —
 * the moment a player is already pausing to move on to the next chapter or
 * back to the map. Kept as its own collection/route (mirroring `activity.ts`)
 * rather than folded into activity events, since this is player-authored
 * text an admin reads, not a passive telemetry point.
 */
export const feedbackRequestSchema = z.object({
  /** Client-generated, mirroring `activity.ts`'s `eventId` so a retried
   *  submission can be deduped rather than stored twice. */
  feedbackId: z.string().min(1).max(128),
  userId: z.string().min(1).max(128),
  /** The player's current display name, captured at submission time rather
   *  than joined from the users collection later — resilient to a deleted
   *  or renamed user, consistent with how activity events are self-contained. */
  displayName: z.string().max(80).optional(),
  /** Optional, so leaving it blank never blocks sending feedback. Empty
   *  string is normalised to "no email" rather than failing validation. */
  email: z.preprocess(
    (value) => (typeof value === 'string' && value.trim() === '' ? undefined : value),
    z.string().email().max(254).optional()
  ),
  message: z.string().trim().min(1).max(4000),
  chapterId: z.string().max(128).optional(),
  /** Client clock, in epoch milliseconds — matches `activity.ts`. */
  timestamp: z.number().int().positive(),
});
export type FeedbackRequest = z.infer<typeof feedbackRequestSchema>;

export interface FeedbackIngestResponse {
  ok: boolean;
  error?: string;
}

/** Shape the admin dashboard reads. */
export interface AdminFeedbackRow {
  feedbackId: string;
  userId: string;
  displayName: string | null;
  email: string | null;
  message: string;
  chapterId: string | null;
  chapterTitle: string | null;
  timestamp: string;
  receivedAt: string;
}
