import { z } from 'zod';

/** Every meaningful in-game event, per Section 9. */
export const activityEventTypeSchema = z.enum([
  'chapter_started',
  'chapter_completed',
  'chapter_unlocked',
  'level_started',
  'level_completed',
  'level_failed',
  'xp_earned',
  'model_download_started',
  'model_download_completed',
  'model_download_failed',
  'cloud_inference_used',
  'onboarding_completed',
  /** Fired when a chapter route loads with `?via=share`, regardless of lock state. */
  'chapter_shared_link_opened',
  /** Fired only when the prerequisite warning dialog was shown and the player chose to proceed. */
  'chapter_jumped_ahead',
  /** Fired on every route change, once onboarded — the sequential path through the app. */
  'page_viewed',
  /** The builder-signature link on the chapter-complete screen was clicked. */
  'portfolio_link_clicked',
  /** The "book 15 min" link on the chapter-complete screen was clicked. */
  'meeting_link_clicked',
]);
export type ActivityEventType = z.infer<typeof activityEventTypeSchema>;

export const activityEventSchema = z.object({
  /** Client-generated, and the key the server acknowledges back. */
  eventId: z.string().min(1).max(128),
  userId: z.string().min(1).max(128),
  type: activityEventTypeSchema,
  /** Client clock, in epoch milliseconds. */
  timestamp: z.number().int().positive(),
  chapterId: z.string().max(128).optional(),
  levelId: z.string().max(128).optional(),
  /** Free-form measurements attached to the event. */
  detail: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional(),
});
export type ActivityEvent = z.infer<typeof activityEventSchema>;

/** Batch shape the sync manager posts. Capped so one request cannot be huge. */
export const activityBatchSchema = z.object({
  events: z.array(activityEventSchema).min(1).max(200),
});
export type ActivityBatch = z.infer<typeof activityBatchSchema>;

export interface ActivityIngestResponse {
  ok: boolean;
  /** Exactly which events were persisted, so the client clears only those. */
  persistedEventIds: string[];
}
