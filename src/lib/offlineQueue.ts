'use client';

/**
 * The local activity queue.
 *
 * Every meaningful event is written here first and synced later, so play works
 * with no connection at all. Events are keyed by a client-generated id, which is
 * what lets the sync manager clear exactly what the server confirmed and nothing
 * else (Section 9).
 */
import { getDb, ACTIVITY_STORE } from '@/models/modelCache';
import type { ActivityEvent, ActivityEventType, ActivityIngestResponse } from '@/types/activity';

/** Hard cap so a long offline stretch cannot fill the user's storage. */
const MAX_QUEUE = 5000;

export function createEventId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `evt-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export interface EnqueueInput {
  userId: string;
  type: ActivityEventType;
  chapterId?: string;
  levelId?: string;
  detail?: Record<string, string | number | boolean>;
}

export async function enqueueActivity(input: EnqueueInput): Promise<ActivityEvent | null> {
  const event: ActivityEvent = {
    eventId: createEventId(),
    userId: input.userId,
    type: input.type,
    timestamp: Date.now(),
    ...(input.chapterId ? { chapterId: input.chapterId } : {}),
    ...(input.levelId ? { levelId: input.levelId } : {}),
    ...(input.detail ? { detail: input.detail } : {}),
  };

  try {
    const db = await getDb();
    const count = await db.count(ACTIVITY_STORE);
    if (count >= MAX_QUEUE) {
      // Drop the oldest rather than the newest — recent play is more useful.
      const oldest = await db.getAllFromIndex(ACTIVITY_STORE, 'byTimestamp', undefined, 1);
      if (oldest[0]) await db.delete(ACTIVITY_STORE, (oldest[0] as ActivityEvent).eventId);
    }
    await db.put(ACTIVITY_STORE, event);
  } catch {
    // Telemetry must never break gameplay.
    return null;
  }

  // Try to deliver right away instead of waiting for the sync manager's next
  // tick. If this fails (offline, server error), the event just stays queued
  // and the existing interval/online/visibility sync picks it up later — no
  // retry logic needed here.
  void sendImmediately(event);
  return event;
}

async function sendImmediately(event: ActivityEvent): Promise<void> {
  try {
    const response = await fetch('/api/activity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: [event] }),
    });
    if (!response.ok) return;

    const payload = (await response.json()) as ActivityIngestResponse;
    if (payload.ok && payload.persistedEventIds.includes(event.eventId)) {
      await clearSynced([event.eventId]);
    }
  } catch {
    // No connection — leave it queued.
  }
}

export async function readQueue(limit = 200): Promise<ActivityEvent[]> {
  try {
    const db = await getDb();
    const all = (await db.getAllFromIndex(ACTIVITY_STORE, 'byTimestamp')) as ActivityEvent[];
    return all.slice(0, limit);
  } catch {
    return [];
  }
}

export async function queueSize(): Promise<number> {
  try {
    const db = await getDb();
    return await db.count(ACTIVITY_STORE);
  } catch {
    return 0;
  }
}

/**
 * Removes exactly the events the server confirmed.
 *
 * Never a blanket clear: events added while a sync was in flight must survive,
 * which is precisely the case Section 9 calls out.
 */
export async function clearSynced(eventIds: readonly string[]): Promise<number> {
  if (eventIds.length === 0) return 0;
  try {
    const db = await getDb();
    const tx = db.transaction(ACTIVITY_STORE, 'readwrite');
    await Promise.all(eventIds.map((id) => tx.store.delete(id)));
    await tx.done;
    return eventIds.length;
  } catch {
    return 0;
  }
}

/** Test and debug seam — drops everything queued. */
export async function clearQueue(): Promise<void> {
  try {
    const db = await getDb();
    await db.clear(ACTIVITY_STORE);
  } catch {
    // Nothing to clear if the database was never reachable.
  }
}
