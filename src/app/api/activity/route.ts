/**
 * Activity ingestion. Write-only, by design (Section 9).
 *
 * Accepts a batch, validates it, inserts it, and answers with exactly which
 * event ids were persisted so the client clears only those from its queue.
 * It never returns anyone's data — this route is one-directional.
 */
import { NextResponse } from 'next/server';
import { activityBatchSchema, type ActivityIngestResponse } from '@/types/activity';
import { activityCollection, isDatabaseConfigured } from '@/lib/mongodb';
import { clientIpFrom, rateLimit, readLimit } from '@/lib/rateLimit';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** The sync manager probes with HEAD to tell "online" from "reachable". */
export async function HEAD(): Promise<Response> {
  return new Response(null, { status: 200 });
}

export async function POST(request: Request): Promise<Response> {
  const ip = clientIpFrom(request.headers) ?? 'unknown';
  const limit = rateLimit(`activity:${ip}`, readLimit('RATE_LIMIT_ACTIVITY_PER_MIN', 60));

  if (!limit.allowed) {
    return NextResponse.json(
      { ok: false, persistedEventIds: [] } satisfies ActivityIngestResponse,
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, persistedEventIds: [] }, { status: 400 });
  }

  const parsed = activityBatchSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, persistedEventIds: [], error: 'Invalid batch shape' },
      { status: 400 }
    );
  }

  // With no database configured the client would otherwise retry forever, so
  // the events are acknowledged and dropped rather than queued indefinitely.
  if (!isDatabaseConfigured()) {
    return NextResponse.json({
      ok: true,
      persistedEventIds: parsed.data.events.map((e) => e.eventId),
    } satisfies ActivityIngestResponse);
  }

  const receivedAt = new Date();
  const documents = parsed.data.events.map((event) => ({ ...event, receivedAt }));

  try {
    const collection = await activityCollection();
    // Unordered so one duplicate id does not abandon the rest of the batch.
    const result = await collection.insertMany(documents, { ordered: false });

    const insertedIndexes = new Set(Object.keys(result.insertedIds).map(Number));
    const persistedEventIds = documents
      .filter((_, index) => insertedIndexes.has(index))
      .map((doc) => doc.eventId);

    return NextResponse.json({ ok: true, persistedEventIds } satisfies ActivityIngestResponse);
  } catch (error) {
    // A duplicate-key error means those events are already stored, so they are
    // safe for the client to clear — resending them forever helps nobody.
    const duplicates = extractDuplicateIds(error, documents.map((d) => d.eventId));
    if (duplicates.length > 0) {
      return NextResponse.json({ ok: true, persistedEventIds: duplicates });
    }
    return NextResponse.json({ ok: false, persistedEventIds: [] }, { status: 503 });
  }
}

interface BulkWriteLike {
  code?: number;
  result?: { insertedIds?: { index: number }[] };
  writeErrors?: { index: number; code: number }[];
}

/** Ids that failed only because they were already inserted. */
function extractDuplicateIds(error: unknown, eventIds: string[]): string[] {
  const bulk = error as BulkWriteLike;
  if (bulk?.code !== 11000 && !bulk?.writeErrors) return [];

  const duplicateIndexes = new Set(
    (bulk.writeErrors ?? []).filter((e) => e.code === 11000).map((e) => e.index)
  );
  if (duplicateIndexes.size === 0) return [];

  return eventIds.filter((_, index) => duplicateIndexes.has(index));
}
