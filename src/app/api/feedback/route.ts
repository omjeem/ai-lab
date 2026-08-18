/**
 * Feedback ingestion. Write-only, by design — same shape as `/api/activity`.
 *
 * Accepts one submission, validates it, and inserts it. There is no GET here;
 * feedback is only ever read back through `/api/admin/feedback`.
 */
import { NextResponse } from 'next/server';
import { feedbackRequestSchema, type FeedbackIngestResponse } from '@/types/feedback';
import { feedbackCollection, isDatabaseConfigured } from '@/lib/mongodb';
import { clientIpFrom, rateLimit, readLimit } from '@/lib/rateLimit';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request): Promise<Response> {
  const ip = clientIpFrom(request.headers) ?? 'unknown';
  const limit = rateLimit(`feedback:${ip}`, readLimit('RATE_LIMIT_FEEDBACK_PER_MIN', 10));

  if (!limit.allowed) {
    return NextResponse.json(
      { ok: false, error: 'Too many submissions' } satisfies FeedbackIngestResponse,
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid JSON' }, { status: 400 });
  }

  const parsed = feedbackRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ ok: false, error: 'Invalid feedback shape' }, { status: 400 });
  }

  // With no database configured the submission is acknowledged and dropped,
  // matching how /api/activity behaves with the same env unset.
  if (!isDatabaseConfigured()) {
    return NextResponse.json({ ok: true } satisfies FeedbackIngestResponse);
  }

  try {
    const collection = await feedbackCollection();
    await collection.insertOne({ ...parsed.data, receivedAt: new Date() });
    return NextResponse.json({ ok: true } satisfies FeedbackIngestResponse);
  } catch (error) {
    // A duplicate feedbackId means this exact submission is already stored —
    // treat a retried request as a success rather than an error.
    const isDuplicate = (error as { code?: number })?.code === 11000;
    if (isDuplicate) return NextResponse.json({ ok: true } satisfies FeedbackIngestResponse);
    return NextResponse.json({ ok: false, error: 'Could not store feedback' }, { status: 503 });
  }
}
