/**
 * User creation at onboarding (Section 8).
 *
 * The client supplies only an id and a display name; everything else is
 * captured here from the request — IP with proxy headers respected, user agent,
 * referrer, language — plus optional geolocation when the operator has
 * configured a lookup of their own.
 */
import { NextResponse } from 'next/server';
import { createUserRequestSchema, type RequestEnrichment, type UserDocument } from '@/types/user';
import { isDatabaseConfigured, usersCollection } from '@/lib/mongodb';
import { clientIpFrom, rateLimit, readLimit } from '@/lib/rateLimit';
import { lookupGeo } from '@/lib/geoLookup';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request): Promise<Response> {
  const ip = clientIpFrom(request.headers);
  const limit = rateLimit(`users:${ip ?? 'unknown'}`, readLimit('RATE_LIMIT_ACTIVITY_PER_MIN', 60));
  if (!limit.allowed) {
    return NextResponse.json(
      { ok: false },
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid JSON' }, { status: 400 });
  }

  const parsed = createUserRequestSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ ok: false, error: 'Invalid request shape' }, { status: 400 });
  }

  // Onboarding must succeed offline-first; without a database the id is still
  // valid locally and the profile is created on a later launch.
  if (!isDatabaseConfigured()) {
    return NextResponse.json({ ok: true, stored: false });
  }

  const enrichment: RequestEnrichment = {
    ip,
    userAgent: request.headers.get('user-agent'),
    referrer: request.headers.get('referer'),
    acceptLanguage: request.headers.get('accept-language'),
    geo: await lookupGeo(ip),
  };

  const now = new Date();
  const { userId, name, client } = parsed.data;

  try {
    const users = await usersCollection();
    await users.updateOne(
      { userId },
      {
        // A returning player refreshes their context without losing firstSeen.
        $set: { name, lastSeen: now, client, enrichment } satisfies Partial<UserDocument>,
        $setOnInsert: { userId, firstSeen: now },
      },
      { upsert: true }
    );
    return NextResponse.json({ ok: true, stored: true });
  } catch {
    return NextResponse.json({ ok: false, error: 'Could not store the profile' }, { status: 503 });
  }
}
