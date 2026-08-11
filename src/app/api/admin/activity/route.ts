/**
 * One user's activity timeline, paginated newest first (Section 10).
 */
import { NextResponse } from 'next/server';
import { requireAdmin, refreshSession } from '@/lib/requireAdmin';
import { activityCollection, isDatabaseConfigured } from '@/lib/mongodb';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_PAGE_SIZE = 200;

export async function GET(request: Request): Promise<Response> {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  if (!isDatabaseConfigured()) {
    return NextResponse.json({ ok: false, error: 'No database configured' }, { status: 503 });
  }

  const url = new URL(request.url);
  const userId = url.searchParams.get('userId');
  if (!userId) {
    return NextResponse.json({ ok: false, error: 'userId is required' }, { status: 400 });
  }

  const page = Math.max(0, Number(url.searchParams.get('page') ?? '0') || 0);
  const size = Math.min(MAX_PAGE_SIZE, Math.max(1, Number(url.searchParams.get('size') ?? '50') || 50));

  try {
    const activity = await activityCollection();
    const [total, events] = await Promise.all([
      activity.countDocuments({ userId }),
      activity
        .find({ userId }, { projection: { _id: 0 } })
        .sort({ timestamp: -1 })
        .skip(page * size)
        .limit(size)
        .toArray(),
    ]);

    return refreshSession(
      NextResponse.json({ ok: true, total, page, size, events }),
      auth.session
    );
  } catch {
    return NextResponse.json({ ok: false, error: 'Query failed' }, { status: 503 });
  }
}
