/**
 * Admin user list. Paginated — never an unbounded scan (Section 10).
 */
import { NextResponse } from 'next/server';
import { requireAdmin, refreshSession } from '@/lib/requireAdmin';
import { activityCollection, isDatabaseConfigured, usersCollection } from '@/lib/mongodb';
import type { AdminUserRow } from '@/types/user';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_PAGE_SIZE = 100;

export async function GET(request: Request): Promise<Response> {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  if (!isDatabaseConfigured()) {
    return NextResponse.json({ ok: false, error: 'No database configured' }, { status: 503 });
  }

  const url = new URL(request.url);
  const page = Math.max(0, Number(url.searchParams.get('page') ?? '0') || 0);
  const size = Math.min(MAX_PAGE_SIZE, Math.max(1, Number(url.searchParams.get('size') ?? '25') || 25));

  try {
    const users = await usersCollection();
    const activity = await activityCollection();

    const [total, documents] = await Promise.all([
      users.countDocuments({}),
      users
        .find({}, { projection: { _id: 0 } })
        .sort({ lastSeen: -1 })
        .skip(page * size)
        .limit(size)
        .toArray(),
    ]);

    // One grouped count for the page, rather than a query per row.
    const ids = documents.map((d) => d.userId);
    const counts = new Map<string, number>();
    if (ids.length > 0) {
      const grouped = await activity
        .aggregate<{ _id: string; count: number }>([
          { $match: { userId: { $in: ids } } },
          { $group: { _id: '$userId', count: { $sum: 1 } } },
        ])
        .toArray();
      for (const row of grouped) counts.set(row._id, row.count);
    }

    const rows: AdminUserRow[] = documents.map((doc) => ({
      userId: doc.userId,
      name: doc.name,
      firstSeen: new Date(doc.firstSeen).toISOString(),
      lastSeen: new Date(doc.lastSeen).toISOString(),
      country: doc.enrichment?.geo?.country ?? null,
      city: doc.enrichment?.geo?.city ?? null,
      ip: doc.enrichment?.ip ?? null,
      userAgent: doc.enrichment?.userAgent ?? null,
      eventCount: counts.get(doc.userId) ?? 0,
    }));

    return refreshSession(
      NextResponse.json({ ok: true, total, page, size, users: rows }),
      auth.session
    );
  } catch {
    return NextResponse.json({ ok: false, error: 'Query failed' }, { status: 503 });
  }
}

/**
 * Deletes a user and every activity event recorded for them.
 *
 * Two collections, no transaction: activity is removed first so a failure
 * partway through never leaves activity orphaned under a deleted user with
 * no way to find it again.
 */
export async function DELETE(request: Request): Promise<Response> {
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

  try {
    const activity = await activityCollection();
    const users = await usersCollection();

    const { deletedCount: deletedActivity } = await activity.deleteMany({ userId });
    const { deletedCount: deletedUsers } = await users.deleteOne({ userId });

    if (deletedUsers === 0) {
      return NextResponse.json({ ok: false, error: 'User not found' }, { status: 404 });
    }

    return refreshSession(
      NextResponse.json({ ok: true, deletedUsers, deletedActivity }),
      auth.session
    );
  } catch {
    return NextResponse.json({ ok: false, error: 'Delete failed' }, { status: 503 });
  }
}
