/**
 * Admin user list. Paginated — never an unbounded scan (Section 10).
 */
import { NextResponse } from 'next/server';
import { requireAdmin, refreshSession } from '@/lib/requireAdmin';
import { activityCollection, isDatabaseConfigured, usersCollection } from '@/lib/mongodb';
import { shapeAdminUserRow, type PageViewEdges } from '@/lib/adminUsers';
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
    const edges = new Map<string, PageViewEdges>();
    if (ids.length > 0) {
      const [countRows, edgeRows] = await Promise.all([
        activity
          .aggregate<{ _id: string; count: number }>([
            { $match: { userId: { $in: ids } } },
            { $group: { _id: '$userId', count: { $sum: 1 } } },
          ])
          .toArray(),
        // Earliest/latest page_viewed per user — landing page and last page.
        activity
          .aggregate<{ _id: string } & PageViewEdges>([
            { $match: { userId: { $in: ids }, type: 'page_viewed' } },
            { $sort: { timestamp: 1 } },
            {
              $group: {
                _id: '$userId',
                landingPage: { $first: '$detail.path' },
                landingTitle: { $first: '$detail.title' },
                lastPage: { $last: '$detail.path' },
              },
            },
          ])
          .toArray(),
      ]);
      for (const row of countRows) counts.set(row._id, row.count);
      for (const row of edgeRows) edges.set(row._id, row);
    }

    const rows: AdminUserRow[] = documents.map((doc) =>
      shapeAdminUserRow(doc, counts.get(doc.userId) ?? 0, edges.get(doc.userId))
    );

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
