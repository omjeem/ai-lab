/**
 * Per-chapter/per-world analytics: how many distinct users started/completed
 * each chapter, and its activity breakdown (level pass/fail, jumped-ahead,
 * shared-link opens). One aggregation over the whole `activity` collection —
 * small enough at this app's scale that a single unfiltered pass is fine.
 */
import { NextResponse } from 'next/server';
import { requireAdmin, refreshSession } from '@/lib/requireAdmin';
import { activityCollection, isDatabaseConfigured } from '@/lib/mongodb';
import { shapeChapterAnalytics, type ChapterAnalyticsAggregateRow } from '@/lib/adminAnalytics';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface GroupedRow {
  _id: { chapterId: string; type: string };
  count: number;
  users: string[];
}

export async function GET(): Promise<Response> {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  if (!isDatabaseConfigured()) {
    return NextResponse.json({ ok: false, error: 'No database configured' }, { status: 503 });
  }

  try {
    const activity = await activityCollection();
    const grouped = await activity
      .aggregate<GroupedRow>([
        { $match: { chapterId: { $exists: true, $ne: null } } },
        {
          $group: {
            _id: { chapterId: '$chapterId', type: '$type' },
            count: { $sum: 1 },
            users: { $addToSet: '$userId' },
          },
        },
      ])
      .toArray();

    const rows: ChapterAnalyticsAggregateRow[] = grouped.map((row) => ({
      chapterId: row._id.chapterId,
      type: row._id.type,
      count: row.count,
      distinctUsers: row.users.length,
    }));

    return refreshSession(
      NextResponse.json({ ok: true, chapters: shapeChapterAnalytics(rows) }),
      auth.session
    );
  } catch {
    return NextResponse.json({ ok: false, error: 'Query failed' }, { status: 503 });
  }
}
