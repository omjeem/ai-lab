/**
 * Admin feedback list, paginated newest first — same shape as
 * `/api/admin/users` and `/api/admin/activity`.
 */
import { NextResponse } from 'next/server';
import { requireAdmin, refreshSession } from '@/lib/requireAdmin';
import { feedbackCollection, isDatabaseConfigured } from '@/lib/mongodb';
import { getManifestChapter } from '@/lib/curriculum';
import type { AdminFeedbackRow } from '@/types/feedback';

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
  const size = Math.min(MAX_PAGE_SIZE, Math.max(1, Number(url.searchParams.get('size') ?? '50') || 50));

  try {
    const feedback = await feedbackCollection();
    const [total, documents] = await Promise.all([
      feedback.countDocuments({}),
      feedback
        .find({}, { projection: { _id: 0 } })
        .sort({ receivedAt: -1 })
        .skip(page * size)
        .limit(size)
        .toArray(),
    ]);

    const rows: AdminFeedbackRow[] = documents.map((doc) => ({
      feedbackId: doc.feedbackId,
      userId: doc.userId,
      displayName: doc.displayName ?? null,
      email: doc.email ?? null,
      message: doc.message,
      chapterId: doc.chapterId ?? null,
      chapterTitle: doc.chapterId ? getManifestChapter(doc.chapterId)?.title ?? null : null,
      timestamp: new Date(doc.timestamp).toISOString(),
      receivedAt: doc.receivedAt.toISOString(),
    }));

    return refreshSession(
      NextResponse.json({ ok: true, total, page, size, feedback: rows }),
      auth.session
    );
  } catch {
    return NextResponse.json({ ok: false, error: 'Query failed' }, { status: 503 });
  }
}
