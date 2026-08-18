/**
 * Shapes raw per-chapter Mongo aggregation rows into the admin dashboard's
 * chapter analytics table. Kept as a pure function, separate from the
 * route/Mongo call, so it's testable without a live database — matching
 * this repo's convention that only pure/shaping logic gets unit tests, not
 * routes that need a live connection.
 */
import { orderedChapters, worldOfChapter } from '@/lib/curriculum';

/** One `{chapterId, type}` group from the activity collection. */
export interface ChapterAnalyticsAggregateRow {
  chapterId: string;
  type: string;
  /** Total events in this group. */
  count: number;
  /** Distinct `userId`s behind those events. */
  distinctUsers: number;
}

export interface ChapterAnalyticsRow {
  chapterId: string;
  chapterTitle: string;
  world: number;
  worldTitle: string;
  /** Distinct users who ever started this chapter. */
  startedUsers: number;
  /** Distinct users who ever completed this chapter. */
  completedUsers: number;
  /** `completedUsers / startedUsers`, 0 when nobody has started it yet. */
  completionRate: number;
  levelCompleted: number;
  levelFailed: number;
  /** Times a visitor chose "Play anyway" past the prerequisite warning. */
  jumpedAhead: number;
  /** Times this chapter was opened via a `?via=share` link. */
  sharedLinkOpened: number;
}

/** One row per chapter in the curriculum, in play order — present even for a chapter with zero activity. */
export function shapeChapterAnalytics(rows: ChapterAnalyticsAggregateRow[]): ChapterAnalyticsRow[] {
  const byChapter = new Map<string, Map<string, ChapterAnalyticsAggregateRow>>();
  for (const row of rows) {
    if (!byChapter.has(row.chapterId)) byChapter.set(row.chapterId, new Map());
    byChapter.get(row.chapterId)!.set(row.type, row);
  }

  return orderedChapters().map((chapter) => {
    const world = worldOfChapter(chapter.id);
    const forChapter = byChapter.get(chapter.id);
    const get = (type: string) => forChapter?.get(type);

    const startedUsers = get('chapter_started')?.distinctUsers ?? 0;
    const completedUsers = get('chapter_completed')?.distinctUsers ?? 0;

    return {
      chapterId: chapter.id,
      chapterTitle: chapter.title,
      world: world?.world ?? 0,
      worldTitle: world?.title ?? '',
      startedUsers,
      completedUsers,
      completionRate: startedUsers > 0 ? completedUsers / startedUsers : 0,
      levelCompleted: get('level_completed')?.count ?? 0,
      levelFailed: get('level_failed')?.count ?? 0,
      jumpedAhead: get('chapter_jumped_ahead')?.count ?? 0,
      sharedLinkOpened: get('chapter_shared_link_opened')?.count ?? 0,
    };
  });
}
