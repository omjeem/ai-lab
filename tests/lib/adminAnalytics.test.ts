import { describe, it, expect } from 'vitest';
import { shapeChapterAnalytics } from '@/lib/adminAnalytics';
import { orderedChapters } from '@/lib/curriculum';

describe('shapeChapterAnalytics', () => {
  it('returns one row per chapter in the curriculum, even with zero activity', () => {
    const rows = shapeChapterAnalytics([]);
    expect(rows).toHaveLength(orderedChapters().length);
    expect(rows.every((r) => r.startedUsers === 0 && r.completionRate === 0)).toBe(true);
  });

  it('joins real aggregation rows onto the right chapter', () => {
    const rows = shapeChapterAnalytics([
      { chapterId: '1-1-vectors', type: 'chapter_started', count: 10, distinctUsers: 8 },
      { chapterId: '1-1-vectors', type: 'chapter_completed', count: 6, distinctUsers: 4 },
      { chapterId: '1-1-vectors', type: 'level_completed', count: 20, distinctUsers: 5 },
      { chapterId: '1-1-vectors', type: 'level_failed', count: 3, distinctUsers: 3 },
      { chapterId: '1-1-vectors', type: 'chapter_jumped_ahead', count: 2, distinctUsers: 2 },
      { chapterId: '1-1-vectors', type: 'chapter_shared_link_opened', count: 5, distinctUsers: 5 },
    ]);

    const vectors = rows.find((r) => r.chapterId === '1-1-vectors')!;
    expect(vectors.chapterTitle).toBe('What is a Vector?');
    expect(vectors.world).toBe(1);
    expect(vectors.startedUsers).toBe(8);
    expect(vectors.completedUsers).toBe(4);
    expect(vectors.completionRate).toBeCloseTo(4 / 8);
    expect(vectors.levelCompleted).toBe(20);
    expect(vectors.levelFailed).toBe(3);
    expect(vectors.jumpedAhead).toBe(2);
    expect(vectors.sharedLinkOpened).toBe(5);

    // A chapter with no rows at all stays zeroed, not dropped.
    const untouched = rows.find((r) => r.chapterId === '1-2-vector-arithmetic')!;
    expect(untouched.startedUsers).toBe(0);
    expect(untouched.completionRate).toBe(0);
  });

  it('never divides by zero when nobody has started a chapter', () => {
    const rows = shapeChapterAnalytics([
      { chapterId: '1-1-vectors', type: 'chapter_completed', count: 1, distinctUsers: 1 },
    ]);
    const vectors = rows.find((r) => r.chapterId === '1-1-vectors')!;
    expect(vectors.completionRate).toBe(0);
  });
});
