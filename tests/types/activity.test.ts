import { describe, it, expect } from 'vitest';
import { activityEventSchema, activityEventTypeSchema } from '@/types/activity';

describe('activityEventTypeSchema', () => {
  it('accepts the chapter-navigation event types added for the sharing/warning gate', () => {
    expect(activityEventTypeSchema.safeParse('chapter_shared_link_opened').success).toBe(true);
    expect(activityEventTypeSchema.safeParse('chapter_jumped_ahead').success).toBe(true);
  });

  it('still rejects an unknown event type', () => {
    expect(activityEventTypeSchema.safeParse('chapter_teleported').success).toBe(false);
  });
});

describe('activityEventSchema with the new types', () => {
  const base = {
    eventId: 'evt-1',
    userId: 'user-1',
    timestamp: Date.now(),
    chapterId: '7-1-retrieval',
  };

  it('accepts a shared-link-open event with its wasLocked detail flag', () => {
    const result = activityEventSchema.safeParse({
      ...base,
      type: 'chapter_shared_link_opened',
      detail: { wasLocked: 1 },
    });
    expect(result.success).toBe(true);
  });

  it('accepts a jumped-ahead event with no detail', () => {
    const result = activityEventSchema.safeParse({ ...base, type: 'chapter_jumped_ahead' });
    expect(result.success).toBe(true);
  });
});
