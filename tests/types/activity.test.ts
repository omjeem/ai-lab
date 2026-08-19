import { describe, it, expect } from 'vitest';
import { activityEventSchema, activityEventTypeSchema } from '@/types/activity';

describe('activityEventTypeSchema', () => {
  it('accepts the chapter-navigation event types added for the sharing/warning gate', () => {
    expect(activityEventTypeSchema.safeParse('chapter_shared_link_opened').success).toBe(true);
    expect(activityEventTypeSchema.safeParse('chapter_jumped_ahead').success).toBe(true);
  });

  it('accepts page_viewed, fired on every route change once onboarded', () => {
    expect(activityEventTypeSchema.safeParse('page_viewed').success).toBe(true);
  });

  it('accepts the chapter-complete builder-links events', () => {
    expect(activityEventTypeSchema.safeParse('portfolio_link_clicked').success).toBe(true);
    expect(activityEventTypeSchema.safeParse('meeting_link_clicked').success).toBe(true);
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

  it('accepts a page_viewed event with its path detail, no chapterId required', () => {
    const result = activityEventSchema.safeParse({
      eventId: 'evt-2',
      userId: 'user-1',
      timestamp: Date.now(),
      type: 'page_viewed',
      detail: { path: '/map' },
    });
    expect(result.success).toBe(true);
  });

  it('accepts portfolio/meeting link clicks scoped to the chapter they were clicked from', () => {
    expect(activityEventSchema.safeParse({ ...base, type: 'portfolio_link_clicked' }).success).toBe(
      true
    );
    expect(activityEventSchema.safeParse({ ...base, type: 'meeting_link_clicked' }).success).toBe(
      true
    );
  });
});
