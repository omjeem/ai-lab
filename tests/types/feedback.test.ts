import { describe, it, expect } from 'vitest';
import { feedbackRequestSchema } from '@/types/feedback';

describe('feedbackRequestSchema', () => {
  const base = {
    feedbackId: 'fbk-1',
    userId: 'user-1',
    message: 'The gradient descent chapter could use a bigger canvas.',
    timestamp: Date.now(),
  };

  it('accepts the minimal shape with no email, displayName or chapterId', () => {
    expect(feedbackRequestSchema.safeParse(base).success).toBe(true);
  });

  it('accepts a fully populated submission', () => {
    const result = feedbackRequestSchema.safeParse({
      ...base,
      displayName: 'Ada',
      email: 'ada@example.com',
      chapterId: '3-1-neuron-tuning',
    });
    expect(result.success).toBe(true);
  });

  it('treats an empty-string email as "no email" rather than rejecting it', () => {
    const result = feedbackRequestSchema.safeParse({ ...base, email: '' });
    expect(result.success).toBe(true);
    if (result.success) expect(result.data.email).toBeUndefined();
  });

  it('rejects a malformed email', () => {
    expect(feedbackRequestSchema.safeParse({ ...base, email: 'not-an-email' }).success).toBe(false);
  });

  it('rejects an empty message', () => {
    expect(feedbackRequestSchema.safeParse({ ...base, message: '   ' }).success).toBe(false);
  });

  it('rejects a message over the length cap', () => {
    expect(
      feedbackRequestSchema.safeParse({ ...base, message: 'x'.repeat(4001) }).success
    ).toBe(false);
  });

  it('rejects a missing feedbackId or userId', () => {
    const rest: Record<string, unknown> = { ...base };
    delete rest.feedbackId;
    expect(feedbackRequestSchema.safeParse(rest).success).toBe(false);
  });
});
