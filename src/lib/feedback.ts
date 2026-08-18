'use client';

/**
 * Client side of the chapter-completion feedback form.
 *
 * Mirrors `userIdentity.ts`'s `registerUser` shape: a best-effort POST that
 * returns a boolean rather than throwing, since a failed send should never
 * block the player from moving on to the next chapter.
 */
export function createFeedbackId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `fbk-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export interface SubmitFeedbackInput {
  userId: string;
  displayName?: string;
  email?: string;
  message: string;
  chapterId?: string;
}

export async function submitFeedback(input: SubmitFeedbackInput): Promise<boolean> {
  try {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        feedbackId: createFeedbackId(),
        userId: input.userId,
        displayName: input.displayName,
        email: input.email,
        message: input.message,
        chapterId: input.chapterId,
        timestamp: Date.now(),
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
