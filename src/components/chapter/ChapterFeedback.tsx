'use client';

/**
 * Optional feedback, offered on the chapter-completion screen — the moment a
 * player is already pausing before moving on to the next chapter or back to
 * the map. Never blocks that navigation: the "next chapter" / "map" buttons
 * work regardless of whether this form is touched, sending, or has failed.
 */
import { useState } from 'react';
import { submitFeedback } from '@/lib/feedback';

const MESSAGE_MAX = 4000;

export function ChapterFeedback({
  chapterId,
  userId,
  displayName,
}: {
  chapterId: string;
  userId: string;
  displayName: string | null;
}) {
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = message.trim();
    if (trimmed.length === 0 || status === 'sending') return;

    setStatus('sending');
    const ok = await submitFeedback({
      userId,
      displayName: displayName ?? undefined,
      email: email.trim() || undefined,
      message: trimmed,
      chapterId,
    });
    setStatus(ok ? 'sent' : 'error');
  };

  if (status === 'sent') {
    return (
      <p className="mb-6 border-l-2 border-good/50 pl-4 text-xs leading-relaxed text-secondary">
        Thanks — we read every note.
      </p>
    );
  }

  return (
    <form onSubmit={submit} className="mb-6 flex flex-col gap-2.5 border-t border-line pt-4">
      <label className="flex flex-col gap-1.5">
        <span className="label">anything about this chapter or the project?</span>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          maxLength={MESSAGE_MAX}
          rows={3}
          placeholder="What's confusing, broken, or missing? (optional)"
          className="resize-none border border-line bg-inset px-3 py-2 text-xs leading-relaxed text-primary placeholder:text-muted focus:border-accent"
          style={{ borderRadius: 'var(--radius)' }}
        />
      </label>

      {message.trim().length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            maxLength={254}
            placeholder="email (optional, if you want a reply)"
            className="min-w-0 flex-1 border border-line bg-inset px-3 py-2 text-xs text-primary placeholder:text-muted focus:border-accent"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <button
            type="submit"
            disabled={status === 'sending'}
            className="inline-flex min-h-[36px] items-center justify-center border border-line bg-transparent px-3 font-mono text-xs uppercase tracking-[0.1em] text-primary transition-colors hover:border-line-strong hover:bg-raised disabled:cursor-not-allowed disabled:opacity-40"
            style={{ borderRadius: 'var(--radius)' }}
          >
            {status === 'sending' ? 'Sending…' : 'Send feedback'}
          </button>
        </div>
      )}

      {status === 'error' && (
        <p className="text-[11px] text-bad">Couldn&apos;t send that — check your connection and try again.</p>
      )}
    </form>
  );
}
