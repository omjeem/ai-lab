'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowRight } from 'lucide-react';
import { generateUserId, registerUser } from '@/lib/userIdentity';
import { enqueueActivity } from '@/lib/offlineQueue';
import { useGameProgressStore } from '@/store/useGameProgressStore';
import { Button, Heading } from '@/components/ui';

export function Onboarding() {
  const router = useRouter();
  const setIdentity = useGameProgressStore((s) => s.setIdentity);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (trimmed.length === 0 || busy) return;

    setBusy(true);
    const userId = generateUserId();
    setIdentity(userId, trimmed);

    void enqueueActivity({ userId, type: 'onboarding_completed' });
    // Registration is best-effort — the id is already valid locally, so a
    // failed call must not block someone from starting.
    void registerUser(userId, trimmed);

    router.push('/map');
  };

  return (
    <div className="grid-field flex flex-1 items-center justify-center px-6 py-16">
      <div className="w-full max-w-md">
        <p className="label mb-3">ai learning lab</p>

        <Heading level={1} className="mb-4 text-balance">
          Learn how models work by running them.
        </Heading>

        <p className="mb-8 text-sm leading-relaxed text-secondary">
          Real embeddings, real attention weights, real gradients — computed in your browser, not
          described to you. No chatbot, no explanations from a model. You will be manipulating the
          internals directly.
        </p>

        <form onSubmit={submit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-2">
            <span className="label">what should we call you</span>
            <input
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={80}
              autoFocus
              autoComplete="nickname"
              placeholder="display name"
              className="border border-line bg-inset px-3 py-3 font-mono text-base text-primary placeholder:text-muted focus:border-accent"
              style={{ borderRadius: 'var(--radius)' }}
            />
          </label>

          <Button type="submit" variant="primary" disabled={name.trim().length === 0 || busy}>
            Start
            <ArrowRight size={13} strokeWidth={2} />
          </Button>
        </form>

        {/*
          Section 8 / 11.1 — one non-blocking line. Collecting IP and device
          details with no disclosure at all is the part of the spec most likely
          to cause real trouble later, so it is stated plainly and briefly.
        */}
        <p className="mt-6 text-[11px] leading-relaxed text-muted">
          We track anonymous play activity, including approximate location and device details, to
          improve this tool.
        </p>
      </div>
    </div>
  );
}
