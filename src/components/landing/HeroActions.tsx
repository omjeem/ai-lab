'use client';

/**
 * The landing page's primary call to action.
 *
 * Progress lives in IndexedDB (Section 7), which hydrates asynchronously, so
 * this starts in the "new visitor" state and swaps in place once the store
 * confirms otherwise. A returning player is sent straight to the map — this
 * page is the pitch, not a lobby they should have to click through again.
 */
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowRight } from 'lucide-react';
import { useGameProgressStore } from '@/store/useGameProgressStore';
import { Button } from '@/components/ui';

export function HeroActions() {
  const router = useRouter();
  const hydrated = useGameProgressStore((s) => s.hydrated);
  const userId = useGameProgressStore((s) => s.userId);
  const displayName = useGameProgressStore((s) => s.displayName);
  const xp = useGameProgressStore((s) => s.xp);

  const returning = hydrated && userId !== null;

  useEffect(() => {
    if (returning) router.replace('/map');
  }, [returning, router]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          onClick={() => router.push('/map')}
          className="px-5"
        >
          {returning ? `Continue as ${displayName ?? 'you'}` : 'Start learning — free, no signup'}
          <ArrowRight size={13} strokeWidth={2} />
        </Button>
        <a
          href="#architecture"
          className="font-mono text-xs uppercase tracking-[0.1em] text-muted transition-colors hover:text-primary"
        >
          See how it runs ↓
        </a>
      </div>
      {returning && (
        <span className="label">
          {xp} xp on this device — taking you back to the map
        </span>
      )}
    </div>
  );
}
