'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { WorldMap } from '@/components/map/WorldMap';
import { useGameProgressStore } from '@/store/useGameProgressStore';

export default function MapPage() {
  const router = useRouter();
  const hydrated = useGameProgressStore((s) => s.hydrated);
  const userId = useGameProgressStore((s) => s.userId);
  const xp = useGameProgressStore((s) => s.xp);
  const chapters = useGameProgressStore((s) => s.chapters);

  // Only redirect once progress has actually loaded, or a returning player
  // would be bounced back through onboarding on every launch.
  useEffect(() => {
    if (hydrated && !userId) router.replace('/onboarding');
  }, [hydrated, userId, router]);

  if (!hydrated) return null;

  const completed: Record<string, number> = {};
  for (const chapter of Object.values(chapters)) {
    if (chapter.completedAt !== null) completed[chapter.chapterId] = chapter.stars;
  }

  return <WorldMap completed={completed} xp={xp} />;
}
