'use client';

import { useEffect } from 'react';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { Wrench } from 'lucide-react';
import { getGame, isChapterUnlocked } from '@/lib/curriculum';
import { ChapterShell, type PrerequisiteGap } from '@/components/chapter/ChapterShell';
import { getGameComponent } from '@/components/games/registry';
import { Button, Heading, Tag } from '@/components/ui';
import { useGameProgressStore } from '@/store/useGameProgressStore';
import { enqueueActivity } from '@/lib/offlineQueue';
import type { GameDefinition } from '@/types/game';

/** Every chapter this one's `unlockRequires` still needs, in order. */
function missingPrerequisites(game: GameDefinition, completed: Record<string, number>) {
  return game.unlockRequires
    .filter((reqId) => completed[reqId] === undefined)
    .map((reqId) => getGame(reqId))
    .filter((g): g is GameDefinition => g !== null);
}

export function ChapterPageClient({
  chapterId,
  sharedEntry,
}: {
  chapterId: string;
  sharedEntry: boolean;
}) {
  const hydrated = useGameProgressStore((s) => s.hydrated);
  const userId = useGameProgressStore((s) => s.userId);
  const chapters = useGameProgressStore((s) => s.chapters);

  const completed: Record<string, number> = {};
  for (const chapter of Object.values(chapters)) {
    if (chapter.completedAt !== null) completed[chapter.chapterId] = chapter.stars;
  }

  // Fires once this chapter's real unlock state is known (post-hydration) —
  // both the "shared into a locked chapter" and the "shared a chapter you'd
  // already unlocked" case are attributed, distinguished only by `wasLocked`.
  useEffect(() => {
    if (!hydrated || !userId || !sharedEntry) return;
    const isUnlocked = Object.keys(completed).length > 0;
    void enqueueActivity({
      userId,
      type: 'chapter_shared_link_opened',
      chapterId,
      detail: { wasLocked: isUnlocked ? 0 : 1 },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, userId, sharedEntry, chapterId]);

  // Progress hasn't loaded from IndexedDB yet — without this, a returning
  // player deep-linking a chapter they already completed would see a false
  // flash before the real state arrives.
  if (!hydrated) return null;

  const game = getGame(chapterId);
  if (!game) notFound();

  const GameComponent = getGameComponent(chapterId);

  const unlocked = isChapterUnlocked(chapterId, completed);

  // Direct URL navigation always renders the chapter immediately — no blocking
  // modal. Instead, a dismissible strip banner is shown via `prerequisiteGap`
  // inside ChapterShell if the chapter's prerequisites haven't been met yet.
  const prerequisiteGap: PrerequisiteGap | undefined = unlocked
    ? undefined
    : {
        missing: missingPrerequisites(game, completed).map((g) => ({
          id: g.id,
          title: g.chapterTitle,
          world: g.world,
        })),
      };

  return (
    <ChapterShell game={game} prerequisiteGap={prerequisiteGap}>
      {(render) =>
        GameComponent ? (
          <GameComponent {...render} game={game} />
        ) : (
          <NotYetBuilt chapterTitle={game.chapterTitle} engineType={render.level.engineType} />
        )
      }
    </ChapterShell>
  );
}

/**
 * Placeholder for chapters whose canvas is not built yet.
 *
 * Says so plainly rather than rendering an empty instrument — the engine and its
 * tests exist, the visualisation does not.
 */
function NotYetBuilt({
  chapterTitle,
  engineType,
}: {
  chapterTitle: string;
  engineType: string;
}) {
  return (
    <div className="grid-field flex flex-1 flex-col items-center justify-center gap-4 px-6 py-16 text-center">
      <span className="text-muted">
        <Wrench size={22} strokeWidth={1.75} />
      </span>
      <Heading level={2}>Instrument not wired up yet</Heading>
      <p className="max-w-prose text-sm leading-relaxed text-secondary">
        The logic for {chapterTitle} is finished and tested — what is missing is the canvas that
        renders it.
      </p>
      <Tag>engine · {engineType}</Tag>
      <Link href="/map">
        <Button>Back to the map</Button>
      </Link>
    </div>
  );
}
