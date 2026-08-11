'use client';

import { use } from 'react';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Wrench } from 'lucide-react';
import { getGame } from '@/lib/curriculum';
import { ChapterShell } from '@/components/chapter/ChapterShell';
import { getGameComponent } from '@/components/games/registry';
import { Button, Heading, Tag } from '@/components/ui';

export default function ChapterPage({
  params,
}: {
  params: Promise<{ worldId: string; chapterId: string }>;
}) {
  const { chapterId } = use(params);
  const game = getGame(chapterId);
  if (!game) notFound();

  const GameComponent = getGameComponent(chapterId);

  return (
    <ChapterShell game={game}>
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
