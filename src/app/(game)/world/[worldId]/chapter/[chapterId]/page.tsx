import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { getGame } from '@/lib/curriculum';
import { ChapterPageClient } from './ChapterPageClient';

type Params = Promise<{ worldId: string; chapterId: string }>;
type SearchParams = Promise<{ via?: string }>;

/**
 * Real per-chapter title and description, pulled from the chapter's own
 * written content (`chapterTitle`, `concept.shortExplanation`) rather than
 * inheriting the site-wide default — this is what gives every chapter its
 * own real, distinct, indexable page instead of 20+ pages sharing one title.
 */
export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { chapterId } = await params;
  const game = getGame(chapterId);
  if (!game) return {};

  return {
    title: game.chapterTitle,
    description: game.concept.shortExplanation,
    alternates: { canonical: `/world/${game.world}/chapter/${game.id}` },
    openGraph: {
      title: `${game.chapterTitle} · World ${game.world}: ${game.worldTitle}`,
      description: game.concept.shortExplanation,
    },
  };
}

export default async function ChapterPage({
  params,
  searchParams,
}: {
  params: Params;
  searchParams: SearchParams;
}) {
  const { chapterId } = await params;
  const game = getGame(chapterId);
  if (!game) notFound();

  const { via } = await searchParams;
  const sharedEntry = via === 'share';

  return <ChapterPageClient chapterId={chapterId} sharedEntry={sharedEntry} />;
}
