import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { getGame } from '@/lib/curriculum';
import { deriveChapterKeywords, SITE_NAME, SITE_URL } from '@/lib/seo';
import { ChapterPageClient } from './ChapterPageClient';

type Params = Promise<{ worldId: string; chapterId: string }>;
type SearchParams = Promise<{ via?: string }>;

/**
 * Real per-chapter title and description, pulled from the chapter's own
 * written content (`chapterTitle`, `concept.shortExplanation`) rather than
 * inheriting the site-wide default — this is what gives every chapter its
 * own real, distinct, indexable page instead of 20+ pages sharing one title.
 *
 * `openGraph`/`twitter` are declared in full here, not partially: Next.js
 * replaces rather than merges a parent layout's metadata object once a route
 * returns its own key, so a partial object silently drops the root layout's
 * `type`/`url`/`siteName`/`summary_large_image` card instead of layering on
 * top of it.
 */
export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { chapterId } = await params;
  const game = getGame(chapterId);
  if (!game) return {};

  const title = `${game.chapterTitle} · World ${game.world}: ${game.worldTitle}`;
  const description = game.concept.shortExplanation;
  const canonical = `/world/${game.world}/chapter/${game.id}`;

  return {
    title: game.chapterTitle,
    description,
    keywords: deriveChapterKeywords(game),
    alternates: { canonical },
    openGraph: {
      type: 'article',
      url: `${SITE_URL}${canonical}`,
      siteName: SITE_NAME,
      locale: 'en_US',
      title,
      description,
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
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

  /**
   * LearningResource, not Course (that's the root page's schema for "learn
   * AI" search intent) — this is what tells a search engine a specific page
   * teaches a specific, named thing, which is the signal topical queries
   * like "vectors explanation" actually key off.
   */
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    name: game.chapterTitle,
    description: game.concept.shortExplanation,
    url: `${SITE_URL}/world/${game.world}/chapter/${game.id}`,
    teaches: game.chapterTitle,
    isAccessibleForFree: true,
    inLanguage: 'en',
    isPartOf: {
      '@type': 'Course',
      name: SITE_NAME,
      url: SITE_URL,
    },
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <ChapterPageClient chapterId={chapterId} sharedEntry={sharedEntry} />
    </>
  );
}
