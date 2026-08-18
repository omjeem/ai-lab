/**
 * Single source of truth for the production origin. Every place that needs
 * an absolute URL — metadataBase, the sitemap, robots.txt, canonical links —
 * reads from here, so there is exactly one line to change if the domain ever
 * does.
 */
export const SITE_URL = 'https://ailab.ommishra.tech';

export const SITE_NAME = 'AI Learning Lab';

/**
 * Deliberately has no chapter/world counts in it — those change as the
 * curriculum grows, and a stale number in a title tag or meta description is
 * worse than no number at all. Written to match "learn AI" / "learn machine
 * learning" search intent rather than to describe the site's own structure.
 */
export const SITE_TITLE = 'Learn AI by Building It, Not Reading About It';

export const SITE_DESCRIPTION =
  'Learn AI and machine learning the way engineers actually learn it: by building it. Run real embeddings, real attention, real gradients, and real neural networks — computed live in your browser, not simulated. No lectures, no chatbot answers, no invented numbers. Free, interactive, and works offline.';

export const SITE_KEYWORDS = [
  'learn AI',
  'learn machine learning',
  'learn artificial intelligence',
  'how do LLMs work',
  'how neural networks work',
  'learn deep learning',
  'interactive AI course',
  'understand transformers',
  'AI for beginners',
  'machine learning from scratch',
  'learn large language models',
];

/**
 * Chapter-specific search terms, derived rather than hand-authored per
 * chapter. Google's `keywords` meta tag has had no ranking effect since
 * ~2009, and the root layout already applies `SITE_KEYWORDS` to every page —
 * so this exists to make each chapter's tag chapter-specific rather than to
 * fill a "zero keywords" gap. A curated per-chapter field in the JSON would
 * need a schema change, a validator rule to keep it from rotting, and 26+
 * files of upkeep for a tag with no measurable ranking benefit; deriving it
 * from data that's already real (and already kept current) costs nothing to
 * maintain as the curriculum grows.
 */
export function deriveChapterKeywords(game: {
  chapterTitle: string;
  worldTitle: string;
}): string[] {
  const terms = [
    game.chapterTitle,
    `${game.chapterTitle} explained`,
    `learn ${game.chapterTitle.toLowerCase()}`,
    game.worldTitle,
    ...SITE_KEYWORDS.slice(0, 5),
  ];
  return Array.from(new Set(terms));
}
