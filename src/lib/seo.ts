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
