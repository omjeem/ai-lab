import { describe, it, expect } from 'vitest';
import { getGame } from '@/lib/curriculum';
import { deriveChapterKeywords, SITE_KEYWORDS } from '@/lib/seo';

describe('deriveChapterKeywords', () => {
  it('includes the chapter title and world title for a real chapter', () => {
    const game = getGame('1-1-vectors')!;
    const keywords = deriveChapterKeywords(game);

    expect(keywords).toContain(game.chapterTitle);
    expect(keywords).toContain(game.worldTitle);
    expect(keywords.length).toBeGreaterThan(0);
  });

  it('never repeats a term', () => {
    const game = getGame('5-2-self-attention')!;
    const keywords = deriveChapterKeywords(game);
    expect(new Set(keywords).size).toBe(keywords.length);
  });

  it('draws its generic terms from the site-wide keyword list', () => {
    const game = getGame('4-1-ngrams')!;
    const keywords = deriveChapterKeywords(game);
    expect(keywords.some((k) => SITE_KEYWORDS.includes(k))).toBe(true);
  });
});
