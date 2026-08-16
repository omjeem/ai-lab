import type { MetadataRoute } from 'next';
import { allGames } from '@/lib/curriculum';
import { SITE_URL } from '@/lib/seo';

/**
 * Generated from the live curriculum rather than a fixed list, so a chapter
 * added or removed later is reflected here automatically — nothing to
 * remember to update by hand.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const staticRoutes: MetadataRoute.Sitemap = [
    { url: SITE_URL, changeFrequency: 'weekly', priority: 1 },
    { url: `${SITE_URL}/onboarding`, changeFrequency: 'monthly', priority: 0.8 },
    { url: `${SITE_URL}/map`, changeFrequency: 'monthly', priority: 0.5 },
  ];

  const chapterRoutes: MetadataRoute.Sitemap = allGames().map((game) => ({
    url: `${SITE_URL}/world/${game.world}/chapter/${game.id}`,
    changeFrequency: 'monthly',
    priority: 0.7,
  }));

  return [...staticRoutes, ...chapterRoutes];
}
