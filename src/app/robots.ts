import type { MetadataRoute } from 'next';
import { SITE_URL } from '@/lib/seo';

/**
 * `/admin` is auth-gated and carries nothing a crawler should index; `/api`
 * is data endpoints, not pages. Everything else — the landing page, the
 * onboarding screen and every chapter — is real, crawlable content.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: ['/admin', '/api'],
    },
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
