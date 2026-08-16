import { Landing } from '@/components/landing/Landing';
import { SITE_DESCRIPTION, SITE_NAME, SITE_URL } from '@/lib/seo';

/**
 * Course schema, not SoftwareApplication — the search intent this targets is
 * "learn AI" / "learn machine learning", and Course is what Google's rich
 * results key off for that intent. No chapter/world counts baked in here;
 * `numberOfCredits`-style specifics would go stale as the curriculum grows.
 */
const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Course',
  name: SITE_NAME,
  description: SITE_DESCRIPTION,
  url: SITE_URL,
  isAccessibleForFree: true,
  inLanguage: 'en',
  provider: {
    '@type': 'Organization',
    name: SITE_NAME,
    url: SITE_URL,
  },
  about: [
    { '@type': 'Thing', name: 'Artificial Intelligence' },
    { '@type': 'Thing', name: 'Machine Learning' },
    { '@type': 'Thing', name: 'Large Language Models' },
    { '@type': 'Thing', name: 'Neural Networks' },
    { '@type': 'Thing', name: 'Deep Learning' },
  ],
  hasCourseInstance: {
    '@type': 'CourseInstance',
    courseMode: 'Online',
    courseWorkload: 'Self-paced',
  },
};

export default function RootPage() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Landing />
    </>
  );
}
