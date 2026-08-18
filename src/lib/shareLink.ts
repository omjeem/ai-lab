/**
 * Chapter share links.
 *
 * A shared link always skips the prerequisite-gap warning for whoever opens
 * it — see the `via=share` handling in `ChapterPageClient.tsx`. Deliberately
 * not named `source`: `src/middleware.ts` already intercepts any `?source=`
 * on every route for an unrelated external-tracking beacon and strips it via
 * redirect, which would destroy this marker before it's ever read.
 */
import { SITE_URL } from '@/lib/seo';

export function buildChapterShareUrl(world: number, chapterId: string): string {
  return `${SITE_URL}/world/${world}/chapter/${chapterId}?via=share`;
}

export type ShareOutcome = 'shared' | 'copied' | 'failed';

/**
 * Tries the native share sheet first (mobile-friendly, gives its own
 * confirmation UI), falling back to the clipboard. A rejected `share()` —
 * most commonly the user cancelling the sheet — is reported as `'failed'`
 * rather than silently falling through to the clipboard, since writing a
 * link to the clipboard after someone dismissed the share sheet would be
 * a surprising thing to do behind their back.
 */
export async function shareChapter(url: string, title: string): Promise<ShareOutcome> {
  if (typeof navigator !== 'undefined' && typeof navigator.share === 'function') {
    try {
      await navigator.share({ title, url });
      return 'shared';
    } catch {
      return 'failed';
    }
  }

  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(url);
      return 'copied';
    } catch {
      return 'failed';
    }
  }

  return 'failed';
}
