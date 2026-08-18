'use client';

/**
 * Bare icon-button used both on the map's chapter nodes and inside a
 * chapter's own header — matches the idiom `ThemeToggle.tsx`/`AppShell.tsx`
 * already use for other single-purpose icon buttons.
 */
import { useEffect, useState } from 'react';
import { Check, Share2 } from 'lucide-react';
import { buildChapterShareUrl, shareChapter } from '@/lib/shareLink';
import { cx } from '@/components/ui';

export function ShareButton({
  world,
  chapterId,
  chapterTitle,
  size = 12,
  className,
}: {
  world: number;
  chapterId: string;
  chapterTitle: string;
  size?: number;
  className?: string;
}) {
  const [justShared, setJustShared] = useState(false);

  useEffect(() => {
    if (!justShared) return;
    const timer = setTimeout(() => setJustShared(false), 1500);
    return () => clearTimeout(timer);
  }, [justShared]);

  const handleShare = async (event: React.MouseEvent) => {
    // Several call sites nest this inside a chapter-node `<Link>` — this
    // button's own action must win, not a navigation to the chapter.
    event.preventDefault();
    event.stopPropagation();
    const outcome = await shareChapter(buildChapterShareUrl(world, chapterId), chapterTitle);
    if (outcome !== 'failed') setJustShared(true);
  };

  return (
    <button
      type="button"
      onClick={handleShare}
      aria-label={`Share ${chapterTitle}`}
      title={justShared ? 'Link copied' : 'Share this chapter'}
      className={cx('p-1 text-muted transition-colors hover:text-primary', className)}
    >
      {justShared ? <Check size={size} strokeWidth={2} /> : <Share2 size={size} strokeWidth={2} />}
    </button>
  );
}
