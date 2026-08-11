'use client';

/**
 * The world map, drawn as a dependency graph.
 *
 * Section 6 is explicit that this must read as a technical systems diagram
 * rather than a platformer path: chapters are nodes on a bus, edges are the real
 * `unlockRequires` relationships, and each world owns a hue so position in the
 * course is legible at a glance.
 */
import Link from 'next/link';
import { motion, useReducedMotion } from 'motion/react';
import { Lock, Check } from 'lucide-react';
import {
  manifest,
  resolveUnlocks,
  summarise,
  type CompletionRecord,
} from '@/lib/curriculum';
import { Heading, Meter, Readout, StarRating, Tag, cx } from '@/components/ui';
import type { ManifestChapter, ManifestWorld } from '@/types/game';

export function WorldMap({ completed, xp }: { completed: CompletionRecord; xp: number }) {
  const states = resolveUnlocks(completed);
  const summary = summarise(completed, xp);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ProgressRail summary={summary} />

      <div className="grid-field flex-1 overflow-x-auto px-3 py-6 sm:px-6">
        <div className="mx-auto flex min-w-[640px] max-w-5xl flex-col gap-0">
          {manifest.worlds.map((world, index) => (
            <WorldRow
              key={world.world}
              world={world}
              states={states}
              isLast={index === manifest.worlds.length - 1}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ProgressRail({ summary }: { summary: ReturnType<typeof summarise> }) {
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-4 border-b border-line px-3 py-4 sm:px-6">
      <div className="flex flex-col gap-1">
        <span className="label">rank</span>
        <span className="font-display text-xl font-semibold text-primary">
          {summary.rank.title}
        </span>
      </div>

      <Readout label="xp" value={summary.xp} size="md" tone="accent" />
      <Readout
        label="chapters"
        value={`${summary.chaptersCompleted}/${summary.chaptersTotal}`}
        size="md"
      />
      <Readout label="stars" value={`${summary.starsEarned}/${summary.starsPossible}`} size="md" />

      {summary.nextRank && (
        <div className="min-w-[180px] flex-1">
          <Meter
            value={summary.xp}
            max={summary.nextRank.minXp}
            label={`${summary.xpToNextRank} xp to ${summary.nextRank.title}`}
          />
        </div>
      )}
    </div>
  );
}

function WorldRow({
  world,
  states,
  isLast,
}: {
  world: ManifestWorld;
  states: Map<string, ReturnType<typeof resolveUnlocks> extends Map<string, infer V> ? V : never>;
  isLast: boolean;
}) {
  const chapters = [...world.chapters].sort((a, b) => a.order - b.order);
  const cleared = chapters.filter((c) => states.get(c.id)?.status === 'completed').length;

  return (
    <section data-world={world.world} className="relative flex gap-4 pb-8">
      {/* The bus running down the left, connecting worlds into one system. */}
      <div className="relative flex w-6 shrink-0 flex-col items-center pt-1">
        <span
          className="block h-2.5 w-2.5 border"
          style={{
            borderColor: 'var(--accent)',
            background: cleared === chapters.length ? 'var(--accent)' : 'transparent',
          }}
          aria-hidden
        />
        {!isLast && <span className="mt-1 w-px flex-1 bg-line" aria-hidden />}
      </div>

      <div className="min-w-0 flex-1">
        <header className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <Heading level={3} className="text-accent">
            World {world.world} · {world.title}
          </Heading>
          <span className="text-xs text-muted">{world.subtitle}</span>
          <span className="label ml-auto">
            {cleared}/{chapters.length} cleared
          </span>
        </header>

        <ol className="flex flex-wrap items-stretch gap-x-0 gap-y-3">
          {chapters.map((chapter, index) => (
            <li key={chapter.id} className="flex items-stretch">
              {index > 0 && <EdgeConnector satisfied={states.get(chapter.id)?.status !== 'locked'} />}
              <ChapterNode
                chapter={chapter}
                world={world}
                state={states.get(chapter.id)!}
              />
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/** The dependency edge between two chapters. */
function EdgeConnector({ satisfied }: { satisfied: boolean }) {
  return (
    <span className="flex w-6 items-center sm:w-8" aria-hidden>
      <span
        className={cx('h-px w-full', satisfied ? 'bg-accent/50' : 'bg-line')}
      />
    </span>
  );
}

function ChapterNode({
  chapter,
  world,
  state,
}: {
  chapter: ManifestChapter;
  world: ManifestWorld;
  state: { status: 'locked' | 'available' | 'completed'; stars: number; missing: string[] };
}) {
  const reduce = useReducedMotion();
  const locked = state.status === 'locked';

  const body = (
    <motion.div
      whileHover={locked || reduce ? undefined : { y: -2 }}
      transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
      className={cx(
        'flex h-full w-[168px] flex-col justify-between gap-2 border px-3 py-2.5',
        locked
          ? 'border-line bg-inset/60 saturate-0'
          : state.status === 'completed'
            ? 'border-accent/40 bg-panel'
            : 'border-line-strong bg-panel'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="readout text-[10px] text-muted">{chapter.id}</span>
        {locked ? (
          <Lock size={11} strokeWidth={2} className="text-muted" />
        ) : state.status === 'completed' ? (
          <Check size={11} strokeWidth={2.5} className="text-accent" />
        ) : null}
      </div>

      <span
        className={cx(
          'text-[13px] leading-snug',
          locked ? 'text-muted' : 'text-primary'
        )}
      >
        {chapter.title}
      </span>

      <div className="flex items-center justify-between gap-2">
        {state.status === 'completed' ? (
          <StarRating stars={state.stars} />
        ) : (
          <span className="readout text-[10px] text-muted">{chapter.xpReward} xp</span>
        )}
        <TierTag tier={chapter.tier} />
      </div>
    </motion.div>
  );

  if (locked) {
    return (
      <div
        title={`Complete ${state.missing.join(', ')} first`}
        aria-label={`${chapter.title}, locked. Requires ${state.missing.join(', ')}`}
      >
        {body}
      </div>
    );
  }

  return (
    <Link
      href={`/world/${world.world}/chapter/${chapter.id}`}
      aria-label={`${chapter.title}, ${state.status}`}
      className="block h-full"
    >
      {body}
    </Link>
  );
}

function TierTag({ tier }: { tier: ManifestChapter['tier'] }) {
  const labels: Record<ManifestChapter['tier'], string> = {
    none: 'math',
    'browser-light': 'light',
    'browser-heavy': 'heavy',
    cloud: 'cloud',
  };
  return <Tag tone={tier === 'none' ? 'neutral' : 'accent'}>{labels[tier]}</Tag>;
}
