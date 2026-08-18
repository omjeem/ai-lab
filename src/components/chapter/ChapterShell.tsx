'use client';

/**
 * The frame every chapter plays inside.
 *
 * Layout follows Section 6: the visualisation owns the screen and the HUD sits
 * around it — a collapsible sheet on mobile, a persistent side rail from
 * tablet up. The shell owns the concept panel, level progression, scoring and
 * the aha reveal; the game component owns only its own canvas and controls.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { ArrowLeft, ArrowRight, Lightbulb, RotateCcw, X } from 'lucide-react';
import type { GameDefinition, GameLevel, ScoreResult } from '@/types/game';
import { nextChapterAfter, worldOfChapter } from '@/lib/curriculum';
import {
  useGameProgressStore,
  type ChapterProgress,
  type LevelResult,
} from '@/store/useGameProgressStore';
import { useSessionStore } from '@/store/useSessionStore';
import { enqueueActivity } from '@/lib/offlineQueue';
import { playChapterCompleteTone, playCorrectTone } from '@/lib/sound';
import { Button, Heading, Panel, Readout, StarRating, Tag, cx } from '@/components/ui';
import { ShareButton } from '@/components/ui/ShareButton';
import { ChapterFeedback } from '@/components/chapter/ChapterFeedback';

export interface GameRenderProps {
  level: GameLevel;
  /** Reports the current score so the HUD can track it live. */
  onScore: (result: ScoreResult) => void;
  /** Submits the run for scoring and progression. */
  onSubmit: (result: ScoreResult) => void;
}

/** The prerequisite chapters this one's `unlockRequires` still needs. */
export interface PrerequisiteGap {
  missing: { id: string; title: string; world: number }[];
}

export interface ChapterShellProps {
  game: GameDefinition;
  /** Present whenever this chapter is being played without its prerequisites met —
   *  shows a dismissible strip banner at the top of the instrument area. */
  prerequisiteGap?: PrerequisiteGap;
  /** Renders the level's own canvas and controls. */
  children: (props: GameRenderProps) => ReactNode;
}

/**
 * Where a returning player picks back up — the first level with no recorded
 * result yet, so re-opening a chapter never re-plays what is already passed.
 * Falls back to the last level once every one of them (including any
 * optional stretch level) has a result, rather than looping back to the
 * first.
 */
function resumeLevelIndex(game: GameDefinition, progress: ChapterProgress | null): number {
  if (!progress) return 0;
  const index = game.levels.findIndex((l) => !progress.levels[l.id]);
  return index === -1 ? game.levels.length - 1 : index;
}

export function ChapterShell({ game, prerequisiteGap, children }: ChapterShellProps) {
  const reduce = useReducedMotion();
  const world = worldOfChapter(game.id);
  const nextChapter = nextChapterAfter(game.id);
  const [gapDismissed, setGapDismissed] = useState(false);

  const userId = useGameProgressStore((s) => s.userId);
  const displayName = useGameProgressStore((s) => s.displayName);
  const recordLevel = useGameProgressStore((s) => s.recordLevel);
  const completeChapter = useGameProgressStore((s) => s.completeChapter);
  const soundEnabled = useGameProgressStore((s) => s.soundEnabled);
  const completedLevels = useGameProgressStore((s) => s.chapters[game.id]?.levels);

  const levelIndex = useSessionStore((s) => s.levelIndex);
  const phase = useSessionStore((s) => s.phase);
  const attempts = useSessionStore((s) => s.attempts);
  const beginChapter = useSessionStore((s) => s.beginChapter);
  const goToLevel = useSessionStore((s) => s.goToLevel);
  const setPhase = useSessionStore((s) => s.setPhase);
  const registerAttempt = useSessionStore((s) => s.registerAttempt);
  const recordResult = useSessionStore((s) => s.recordResult);

  const [liveScore, setLiveScore] = useState<ScoreResult | null>(null);
  const [chapterDone, setChapterDone] = useState(false);
  const [hintsRevealed, setHintsRevealed] = useState(0);

  const level = game.levels[levelIndex];
  const requiredLevels = useMemo(() => game.levels.filter((l) => !l.optional), [game.levels]);

  useEffect(() => {
    beginChapter(game.id);
    // A returning player resumes past what they have already cleared, rather
    // than always landing back on level 1 — the persisted result is the
    // source of truth here, not the (deliberately unpersisted) session state.
    const progress = useGameProgressStore.getState().chapterProgress(game.id);
    const resumeIndex = resumeLevelIndex(game, progress);
    if (resumeIndex > 0) goToLevel(resumeIndex);
    if (userId) void enqueueActivity({ userId, type: 'chapter_started', chapterId: game.id });
  }, [game, userId, beginChapter, goToLevel]);

  // A new level starts with none of its hints shown. Not tied to `phase`, so a
  // retry within the same level keeps whatever the player already unlocked.
  useEffect(() => setHintsRevealed(0), [level?.id]);

  const handleSubmit = useCallback(
    (result: ScoreResult) => {
      if (!level) return;
      registerAttempt();
      recordResult(result);
      setLiveScore(result);

      if (result.passed) {
        recordLevel(game.id, {
          levelId: level.id,
          stars: result.stars,
          metric: result.metric,
          value: result.value,
          xpEarned: result.xpEarned,
          attempts: attempts + 1,
          completedAt: Date.now(),
        });
        if (soundEnabled) playCorrectTone();
      }

      if (userId) {
        void enqueueActivity({
          userId,
          type: result.passed ? 'level_completed' : 'level_failed',
          chapterId: game.id,
          levelId: level.id,
          detail: {
            metric: result.metric,
            value: result.value,
            stars: result.stars,
            xp: result.xpEarned,
            attempts: attempts + 1,
          },
        });
      }
    },
    [level, game.id, userId, attempts, soundEnabled, registerAttempt, recordResult, recordLevel]
  );

  const advance = useCallback(() => {
    const isLastRequired = levelIndex >= requiredLevels.length - 1;
    if (!isLastRequired) {
      goToLevel(levelIndex + 1);
      setLiveScore(null);
      return;
    }

    // Chapter stars are the weakest link across its required levels — clearing
    // two perfectly and scraping the third is a two-star chapter.
    const progress = useGameProgressStore.getState().chapterProgress(game.id);
    const earned = requiredLevels.map((l) => progress?.levels[l.id]?.stars ?? 0);
    const chapterStars = earned.length === 0 ? 0 : Math.min(...earned);

    completeChapter(game.id, chapterStars);
    setChapterDone(true);
    if (soundEnabled) playChapterCompleteTone();

    if (userId) {
      void enqueueActivity({
        userId,
        type: 'chapter_completed',
        chapterId: game.id,
        detail: { stars: chapterStars },
      });
    }
  }, [levelIndex, requiredLevels, game.id, userId, soundEnabled, goToLevel, completeChapter]);

  if (!level) return null;

  return (
    <div data-world={game.world} className="flex min-h-0 flex-1 flex-col lg:flex-row">
      {/* ── the instrument: dominant, always visible ── */}
      <div className="relative flex min-h-[55vh] min-w-0 flex-1 flex-col">
        <ChapterBar
          game={game}
          level={level}
          levelIndex={levelIndex}
          totalLevels={game.levels.length}
          worldNumber={world?.world ?? game.world}
        />

        {prerequisiteGap && prerequisiteGap.missing.length > 0 && !gapDismissed && (
          <PrerequisiteGapBanner gap={prerequisiteGap} onDismiss={() => setGapDismissed(true)} />
        )}

        <div className="relative flex min-h-0 flex-1 flex-col">
          {phase === 'concept' ? (
            <ConceptPanel game={game} level={level} onStart={() => setPhase('playing')} />
          ) : (
            children({ level, onScore: setLiveScore, onSubmit: handleSubmit })
          )}
        </div>
      </div>

      {/* ── HUD: below the canvas on mobile, a persistent rail from lg up.
          Always rendered — there is no toggle to find the score or hints. ── */}
      <aside
        className={cx(
          'flex flex-col border-t border-line bg-panel/40',
          'lg:min-h-0 lg:w-[320px] lg:shrink-0 lg:border-l lg:border-t-0'
        )}
      >
        <Hud
          game={game}
          level={level}
          levelIndex={levelIndex}
          liveScore={liveScore}
          attempts={attempts}
          phase={phase}
          completedLevels={completedLevels}
          hintsRevealed={hintsRevealed}
          onRevealHint={() =>
            setHintsRevealed((n) => Math.min(n + 1, level.hints?.length ?? 0))
          }
          onRetry={() => {
            setLiveScore(null);
            setPhase('playing');
          }}
          onSelectLevel={(index) => {
            setLiveScore(null);
            goToLevel(index);
          }}
          onAdvance={advance}
        />
      </aside>

      <AnimatePresence>
        {chapterDone && (
          <AhaReveal
            game={game}
            reduce={reduce ?? false}
            userId={userId}
            displayName={displayName}
            nextHref={
              nextChapter
                ? `/world/${nextChapter.id.split('-')[0]}/chapter/${nextChapter.id}`
                : '/map'
            }
            nextLabel={nextChapter ? nextChapter.title : 'Back to the map'}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── top bar ────────────────────────────────────────────────── */

function ChapterBar({
  game,
  level,
  levelIndex,
  totalLevels,
  worldNumber,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  totalLevels: number;
  worldNumber: number;
}) {
  return (
    <div className="flex items-center gap-3 border-b border-line px-3 py-2">
      <Link
        href="/map"
        className="flex items-center gap-1.5 text-muted transition-colors hover:text-primary"
        aria-label="Back to the map"
      >
        <ArrowLeft size={14} strokeWidth={2} />
        <span className="label hidden sm:inline">map</span>
      </Link>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="readout text-[10px] text-accent">W{worldNumber}</span>
          <span className="truncate text-[13px] text-primary">{game.chapterTitle}</span>
        </div>
      </div>

      <ShareButton world={worldNumber} chapterId={game.id} chapterTitle={game.chapterTitle} size={13} />

      <span className="label whitespace-nowrap">
        {levelIndex + 1}/{totalLevels} · {level.difficulty}
      </span>
    </div>
  );
}


/* ── prerequisite-gap banner ────────────────────────────────── */

/**
 * A dismissible one-line strip shown at the top of the instrument area when a
 * player opens a chapter without completing its prerequisites (via direct URL,
 * bookmark, back-button, or share link). It stays up for the session; dismissing
 * hides it for this visit only, since the underlying gap hasn't closed.
 */
function PrerequisiteGapBanner({
  gap,
  onDismiss,
}: {
  gap: PrerequisiteGap;
  onDismiss: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
      <span className="min-w-0 flex-1 leading-relaxed">
        You haven&apos;t finished{' '}
        {gap.missing.map((m, i) => (
          <span key={m.id}>
            {i > 0 && ', '}
            <Link
              href={`/world/${m.world}/chapter/${m.id}`}
              className="underline underline-offset-2 hover:text-primary"
            >
              {m.title}
            </Link>
          </span>
        ))}{' '}
        yet — finish {gap.missing.length === 1 ? 'it' : 'them'} first to build on this properly.
      </span>
      <Link href="/map" className="label shrink-0 underline underline-offset-2 hover:text-primary">
        map
      </Link>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="shrink-0 text-warn transition-colors hover:text-primary"
      >
        <X size={13} strokeWidth={2} />
      </button>
    </div>
  );
}

/* ── concept panel ──────────────────────────────────────────── */

function ConceptPanel({
  game,
  level,
  onStart,
}: {
  game: GameDefinition;
  level: GameLevel;
  onStart: () => void;
}) {
  return (
    <div className="grid-field flex flex-1 items-center justify-center px-6 py-10">
      <div className="w-full max-w-xl">
        <p className="label mb-2">{game.id}</p>
        <Heading level={2} className="mb-4">
          {game.chapterTitle}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">
          {game.concept.shortExplanation}
        </p>

        <div className="mb-6 border-l-2 border-accent/50 pl-4">
          <p className="label mb-1">level {level.id} · {level.title}</p>
          <p className="text-sm leading-relaxed text-primary">{level.instructions}</p>
        </div>

        <Button variant="primary" onClick={onStart}>
          Begin
          <ArrowRight size={13} strokeWidth={2} />
        </Button>
      </div>
    </div>
  );
}

/* ── HUD ────────────────────────────────────────────────────── */

function Hud({
  game,
  level,
  levelIndex,
  liveScore,
  attempts,
  phase,
  completedLevels,
  hintsRevealed,
  onRevealHint,
  onRetry,
  onAdvance,
  onSelectLevel,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  liveScore: ScoreResult | null;
  attempts: number;
  phase: string;
  /** Which of this chapter's levels already have a passing result, keyed by level id. */
  completedLevels: Record<string, LevelResult> | undefined;
  hintsRevealed: number;
  onRevealHint: () => void;
  onRetry: () => void;
  onAdvance: () => void;
  /** Jumps straight to an already-cleared level, replaying it. */
  onSelectLevel: (index: number) => void;
}) {
  const comparator = level.passCriteria.comparator;
  const target = level.passCriteria.threshold;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <Panel label="score">
        <div className="flex flex-col gap-4">
          <Readout
            label={level.passCriteria.metric}
            value={liveScore?.value ?? '—'}
            size="lg"
            tone={liveScore === null ? 'neutral' : liveScore.passed ? 'good' : 'bad'}
          />

          <div className="flex items-center justify-between gap-3">
            <span className="label">
              target {comparator === 'lte' ? '≤' : comparator === 'gte' ? '≥' : '='} {target}
            </span>
            <StarRating stars={liveScore?.stars ?? 0} size={14} />
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="label">attempts</span>
            <span className="readout text-sm text-secondary">{attempts}</span>
          </div>
        </div>
      </Panel>

      {level.hints && level.hints.length > 0 && (
        <HintPanel hints={level.hints} revealed={hintsRevealed} onReveal={onRevealHint} />
      )}

      {liveScore && Object.keys(liveScore.breakdown).length > 0 && (
        <Panel label="breakdown">
          <dl className="flex flex-col gap-1.5">
            {Object.entries(liveScore.breakdown).map(([key, value]) => (
              <div key={key} className="flex items-baseline justify-between gap-3">
                <dt className="truncate text-[11px] text-muted">{key}</dt>
                <dd className="readout text-xs text-secondary">
                  {Number.isFinite(value) ? Number(value).toFixed(3) : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}

      <Panel label="level">
        <ol className="flex flex-col gap-1.5">
          {game.levels.map((l, index) => {
            const result = completedLevels?.[l.id];
            const isCurrent = index === levelIndex;
            const row = (
              <>
                <span className="truncate">
                  {index + 1}. {l.title}
                </span>
                <span className="flex items-center gap-1.5">
                  {result && <StarRating stars={result.stars} size={10} />}
                  {l.optional && <Tag>stretch</Tag>}
                </span>
              </>
            );

            // Only a level that already has a result is a re-entry point —
            // this is a replay shortcut, not a way to skip ahead.
            if (result && !isCurrent) {
              return (
                <li key={l.id}>
                  <button
                    type="button"
                    onClick={() => onSelectLevel(index)}
                    className="flex w-full items-center justify-between gap-2 text-[11px] text-muted transition-colors hover:text-primary"
                  >
                    {row}
                  </button>
                </li>
              );
            }

            return (
              <li
                key={l.id}
                className={cx(
                  'flex items-center justify-between gap-2 text-[11px]',
                  isCurrent ? 'text-accent' : 'text-muted'
                )}
              >
                {row}
              </li>
            );
          })}
        </ol>
      </Panel>

      <div className="mt-auto flex flex-col gap-2">
        {phase === 'scored' && (
          <Button variant="primary" onClick={onAdvance}>
            {levelIndex >= game.levels.filter((l) => !l.optional).length - 1
              ? 'Finish chapter'
              : 'Next level'}
            <ArrowRight size={13} strokeWidth={2} />
          </Button>
        )}
        {(phase === 'scored' || phase === 'failed') && (
          <Button onClick={onRetry}>
            <RotateCcw size={13} strokeWidth={2} />
            Try again
          </Button>
        )}
      </div>
    </div>
  );
}

/* ── hints ──────────────────────────────────────────────────── */

/**
 * Progressive hints, one level's worth at a time.
 *
 * Revealed hints stay visible — nothing here re-hides them on a retry or a
 * score change, since `hintsRevealed` only resets when `ChapterShell` sees a
 * new `level.id`. The last hint is the level's own worked answer, not another
 * nudge, so it gets a distinct "answer" tag rather than a hint number.
 *
 * The list has a fixed max height and scrolls internally past that, rather
 * than growing — revealing hints one after another used to push the whole
 * HUD (and with it the canvas beside it) further down the page every click.
 */
function HintPanel({
  hints,
  revealed,
  onReveal,
}: {
  hints: string[];
  revealed: number;
  onReveal: () => void;
}) {
  const shown = hints.slice(0, revealed);
  const exhausted = revealed >= hints.length;
  const listRef = useRef<HTMLOListElement>(null);

  // A freshly-revealed hint should be the thing you see, not something you
  // have to go find below the fold of the box that just capped its height.
  useEffect(() => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [revealed]);

  return (
    <Panel label="hints" actions={<span className="label">{revealed}/{hints.length}</span>}>
      <div className="flex flex-col gap-3">
        {shown.length > 0 && (
          <ol
            ref={listRef}
            className="flex max-h-64 flex-col gap-2.5 overflow-y-auto pr-1"
          >
            {shown.map((hint, i) => {
              const isAnswer = i === hints.length - 1;
              return (
                <li key={i} className="flex items-start gap-2.5">
                  {isAnswer ? (
                    <Tag tone="accent">answer</Tag>
                  ) : (
                    <span className="label mt-0.5 shrink-0">{i + 1}</span>
                  )}
                  <p className="flex-1 text-xs leading-relaxed text-secondary">{hint}</p>
                </li>
              );
            })}
          </ol>
        )}

        {!exhausted && (
          <Button onClick={onReveal}>
            <Lightbulb size={13} strokeWidth={2} />
            {revealed === 0
              ? 'Show a hint'
              : revealed === hints.length - 1
                ? 'Reveal the answer'
                : 'Show another hint'}
          </Button>
        )}
      </div>
    </Panel>
  );
}

/* ── aha reveal ─────────────────────────────────────────────── */

/** Chapter completion. Section 6: satisfying but quick, and always skippable. */
function AhaReveal({
  game,
  reduce,
  userId,
  displayName,
  nextHref,
  nextLabel,
}: {
  game: GameDefinition;
  reduce: boolean;
  userId: string | null;
  displayName: string | null;
  nextHref: string;
  nextLabel: string;
}) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-void/95 px-6 backdrop-blur-sm"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.22 }}
      role="dialog"
      aria-label="Chapter complete"
    >
      <motion.div
        data-world={game.world}
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto border border-line bg-panel p-6"
        style={{ borderRadius: 'var(--radius-lg)' }}
        initial={reduce ? false : { y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: reduce ? 0 : 0.32, ease: [0.16, 1, 0.3, 1] }}
      >
        <p className="label mb-3">chapter complete</p>
        <Heading level={2} className="mb-4">
          {game.chapterTitle}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">{game.concept.ahaMoment}</p>

        <div className="mb-6 flex flex-wrap items-center gap-2">
          <Link href={nextHref}>
            <Button variant="primary">
              {nextLabel}
              <ArrowRight size={13} strokeWidth={2} />
            </Button>
          </Link>
          <Link href="/map">
            <Button>Map</Button>
          </Link>
        </div>

        {userId && <ChapterFeedback chapterId={game.id} userId={userId} displayName={displayName} />}
      </motion.div>
    </motion.div>
  );
}
