'use client';

/**
 * The frame every chapter plays inside.
 *
 * Layout follows Section 6: the visualisation owns the screen and the HUD sits
 * around it — a collapsible sheet on mobile, a persistent side rail from
 * tablet up. The shell owns the concept panel, level progression, scoring and
 * the aha reveal; the game component owns only its own canvas and controls.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { ArrowLeft, ArrowRight, Lightbulb, PanelRightClose, PanelRightOpen, RotateCcw } from 'lucide-react';
import type { GameDefinition, GameLevel, ScoreResult } from '@/types/game';
import { nextChapterAfter, worldOfChapter } from '@/lib/curriculum';
import { useGameProgressStore } from '@/store/useGameProgressStore';
import { useSessionStore } from '@/store/useSessionStore';
import { enqueueActivity } from '@/lib/offlineQueue';
import { Button, Heading, Panel, Readout, StarRating, Tag, cx } from '@/components/ui';

export interface GameRenderProps {
  level: GameLevel;
  /** Reports the current score so the HUD can track it live. */
  onScore: (result: ScoreResult) => void;
  /** Submits the run for scoring and progression. */
  onSubmit: (result: ScoreResult) => void;
}

export interface ChapterShellProps {
  game: GameDefinition;
  /** Renders the level's own canvas and controls. */
  children: (props: GameRenderProps) => ReactNode;
}

export function ChapterShell({ game, children }: ChapterShellProps) {
  const reduce = useReducedMotion();
  const world = worldOfChapter(game.id);
  const nextChapter = nextChapterAfter(game.id);

  const userId = useGameProgressStore((s) => s.userId);
  const recordLevel = useGameProgressStore((s) => s.recordLevel);
  const completeChapter = useGameProgressStore((s) => s.completeChapter);

  const levelIndex = useSessionStore((s) => s.levelIndex);
  const phase = useSessionStore((s) => s.phase);
  const attempts = useSessionStore((s) => s.attempts);
  const beginChapter = useSessionStore((s) => s.beginChapter);
  const goToLevel = useSessionStore((s) => s.goToLevel);
  const setPhase = useSessionStore((s) => s.setPhase);
  const registerAttempt = useSessionStore((s) => s.registerAttempt);
  const recordResult = useSessionStore((s) => s.recordResult);

  const [liveScore, setLiveScore] = useState<ScoreResult | null>(null);
  const [hudOpen, setHudOpen] = useState(false);
  const [chapterDone, setChapterDone] = useState(false);
  const [hintsRevealed, setHintsRevealed] = useState(0);

  const level = game.levels[levelIndex];
  const requiredLevels = useMemo(() => game.levels.filter((l) => !l.optional), [game.levels]);

  useEffect(() => {
    beginChapter(game.id);
    if (userId) void enqueueActivity({ userId, type: 'chapter_started', chapterId: game.id });
  }, [game.id, userId, beginChapter]);

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
    [level, game.id, userId, attempts, registerAttempt, recordResult, recordLevel]
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

    if (userId) {
      void enqueueActivity({
        userId,
        type: 'chapter_completed',
        chapterId: game.id,
        detail: { stars: chapterStars },
      });
    }
  }, [levelIndex, requiredLevels, game.id, userId, goToLevel, completeChapter]);

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
          onToggleHud={() => setHudOpen((open) => !open)}
          hudOpen={hudOpen}
        />

        <div className="relative flex min-h-0 flex-1 flex-col">
          {phase === 'concept' ? (
            <ConceptPanel game={game} level={level} onStart={() => setPhase('playing')} />
          ) : (
            children({ level, onScore: setLiveScore, onSubmit: handleSubmit })
          )}
        </div>
      </div>

      {/* ── HUD: sheet on mobile, rail from lg up ── */}
      <aside
        className={cx(
          'border-line bg-panel/40',
          'lg:flex lg:w-[320px] lg:shrink-0 lg:flex-col lg:border-l',
          hudOpen ? 'flex flex-col border-t' : 'hidden lg:flex'
        )}
      >
        <Hud
          game={game}
          level={level}
          levelIndex={levelIndex}
          liveScore={liveScore}
          attempts={attempts}
          phase={phase}
          hintsRevealed={hintsRevealed}
          onRevealHint={() =>
            setHintsRevealed((n) => Math.min(n + 1, level.hints?.length ?? 0))
          }
          onRetry={() => {
            setLiveScore(null);
            setPhase('playing');
          }}
          onAdvance={advance}
        />
      </aside>

      <AnimatePresence>
        {chapterDone && (
          <AhaReveal
            game={game}
            reduce={reduce ?? false}
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
  onToggleHud,
  hudOpen,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  totalLevels: number;
  worldNumber: number;
  onToggleHud: () => void;
  hudOpen: boolean;
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

      <span className="label whitespace-nowrap">
        {levelIndex + 1}/{totalLevels} · {level.difficulty}
      </span>

      <button
        type="button"
        onClick={onToggleHud}
        aria-label={hudOpen ? 'Hide controls' : 'Show controls'}
        className="p-1 text-muted transition-colors hover:text-primary lg:hidden"
      >
        {hudOpen ? <PanelRightClose size={15} strokeWidth={1.75} /> : <PanelRightOpen size={15} strokeWidth={1.75} />}
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
  hintsRevealed,
  onRevealHint,
  onRetry,
  onAdvance,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  liveScore: ScoreResult | null;
  attempts: number;
  phase: string;
  hintsRevealed: number;
  onRevealHint: () => void;
  onRetry: () => void;
  onAdvance: () => void;
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
          {game.levels.map((l, index) => (
            <li
              key={l.id}
              className={cx(
                'flex items-center justify-between gap-2 text-[11px]',
                index === levelIndex ? 'text-accent' : 'text-muted'
              )}
            >
              <span className="truncate">
                {index + 1}. {l.title}
              </span>
              {l.optional && <Tag>stretch</Tag>}
            </li>
          ))}
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

  return (
    <Panel label="hints" actions={<span className="label">{revealed}/{hints.length}</span>}>
      <div className="flex flex-col gap-3">
        {shown.length > 0 && (
          <ol className="flex flex-col gap-2.5">
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
  nextHref,
  nextLabel,
}: {
  game: GameDefinition;
  reduce: boolean;
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
        className="w-full max-w-lg border border-line bg-panel p-6"
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

        <div className="flex flex-wrap items-center gap-2">
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
      </motion.div>
    </motion.div>
  );
}
