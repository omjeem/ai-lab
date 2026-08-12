'use client';

/**
 * Chapter 1.2 — Vector Arithmetic.
 *
 * The instrument is the equation itself. Every term carries a strip of its own
 * real coordinates, and the result strip beneath the `=` is the live sum, so the
 * arithmetic is something you watch happen rather than something you are told
 * about. Nothing here holds an answer key: the ranking comes from the engine,
 * which recomputes cosine similarity over live embeddings on every change.
 *
 * The three modes share the equation and differ only in what is being asked —
 * guess the nearest neighbour, build the terms yourself, or estimate the
 * similarity before it is shown.
 *
 * Keyboard operability (Section 11.5): candidates are buttons, operators are
 * buttons, estimates are native range inputs. There is nothing drag-only here.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, Eye, RotateCcw, Sigma } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ActiveTerm,
  type RankedCandidate,
  type VectorArithmeticAction,
  type VectorArithmeticConfig,
  type VectorArithmeticState,
} from '@/engines/vectorArithmeticEngine';
import { norm } from '@/engines/shared';
import { embeddingModel, EMBEDDING_MODEL_ID } from '@/models/embeddingModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Slider, Tag, cx } from '@/components/ui';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

/** Dimensions drawn in a coordinate strip. 384 bars would be a texture, not a reading. */
const TRACE_DIMS = 48;

export function ArithmeticCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as VectorArithmeticConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<VectorArithmeticState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { embedder: embeddingModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  /**
   * Every action reduces from the latest state rather than one captured at call
   * time — a term's embedding resolves asynchronously, so an in-flight word can
   * land after the player has already changed an operator.
   */
  const dispatch = useCallback((action: VectorArithmeticAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={EMBEDDING_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: VectorArithmeticState;
  dispatch: (action: VectorArithmeticAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  // The HUD tracks the score live, so every action updates it.
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const editable = state.mode === 'free-terms';
  const [drafts, setDrafts] = useState<string[]>(() => state.terms.map((t) => t.word ?? ''));
  const [pending, setPending] = useState<number | null>(null);
  const [embedError, setEmbedError] = useState<string | null>(null);

  /** Embeds whatever the player typed and folds it into the chain. */
  const commitTerm = useCallback(
    async (index: number, raw: string) => {
      const word = raw.trim().toLowerCase();
      if (!word || word === state.terms[index]?.word) return;

      setPending(index);
      setEmbedError(null);
      try {
        const [vector] = await embeddingModel.embed([word]);
        if (vector) dispatch({ type: 'SET_TERM', index, word, vector });
      } catch (caught) {
        setEmbedError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setPending(null);
      }
    },
    [dispatch, state.terms]
  );

  const reset = () => {
    dispatch({ type: 'RESET' });
    setDrafts(state.terms.map(() => ''));
    setEmbedError(null);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* ── the instrument: the equation and whatever it is being asked ── */}
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {/* `my-auto` centres a short instrument without clipping a tall one. */}
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          <Equation
            terms={state.terms}
            resultVector={state.resultVector}
            editable={editable}
            drafts={drafts}
            pending={pending}
            onDraft={(index, value) =>
              setDrafts((prev) => prev.map((d, i) => (i === index ? value : d)))
            }
            onCommit={(index, value) => void commitTerm(index, value)}
            onToggleOp={(index, op) => dispatch({ type: 'SET_OP', index, op })}
          />

          {embedError && (
            <p className="readout text-xs text-bad">could not embed that word — {embedError}</p>
          )}

          {state.mode === 'fixed-analogy' && <AnalogyBoard state={state} dispatch={dispatch} />}
          {state.mode === 'free-terms' && <FreeTermsBoard state={state} />}
          {state.mode === 'estimate-similarity' && (
            <EstimateBoard state={state} dispatch={dispatch} />
          )}
        </div>
      </div>

      {/* ── controls: secondary layer under the instrument ── */}
      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={reset} title="Clear the run and start this level over">
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Actions state={state} dispatch={dispatch} onSubmit={onSubmit} />
      </div>
    </div>
  );
}

/* ── the equation ───────────────────────────────────────────── */

function Equation({
  terms,
  resultVector,
  editable,
  drafts,
  pending,
  onDraft,
  onCommit,
  onToggleOp,
}: {
  terms: ActiveTerm[];
  resultVector: number[] | null;
  editable: boolean;
  drafts: string[];
  pending: number | null;
  onDraft: (index: number, value: string) => void;
  onCommit: (index: number, value: string) => void;
  onToggleOp: (index: number, op: 'add' | 'subtract') => void;
}) {
  return (
    <div className="flex flex-wrap items-stretch justify-center gap-2">
      {terms.map((term, index) => (
        <div key={index} className="flex items-stretch gap-2">
          <Operator
            op={term.op}
            // A leading `+` is noise; a leading `−` negates the term and is not.
            hidden={index === 0 && term.op === 'add' && !editable}
            editable={editable}
            onToggle={() => onToggleOp(index, term.op === 'add' ? 'subtract' : 'add')}
            label={`Term ${index + 1} sign`}
          />
          <TermChip
            term={term}
            editable={editable}
            draft={drafts[index] ?? ''}
            busy={pending === index}
            onDraft={(value) => onDraft(index, value)}
            onCommit={(value) => onCommit(index, value)}
          />
        </div>
      ))}

      <div className="flex items-stretch gap-2">
        <Operator op="equals" editable={false} onToggle={() => {}} label="equals" />
        <ResultChip vector={resultVector} />
      </div>
    </div>
  );
}

function Operator({
  op,
  hidden,
  editable,
  onToggle,
  label,
}: {
  op: 'add' | 'subtract' | 'equals';
  hidden?: boolean;
  editable: boolean;
  onToggle: () => void;
  label: string;
}) {
  const glyph = op === 'add' ? '+' : op === 'subtract' ? '−' : '=';

  if (hidden) return <span className="w-4" aria-hidden />;

  if (!editable) {
    return (
      <span
        className={cx(
          'readout flex w-5 items-center justify-center text-xl',
          op === 'equals' ? 'text-accent' : 'text-secondary'
        )}
        aria-hidden
      >
        {glyph}
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={`${label}: currently ${op === 'add' ? 'plus' : 'minus'}. Toggle.`}
      className={cx(
        'readout flex w-7 items-center justify-center border text-xl transition-colors',
        'border-line text-secondary hover:border-accent hover:text-accent'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {glyph}
    </button>
  );
}

function TermChip({
  term,
  editable,
  draft,
  busy,
  onDraft,
  onCommit,
}: {
  term: ActiveTerm;
  editable: boolean;
  draft: string;
  busy: boolean;
  onDraft: (value: string) => void;
  onCommit: (value: string) => void;
}) {
  return (
    <div
      className="flex w-[132px] flex-col gap-1.5 border border-line-strong bg-raised px-2 py-2"
      style={{ borderRadius: 'var(--radius)' }}
    >
      {editable ? (
        <input
          value={draft}
          onChange={(event) => onDraft(event.target.value)}
          onBlur={() => onCommit(draft)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') onCommit(draft);
          }}
          placeholder="any word"
          maxLength={40}
          aria-label="Term word"
          className="w-full bg-transparent font-mono text-xs text-primary placeholder:text-muted focus:outline-none"
        />
      ) : (
        <span className="truncate font-mono text-xs text-primary">{term.word}</span>
      )}

      <Trace vector={term.vector} tone="muted" height={20} />

      <span className="label text-[9px]">
        {busy ? 'embedding…' : term.vector ? `${term.vector.length}d` : 'empty'}
      </span>
    </div>
  );
}

function ResultChip({ vector }: { vector: number[] | null }) {
  return (
    <div
      className="flex min-w-[180px] flex-1 flex-col gap-1.5 border border-accent/40 bg-accent-faint px-2 py-2"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <span className="flex items-center gap-1.5 font-mono text-xs text-accent">
        <Sigma size={11} strokeWidth={2} />
        result
      </span>

      <Trace vector={vector} tone="accent" height={20} />

      <span className="label text-[9px]">
        {vector ? `‖v‖ ${norm(vector).toFixed(3)}` : 'set every term'}
      </span>
    </div>
  );
}

/**
 * A vector's leading coordinates, drawn signed around zero.
 *
 * Scaled by the largest magnitude in the slice, so a strip shows shape rather
 * than absolute size — the readout beside it carries the actual norm.
 */
function Trace({
  vector,
  tone,
  height,
}: {
  vector: number[] | null;
  tone: 'accent' | 'muted';
  height: number;
}) {
  if (!vector || vector.length === 0) {
    return <div className="w-full bg-inset" style={{ height, borderRadius: '2px' }} aria-hidden />;
  }

  const slice = vector.slice(0, TRACE_DIMS);
  const scale = Math.max(...slice.map(Math.abs)) || 1;
  const mid = height / 2;
  const fill = tone === 'accent' ? 'var(--accent)' : 'var(--text-muted)';

  return (
    <svg
      viewBox={`0 0 ${TRACE_DIMS} ${height}`}
      preserveAspectRatio="none"
      className="block w-full"
      style={{ height }}
      role="img"
      aria-label={`First ${slice.length} of ${vector.length} coordinates`}
    >
      <line x1={0} y1={mid} x2={TRACE_DIMS} y2={mid} stroke="var(--line)" strokeWidth={0.4} />
      {slice.map((value, i) => {
        const magnitude = (Math.abs(value) / scale) * (mid - 0.5);
        return (
          <rect
            key={i}
            x={i + 0.15}
            y={value >= 0 ? mid - magnitude : mid}
            width={0.7}
            height={Math.max(magnitude, 0.3)}
            fill={fill}
            opacity={value >= 0 ? 0.9 : 0.5}
          />
        );
      })}
    </svg>
  );
}

/* ── mode: fixed analogy ────────────────────────────────────── */

function AnalogyBoard({
  state,
  dispatch,
}: {
  state: VectorArithmeticState;
  dispatch: (action: VectorArithmeticAction) => void;
}) {
  if (state.revealed) {
    return (
      <Panel label={`ranked by cosine similarity to the result · ${state.ranked.length} candidates`}>
        <RankedList ranked={state.ranked} highlight={state.guess} />
      </Panel>
    );
  }

  return (
    <Panel label="which candidate is nearest to the result?">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-1.5">
          {state.config.candidatePool.map((word) => {
            const chosen = state.guess === word;
            return (
              <button
                key={word}
                type="button"
                onClick={() => dispatch({ type: 'GUESS', word })}
                aria-pressed={chosen}
                className={cx(
                  'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors',
                  chosen
                    ? 'border-accent bg-accent-dim text-primary'
                    : 'border-line-strong bg-raised text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {word}
              </button>
            );
          })}
        </div>
        <p className="text-xs leading-relaxed text-muted">
          The candidates are listed in the order they were written, not by similarity. Commit to a
          guess, then reveal the real ranking — top three all score, first place scores most.
        </p>
      </div>
    </Panel>
  );
}

/* ── mode: free terms ───────────────────────────────────────── */

function FreeTermsBoard({ state }: { state: VectorArithmeticState }) {
  const target = state.config.target ?? '';
  const rank = state.ranked.findIndex((r) => r.word === target) + 1;
  const maxAttempts = state.config.maxAttempts ?? Infinity;
  const shown = state.ranked.slice(0, state.config.topK);
  const targetOutside = rank > state.config.topK;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <Readout
          label={`rank of "${target}"`}
          value={rank > 0 ? `${rank} / ${state.ranked.length}` : '—'}
          size="md"
          tone={rank === 1 ? 'good' : rank > 0 ? 'accent' : 'neutral'}
        />
        <Readout
          label="attempts committed"
          value={
            Number.isFinite(maxAttempts) ? `${state.attempts} / ${maxAttempts}` : state.attempts
          }
          size="sm"
        />
        <Readout label="best locked score" value={state.bestTargetRankScore} size="sm" tone="accent" />
      </div>

      <Panel label={`nearest ${state.config.topK} of ${state.config.candidatePool.length}`}>
        {shown.length === 0 ? (
          <p className="text-xs text-muted">
            Fill all three terms to compute a result. Every word you type is embedded live.
          </p>
        ) : (
          <RankedList
            ranked={targetOutside ? [...shown, state.ranked[rank - 1]!] : shown}
            highlight={target}
            gapBefore={targetOutside ? rank : undefined}
          />
        )}
      </Panel>

      <p className="text-xs leading-relaxed text-muted">
        Committing locks the current ranking into your score and spends an attempt. Only your best
        attempt counts, so a worse experiment can never cost you progress.
      </p>
    </div>
  );
}

/* ── mode: estimate similarity ──────────────────────────────── */

function EstimateBoard({
  state,
  dispatch,
}: {
  state: VectorArithmeticState;
  dispatch: (action: VectorArithmeticAction) => void;
}) {
  return (
    <Panel
      label={
        state.revealed
          ? 'your estimate against the real cosine similarity'
          : 'estimate the cosine similarity to the result'
      }
    >
      <div className="flex flex-col gap-4">
        {state.rounds.map((round, index) => {
          const error =
            round.estimate === null ? null : Math.abs(round.estimate - round.actualSimilarity);
          return (
            <div key={round.word} className="flex flex-col gap-1.5">
              <Slider
                label={round.word}
                value={round.estimate ?? 0}
                min={-1}
                max={1}
                step={0.01}
                onChange={(value) => dispatch({ type: 'ESTIMATE', roundIndex: index, value })}
                format={(value) => value.toFixed(2)}
                disabled={state.revealed}
              />

              {state.revealed && (
                <div className="flex items-baseline justify-between gap-3">
                  <span className="label">
                    actual {round.actualSimilarity.toFixed(3)}
                  </span>
                  <span
                    className={cx(
                      'readout text-xs',
                      error === null ? 'text-muted' : error <= 0.06 ? 'text-good' : error <= 0.2 ? 'text-warn' : 'text-bad'
                    )}
                  >
                    {error === null ? 'not estimated' : `error ${error.toFixed(3)}`}
                  </span>
                </div>
              )}
            </div>
          );
        })}

        {!state.revealed && (
          <p className="text-xs leading-relaxed text-muted">
            Cosine similarity runs from −1 to 1. Estimate every round before revealing — a round left
            unanswered is scored as the maximum possible error.
          </p>
        )}
      </div>
    </Panel>
  );
}

/* ── ranked list ────────────────────────────────────────────── */

function RankedList({
  ranked,
  highlight,
  gapBefore,
}: {
  ranked: RankedCandidate[];
  highlight: string | null;
  /** Rank of a row pulled in from outside the visible window, drawn after a break. */
  gapBefore?: number;
}) {
  const reduce = useReducedMotion();

  return (
    <ol className="flex flex-col gap-2">
      {ranked.map((candidate, index) => {
        const outsider = gapBefore !== undefined && index === ranked.length - 1;
        const rank = outsider ? gapBefore : index + 1;
        const marked = candidate.word === highlight;

        return (
          <li key={candidate.word} className="flex flex-col gap-1">
            {outsider && <span className="label text-center text-[9px]">···</span>}
            <div className="flex items-center gap-3">
              <span className="readout w-5 shrink-0 text-[10px] text-muted">{rank}</span>
              <span
                className={cx(
                  'w-24 shrink-0 truncate font-mono text-xs',
                  marked ? 'text-accent' : 'text-primary'
                )}
              >
                {candidate.word}
              </span>

              <div className="min-w-0 flex-1">
                <div className="relative h-1.5 w-full bg-inset">
                  <motion.div
                    className={cx('absolute inset-y-0 left-0', marked ? 'bg-accent' : 'bg-line-strong')}
                    initial={false}
                    animate={{ width: `${Math.max(0, Math.min(1, candidate.similarity)) * 100}%` }}
                    transition={reduce ? { duration: 0 } : { duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
              </div>

              <span
                className={cx(
                  'readout w-14 shrink-0 text-right text-xs',
                  marked ? 'text-accent' : 'text-secondary'
                )}
              >
                {candidate.similarity.toFixed(3)}
              </span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  dispatch,
  onSubmit,
}: {
  state: VectorArithmeticState;
  dispatch: (action: VectorArithmeticAction) => void;
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const submit = (
    <Button
      variant="primary"
      className="ml-auto"
      disabled={submitBlocked(state)}
      title={submitHint(state)}
      onClick={() => onSubmit(evaluate(state))}
    >
      <Check size={13} strokeWidth={2} />
      Submit
    </Button>
  );

  if (state.mode === 'fixed-analogy') {
    return (
      <>
        <Button
          disabled={state.guess === null || state.revealed}
          title="Locks your guess and shows the real ranking"
          onClick={() => dispatch({ type: 'REVEAL' })}
        >
          <Eye size={13} strokeWidth={2} />
          Reveal ranking
        </Button>
        {state.guess === null && <span className="label">pick a candidate first</span>}
        {submit}
      </>
    );
  }

  if (state.mode === 'free-terms') {
    const spent = state.attempts >= (state.config.maxAttempts ?? Infinity);
    return (
      <>
        <Button
          disabled={state.resultVector === null || spent}
          title={spent ? 'No attempts left' : 'Locks the current ranking into your score'}
          onClick={() => dispatch({ type: 'COMMIT_ATTEMPT' })}
        >
          <Check size={13} strokeWidth={2} />
          Commit attempt
        </Button>
        {spent && <Tag tone="warn">no attempts left</Tag>}
        {submit}
      </>
    );
  }

  const unanswered = state.rounds.filter((r) => r.estimate === null).length;
  return (
    <>
      <Button
        disabled={state.revealed || unanswered > 0}
        title="Shows the real cosine similarities"
        onClick={() => dispatch({ type: 'REVEAL' })}
      >
        <Eye size={13} strokeWidth={2} />
        Reveal actuals
      </Button>
      {unanswered > 0 && (
        <span className="label">
          {unanswered} round{unanswered === 1 ? '' : 's'} left to estimate
        </span>
      )}
      {submit}
    </>
  );
}

/** Submitting a run that cannot score anything is a mistake, not a choice. */
function submitBlocked(state: VectorArithmeticState): boolean {
  switch (state.mode) {
    case 'fixed-analogy':
      return state.guess === null;
    case 'free-terms':
      return state.attempts === 0;
    case 'estimate-similarity':
      return state.rounds.some((r) => r.estimate === null);
  }
}

function submitHint(state: VectorArithmeticState): string | undefined {
  if (!submitBlocked(state)) return undefined;
  return {
    'fixed-analogy': 'Pick a candidate first',
    'free-terms': 'Commit at least one attempt first',
    'estimate-similarity': 'Estimate every round first',
  }[state.mode];
}
