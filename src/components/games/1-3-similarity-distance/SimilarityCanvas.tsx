'use client';

/**
 * Chapter 1.3 — Similarity & Distance.
 *
 * Three instruments over one idea. Ranking and odd-one-out ask the player to
 * commit before the geometry is shown, because a visible similarity matrix
 * turns either into a copying exercise; the matrix is the reward for guessing,
 * and the place the intuition gets corrected. The free set is the opposite —
 * its whole subject is the disagreement between two metrics, so both matrices
 * are live from the first word.
 *
 * Nothing here holds an answer key. The true ranking, the outlier and the
 * disagreements all come from the engine, which derives them from live vectors.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { ArrowDown, ArrowUp, Check, GripVertical, Plus, RotateCcw, X } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type DisagreementQuestion,
  type OddOneOutSet,
  type SimilarityRankAction,
  type SimilarityRankConfig,
  type SimilarityRankState,
} from '@/engines/similarityRankEngine';
import { cosineSimilarity, euclideanDistance, norm } from '@/engines/shared';
import {
  embeddingModel,
  rawEmbeddingModel,
  preloadEmbeddingModel,
  EMBEDDING_MODEL_ID,
} from '@/models/embeddingModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Tag, cx } from '@/components/ui';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

/**
 * Levels that weigh two metrics against each other need the magnitudes kept.
 * `metric: 'both'` in the level JSON is what says so.
 */
function embedderFor(config: SimilarityRankConfig) {
  return config.metric === 'both' ? rawEmbeddingModel : embeddingModel;
}

export function SimilarityCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as SimilarityRankConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<SimilarityRankState | null>(null);

  const load = useCallback(async () => {
    // The free set starts with no words, so `prepare` embeds nothing and the
    // gate would pass without ever fetching the model. Warm it explicitly, or
    // the first word the player adds triggers a silent 23MB download.
    await preloadEmbeddingModel();
    const prepared = await prepare(config, { embedder: embedderFor(config) });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: SimilarityRankAction) => {
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
  state: SimilarityRankState;
  dispatch: (action: SimilarityRankAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  // The engine's own `complete` status drives the reveal, so "what the model
  // really thinks" appears exactly once the answer is locked in.
  const revealed = state.status === 'complete';

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'rank' && (
            <RankBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'odd-one-out' && (
            <OddOneOutBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'free-set' && <FreeSetBoard state={state} dispatch={dispatch} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={() => dispatch({ type: 'RESET' })}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Actions state={state} revealed={revealed} onSubmit={submit} />
      </div>
    </div>
  );
}

/* ── level 1: rank by similarity ────────────────────────────── */

function RankBoard({
  state,
  dispatch,
  revealed,
}: {
  state: SimilarityRankState;
  dispatch: (action: SimilarityRankAction) => void;
  revealed: boolean;
}) {
  const anchor = state.config.anchor ?? '';
  const dragging = useRef<number | null>(null);

  const truePosition = useMemo(
    () => new Map(state.trueOrder.map((word, i) => [word, i])),
    [state.trueOrder]
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col items-center gap-1">
        <span className="label">anchor</span>
        <div
          className="border border-accent/50 bg-accent-faint px-4 py-2 font-mono text-sm text-accent"
          style={{ borderRadius: 'var(--radius)' }}
        >
          {anchor}
        </div>
        <p className="mt-1 text-xs text-muted">nearest at the top</p>
      </div>

      <Panel label={revealed ? 'your order against the model’s' : 'drag into order, nearest first'}>
        <ol className="flex flex-col gap-1.5">
          {state.ordering.map((word, index) => {
            const actual = truePosition.get(word) ?? 0;
            const offBy = revealed ? index - actual : 0;

            return (
              <li
                key={word}
                draggable={!revealed}
                onDragStart={() => (dragging.current = index)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  const from = dragging.current;
                  dragging.current = null;
                  if (from !== null && from !== index) {
                    dispatch({ type: 'MOVE_ITEM', from, to: index });
                  }
                }}
                className={cx(
                  'flex items-center gap-3 border px-2 py-2',
                  revealed && offBy === 0
                    ? 'border-good/40 bg-good/5'
                    : revealed
                      ? 'border-line bg-raised'
                      : 'border-line-strong bg-raised'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                <span className="readout w-5 shrink-0 text-xs text-muted">{index + 1}</span>

                {!revealed && (
                  <span className="text-muted" aria-hidden>
                    <GripVertical size={13} strokeWidth={1.75} />
                  </span>
                )}

                <span className="min-w-0 flex-1 truncate font-mono text-sm text-primary">
                  {word}
                </span>

                {revealed ? (
                  <RevealedRank
                    word={word}
                    anchor={anchor}
                    vectors={state.vectors}
                    actualPosition={actual}
                    offBy={offBy}
                  />
                ) : (
                  /* Dragging is the fast path; these are the whole path on a
                     keyboard, per Section 11.5. */
                  <span className="flex shrink-0 items-center gap-1">
                    <MoveButton
                      label={`Move ${word} up`}
                      disabled={index === 0}
                      onClick={() => dispatch({ type: 'MOVE_ITEM', from: index, to: index - 1 })}
                    >
                      <ArrowUp size={12} strokeWidth={2} />
                    </MoveButton>
                    <MoveButton
                      label={`Move ${word} down`}
                      disabled={index === state.ordering.length - 1}
                      onClick={() => dispatch({ type: 'MOVE_ITEM', from: index, to: index + 1 })}
                    >
                      <ArrowDown size={12} strokeWidth={2} />
                    </MoveButton>
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </Panel>

      {revealed && (
        <Heatmap
          label="cosine similarity, every pair"
          words={[anchor, ...state.trueOrder]}
          vectors={state.vectors}
          metric="cosine"
          highlightRow={anchor}
        />
      )}
    </div>
  );
}

function RevealedRank({
  word,
  anchor,
  vectors,
  actualPosition,
  offBy,
}: {
  word: string;
  anchor: string;
  vectors: Record<string, number[]>;
  actualPosition: number;
  offBy: number;
}) {
  const similarity = cosineSimilarity(vectors[anchor] ?? [], vectors[word] ?? []);
  return (
    <span className="flex shrink-0 items-center gap-3">
      <span className="readout text-xs text-secondary">{similarity.toFixed(3)}</span>
      <span className="label w-20 text-right">
        {offBy === 0 ? 'exact' : `really #${actualPosition + 1}`}
      </span>
    </span>
  );
}

function MoveButton({
  children,
  label,
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        'flex h-7 w-7 items-center justify-center border border-line text-muted transition-colors',
        'hover:border-accent hover:text-accent disabled:opacity-30 disabled:hover:border-line disabled:hover:text-muted'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {children}
    </button>
  );
}

/* ── level 2: odd one out ───────────────────────────────────── */

function OddOneOutBoard({
  state,
  dispatch,
  revealed,
}: {
  state: SimilarityRankState;
  dispatch: (action: SimilarityRankAction) => void;
  revealed: boolean;
}) {
  return (
    <div className="flex flex-col gap-4">
      {state.sets.map((set, index) => (
        <OddOneOutSetPanel
          key={set.words.join('|')}
          set={set}
          index={index}
          vectors={state.vectors}
          revealed={revealed}
          onAnswer={(word) => dispatch({ type: 'ANSWER_ODD', setIndex: index, word })}
        />
      ))}
      {!revealed && (
        <p className="text-xs leading-relaxed text-muted">
          The outlier is whichever word has the lowest mean cosine similarity to the other three,
          computed live. Commit to all three sets, then the full matrix opens up.
        </p>
      )}
    </div>
  );
}

function OddOneOutSetPanel({
  set,
  index,
  vectors,
  revealed,
  onAnswer,
}: {
  set: OddOneOutSet;
  index: number;
  vectors: Record<string, number[]>;
  revealed: boolean;
  onAnswer: (word: string) => void;
}) {
  const correct = set.answer === set.trueOutlier;

  /** Mean similarity of each word to the rest — the quantity being ranked. */
  const means = useMemo(
    () =>
      new Map(
        set.words.map((word) => {
          const others = set.words.filter((w) => w !== word);
          const total = others.reduce(
            (sum, other) => sum + cosineSimilarity(vectors[word] ?? [], vectors[other] ?? []),
            0
          );
          return [word, others.length === 0 ? 0 : total / others.length];
        })
      ),
    [set.words, vectors]
  );

  return (
    <Panel
      label={`set ${index + 1}`}
      actions={
        revealed ? (
          <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : 'missed'}</Tag>
        ) : set.answer ? (
          <Tag tone="accent">answered</Tag>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-1.5">
          {set.words.map((word) => {
            const chosen = set.answer === word;
            const truth = revealed && word === set.trueOutlier;
            return (
              <button
                key={word}
                type="button"
                disabled={revealed}
                onClick={() => onAnswer(word)}
                aria-pressed={chosen}
                className={cx(
                  'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors disabled:cursor-default',
                  truth
                    ? 'border-good bg-good/10 text-good'
                    : chosen
                      ? revealed
                        ? 'border-bad bg-bad/10 text-bad'
                        : 'border-accent bg-accent-dim text-primary'
                      : 'border-line-strong bg-raised text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {word}
                {revealed && (
                  <span className="readout ml-2 text-[10px] opacity-70">
                    {(means.get(word) ?? 0).toFixed(3)}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {revealed && (
          <Heatmap
            label="cosine similarity within the set"
            words={set.words}
            vectors={vectors}
            metric="cosine"
            highlightRow={set.trueOutlier}
            compact
          />
        )}
      </div>
    </Panel>
  );
}

/* ── level 3: cosine vs euclidean ───────────────────────────── */

function FreeSetBoard({
  state,
  dispatch,
}: {
  state: SimilarityRankState;
  dispatch: (action: SimilarityRankAction) => void;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const min = state.config.minWords ?? 3;
  const max = state.config.maxWords ?? 8;
  const full = state.words.length >= max;

  const addWord = async () => {
    const word = draft.trim().toLowerCase();
    if (!word || busy || full || state.words.includes(word)) return;

    setBusy(true);
    setError(null);
    try {
      // Raw, unnormalised vectors — the two metrics only differ when length does.
      const [vector] = await rawEmbeddingModel.embed([word]);
      if (vector) dispatch({ type: 'ADD_WORD', word, vector });
      setDraft('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel
        label={`your set · ${state.words.length} of ${max}`}
        actions={<span className="label">{min} needed</span>}
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-1.5">
            {state.words.map((word) => (
              <span
                key={word}
                className="inline-flex items-center gap-2 border border-line-strong bg-raised px-2 py-1.5"
                style={{ borderRadius: 'var(--radius)' }}
              >
                <span className="font-mono text-xs text-primary">{word}</span>
                {/* Magnitude is the whole reason the metrics can disagree. */}
                <span className="readout text-[10px] text-muted">
                  ‖v‖ {norm(state.vectors[word] ?? []).toFixed(2)}
                </span>
                <button
                  type="button"
                  onClick={() => dispatch({ type: 'REMOVE_WORD', word })}
                  aria-label={`Remove ${word}`}
                  className="text-muted transition-colors hover:text-bad"
                >
                  <X size={12} strokeWidth={2} />
                </button>
              </span>
            ))}
            {state.words.length === 0 && (
              <p className="text-xs text-muted">Add {min} or more words to build the matrices.</p>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && void addWord()}
              placeholder={full ? 'set is full' : 'any word'}
              maxLength={40}
              disabled={full}
              aria-label="Word to add"
              className="w-40 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-50"
              style={{ borderRadius: 'var(--radius)' }}
            />
            <Button onClick={() => void addWord()} disabled={busy || full || draft.trim() === ''}>
              <Plus size={13} strokeWidth={2} />
              embed
            </Button>
          </div>

          {error && <p className="readout text-xs text-bad">could not embed that word — {error}</p>}
        </div>
      </Panel>

      {state.words.length >= 2 && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Heatmap
            label="cosine · higher is nearer"
            words={state.words}
            vectors={state.vectors}
            metric="cosine"
            compact
          />
          <Heatmap
            label="euclidean · lower is nearer"
            words={state.words}
            vectors={state.vectors}
            metric="euclidean"
            compact
          />
        </div>
      )}

      {state.questions.length > 0 ? (
        <Panel label="where do the two metrics disagree?">
          <div className="flex flex-col gap-3">
            {state.questions.map((question, index) => (
              <QuestionRow
                key={`${question.anchor}|${question.x}|${question.y}`}
                question={question}
                vectors={state.vectors}
                onAnswer={(value) =>
                  dispatch({ type: 'ANSWER_DISAGREEMENT', questionIndex: index, value })
                }
              />
            ))}
          </div>
        </Panel>
      ) : (
        <p className="text-xs leading-relaxed text-muted">
          Questions appear once the set reaches {min} words. They are drawn from your own words, and
          the answer is computed from the vectors — there is nothing to look up.
        </p>
      )}
    </div>
  );
}

/**
 * One triple, with the orderings the two metrics give it side by side.
 *
 * Both metrics are shown deliberately: the level's subject is that they can
 * diverge, and the divergence is only legible if you can watch both at once.
 */
function QuestionRow({
  question,
  vectors,
  onAnswer,
}: {
  question: DisagreementQuestion;
  vectors: Record<string, number[]>;
  onAnswer: (value: boolean) => void;
}) {
  const { anchor, x, y } = question;
  const cosX = cosineSimilarity(vectors[anchor] ?? [], vectors[x] ?? []);
  const cosY = cosineSimilarity(vectors[anchor] ?? [], vectors[y] ?? []);
  const eucX = euclideanDistance(vectors[anchor] ?? [], vectors[x] ?? []);
  const eucY = euclideanDistance(vectors[anchor] ?? [], vectors[y] ?? []);

  return (
    <div
      className="flex flex-col gap-2 border border-line bg-inset/40 p-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-xs leading-relaxed text-secondary">
        From <span className="font-mono text-accent">{anchor}</span> — do cosine and Euclidean
        disagree about whether <span className="font-mono text-primary">{x}</span> or{' '}
        <span className="font-mono text-primary">{y}</span> is nearer?
      </p>

      <div className="grid gap-2 sm:grid-cols-2">
        <MetricVerdict label="cosine picks" winner={cosX > cosY ? x : y} a={`${x} ${cosX.toFixed(3)}`} b={`${y} ${cosY.toFixed(3)}`} />
        <MetricVerdict label="euclidean picks" winner={eucX < eucY ? x : y} a={`${x} ${eucX.toFixed(2)}`} b={`${y} ${eucY.toFixed(2)}`} />
      </div>

      <div className="flex items-center gap-1.5">
        {[true, false].map((value) => (
          <button
            key={String(value)}
            type="button"
            onClick={() => onAnswer(value)}
            aria-pressed={question.answer === value}
            className={cx(
              'min-h-[36px] border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors',
              question.answer === value
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line text-muted hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {value ? 'they disagree' : 'they agree'}
          </button>
        ))}
      </div>
    </div>
  );
}

function MetricVerdict({
  label,
  winner,
  a,
  b,
}: {
  label: string;
  winner: string;
  a: string;
  b: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="label text-[9px]">
        {label} <span className="text-accent">{winner}</span>
      </span>
      <span className="readout text-[10px] text-muted">
        {a} · {b}
      </span>
    </div>
  );
}

/* ── heatmap ────────────────────────────────────────────────── */

/**
 * A pairwise metric matrix.
 *
 * Both metrics are drawn so that a brighter cell always means "nearer", which
 * is what makes a disagreement between them show up as a difference in pattern
 * rather than a difference in direction.
 */
function Heatmap({
  label,
  words,
  vectors,
  metric,
  highlightRow,
  compact,
}: {
  label: string;
  words: string[];
  vectors: Record<string, number[]>;
  metric: 'cosine' | 'euclidean';
  highlightRow?: string;
  compact?: boolean;
}) {
  const reduce = useReducedMotion();

  const { cells, min, max } = useMemo(() => {
    const cells = words.map((row) =>
      words.map((column) =>
        metric === 'cosine'
          ? cosineSimilarity(vectors[row] ?? [], vectors[column] ?? [])
          : euclideanDistance(vectors[row] ?? [], vectors[column] ?? [])
      )
    );
    const offDiagonal = cells.flatMap((row, i) => row.filter((_, j) => i !== j));
    return {
      cells,
      min: offDiagonal.length ? Math.min(...offDiagonal) : 0,
      max: offDiagonal.length ? Math.max(...offDiagonal) : 1,
    };
  }, [words, vectors, metric]);

  /** 0 = furthest, 1 = nearest, whichever direction the metric runs in. */
  const intensity = (value: number) => {
    if (max === min) return 0.5;
    const scaled = (value - min) / (max - min);
    return metric === 'cosine' ? scaled : 1 - scaled;
  };

  return (
    <Panel label={label} flush>
      <div className="overflow-x-auto p-2">
        <table className="w-full border-collapse">
          <caption className="sr-only">
            {metric} between every pair of {words.length} words
          </caption>
          <thead>
            <tr>
              <th className="p-1" />
              {words.map((word) => (
                <th
                  key={word}
                  scope="col"
                  className="p-1 text-center font-mono text-[9px] font-normal text-muted"
                >
                  {compact ? word.slice(0, 4) : word}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {words.map((row, i) => (
              <tr key={row}>
                <th
                  scope="row"
                  className={cx(
                    'whitespace-nowrap py-1 pr-2 text-right font-mono text-[10px] font-normal',
                    row === highlightRow ? 'text-accent' : 'text-secondary'
                  )}
                >
                  {row}
                </th>
                {words.map((column, j) => {
                  const value = cells[i]![j]!;
                  const self = i === j;
                  return (
                    <td key={column} className="p-[2px]">
                      <motion.div
                        initial={false}
                        animate={{ opacity: 1 }}
                        transition={reduce ? { duration: 0 } : { duration: 0.18 }}
                        title={`${row} → ${column}: ${value.toFixed(4)}`}
                        className="flex h-7 items-center justify-center"
                        style={{
                          borderRadius: '2px',
                          background: self
                            ? 'var(--surface-inset)'
                            : `color-mix(in oklab, var(--accent) ${Math.round(
                                intensity(value) * 82 + 6
                              )}%, var(--surface-inset))`,
                        }}
                      >
                        <span
                          className={cx(
                            'readout text-[9px]',
                            self ? 'text-muted' : intensity(value) > 0.55 ? 'text-inverse' : 'text-primary'
                          )}
                        >
                          {self ? '—' : value.toFixed(2)}
                        </span>
                      </motion.div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: SimilarityRankState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = submitBlocked(state);

  return (
    <>
      {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
      {revealed && <Tag tone="accent">answers revealed</Tag>}

      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || revealed}
        title={blocked ? submitHint(state) : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

/** Submitting a run that cannot be scored is a mistake, not a choice. */
function submitBlocked(state: SimilarityRankState): boolean {
  switch (state.mode) {
    case 'rank':
      return state.ordering.length < 2;
    case 'odd-one-out':
      return state.sets.some((set) => set.answer === null);
    case 'free-set':
      return (
        state.questions.length === 0 || state.questions.some((question) => question.answer === null)
      );
  }
}

function submitHint(state: SimilarityRankState): string {
  switch (state.mode) {
    case 'rank':
      return 'nothing to rank';
    case 'odd-one-out': {
      const left = state.sets.filter((set) => set.answer === null).length;
      return `${left} set${left === 1 ? '' : 's'} left`;
    }
    case 'free-set': {
      const min = state.config.minWords ?? 3;
      if (state.questions.length === 0) return `add ${Math.max(0, min - state.words.length)} more`;
      const left = state.questions.filter((question) => question.answer === null).length;
      return `${left} question${left === 1 ? '' : 's'} left`;
    }
  }
}
