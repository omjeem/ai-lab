'use client';

/**
 * Chapter 7.1 — Retrieval.
 *
 * Three boards over one real embedder. Ranking hides the geometry until the
 * player commits, same reveal discipline as `SimilarityCanvas`'s rank mode —
 * a visible similarity bar turns ranking into copying. "Break the retriever"
 * and "chunking strategy" are both live from the first interaction instead,
 * because their whole subject is watching the real number move as you act.
 * Nothing here holds an answer key: the true order, the margin and the
 * chunking precision are all read straight off `retrievalRankEngine`.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, Check, GripVertical, RotateCcw, Search } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  roundPrecision,
  type ChunkRound,
  type RetrievalFact,
  type RetrievalRankAction,
  type RetrievalRankConfig,
  type RetrievalRankState,
} from '@/engines/retrievalRankEngine';
import { cosineSimilarity } from '@/engines/shared';
import { embeddingModel, EMBEDDING_MODEL_ID } from '@/models/embeddingModel';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import { usePointerDrag } from '../usePointerDrag';
import { useDragReorder } from '../useDragReorder';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function RetrievalCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as RetrievalRankConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<RetrievalRankState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { corpus: corpusLoader, embedder: embeddingModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: RetrievalRankAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={EMBEDDING_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: RetrievalRankState;
  dispatch: (action: RetrievalRankAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status`: every action bumps status back to 'active', so
  // the rank board's own reordering after submit would un-reveal the answer.
  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    setRevealed(true);
  };

  const blocked = submitBlocked(state);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'rank' && <RankBoard state={state} dispatch={dispatch} revealed={revealed} />}
          {state.mode === 'break-retriever' && <BreakRetrieverBoard state={state} dispatch={dispatch} result={result} />}
          {state.mode === 'chunking' && <ChunkingBoard state={state} dispatch={dispatch} result={result} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
        {revealed && <Tag tone="accent">answers revealed</Tag>}
        <Button
          variant="primary"
          className="ml-auto"
          disabled={blocked || revealed}
          title={blocked ? submitHint(state) : undefined}
          onClick={submit}
        >
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

function submitBlocked(state: RetrievalRankState): boolean {
  switch (state.mode) {
    case 'rank':
      return state.ordering.length < 2;
    case 'break-retriever':
      return state.submittedVector === null;
    case 'chunking':
      return state.chunkRounds.some((r) => r.strategy === null);
  }
}

function submitHint(state: RetrievalRankState): string {
  switch (state.mode) {
    case 'rank':
      return 'nothing to rank';
    case 'break-retriever':
      return 'test a query first';
    case 'chunking': {
      const left = state.chunkRounds.filter((r) => r.strategy === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
  }
}

function factLookup(facts: readonly RetrievalFact[], id: string): RetrievalFact | undefined {
  return facts.find((f) => f.id === id);
}

/* ── level 1: rank by relevance ────────────────────────────────── */

function RankBoard({
  state,
  dispatch,
  revealed,
}: {
  state: RetrievalRankState;
  dispatch: (action: RetrievalRankAction) => void;
  revealed: boolean;
}) {
  const reorder = useDragReorder((from, to) => dispatch({ type: 'MOVE_ITEM', from, to }));
  const truePosition = useMemo(
    () => new Map(state.trueOrder.map((id, i) => [id, i])),
    [state.trueOrder]
  );
  const maxSim = useMemo(
    () => Math.max(...state.ordering.map((id) => cosineSimilarity(state.queryVector, state.factVectors[id] ?? [])), 1e-9),
    [state.ordering, state.queryVector, state.factVectors]
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col items-center gap-1 text-center">
        <span className="label">query</span>
        <div
          className="border border-accent/50 bg-accent-faint px-4 py-2 text-sm text-accent"
          style={{ borderRadius: 'var(--radius)' }}
        >
          {state.query}
        </div>
        <p className="mt-1 text-xs text-muted">most relevant at the top</p>
      </div>

      <Panel label={revealed ? 'your order against the real ranking' : 'drag into order, most relevant first'}>
        <ol className="flex flex-col gap-1.5">
          {state.ordering.map((factId, index) => {
            const fact = factLookup(state.facts, factId);
            if (!fact) return null;
            const actual = truePosition.get(factId) ?? 0;
            const offBy = revealed ? index - actual : 0;
            const similarity = cosineSimilarity(state.queryVector, state.factVectors[factId] ?? []);

            return (
              <RankRow
                key={factId}
                fact={fact}
                index={index}
                revealed={revealed}
                offBy={offBy}
                actual={actual}
                similarity={similarity}
                maxSim={maxSim}
                totalItems={state.ordering.length}
                dragging={reorder.dragIndex === index}
                dropTarget={reorder.overIndex === index && reorder.dragIndex !== null && reorder.dragIndex !== index}
                registerRef={reorder.registerItem(index)}
                onGripDragStart={() => reorder.startDrag(index)}
                onGripDragMove={reorder.dragTo}
                onGripDragEnd={reorder.dropAt}
                onMoveUp={() => dispatch({ type: 'MOVE_ITEM', from: index, to: index - 1 })}
                onMoveDown={() => dispatch({ type: 'MOVE_ITEM', from: index, to: index + 1 })}
              />
            );
          })}
        </ol>
      </Panel>
    </div>
  );
}

function RankRow({
  fact,
  index,
  revealed,
  offBy,
  actual,
  similarity,
  maxSim,
  totalItems,
  dragging,
  dropTarget,
  registerRef,
  onGripDragStart,
  onGripDragMove,
  onGripDragEnd,
  onMoveUp,
  onMoveDown,
}: {
  fact: RetrievalFact;
  index: number;
  revealed: boolean;
  offBy: number;
  actual: number;
  similarity: number;
  maxSim: number;
  totalItems: number;
  dragging: boolean;
  dropTarget: boolean;
  registerRef: (element: HTMLElement | null) => void;
  onGripDragStart: () => void;
  onGripDragMove: (clientY: number) => void;
  onGripDragEnd: (clientY: number) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}) {
  const grip = usePointerDrag({
    onDragStart: onGripDragStart,
    onDragMove: (_x, y) => onGripDragMove(y),
    onDragEnd: (_x, y) => onGripDragEnd(y),
    threshold: 6,
  });

  return (
    <li
      ref={registerRef}
      className={cx(
        'flex flex-col gap-2 border px-2 py-2 transition-opacity',
        dragging && 'opacity-40',
        dropTarget && 'border-accent',
        revealed && offBy === 0 ? 'border-good/40 bg-good/5' : 'border-line-strong bg-raised'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      <div className="flex items-center gap-3">
        <span className="readout w-5 shrink-0 text-xs text-muted">{index + 1}</span>

        {!revealed && (
          <span {...grip} style={{ ...grip.style, cursor: 'grab' }} className="text-muted active:cursor-grabbing">
            <GripVertical size={13} strokeWidth={1.75} aria-hidden />
          </span>
        )}

        <span className="min-w-0 flex-1 truncate font-mono text-sm text-primary">{fact.topic}</span>

        {revealed ? (
          <span className="flex shrink-0 items-center gap-3">
            <span className="readout text-xs text-secondary">{similarity.toFixed(3)}</span>
            <span className="label w-20 text-right">{offBy === 0 ? 'exact' : `really #${actual + 1}`}</span>
          </span>
        ) : (
          <span className="flex shrink-0 items-center gap-1">
            <MoveButton label={`Move ${fact.topic} up`} disabled={index === 0} onClick={onMoveUp}>
              <ArrowUp size={12} strokeWidth={2} />
            </MoveButton>
            <MoveButton label={`Move ${fact.topic} down`} disabled={index === totalItems - 1} onClick={onMoveDown}>
              <ArrowDown size={12} strokeWidth={2} />
            </MoveButton>
          </span>
        )}
      </div>

      {revealed && (
        <div
          className="relative h-1.5 w-full bg-inset"
          role="img"
          aria-label={`${fact.topic} similarity ${similarity.toFixed(3)}`}
        >
          <div
            className="absolute inset-y-0 left-0 bg-accent"
            style={{ width: `${Math.max(2, (similarity / maxSim) * 100)}%` }}
          />
        </div>
      )}
    </li>
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

/* ── level 2: break the retriever ──────────────────────────────── */

function BreakRetrieverBoard({
  state,
  dispatch,
  result,
}: {
  state: RetrievalRankState;
  dispatch: (action: RetrievalRankAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const target = factLookup(state.facts, state.targetFactId);

  const test = async () => {
    const query = draft.trim();
    if (!query || busy) return;
    setBusy(true);
    setError(null);
    try {
      const [vector] = await embeddingModel.embed([query]);
      if (vector) dispatch({ type: 'SUBMIT_QUERY', query, vector });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const ranked = useMemo(() => {
    if (!state.submittedVector) return [];
    return [...state.facts]
      .map((fact) => ({ fact, sim: cosineSimilarity(state.submittedVector!, state.factVectors[fact.id] ?? []) }))
      .sort((a, b) => b.sim - a.sim);
  }, [state.facts, state.factVectors, state.submittedVector]);

  return (
    <div className="flex flex-col gap-4">
      <Panel label="target">
        <p className="text-sm text-primary">
          Make <span className="text-accent">{target?.topic ?? state.targetFactId}</span> rank #1.
        </p>
      </Panel>

      <Panel label="your query">
        <div className="flex items-center gap-1.5">
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void test()}
            placeholder="type a query to test"
            maxLength={120}
            aria-label="Retrieval query"
            className="w-full min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={() => void test()} disabled={busy || draft.trim() === ''}>
            <Search size={13} strokeWidth={2} />
            test
          </Button>
        </div>
        {error && <p className="readout mt-2 text-xs text-bad">could not embed that query — {error}</p>}
      </Panel>

      {state.submittedQuery && (
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="retrievalMargin" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <span className="label max-w-[16rem] text-right">for “{state.submittedQuery}”</span>
            </div>
            <Meter value={Math.max(0, result.value)} max={Math.max(0.4, result.value)} threshold={0} tone={result.passed ? 'good' : 'accent'} />
          </div>
        </Panel>
      )}

      {ranked.length > 0 && (
        <Panel label="real similarity, every passage" flush>
          <ol className="flex flex-col gap-1.5 p-2">
            {ranked.map(({ fact, sim }, i) => (
              <li
                key={fact.id}
                className={cx(
                  'flex items-center gap-3 px-1 py-1',
                  fact.id === state.targetFactId && 'text-accent'
                )}
              >
                <span className="readout w-5 shrink-0 text-xs text-muted">{i + 1}</span>
                <span className="w-40 shrink-0 truncate font-mono text-xs">{fact.topic}</span>
                <div className="relative h-1.5 flex-1 bg-inset">
                  <div
                    className={cx('absolute inset-y-0 left-0', fact.id === state.targetFactId ? 'bg-accent' : 'bg-line-strong')}
                    style={{ width: `${Math.max(1, ((sim + 1) / 2) * 100)}%` }}
                  />
                </div>
                <span className="readout w-14 shrink-0 text-right text-xs text-secondary">{sim.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        </Panel>
      )}
    </div>
  );
}

/* ── level 3: chunking strategy ────────────────────────────────── */

function ChunkingBoard({
  state,
  dispatch,
  result,
}: {
  state: RetrievalRankState;
  dispatch: (action: RetrievalRankAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  return (
    <div className="flex flex-col gap-4">
      <Panel label="score">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <Readout label="retrievalPrecision" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
            <Readout label="answered" value={`${result.breakdown.answered ?? 0}/${result.breakdown.total ?? 0}`} size="sm" />
          </div>
          <Meter value={result.value} max={1} threshold={state.rules.passCriteria.threshold} tone={result.passed ? 'good' : 'accent'} />
        </div>
      </Panel>

      <Panel label="per query — choose paragraph or sentence chunking" flush>
        <ol className="flex flex-col gap-2 p-2">
          {state.chunkRounds.map((round, index) => (
            <ChunkRoundRow
              key={`${round.docId}-${round.factId}`}
              state={state}
              round={round}
              onSetStrategy={(strategy) => dispatch({ type: 'SET_STRATEGY', roundIndex: index, strategy })}
            />
          ))}
        </ol>
      </Panel>
    </div>
  );
}

function ChunkRoundRow({
  state,
  round,
  onSetStrategy,
}: {
  state: RetrievalRankState;
  round: ChunkRound;
  onSetStrategy: (strategy: 'paragraph' | 'sentence') => void;
}) {
  const fact = factLookup(state.facts, round.factId);
  const precision = round.strategy ? roundPrecision(state, round) : null;

  return (
    <li className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5" style={{ borderRadius: 'var(--radius)' }}>
      <p className="font-mono text-xs text-primary">{fact?.query ?? round.factId}</p>
      <div className="flex flex-wrap items-center gap-2">
        {(['paragraph', 'sentence'] as const).map((strategy) => (
          <button
            key={strategy}
            type="button"
            aria-pressed={round.strategy === strategy}
            onClick={() => onSetStrategy(strategy)}
            className={cx(
              'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
              round.strategy === strategy
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-transparent text-secondary hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {strategy}
          </button>
        ))}
        {precision !== null && (
          <Tag tone={precision >= 1 ? 'good' : precision > 0 ? 'warn' : 'bad'}>
            precision {formatNumber(precision, 2)}
          </Tag>
        )}
      </div>
    </li>
  );
}
