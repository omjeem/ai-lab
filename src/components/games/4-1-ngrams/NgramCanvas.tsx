'use client';

/**
 * Chapter 4.1 — N-grams.
 *
 * The count table is the whole model, so it is the dominant visual on every
 * level: the tune and sparsity boards show the real `NgramTable` (context →
 * word → count) built live from the corpus, and level 1's reveal shows the
 * real smoothed probability behind each candidate rather than just marking
 * it right or wrong. Nothing here is precomputed — `buildTable` runs on the
 * bundled corpus at load time, same as the engine's own `prepare`.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  buildTable,
  probabilityOf,
  heldOutPerplexity,
  contextCoverage,
  findHighestViableOrder,
  type NgramPredictionAction,
  type NgramPredictionConfig,
  type NgramPredictionState,
  type NgramTable,
} from '@/engines/ngramPredictionEngine';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Slider, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function NgramCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as NgramPredictionConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<NgramPredictionState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { corpus: corpusLoader });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: NgramPredictionAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={game.modelRequirement.modelId}
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
  state: NgramPredictionState;
  dispatch: (action: NgramPredictionAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status === 'complete'`: every action bumps status back
  // to `active`, so answering another round after submit would un-reveal.
  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const isQuiz = state.mode === 'beat-the-model';

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    if (isQuiz) setRevealed(true);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'beat-the-model' && <BeatBoard state={state} dispatch={dispatch} revealed={revealed} />}
        {state.mode === 'tune-order' && <TuneOrderBoard state={state} dispatch={dispatch} result={result} />}
        {state.mode === 'sparsity-wall' && <SparsityBoard state={state} dispatch={dispatch} result={result} />}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        {isQuiz ? (
          <QuizSubmit state={state} revealed={revealed} onSubmit={submit} />
        ) : (
          <Button variant="primary" className="ml-auto" onClick={submit}>
            <Check size={13} strokeWidth={2} />
            Submit
          </Button>
        )}
      </div>
    </div>
  );
}

function QuizSubmit({
  state,
  revealed,
  onSubmit,
}: {
  state: NgramPredictionState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const left = state.rounds.filter((r) => r.answer === null).length;
  const blocked = left > 0;

  return (
    <>
      {!revealed && blocked && <span className="label">{left} round{left === 1 ? '' : 's'} left</span>}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button variant="primary" className="ml-auto" disabled={blocked || revealed} onClick={onSubmit}>
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

/* ── level 1: beat the bigram ──────────────────────────────────── */

function BeatBoard({
  state,
  dispatch,
  revealed,
}: {
  state: NgramPredictionState;
  dispatch: (action: NgramPredictionAction) => void;
  revealed: boolean;
}) {
  return (
    <div className="mx-auto my-auto w-full max-w-3xl">
      <p className="mb-3 text-xs leading-relaxed text-muted">
        A {state.n}-gram table trained on {state.trainTokens.length} tokens from the corpus. Pick the
        word that really follows each context in the held-out text.
      </p>
      <Panel label="which word comes next?">
        <ol className="flex flex-col gap-2">
          {state.rounds.map((round, i) => {
            const correct = round.answer === round.trueNext;
            const contextText = round.context.join(' ');
            return (
              <li
                key={i}
                className={cx(
                  'flex flex-col gap-2 border px-3 py-2.5',
                  revealed
                    ? correct
                      ? 'border-good/40 bg-good/5'
                      : 'border-bad/40 bg-bad/5'
                    : round.answer
                      ? 'border-accent/40 bg-accent-faint'
                      : 'border-line-strong bg-raised'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                <div className="flex flex-wrap items-center gap-3">
                  <span className="readout w-5 shrink-0 text-xs text-accent">{i + 1}</span>
                  <span className="min-w-0 flex-1 font-mono text-sm text-secondary">
                    … {contextText} <span className="text-primary">___</span>
                  </span>
                  {revealed && <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : `was "${round.trueNext}"`}</Tag>}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {round.candidates.map((word) => {
                    const chosen = round.answer === word;
                    const isTruth = revealed && word === round.trueNext;
                    const prob = revealed ? probabilityOf(state.table, contextText, word, state.alpha) : null;
                    return (
                      <button
                        key={word}
                        type="button"
                        disabled={revealed}
                        onClick={() => dispatch({ type: 'ANSWER', roundIndex: i, word })}
                        aria-pressed={chosen}
                        className={cx(
                          'flex min-h-[36px] flex-1 items-center justify-center gap-1.5 border px-2 py-1.5 font-mono text-xs transition-colors disabled:cursor-default',
                          isTruth
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
                        {prob !== null && <span className="readout opacity-70">{(prob * 100).toFixed(1)}%</span>}
                      </button>
                    );
                  })}
                </div>
              </li>
            );
          })}
        </ol>
      </Panel>
    </div>
  );
}

/* ── level 2: tune the order ───────────────────────────────────── */

function TuneOrderBoard({
  state,
  dispatch,
  result,
}: {
  state: NgramPredictionState;
  dispatch: (action: NgramPredictionAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [nMin, nMax] = state.config.nRange ?? [state.n, state.n];
  const [aMin, aMax] = state.config.smoothingAlphaRange ?? [0, 1];

  const sweep = useMemo(() => {
    const vocabularySize = new Set(state.trainTokens).size;
    const points: { n: number; perplexity: number }[] = [];
    for (let n = nMin; n <= nMax; n++) {
      const table = buildTable(state.trainTokens, n, vocabularySize);
      points.push({ n, perplexity: heldOutPerplexity(table, state.heldOutTokens, state.alpha) });
    }
    return points;
  }, [state.trainTokens, state.heldOutTokens, state.alpha, nMin, nMax]);

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[420px]">
        <Panel label="perplexity by order, at the current smoothing">
          <MetricSweep
            points={sweep.map((p) => ({ label: String(p.n), value: p.perplexity, active: p.n === state.n }))}
            threshold={state.rules.passCriteria.threshold}
            ariaLabel="Held-out perplexity for each n-gram order at the current smoothing"
          />
        </Panel>
        <CountTable table={state.table} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="perplexity" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <Readout label="coverage" value={result.breakdown.coverage ?? 0} size="sm" />
              <Readout label="contexts" value={result.breakdown.contexts ?? 0} size="sm" />
            </div>
            <Meter
              value={result.value}
              max={Math.max(result.value, state.rules.passCriteria.threshold) * 1.3}
              threshold={state.rules.passCriteria.threshold}
              tone={result.passed ? 'good' : 'accent'}
            />
          </div>
        </Panel>

        <Panel label="order (n)">
          <Slider
            label="n"
            value={state.n}
            min={nMin}
            max={nMax}
            step={1}
            onChange={(value) => dispatch({ type: 'SET_N', value })}
            format={(value) => String(Math.round(value))}
          />
        </Panel>

        <Panel label="smoothing (alpha)">
          <Slider
            label="α"
            value={state.alpha}
            min={aMin}
            max={aMax}
            step={0.001}
            onChange={(value) => dispatch({ type: 'SET_ALPHA', value })}
            format={(value) => value.toFixed(3)}
          />
        </Panel>
      </div>
    </div>
  );
}

/* ── level 3: the sparsity wall ────────────────────────────────── */

function SparsityBoard({
  state,
  dispatch,
  result,
}: {
  state: NgramPredictionState;
  dispatch: (action: NgramPredictionAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [nMin, nMax] = state.config.nRange ?? [state.n, state.n];
  const minCoverage = state.config.minCoverage ?? 0.5;
  const highestViable = useMemo(() => findHighestViableOrder(state), [state]);

  const sweep = useMemo(() => {
    const vocabularySize = new Set(state.trainTokens).size;
    const points: { n: number; coverage: number }[] = [];
    for (let n = nMin; n <= nMax; n++) {
      const table = buildTable(state.trainTokens, n, vocabularySize);
      points.push({ n, coverage: contextCoverage(table, state.heldOutTokens) });
    }
    return points;
  }, [state.trainTokens, state.heldOutTokens, nMin, nMax]);

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[420px]">
        <Panel label={`held-out context coverage by order — floor ${minCoverage.toFixed(2)}`}>
          <MetricSweep
            points={sweep.map((p) => ({ label: String(p.n), value: p.coverage, active: p.n === state.n }))}
            threshold={minCoverage}
            ariaLabel="Held-out context coverage for each n-gram order"
          />
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Highest order that still clears the floor: <span className="readout text-accent">n = {highestViable}</span>
          </p>
        </Panel>
        <CountTable table={state.table} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="sparsity score" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <Readout label="coverage" value={result.breakdown.coverage ?? 0} size="sm" tone={(result.breakdown.coverage ?? 0) >= minCoverage ? 'good' : 'bad'} />
            </div>
            <Meter
              value={result.breakdown.coverage ?? 0}
              max={1}
              threshold={minCoverage}
              tone={(result.breakdown.coverage ?? 0) >= minCoverage ? 'good' : 'bad'}
            />
            {(result.breakdown.coverage ?? 0) < minCoverage && (
              <Tag tone="bad">below the coverage floor — this order can&apos;t generalise</Tag>
            )}
          </div>
        </Panel>

        <Panel label="order (n)">
          <Slider
            label="n"
            value={state.n}
            min={nMin}
            max={nMax}
            step={1}
            onChange={(value) => dispatch({ type: 'SET_N', value })}
            format={(value) => String(Math.round(value))}
          />
        </Panel>
      </div>
    </div>
  );
}

/* ── the count table ───────────────────────────────────────────── */

const COUNT_TABLE_ROWS = 10;

function CountTable({ table }: { table: NgramTable }) {
  const rows = useMemo(() => {
    return [...table.contextTotals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, COUNT_TABLE_ROWS)
      .map(([context, total]) => {
        const bucket = table.counts.get(context);
        const distinct = bucket?.size ?? 0;
        const top = bucket ? [...bucket.entries()].sort((a, b) => b[1] - a[1])[0] : undefined;
        return { context: context === '' ? '(unigram)' : context, total, distinct, top };
      });
  }, [table]);

  return (
    <Panel label={`count table — top ${rows.length} of ${table.counts.size} contexts, order n = ${table.n}`} flush>
      <div className="overflow-x-auto p-2">
        <table className="w-full min-w-[320px] border-collapse text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="px-2 py-1 font-normal">context</th>
              <th className="px-2 py-1 text-right font-normal">seen</th>
              <th className="px-2 py-1 text-right font-normal">next words</th>
              <th className="px-2 py-1 font-normal">top continuation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="text-secondary">
                <td className="readout truncate px-2 py-1 text-primary">{row.context}</td>
                <td className="readout px-2 py-1 text-right">{row.total}</td>
                <td className="readout px-2 py-1 text-right">{row.distinct}</td>
                <td className="readout px-2 py-1">{row.top ? `${row.top[0]} (${row.top[1]})` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* ── shared sweep bar chart ────────────────────────────────────── */

function MetricSweep({
  points,
  threshold,
  ariaLabel,
}: {
  points: { label: string; value: number; active: boolean }[];
  /** Drawn as a line across the bars — the pass floor or ceiling for this metric. */
  threshold?: number;
  ariaLabel: string;
}) {
  const values = points.map((p) => p.value).filter(Number.isFinite);
  const hi = values.length ? Math.max(...values, threshold ?? 0) : 1;
  const thresholdPct = threshold !== undefined ? Math.max(0, Math.min(1, threshold / Math.max(hi, 1e-9))) : null;

  return (
    <div className="flex items-end gap-2" role="img" aria-label={ariaLabel}>
      {points.map((p) => {
        const finite = Number.isFinite(p.value);
        const pct = finite ? Math.max(0.02, Math.min(1, p.value / Math.max(hi, 1e-9))) : 1;
        return (
          <div key={p.label} className="flex flex-1 flex-col items-center gap-1">
            <span className="readout text-[10px] text-muted">{finite ? formatNumber(p.value, 2) : '∞'}</span>
            <div className="relative h-20 w-full bg-inset">
              {thresholdPct !== null && (
                <div
                  className="absolute inset-x-0 h-px bg-text-muted"
                  style={{ bottom: `${thresholdPct * 100}%`, background: 'var(--text-muted)' }}
                  aria-hidden
                />
              )}
              <div
                className={cx('absolute inset-x-0 bottom-0', p.active ? 'bg-accent' : 'bg-line-strong')}
                style={{ height: `${pct * 100}%` }}
              />
            </div>
            <span className={cx('readout text-xs', p.active ? 'text-accent' : 'text-muted')}>{p.label}</span>
          </div>
        );
      })}
    </div>
  );
}
