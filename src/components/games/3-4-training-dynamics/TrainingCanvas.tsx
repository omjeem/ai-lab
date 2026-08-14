'use client';

/**
 * Chapter 3.4 — Training Dynamics.
 *
 * Every curve on screen was produced by an actual `TinyNet` training run, not
 * an animation standing in for one. The diagnose level trains four real nets
 * (one per candidate learning rate) inside `initState` before the player sees
 * anything; the tune and budget levels train one fresh net per `RUN` from the
 * level's own seed, so the same settings always reproduce the same curve.
 *
 * Model-free like the rest of World 3: `modelRequirement.modelId` is
 * `local:tiny-net`, so there is nothing for `ModelGate` to gate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, Minus, Play, Plus, RotateCcw, X } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  projectedUpdates,
  type LossCurve,
  type TrainingDashboardAction,
  type TrainingDashboardConfig,
  type TrainingDashboardState,
} from '@/engines/trainingDashboardEngine';
import type { Activation } from '@/models/tinyNetTrainer';
import { Button, Meter, Panel, Readout, Slider, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules, ScoreResult } from '@/types/game';

const ACTIVATION_LABEL: Partial<Record<Activation, string>> = {
  tanh: 'tanh',
  relu: 'ReLU',
  leakyRelu: 'leaky ReLU',
  sigmoid: 'sigmoid',
  gelu: 'GELU',
};

export function TrainingCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as TrainingDashboardConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<TrainingDashboardState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: TrainingDashboardAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">running the training sweep</span>
      </div>
    );
  }

  return <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />;
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status === 'complete'`: every action bumps status back
  // to `active`, so a post-reveal answer change would silently un-reveal.
  const [revealed, setRevealed] = useState(false);
  // Bumped on reset so the per-run comparison log in Tune/Budget boards — kept
  // in local state because the engine only remembers the most recent run —
  // clears by remounting rather than needing its own reset wiring.
  const [resetKey, setResetKey] = useState(0);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
    setResetKey((k) => k + 1);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const isDiagnose = state.mode === 'diagnose-curve';

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    if (isDiagnose) setRevealed(true);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'diagnose-curve' && (
          <DiagnoseBoard state={state} dispatch={dispatch} revealed={revealed} />
        )}
        {state.mode === 'tune-run' && <TuneBoard key={resetKey} state={state} dispatch={dispatch} result={result} />}
        {state.mode === 'budget-run' && (
          <BudgetBoard key={resetKey} state={state} dispatch={dispatch} result={result} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {!isDiagnose && (
          <Button onClick={() => dispatch({ type: 'RUN' })}>
            <Play size={13} strokeWidth={2} />
            Run
          </Button>
        )}

        {isDiagnose ? (
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
  state: TrainingDashboardState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = state.curves.some((c) => c.answer === null);
  const left = state.curves.filter((c) => c.answer === null).length;

  return (
    <>
      {!revealed && blocked && (
        <span className="label">{left} run{left === 1 ? '' : 's'} left</span>
      )}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button variant="primary" className="ml-auto" disabled={blocked || revealed} onClick={onSubmit}>
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

/* ── level 1: diagnose the curve ───────────────────────────────── */

function DiagnoseBoard({
  state,
  dispatch,
  revealed,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
  revealed: boolean;
}) {
  const candidates = state.config.candidateLearningRates ?? [];

  return (
    <div className="mx-auto my-auto w-full max-w-4xl">
      <p className="mb-3 text-xs leading-relaxed text-muted">
        Four real training runs on this dataset, one per candidate learning rate. Match each curve to
        the rate that produced it.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {state.curves.map((curve, i) => (
          <CurvePanel
            key={curve.id}
            index={i}
            curve={curve}
            candidates={candidates}
            revealed={revealed}
            onAnswer={(rate) => dispatch({ type: 'MATCH_CURVE', curveId: curve.id, learningRate: rate })}
          />
        ))}
      </div>
    </div>
  );
}

function CurvePanel({
  index,
  curve,
  candidates,
  revealed,
  onAnswer,
}: {
  index: number;
  curve: LossCurve;
  candidates: number[];
  revealed: boolean;
  onAnswer: (rate: number) => void;
}) {
  const correct = curve.answer === curve.trueLearningRate;

  return (
    <Panel
      label={`run ${index + 1}`}
      actions={
        revealed ? (
          <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : `it was ${curve.trueLearningRate}`}</Tag>
        ) : curve.answer !== null ? (
          <Tag tone="accent">answered</Tag>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <LineChart
          series={[{ values: curve.losses, color: 'var(--accent)' }]}
          ariaLabel={`Validation loss over ${curve.losses.length} epochs, run ${index + 1}`}
        />
        <div className="flex flex-wrap gap-1.5">
          {candidates.map((rate) => {
            const chosen = curve.answer === rate;
            const isTruth = revealed && rate === curve.trueLearningRate;
            return (
              <button
                key={rate}
                type="button"
                disabled={revealed}
                onClick={() => onAnswer(rate)}
                aria-pressed={chosen}
                className={cx(
                  'min-h-[36px] flex-1 border px-2 py-1.5 font-mono text-xs transition-colors disabled:cursor-default',
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
                lr = {rate}
              </button>
            );
          })}
        </div>
      </div>
    </Panel>
  );
}

/* ── level 2: batch size trade-off ─────────────────────────────── */

interface RunLogEntry {
  batchSize: number;
  learningRate: number;
  finalValidationLoss: number;
  finalValidationAccuracy: number;
  updatesUsed: number;
}

function useRunLog(state: TrainingDashboardState): RunLogEntry[] {
  const [log, setLog] = useState<RunLogEntry[]>([]);
  useEffect(() => {
    if (!state.lastRun) return;
    setLog((prev) =>
      [
        ...prev,
        {
          batchSize: state.batchSize,
          learningRate: state.learningRate,
          finalValidationLoss: state.lastRun!.finalValidationLoss,
          finalValidationAccuracy: state.lastRun!.finalValidationAccuracy,
          updatesUsed: state.lastRun!.updatesUsed,
        },
      ].slice(-8)
    );
    // Only a new run (a new `lastRun` reference) should append.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.lastRun]);
  return log;
}

function TuneBoard({
  state,
  dispatch,
  result,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
  result: ScoreResult;
}) {
  const log = useRunLog(state);

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[440px]">
        <Panel label="validation loss">
          {state.lastRun ? (
            <LineChart
              series={[{ values: state.lastRun.lossHistory, color: 'var(--accent)' }]}
              ariaLabel="Validation loss curve for the current run"
            />
          ) : (
            <p className="text-xs text-muted">Run to see the loss curve.</p>
          )}
        </Panel>
        <RunLog entries={log} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout
                label="validation loss"
                value={state.lastRun ? result.value : '—'}
                size="lg"
                tone={result.passed ? 'good' : 'accent'}
              />
              {state.lastRun && <Readout label="accuracy" value={result.breakdown.accuracy ?? 0} size="sm" />}
            </div>
            {state.lastRun && (
              <Meter
                value={result.value}
                max={Math.max(result.value, state.rules.passCriteria.threshold) * 1.4}
                threshold={state.rules.passCriteria.threshold}
                tone={result.passed ? 'good' : 'accent'}
              />
            )}
          </div>
        </Panel>

        <Panel label="batch size">
          <BatchSizeButtons state={state} dispatch={dispatch} />
        </Panel>

        <Panel label="learning rate">
          <Slider
            label="η"
            value={state.learningRate}
            min={state.config.learningRateRange?.[0] ?? 0}
            max={state.config.learningRateRange?.[1] ?? 1}
            step={0.001}
            onChange={(value) => dispatch({ type: 'SET_LEARNING_RATE', value })}
            format={(value) => value.toFixed(3)}
          />
        </Panel>

        <div
          className="flex items-center justify-between gap-3 border border-line-strong bg-raised px-3 py-2"
          style={{ borderRadius: 'var(--radius)' }}
        >
          <span className="label">projected updates this run</span>
          <span className="readout text-sm text-accent">{projectedUpdates(state)}</span>
        </div>
      </div>
    </div>
  );
}

/* ── level 3: the compute budget ───────────────────────────────── */

function BudgetBoard({
  state,
  dispatch,
  result,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
  result: ScoreResult;
}) {
  const log = useRunLog(state);
  const budget = state.config.updateBudget ?? Infinity;
  const projected = projectedUpdates(state);
  const overBudget = projected > budget;

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[440px]">
        <Panel label="validation loss">
          {state.lastRun ? (
            <LineChart
              series={[{ values: state.lastRun.lossHistory, color: 'var(--accent)' }]}
              ariaLabel="Validation loss curve for the current run"
            />
          ) : (
            <p className="text-xs text-muted">Run to see the loss curve.</p>
          )}
        </Panel>

        <Panel label="score">
          <div className="flex flex-col gap-3">
            <Readout
              label="validation loss"
              value={state.lastRun ? result.value : '—'}
              size="lg"
              tone={result.passed ? 'good' : 'accent'}
            />
            {state.lastRun && (
              <div className="flex flex-wrap gap-4">
                <Readout label="accuracy" value={result.breakdown.accuracy ?? 0} size="sm" />
                <Readout label="updates used" value={result.breakdown.updatesUsed ?? 0} size="sm" />
                {(result.breakdown.budgetPenalty ?? 0) > 0 && (
                  <Readout label="overspend penalty" value={result.breakdown.budgetPenalty ?? 0} size="sm" tone="bad" />
                )}
              </div>
            )}
          </div>
        </Panel>

        <RunLog entries={log} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="compute budget">
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="label">projected updates</span>
              <span className={cx('readout text-sm', overBudget ? 'text-bad' : 'text-primary')}>
                {projected} <span className="text-muted">/ {budget}</span>
              </span>
            </div>
            <Meter
              value={projected}
              max={Math.max(projected, budget) * 1.15}
              threshold={budget}
              tone={overBudget ? 'bad' : 'good'}
            />
            {overBudget && (
              <Tag tone="warn">over budget — the overspend is charged back into your score on Run</Tag>
            )}
          </div>
        </Panel>

        <Panel label="architecture">
          <ArchitectureEditor state={state} dispatch={dispatch} />
        </Panel>

        {state.config.activationOptions && (
          <Panel label="activation">
            <div className="flex flex-wrap gap-1.5">
              {state.config.activationOptions.map((option) => {
                const active = state.activation === option;
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => dispatch({ type: 'SET_ACTIVATION', value: option })}
                    aria-pressed={active}
                    className={cx(
                      'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors',
                      active
                        ? 'border-accent bg-accent-dim text-primary'
                        : 'border-line-strong bg-raised text-secondary hover:border-accent'
                    )}
                    style={{ borderRadius: 'var(--radius)' }}
                  >
                    {ACTIVATION_LABEL[option] ?? option}
                  </button>
                );
              })}
            </div>
          </Panel>
        )}

        <Panel label="learning rate">
          <Slider
            label="η"
            value={state.learningRate}
            min={state.config.learningRateRange?.[0] ?? 0}
            max={state.config.learningRateRange?.[1] ?? 1}
            step={0.001}
            onChange={(value) => dispatch({ type: 'SET_LEARNING_RATE', value })}
            format={(value) => value.toFixed(3)}
          />
        </Panel>

        <Panel label="batch size">
          <BatchSizeButtons state={state} dispatch={dispatch} />
        </Panel>

        <Panel label="epochs">
          <Slider
            label="epochs"
            value={state.epochs}
            min={1}
            max={state.config.maxEpochs ?? state.epochs}
            step={1}
            onChange={(value) => dispatch({ type: 'SET_EPOCHS', value })}
            format={(value) => String(Math.round(value))}
          />
        </Panel>
      </div>
    </div>
  );
}

function BatchSizeButtons({
  state,
  dispatch,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {(state.config.batchSizeOptions ?? []).map((bs) => {
        const active = state.batchSize === bs;
        return (
          <button
            key={bs}
            type="button"
            onClick={() => dispatch({ type: 'SET_BATCH_SIZE', value: bs })}
            aria-pressed={active}
            className={cx(
              'min-h-[40px] flex-1 border px-3 py-2 font-mono text-xs transition-colors',
              active
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-raised text-secondary hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {bs}
          </button>
        );
      })}
    </div>
  );
}

function ArchitectureEditor({
  state,
  dispatch,
}: {
  state: TrainingDashboardState;
  dispatch: (action: TrainingDashboardAction) => void;
}) {
  const limits = state.config.architectureRange;
  const hidden = state.architecture.slice(1, -1);

  if (!limits) {
    return <p className="text-xs text-muted">fixed architecture: {state.architecture.join(' – ')}</p>;
  }

  const setUnits = (next: number[]) => dispatch({ type: 'SET_HIDDEN_LAYERS', units: next });

  return (
    <div className="flex flex-col gap-3">
      <span className="label">{state.architecture.join(' – ')}</span>
      <div className="flex flex-wrap items-stretch gap-2">
        {hidden.map((units, i) => (
          <div
            key={i}
            className="flex flex-col items-center gap-1.5 border border-line-strong px-2 py-2"
            style={{ borderRadius: 'var(--radius)' }}
          >
            <span className="label">layer {i + 1}</span>
            <div className="flex items-center gap-1">
              <Button
                disabled={units <= 1}
                onClick={() => setUnits(hidden.map((u, j) => (j === i ? u - 1 : u)))}
                className="min-h-0! px-2! py-1.5!"
              >
                <Minus size={12} strokeWidth={2} />
                <span className="sr-only">fewer units in layer {i + 1}</span>
              </Button>
              <span className="readout w-6 text-center text-sm text-primary">{units}</span>
              <Button
                disabled={units >= limits.maxUnits}
                onClick={() => setUnits(hidden.map((u, j) => (j === i ? u + 1 : u)))}
                className="min-h-0! px-2! py-1.5!"
              >
                <Plus size={12} strokeWidth={2} />
                <span className="sr-only">more units in layer {i + 1}</span>
              </Button>
            </div>
            <Button
              variant="danger"
              onClick={() => setUnits(hidden.filter((_, j) => j !== i))}
              className="min-h-0! px-2! py-1!"
            >
              <X size={11} strokeWidth={2} />
              <span className="sr-only">remove layer {i + 1}</span>
            </Button>
          </div>
        ))}
        <Button disabled={hidden.length >= limits.maxHidden} onClick={() => setUnits([...hidden, Math.min(4, limits.maxUnits)])}>
          <Plus size={13} strokeWidth={2} />
          layer
        </Button>
      </div>
    </div>
  );
}

function RunLog({ entries }: { entries: RunLogEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <Panel label="past runs this session" flush>
      <div className="overflow-x-auto p-2">
        <table className="w-full min-w-[280px] border-collapse text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="px-2 py-1 font-normal">batch</th>
              <th className="px-2 py-1 font-normal">η</th>
              <th className="px-2 py-1 font-normal">loss</th>
              <th className="px-2 py-1 font-normal">acc</th>
              <th className="px-2 py-1 font-normal">updates</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, i) => (
              <tr key={i} className={i === entries.length - 1 ? 'text-accent' : 'text-secondary'}>
                <td className="readout px-2 py-1">{entry.batchSize}</td>
                <td className="readout px-2 py-1">{entry.learningRate.toFixed(3)}</td>
                <td className="readout px-2 py-1">{formatNumber(entry.finalValidationLoss)}</td>
                <td className="readout px-2 py-1">{(entry.finalValidationAccuracy * 100).toFixed(0)}%</td>
                <td className="readout px-2 py-1">{entry.updatesUsed}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* ── shared line chart ─────────────────────────────────────────── */

function LineChart({
  series,
  ariaLabel,
}: {
  series: { values: number[]; color: string }[];
  ariaLabel: string;
}) {
  const width = 100;
  const height = 30;
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const hi = all.length ? Math.max(...all, 1e-6) : 1;
  const n = series[0]?.values.length ?? 0;
  const toX = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * width);
  const toY = (v: number) => height - Math.min(1, Math.max(0, v / hi)) * height;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block h-24 w-full" role="img" aria-label={ariaLabel}>
      <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line-faint)" strokeWidth={0.3} />
      {series.map((s, si) => (
        <path
          key={si}
          d={s.values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(v).toFixed(2)}`).join(' ')}
          fill="none"
          stroke={s.color}
          strokeWidth={0.8}
        />
      ))}
    </svg>
  );
}
