'use client';

/**
 * Chapter 8.1 — Quantization Tradeoffs.
 *
 * Three boards over the same two real model instances (`referencePrecisionModel`
 * = fp32, `quantizedPrecisionModel` = q8, both the exact model `tinyCausalLM`
 * already ships — see `src/models/quantizationModel.ts`). `ModelGate` tracks
 * only the fp32 download's progress bar, since it's the larger, slower of the
 * two real downloads and so dominates the wait; q8 downloads immediately after
 * inside the same `load()` call, just without its own visible progress bar —
 * a real, honest tradeoff for a chapter that needs two full model instances
 * rather than the one every other `ModelGate` usage in this app assumes.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw, X } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ConfidenceRound,
  type DivergenceRound,
  type QuantizationTradeoffAction,
  type QuantizationTradeoffConfig,
  type QuantizationTradeoffState,
  type SpeedPredictionField,
} from '@/engines/quantizationTradeoffEngine';
import { referencePrecisionModel, quantizedPrecisionModel, REFERENCE_MODEL_ID } from '@/models/quantizationModel';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function QuantizationCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as QuantizationTradeoffConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<QuantizationTradeoffState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, {
      reference: referencePrecisionModel,
      quantized: quantizedPrecisionModel,
      corpus: corpusLoader,
    });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: QuantizationTradeoffAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={REFERENCE_MODEL_ID}
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
  state: QuantizationTradeoffState;
  dispatch: (action: QuantizationTradeoffAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status`: every action bumps status back to 'active', so
  // continuing to interact with an already-revealed board (e.g. the speed
  // board's toggles) would silently un-reveal what was just shown.
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
          {state.mode === 'compare-outputs' && <DivergenceBoard state={state} dispatch={dispatch} />}
          {state.mode === 'compare-speed' && <SpeedBoard state={state} dispatch={dispatch} revealed={revealed} result={result} />}
          {state.mode === 'compare-confidence' && <ConfidenceBoard state={state} dispatch={dispatch} />}
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

function submitBlocked(state: QuantizationTradeoffState): boolean {
  switch (state.mode) {
    case 'compare-outputs':
      return state.divergenceRounds.some((r) => r.pick === null);
    case 'compare-speed':
      return Object.values(state.speedPredictions).some((v) => v === null);
    case 'compare-confidence':
      return state.confidenceRounds.some((r) => r.pick === null);
  }
}

function submitHint(state: QuantizationTradeoffState): string {
  switch (state.mode) {
    case 'compare-outputs': {
      const left = state.divergenceRounds.filter((r) => r.pick === null).length;
      return `${left} prompt${left === 1 ? '' : 's'} left`;
    }
    case 'compare-speed': {
      const left = Object.values(state.speedPredictions).filter((v) => v === null).length;
      return `${left} prediction${left === 1 ? '' : 's'} left`;
    }
    case 'compare-confidence': {
      const left = state.confidenceRounds.filter((r) => r.pick === null).length;
      return `${left} fact${left === 1 ? '' : 's'} left`;
    }
  }
}

/* ── level 1: spot the divergence ──────────────────────────────── */

function DivergenceBoard({
  state,
  dispatch,
}: {
  state: QuantizationTradeoffState;
  dispatch: (action: QuantizationTradeoffAction) => void;
}) {
  return (
    <Panel label="fp32's real answer is shown — will q8 land on the same top token?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.divergenceRounds.map((round, index) => (
          <DivergenceRow
            key={round.prompt}
            round={round}
            onGuess={(guess) => dispatch({ type: 'PICK_DIVERGENCE', roundIndex: index, guess })}
          />
        ))}
      </ol>
    </Panel>
  );
}

function DivergenceRow({
  round,
  onGuess,
}: {
  round: DivergenceRound;
  onGuess: (guess: boolean) => void;
}) {
  const revealed = round.pick !== null;
  const correct = revealed && round.pick === round.diverges;

  return (
    <li className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5" style={{ borderRadius: 'var(--radius)' }}>
      <p className="font-mono text-xs text-primary">
        “{round.prompt}” <span className="text-accent">{round.referenceTop}</span>{' '}
        <span className="text-muted">({formatNumber(round.referenceProb, 2)})</span>
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'same top token', guess: false },
            { label: 'different top token', guess: true },
          ] as const
        ).map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={round.pick === option.guess}
            disabled={revealed}
            onClick={() => onGuess(option.guess)}
            className={cx(
              'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
              round.pick === option.guess
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-transparent text-secondary hover:border-accent',
              revealed && 'disabled:opacity-70'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {option.label}
          </button>
        ))}
        {revealed && (
          <Tag tone={correct ? 'good' : 'bad'}>
            {correct ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            q8: {round.quantizedTop} ({formatNumber(round.quantizedProb, 2)})
          </Tag>
        )}
      </div>
    </li>
  );
}

/* ── level 2: predict the tradeoff ─────────────────────────────── */

const SPEED_QUESTIONS: { field: SpeedPredictionField; label: string }[] = [
  { field: 'smallerIsQuantized', label: 'q8 is the smaller real download' },
  { field: 'fasterLoadIsQuantized', label: 'q8 loaded faster, real wall-clock time' },
  { field: 'fasterInferenceIsQuantized', label: 'q8 ran the probe prompt faster, real wall-clock time' },
];

function SpeedBoard({
  state,
  dispatch,
  revealed,
  result,
}: {
  state: QuantizationTradeoffState;
  dispatch: (action: QuantizationTradeoffAction) => void;
  revealed: boolean;
  result: ReturnType<typeof evaluate>;
}) {
  const m = state.speedMeasurement;

  return (
    <div className="flex flex-col gap-4">
      <Panel label="predict, then reveal the real measurements" flush>
        <ol className="flex flex-col gap-2 p-2">
          {SPEED_QUESTIONS.map(({ field, label }) => (
            <li key={field} className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5" style={{ borderRadius: 'var(--radius)' }}>
              <p className="font-mono text-xs text-primary">{label}</p>
              <div className="flex flex-wrap items-center gap-2">
                {([true, false] as const).map((value) => (
                  <button
                    key={String(value)}
                    type="button"
                    aria-pressed={state.speedPredictions[field] === value}
                    disabled={revealed}
                    onClick={() => dispatch({ type: 'PREDICT_SPEED', field, value })}
                    className={cx(
                      'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                      state.speedPredictions[field] === value
                        ? 'border-accent bg-accent-dim text-primary'
                        : 'border-line-strong bg-transparent text-secondary hover:border-accent'
                    )}
                    style={{ borderRadius: 'var(--radius)' }}
                  >
                    {value ? 'true' : 'false'}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ol>
      </Panel>

      {revealed && m && (
        <Panel label="real measurements">
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <Readout label="fp32 size" value={m.referenceSizeMB} unit="MB" size="sm" />
              <Readout label="q8 size" value={m.quantizedSizeMB} unit="MB" size="sm" tone="accent" />
              <Readout label="fp32 load" value={m.referenceLoadMs} unit="ms" size="sm" />
              <Readout label="q8 load" value={m.quantizedLoadMs} unit="ms" size="sm" tone="accent" />
              <Readout label="fp32 inference" value={m.referenceInferenceMs} unit="ms" size="sm" />
              <Readout label="q8 inference" value={m.quantizedInferenceMs} unit="ms" size="sm" tone="accent" />
            </div>
            <div className="flex items-end justify-between gap-4">
              <Readout label="tradeoffPredictionAccuracy" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <Readout label="correct" value={`${result.breakdown.correct ?? 0}/${result.breakdown.total ?? 0}`} size="sm" />
            </div>
            <Meter value={result.value} max={1} threshold={state.rules.passCriteria.threshold} tone={result.passed ? 'good' : 'accent'} />
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ── level 3: numeric confidence ───────────────────────────────── */

function ConfidenceBoard({
  state,
  dispatch,
}: {
  state: QuantizationTradeoffState;
  dispatch: (action: QuantizationTradeoffAction) => void;
}) {
  return (
    <Panel label="which precision is really more confident in the true digits?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.confidenceRounds.map((round, index) => (
          <ConfidenceRoundRow
            key={round.factId}
            round={round}
            onPick={(pick) => dispatch({ type: 'PICK_CONFIDENCE', roundIndex: index, pick })}
          />
        ))}
      </ol>
    </Panel>
  );
}

function ConfidenceRoundRow({
  round,
  onPick,
}: {
  round: ConfidenceRound;
  onPick: (pick: 'reference' | 'quantized') => void;
}) {
  const revealed = round.pick !== null;
  const truth: 'reference' | 'quantized' = round.referenceMoreConfident ? 'reference' : 'quantized';
  const correct = revealed && round.pick === truth;
  const maxBits = Math.max(round.referenceMeanBits, round.quantizedMeanBits, 1e-9);

  return (
    <li className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5" style={{ borderRadius: 'var(--radius)' }}>
      <p className="font-mono text-xs text-primary">
        {round.topic} — <span className="text-secondary">{round.query}</span> ({round.answer})
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'fp32 more confident', pick: 'reference' as const },
            { label: 'q8 more confident', pick: 'quantized' as const },
          ]
        ).map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={round.pick === option.pick}
            disabled={revealed}
            onClick={() => onPick(option.pick)}
            className={cx(
              'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
              round.pick === option.pick
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-transparent text-secondary hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {option.label}
          </button>
        ))}
        {revealed && (
          <Tag tone={correct ? 'good' : 'bad'}>
            {correct ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            {formatNumber(round.referenceMeanBits, 2)} vs {formatNumber(round.quantizedMeanBits, 2)} bits
          </Tag>
        )}
      </div>
      {revealed && (
        <div className="flex flex-col gap-1">
          <BitsBar label="fp32" bits={round.referenceMeanBits} max={maxBits} />
          <BitsBar label="q8" bits={round.quantizedMeanBits} max={maxBits} />
        </div>
      )}
    </li>
  );
}

function BitsBar({ label, bits, max }: { label: string; bits: number; max: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="label w-10 shrink-0">{label}</span>
      <div className="relative h-1.5 flex-1 bg-inset" role="img" aria-label={`${label} surprisal ${bits.toFixed(3)} bits`}>
        <div className="absolute inset-y-0 left-0 bg-accent" style={{ width: `${Math.max(2, (bits / max) * 100)}%` }} />
      </div>
      <span className="readout w-14 shrink-0 text-right text-[11px] text-secondary">{formatNumber(bits, 2)}</span>
    </div>
  );
}
