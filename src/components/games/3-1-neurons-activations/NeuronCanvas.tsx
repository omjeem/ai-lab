'use client';

/**
 * Chapter 3.1 — Neurons & Activations.
 *
 * One neuron — weighted sum, bias, non-linearity — evaluated for real over a
 * fixed probe set on every render. The "curve" in every mode here is just that
 * neuron's actual output at each probe, plotted by index rather than by a
 * smooth input axis, because the probes are genuinely random points, not a
 * sweep.
 *
 * Model-free (the neuron is arithmetic, not a download), same as the rest of
 * World 2 and 3: nothing for `ModelGate` to gate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  currentOutputs,
  preActivation,
  type NeuronTuningAction,
  type NeuronTuningConfig,
  type NeuronTuningState,
} from '@/engines/neuronTuningEngine';
import type { Activation } from '@/models/tinyNetTrainer';
import { Button, Panel, Readout, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules, ScoreResult } from '@/types/game';

const ACTIVATION_LABEL: Record<Activation, string> = {
  relu: 'ReLU',
  sigmoid: 'sigmoid',
  tanh: 'tanh',
  leakyRelu: 'leaky ReLU',
  gelu: 'GELU',
  linear: 'linear',
};

export function NeuronCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as NeuronTuningConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<NeuronTuningState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: NeuronTuningAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">wiring up the neuron</span>
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
  state: NeuronTuningState;
  dispatch: (action: NeuronTuningAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status === 'complete'`: ANSWER_ACTIVATION bumps status
  // back to `active`, same latent trap as every other quiz-style board.
  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'identify-activation' ? (
          <IdentifyBoard state={state} dispatch={dispatch} revealed={revealed} />
        ) : (
          <TuneBoard state={state} dispatch={dispatch} result={result} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        {state.mode === 'identify-activation' ? (
          <IdentifySubmit
            state={state}
            revealed={revealed}
            onSubmit={() => {
              onSubmit(evaluate(state));
              dispatch({ type: 'SUBMIT' });
              setRevealed(true);
            }}
          />
        ) : (
          <Button variant="primary" className="ml-auto" onClick={() => onSubmit(evaluate(state))}>
            <Check size={13} strokeWidth={2} />
            Submit
          </Button>
        )}
      </div>
    </div>
  );
}

/* ── match-output / revive-dead-neuron ─────────────────────────── */

function TuneBoard({
  state,
  dispatch,
  result,
}: {
  state: NeuronTuningState;
  dispatch: (action: NeuronTuningAction) => void;
  result: ScoreResult;
}) {
  const outputs = useMemo(() => currentOutputs(state), [state]);
  const revive = state.mode === 'revive-dead-neuron';

  const preActivations = useMemo(
    () => state.probes.map((p) => preActivation(state.weights, state.bias, p)),
    [state.probes, state.weights, state.bias]
  );

  return (
    <div className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-3 lg:flex-row lg:items-start">
      <Panel label={revive ? 'pre-activation sum' : 'output vs target'} className="w-full shrink-0 lg:w-[440px]">
        {revive ? (
          <PreActivationTrace values={preActivations} />
        ) : (
          <OutputTrace mine={outputs} target={state.targetOutputs} />
        )}
      </Panel>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <Readout
              label={revive ? 'active fraction' : 'match'}
              value={result.value}
              size="lg"
              tone={result.passed ? 'good' : 'accent'}
            />
            {revive && (
              <Tag tone={result.passed ? 'good' : 'warn'}>
                needs ≥ {(state.config.minActiveFraction ?? 0.35).toFixed(2)} of probes active
              </Tag>
            )}
          </div>
        </Panel>

        <Panel label="weights and bias">
          <WeightControls state={state} dispatch={dispatch} />
        </Panel>

        {revive && (
          <p className="text-xs leading-relaxed text-muted">
            ReLU is dead when the sum going in never clears zero — the output is flat regardless, so
            the sum is the only place a fix is visible.
          </p>
        )}
      </div>
    </div>
  );
}

function WeightControls({
  state,
  dispatch,
}: {
  state: NeuronTuningState;
  dispatch: (action: NeuronTuningAction) => void;
}) {
  const [wMin, wMax] = state.config.weightRange;
  const [bMin, bMax] = state.config.biasRange;
  return (
    <div className="flex flex-col gap-3">
      {state.weights.map((weight, i) => (
        <Slider
          key={i}
          label={`w${i}`}
          value={weight}
          min={wMin}
          max={wMax}
          step={(wMax - wMin) / 200}
          onChange={(value) => dispatch({ type: 'SET_WEIGHT', index: i, value })}
          format={(value) => value.toFixed(3)}
        />
      ))}
      <Slider
        label="bias"
        value={state.bias}
        min={bMin}
        max={bMax}
        step={(bMax - bMin) / 200}
        onChange={(value) => dispatch({ type: 'SET_BIAS', value })}
        format={(value) => value.toFixed(3)}
      />
    </div>
  );
}

function OutputTrace({ mine, target }: { mine: number[]; target: number[] }) {
  const all = [...mine, ...target].filter(Number.isFinite);
  const lo = all.length ? Math.min(...all, 0) : -1;
  const hi = all.length ? Math.max(...all, 0) : 1;
  const pad = Math.max((hi - lo) * 0.1, 0.2);
  const yDomain: [number, number] = [lo - pad, hi + pad];

  return (
    <div className="flex flex-col gap-2">
      <Trace width={100} height={34} series={[
        target.length > 0 ? { values: target, color: 'var(--text-secondary)', dashed: true } : null,
        { values: mine, color: 'var(--accent)' },
      ].filter((s): s is { values: number[]; color: string; dashed?: boolean } => s !== null)}
        yDomain={yDomain}
        zeroLine
      />
      <div className="flex items-center gap-4 px-1">
        <span className="flex items-center gap-1.5">
          <svg width={12} height={10} aria-hidden>
            <line x1={0} y1={5} x2={12} y2={5} stroke="var(--accent)" strokeWidth={1.5} />
          </svg>
          <span className="label">yours</span>
        </span>
        {target.length > 0 && (
          <span className="flex items-center gap-1.5">
            <svg width={12} height={10} aria-hidden>
              <line x1={0} y1={5} x2={12} y2={5} stroke="var(--text-secondary)" strokeWidth={1.5} strokeDasharray="2 1.5" />
            </svg>
            <span className="label">target</span>
          </span>
        )}
      </div>
    </div>
  );
}

function PreActivationTrace({ values }: { values: number[] }) {
  const hi = Math.max(...values, 0.1);
  const lo = Math.min(...values, -0.1);
  const pad = Math.max((hi - lo) * 0.1, 0.2);
  const yDomain: [number, number] = [lo - pad, hi + pad];
  const active = values.filter((v) => v > 0).length;

  return (
    <div className="flex flex-col gap-2">
      <Trace
        width={100}
        height={34}
        series={[{ values, color: 'var(--accent)' }]}
        yDomain={yDomain}
        zeroLine
        pointTone={(v) => (v > 0 ? 'var(--signal-good)' : 'var(--signal-bad)')}
      />
      <div className="flex items-center justify-between px-1">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: 'var(--signal-good)' }} aria-hidden />
          <span className="label">active ({active})</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: 'var(--signal-bad)' }} aria-hidden />
          <span className="label">dead ({values.length - active})</span>
        </span>
      </div>
    </div>
  );
}

/** Small line-plus-points chart over an arbitrary index axis. */
function Trace({
  width,
  height,
  series,
  yDomain,
  zeroLine,
  pointTone,
}: {
  width: number;
  height: number;
  series: { values: number[]; color: string; dashed?: boolean }[];
  yDomain: [number, number];
  zeroLine?: boolean;
  pointTone?: (value: number) => string;
}) {
  const [lo, hi] = yDomain;
  const toY = (v: number) => height - ((v - lo) / (hi - lo)) * height;
  const n = series[0]?.values.length ?? 0;
  const toX = (i: number) => (n <= 1 ? width / 2 : (i / (n - 1)) * width);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="block h-24 w-full"
      role="img"
      aria-label="Neuron output across the probe set"
    >
      {zeroLine && lo < 0 && hi > 0 && (
        <line x1={0} y1={toY(0)} x2={width} y2={toY(0)} stroke="var(--line)" strokeWidth={0.4} />
      )}
      <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line-faint)" strokeWidth={0.3} />

      {series.map((s, si) => {
        const path = s.values
          .map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(v).toFixed(2)}`)
          .join(' ');
        return (
          <path
            key={si}
            d={path}
            fill="none"
            stroke={s.color}
            strokeWidth={0.8}
            strokeDasharray={s.dashed ? '2 1.5' : undefined}
          />
        );
      })}

      {pointTone &&
        series[0]?.values.map((v, i) => (
          <circle key={i} cx={toX(i)} cy={toY(v)} r={1.1} fill={pointTone(v)} />
        ))}
    </svg>
  );
}

/* ── identify-activation ────────────────────────────────────── */

function IdentifyBoard({
  state,
  dispatch,
  revealed,
}: {
  state: NeuronTuningState;
  dispatch: (action: NeuronTuningAction) => void;
  revealed: boolean;
}) {
  const candidates = state.config.candidates ?? [];

  return (
    <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-3">
      <p className="text-xs leading-relaxed text-muted">
        Same weights and bias, five different activations run over them for real. Read the shape.
      </p>
      {state.rounds.map((round, index) => {
        const correct = round.answer === round.trueActivation;
        const values = round.trace;
        const hi = Math.max(...values);
        const lo = Math.min(...values);
        const pad = Math.max((hi - lo) * 0.1, 0.1);
        return (
          <Panel
            key={index}
            label={`round ${index + 1}`}
            actions={
              revealed ? (
                <Tag tone={correct ? 'good' : 'bad'}>
                  {correct ? 'correct' : `it was ${ACTIVATION_LABEL[round.trueActivation]}`}
                </Tag>
              ) : round.answer ? (
                <Tag tone="accent">answered</Tag>
              ) : null
            }
          >
            <div className="flex flex-col gap-3">
              <Trace
                width={100}
                height={26}
                series={[{ values, color: 'var(--accent)' }]}
                yDomain={[lo - pad, hi + pad]}
                zeroLine
              />
              <div className="flex flex-wrap gap-1.5">
                {candidates.map((option) => {
                  const chosen = round.answer === option;
                  const isAnswer = revealed && option === round.trueActivation;
                  return (
                    <button
                      key={option}
                      type="button"
                      disabled={revealed}
                      onClick={() => dispatch({ type: 'ANSWER_ACTIVATION', roundIndex: index, value: option })}
                      aria-pressed={chosen}
                      className={cx(
                        'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors disabled:cursor-default',
                        isAnswer
                          ? 'border-good bg-good/10 text-good'
                          : chosen
                            ? revealed
                              ? 'border-bad bg-bad/10 text-bad'
                              : 'border-accent bg-accent-dim text-primary'
                            : 'border-line-strong bg-raised text-secondary hover:border-accent'
                      )}
                      style={{ borderRadius: 'var(--radius)' }}
                    >
                      {ACTIVATION_LABEL[option]}
                    </button>
                  );
                })}
              </div>
            </div>
          </Panel>
        );
      })}
    </div>
  );
}

function IdentifySubmit({
  state,
  revealed,
  onSubmit,
}: {
  state: NeuronTuningState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const answered = state.rounds.filter((r) => r.answer !== null).length;
  const blocked = answered < state.rounds.length;

  return (
    <>
      {!revealed && blocked && <span className="label">{state.rounds.length - answered} round(s) left</span>}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || revealed}
        title={blocked ? `${state.rounds.length - answered} round(s) left` : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}
