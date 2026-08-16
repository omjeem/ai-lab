'use client';

/**
 * Chapter 5.4 — Residuals & Layer Norm.
 *
 * The trace is a real per-layer L2 norm read straight from the model's
 * captured hidden states — see `residualToggleEngine.computeTrace`. Toggling
 * the residual path or layer norm off doesn't fabricate a hypothetical
 * network; it recomputes the trace from the same measured activations using
 * only what that configuration would actually carry forward.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import { Check, Play, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ResidualToggleAction,
  type ResidualToggleConfig,
  type ResidualToggleState,
} from '@/engines/residualToggleEngine';
import { hiddenStateModel, HIDDEN_STATE_MODEL_ID } from '@/models/hiddenStateModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Slider, Tag, Toggle } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function ResidualCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as ResidualToggleConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<ResidualToggleState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { hiddenStates: hiddenStateModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const loadSentence = useCallback(
    async (sentence: string) => {
      const nextConfig: ResidualToggleConfig = { ...config, sentence };
      const prepared = await prepare(nextConfig, { hiddenStates: hiddenStateModel });
      setState(initState(nextConfig, rules, prepared));
    },
    [config, rules]
  );

  const dispatch = useCallback((action: ResidualToggleAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={HIDDEN_STATE_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} loadSentence={loadSentence} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
  loadSentence,
}: {
  state: ResidualToggleState;
  dispatch: (action: ResidualToggleAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
  loadSentence: (sentence: string) => Promise<void>;
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const [revealed, setRevealed] = useState(false);
  const [runId, setRunId] = useState(0);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
    setRunId((n) => n + 1);
  }, [dispatch]);

  useRetrySignal(resetRun);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div key={runId} className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'predict-trace' && (
            <PredictTraceBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'toggle-stability' && <ToggleStabilityBoard state={state} dispatch={dispatch} />}
          {state.mode === 'minimise-drift' && (
            <MinimiseDriftBoard state={state} dispatch={dispatch} loadSentence={loadSentence} />
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Actions
          state={state}
          revealed={revealed}
          onSubmit={() => {
            onSubmit(evaluate(state));
            dispatch({ type: 'SUBMIT' });
            setRevealed(true);
          }}
        />
      </div>
    </div>
  );
}

/* ── the trace chart ────────────────────────────────────────── */

const CHART_W = 640;
const CHART_H = 220;
const PAD_X = 36;
const PAD_Y = 16;

function layerX(layer: number, layerCount: number): number {
  return PAD_X + (layer / Math.max(1, layerCount - 1)) * (CHART_W - 2 * PAD_X);
}

function valueY(value: number, yMin: number, yMax: number): number {
  const span = Math.max(yMax - yMin, 1e-9);
  return CHART_H - PAD_Y - ((value - yMin) / span) * (CHART_H - 2 * PAD_Y);
}

function polyline(values: number[], yMin: number, yMax: number): string {
  return values.map((v, i) => `${layerX(i, values.length)},${valueY(v, yMin, yMax)}`).join(' ');
}

/**
 * The dominant visual for all three levels. `trace` may be `null` to draw the
 * axes without revealing the real line (predict-trace, pre-reveal) — guess
 * markers still render on top so the player sees where their own estimates
 * land relative to each other and to the (still-hidden) real shape.
 */
function TraceChart({
  layerCount,
  trace,
  reference,
  band,
  markers,
  yDomain,
}: {
  layerCount: number;
  trace: number[] | null;
  reference?: number[];
  band?: [number, number];
  markers?: { layer: number; value: number; tone: 'accent' | 'good' | 'bad' }[];
  yDomain: [number, number];
}) {
  const reduce = useReducedMotion();
  const [yMin, yMax] = yDomain;
  const toneColor = { accent: 'var(--accent)', good: 'var(--good)', bad: 'var(--bad)' };

  return (
    <Panel label="activation trace" flush>
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="w-full" role="img" aria-label="Activation norm by layer">
        {band && (
          <rect
            x={PAD_X}
            y={valueY(band[1], yMin, yMax)}
            width={CHART_W - 2 * PAD_X}
            height={Math.max(0, valueY(band[0], yMin, yMax) - valueY(band[1], yMin, yMax))}
            fill="var(--good)"
            opacity={0.08}
          />
        )}
        {Array.from({ length: layerCount }, (_, l) => (
          <line
            key={l}
            x1={layerX(l, layerCount)}
            x2={layerX(l, layerCount)}
            y1={PAD_Y}
            y2={CHART_H - PAD_Y}
            stroke="var(--line-faint)"
            strokeWidth={1}
          />
        ))}
        {reference && (
          <polyline
            points={polyline(reference, yMin, yMax)}
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth={1.5}
            strokeDasharray="3 3"
          />
        )}
        {trace && (
          <polyline
            points={polyline(trace, yMin, yMax)}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
            style={{ transition: reduce ? undefined : 'all 160ms ease-out' }}
          />
        )}
        {trace &&
          trace.map((v, i) => (
            <circle key={i} cx={layerX(i, layerCount)} cy={valueY(v, yMin, yMax)} r={3} fill="var(--accent)" />
          ))}
        {markers?.map((m, i) => (
          <circle
            key={i}
            cx={layerX(m.layer, layerCount)}
            cy={valueY(m.value, yMin, yMax)}
            r={4}
            fill={toneColor[m.tone]}
            stroke="var(--surface-panel)"
            strokeWidth={1.5}
          />
        ))}
        {Array.from({ length: layerCount }, (_, l) => (
          <text
            key={l}
            x={layerX(l, layerCount)}
            y={CHART_H - 2}
            textAnchor="middle"
            fontSize={9}
            fontFamily="var(--font-mono)"
            fill="var(--text-muted)"
          >
            {l}
          </text>
        ))}
      </svg>
    </Panel>
  );
}

/* ── level 1: predict the trace ─────────────────────────────── */

const GUESS_DOMAIN: [number, number] = [0, 40];

function PredictTraceBoard({
  state,
  dispatch,
  revealed,
}: {
  state: ResidualToggleState;
  dispatch: (action: ResidualToggleAction) => void;
  revealed: boolean;
}) {
  const [roundIndex, setRoundIndex] = useState(0);
  const round = state.rounds[roundIndex];

  const markers = useMemo(
    () =>
      state.rounds
        .filter((r) => r.estimate !== null)
        .map((r) => ({
          layer: r.layer,
          value: r.estimate!,
          tone: revealed ? (isRoundCorrect(r) ? ('good' as const) : ('bad' as const)) : ('accent' as const),
        })),
    [state.rounds, revealed]
  );

  if (!round) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <span className="label">
          round {roundIndex + 1} / {state.rounds.length} — layer {round.layer}
        </span>
        <div className="flex items-center gap-1">
          {state.rounds.map((r, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setRoundIndex(i)}
              aria-label={`Round ${i + 1}`}
              aria-current={i === roundIndex}
              className="h-2 w-2"
              style={{
                borderRadius: '50%',
                background:
                  r.estimate !== null ? 'var(--accent)' : i === roundIndex ? 'var(--text-secondary)' : 'var(--line-strong)',
              }}
            />
          ))}
        </div>
      </div>

      <TraceChart
        layerCount={state.config.layerCount}
        trace={revealed ? state.trace : null}
        markers={markers}
        yDomain={GUESS_DOMAIN}
      />

      <Slider
        label={`your estimate — layer ${round.layer}`}
        value={round.estimate ?? GUESS_DOMAIN[0]}
        min={GUESS_DOMAIN[0]}
        max={GUESS_DOMAIN[1]}
        step={0.1}
        disabled={revealed}
        onChange={(value) => dispatch({ type: 'ESTIMATE_TRACE', roundIndex, value })}
      />

      {revealed && (
        <p className="readout text-xs text-secondary">
          real value at layer {round.layer}: <span className="text-accent">{round.trueValue.toFixed(3)}</span> — yours:{' '}
          {(round.estimate ?? 0).toFixed(3)}
        </p>
      )}

      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? 'The real trace is now drawn in full. Both the residual path and layer norm are on here — the same combination every trained transformer ships with.'
          : 'The chart hides the real line until you submit. Drag the slider to place your estimate for this layer, then move to the next round.'}
      </p>
    </div>
  );
}

function isRoundCorrect(round: { estimate: number | null; trueValue: number }): boolean {
  if (round.estimate === null) return false;
  return Math.abs(round.estimate - round.trueValue) / Math.max(round.trueValue, 1e-6) < 0.15;
}

/* ── level 2: pull the plumbing ─────────────────────────────── */

function ToggleStabilityBoard({
  state,
  dispatch,
}: {
  state: ResidualToggleState;
  dispatch: (action: ResidualToggleAction) => void;
}) {
  const [low, high] = state.config.stableBand ?? [0.5, 2];
  const reference = Math.max(state.trace.reduce((a, b) => a + b, 0) / Math.max(state.trace.length, 1), 1e-9);
  const band: [number, number] = [reference * low, reference * high];

  const allValues = [...state.trace, ...state.referenceTrace, band[0], band[1]];
  const yDomain: [number, number] = [Math.min(...allValues) * 0.85, Math.max(...allValues) * 1.1];

  return (
    <div className="flex flex-col gap-3">
      <TraceChart
        layerCount={state.config.layerCount}
        trace={state.trace}
        reference={state.referenceTrace}
        band={band}
        yDomain={yDomain}
      />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="border border-line px-3" style={{ borderRadius: 'var(--radius)' }}>
          <Toggle
            label="residual path"
            checked={state.residual}
            onChange={(value) => dispatch({ type: 'SET_RESIDUAL', value })}
          />
        </div>
        <div className="border border-line px-3" style={{ borderRadius: 'var(--radius)' }}>
          <Toggle
            label="layer norm"
            checked={state.layerNorm}
            onChange={(value) => dispatch({ type: 'SET_LAYER_NORM', value })}
          />
        </div>
      </div>

      <p className="text-xs leading-relaxed text-muted">
        The shaded band is the stable zone — within {low}×–{high}× this configuration&apos;s own average magnitude.
        The dashed line is the reference trace with both mechanisms on, always shown for comparison. With both
        toggles off, only each block&apos;s own delta survives from one layer to the next; the trained model never
        actually runs that way.
      </p>
    </div>
  );
}

/* ── level 3: minimise the drift ────────────────────────────── */

function logSliderValue(value: number, min: number, max: number): number {
  const clamped = Math.max(min, Math.min(max, value));
  return (Math.log10(clamped) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
}

function logSliderToValue(t: number, min: number, max: number): number {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return Math.pow(10, logMin + t * (logMax - logMin));
}

function MinimiseDriftBoard({
  state,
  dispatch,
  loadSentence,
}: {
  state: ResidualToggleState;
  dispatch: (action: ResidualToggleAction) => void;
  loadSentence: (sentence: string) => Promise<void>;
}) {
  const [sentence, setSentence] = useState(state.config.sentence);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [scaleMin, scaleMax] = state.config.residualScaleRange ?? [0, 2];
  const [epsMin, epsMax] = state.config.normEpsilonRange ?? [1e-7, 1];

  const result = evaluate(state);
  const collapsing = (result.breakdown.collapsePenalty ?? 0) > 0;

  const allValues = [...state.trace, ...state.referenceTrace];
  const yDomain: [number, number] = [Math.min(...allValues, 0) * 0.9, Math.max(...allValues, 1) * 1.15];

  const tryCustomSentence = async () => {
    const text = sentence.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    try {
      await loadSentence(text);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      {state.config.allowUserSentence && (
        <div className="flex flex-wrap items-center gap-1.5">
          <input
            value={sentence}
            onChange={(event) => setSentence(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void tryCustomSentence()}
            disabled={busy}
            maxLength={220}
            aria-label="Your sentence"
            className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-60"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={() => void tryCustomSentence()} disabled={busy || sentence.trim() === ''}>
            <Play size={13} strokeWidth={2} />
            {busy ? 'running' : 'load sentence'}
          </Button>
        </div>
      )}
      {error && <p className="readout text-xs text-bad">the model could not run that — {error}</p>}

      <TraceChart layerCount={state.config.layerCount} trace={state.trace} reference={state.referenceTrace} yDomain={yDomain} />

      <Slider
        label="residual scale"
        value={state.residualScale}
        min={scaleMin}
        max={scaleMax}
        step={0.01}
        onChange={(value) => dispatch({ type: 'SET_RESIDUAL_SCALE', value })}
      />

      <Slider
        label="normalisation epsilon (log scale)"
        value={logSliderValue(state.normEpsilon, epsMin, epsMax)}
        min={0}
        max={1}
        step={0.001}
        format={() => state.normEpsilon.toExponential(2)}
        onChange={(t) => dispatch({ type: 'SET_NORM_EPSILON', value: logSliderToValue(t, epsMin, epsMax) })}
      />

      <div className="flex flex-wrap items-center gap-2">
        <span className="label">
          drift {(result.breakdown.drift ?? 0).toFixed(5)} · variance {(result.breakdown.variance ?? 0).toFixed(3)}
        </span>
        {collapsing && (
          <Tag tone="bad">representation collapsing — penalty {(result.breakdown.collapsePenalty ?? 0).toFixed(2)}</Tag>
        )}
      </div>

      <p className="text-xs leading-relaxed text-muted">
        A smaller epsilon keeps drift down. Pushing the residual scale toward zero looks tempting — it flattens the
        trace, which looks stable — but the representation collapses along with it, and that&apos;s measured and
        penalised separately.
      </p>
    </div>
  );
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: ResidualToggleState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = state.mode === 'predict-trace' && state.rounds.some((r) => r.estimate === null);
  const hint =
    state.mode === 'predict-trace'
      ? `${state.rounds.filter((r) => r.estimate === null).length} round${
          state.rounds.filter((r) => r.estimate === null).length === 1 ? '' : 's'
        } left`
      : '';

  return (
    <>
      {!revealed && blocked && <span className="label">{hint}</span>}
      {revealed && <Tag tone="accent">revealed</Tag>}
      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || (state.mode === 'predict-trace' && revealed)}
        title={blocked ? hint : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}
