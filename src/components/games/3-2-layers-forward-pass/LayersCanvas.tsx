'use client';

/**
 * Chapter 3.2 — Layers & Forward Pass.
 *
 * A real `TinyNet`, trained with real backprop in the browser — no framework,
 * no animation standing in for training. The boundary heatmap is sampled from
 * actual forward passes on every redraw (`boundaryGrid`), and training runs in
 * short chunks scheduled across animation frames so the boundary is visibly
 * moving rather than freezing the tab for a few hundred epochs.
 *
 * Model-free in the sense the other World 2/3 canvases are: the net trains
 * synchronously from a seed, so there is nothing for `ModelGate` to gate.
 */
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { Check, Minus, Play, Plus, RotateCcw, Square, StepForward, X } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  boundaryGrid,
  hiddenRepresentation,
  type NetworkBoundaryAction,
  type NetworkBoundaryConfig,
  type NetworkBoundaryState,
} from '@/engines/networkBoundaryEngine';
import type { Activation } from '@/models/tinyNetTrainer';
import type { ToyPoint } from '@/engines/toyDatasets';
import { Button, Meter, Panel, Readout, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const PLOT = 100;
const GRID_RES = 26;
const BOUNDARY_DOMAIN = 1.2;
const TRAIN_CHUNK = 15;
const HIDDEN_SAMPLE_CAP = 160;

const ACTIVATION_LABEL: Partial<Record<Activation, string>> = {
  tanh: 'tanh',
  relu: 'ReLU',
  leakyRelu: 'leaky ReLU',
  sigmoid: 'sigmoid',
  gelu: 'GELU',
};

export function LayersCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as NetworkBoundaryConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<NetworkBoundaryState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: NetworkBoundaryAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">initialising the network</span>
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
  state: NetworkBoundaryState;
  dispatch: (action: NetworkBoundaryAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  const [isTraining, setIsTraining] = useState(false);
  const exhausted = state.epochsTrained >= state.config.maxEpochs;

  // Chains itself across animation frames: each tick trains one short chunk,
  // commits, and — because `state.epochsTrained` is a dependency — schedules
  // the next tick. Stops itself at the epoch budget or when toggled off.
  useEffect(() => {
    if (!isTraining || exhausted) {
      if (exhausted) setIsTraining(false);
      return;
    }
    const id = requestAnimationFrame(() => dispatch({ type: 'TRAIN', epochs: TRAIN_CHUNK }));
    return () => cancelAnimationFrame(id);
  }, [isTraining, exhausted, state.epochsTrained, dispatch]);

  const resetRun = useCallback(() => {
    setIsTraining(false);
    dispatch({ type: 'RESET' });
  }, [dispatch]);
  useRetrySignal(resetRun);

  const budget = state.config.parameterBudget;
  const params = state.net.parameterCount;
  const overBudget = budget !== undefined && params > budget;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
          <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[440px]">
            <BoundaryPlot state={state} />
            {state.config.showHiddenRepresentations && <HiddenLayers state={state} />}
          </div>

          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <Panel label="training">
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <Readout
                    label={result.metric === 'efficiencyScore' ? 'efficiency' : 'accuracy'}
                    value={result.value}
                    size="lg"
                    tone={result.passed ? 'good' : 'accent'}
                  />
                  {result.metric === 'efficiencyScore' && (
                    <Readout label="raw accuracy" value={result.breakdown.accuracy ?? 0} size="sm" />
                  )}
                  <Readout label="epochs" value={`${state.epochsTrained} / ${state.config.maxEpochs}`} size="sm" />
                </div>

                <div className="flex items-baseline justify-between gap-3">
                  <span className="label">parameters</span>
                  <span className={cx('readout text-sm', overBudget ? 'text-bad' : 'text-primary')}>
                    {params}
                    {budget !== undefined && <span className="text-muted"> / {budget}</span>}
                  </span>
                </div>
                {budget !== undefined && (
                  <Meter value={params} max={Math.max(params, budget) * 1.15} threshold={budget} tone={overBudget ? 'bad' : 'good'} />
                )}

                <LossHistory state={state} />

                {exhausted && <Tag tone="warn">epoch budget spent — reset to train again</Tag>}
              </div>
            </Panel>

            <Panel label="architecture">
              <ArchitectureEditor state={state} dispatch={dispatch} disabled={isTraining} />
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

            {state.config.activationOptions && (
              <Panel label="activation">
                <div className="flex flex-wrap gap-1.5">
                  {state.config.activationOptions.map((option) => {
                    const active = state.activation === option;
                    return (
                      <button
                        key={option}
                        type="button"
                        disabled={isTraining}
                        onClick={() => dispatch({ type: 'SET_ACTIVATION', value: option })}
                        aria-pressed={active}
                        className={cx(
                          'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40',
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
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Button
          onClick={() => dispatch({ type: 'TRAIN', epochs: TRAIN_CHUNK })}
          disabled={exhausted || isTraining}
        >
          <StepForward size={13} strokeWidth={2} />
          ×{TRAIN_CHUNK}
        </Button>
        <Button onClick={() => setIsTraining((t) => !t)} disabled={exhausted && !isTraining}>
          {isTraining ? (
            <>
              <Square size={13} strokeWidth={2} />
              Stop
            </>
          ) : (
            <>
              <Play size={13} strokeWidth={2} />
              Train
            </>
          )}
        </Button>

        <Button variant="primary" className="ml-auto" onClick={() => onSubmit(evaluate(state))}>
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

/* ── the decision boundary ─────────────────────────────────────── */

function BoundaryPlot({ state }: { state: NetworkBoundaryState }) {
  const toScreen = useCallback(
    (x: number, y: number): [number, number] => [
      ((x + BOUNDARY_DOMAIN) / (2 * BOUNDARY_DOMAIN)) * PLOT,
      PLOT - ((y + BOUNDARY_DOMAIN) / (2 * BOUNDARY_DOMAIN)) * PLOT,
    ],
    []
  );

  const points = useMemo(
    () => [...state.trainSet.map((p) => ({ ...p, held: false })), ...state.validationSet.map((p) => ({ ...p, held: true }))],
    [state.trainSet, state.validationSet]
  );

  return (
    <Panel label="decision boundary" flush className="w-full">
      <div className="p-2">
        <svg
          viewBox={`0 0 ${PLOT} ${PLOT}`}
          className="block w-full"
          role="img"
          aria-label={`Network decision boundary over ${points.length} points, ${state.epochsTrained} epochs trained`}
        >
          <BoundaryGridLayer net={state.net} />

          <g>
            {points.map((p, i) => {
              const [sx, sy] = toScreen(p.x[0], p.x[1]);
              return p.label === 1 ? (
                <circle
                  key={i}
                  cx={sx}
                  cy={sy}
                  r={p.held ? 1.6 : 1.2}
                  fill="var(--text-primary)"
                  opacity={p.held ? 1 : 0.75}
                  stroke={p.held ? 'var(--surface-base)' : 'none'}
                  strokeWidth={p.held ? 0.5 : 0}
                />
              ) : (
                <rect
                  key={i}
                  x={sx - (p.held ? 1.5 : 1.1)}
                  y={sy - (p.held ? 1.5 : 1.1)}
                  width={p.held ? 3 : 2.2}
                  height={p.held ? 3 : 2.2}
                  fill="none"
                  stroke="var(--text-primary)"
                  strokeWidth={p.held ? 0.9 : 0.7}
                  opacity={p.held ? 1 : 0.75}
                />
              );
            })}
          </g>
        </svg>

        <div className="flex flex-wrap items-center gap-3 px-1 pt-2">
          <Legend />
        </div>
      </div>
    </Panel>
  );
}

/**
 * The heatmap itself, isolated so the ~700-cell grid only redraws when the
 * net's own reference changes — training dispatches a new cloned net every
 * chunk, so this still refreshes every step without re-diffing on unrelated
 * renders of the parent.
 */
const BoundaryGridLayer = memo(function BoundaryGridLayer({ net }: { net: NetworkBoundaryState['net'] }) {
  const grid = useMemo(
    () => boundaryGrid({ net } as unknown as NetworkBoundaryState, GRID_RES),
    [net]
  );
  const cell = PLOT / GRID_RES;

  return (
    <g>
      {grid.map((row, r) =>
        row.map((value, c) => {
          const x = -BOUNDARY_DOMAIN + (2 * BOUNDARY_DOMAIN * c) / (GRID_RES - 1);
          const y = -BOUNDARY_DOMAIN + (2 * BOUNDARY_DOMAIN * r) / (GRID_RES - 1);
          const sx = ((x + BOUNDARY_DOMAIN) / (2 * BOUNDARY_DOMAIN)) * PLOT;
          const sy = PLOT - ((y + BOUNDARY_DOMAIN) / (2 * BOUNDARY_DOMAIN)) * PLOT;
          const mix = Math.max(4, Math.min(96, value * 100));
          return (
            <rect
              key={`${r}-${c}`}
              x={sx - cell / 2}
              y={sy - cell / 2}
              width={cell + 0.5}
              height={cell + 0.5}
              fill={`color-mix(in oklab, var(--accent) ${mix.toFixed(0)}%, var(--surface-inset))`}
            />
          );
        })
      )}
    </g>
  );
});

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3.5} fill="var(--text-primary)" />
        </svg>
        <span className="label">label 1</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <rect x={1} y={1} width={8} height={8} fill="none" stroke="var(--text-primary)" strokeWidth={1.4} />
        </svg>
        <span className="label">label 0</span>
      </span>
      <span className="label">larger marks = held out for validation</span>
    </div>
  );
}

/* ── loss / accuracy history ───────────────────────────────────── */

function LossHistory({ state }: { state: NetworkBoundaryState }) {
  const history = state.history;
  if (history.length === 0) {
    return <p className="text-xs text-muted">train to see the loss curve</p>;
  }

  const width = 100;
  const height = 26;
  const losses = history.flatMap((h) => [h.trainLoss, h.validationLoss]).filter(Number.isFinite);
  const hi = losses.length ? Math.max(...losses) : 1;
  const toX = (i: number) => (history.length <= 1 ? 0 : (i / (history.length - 1)) * width);
  const toY = (v: number) => height - Math.min(1, v / Math.max(hi, 1e-6)) * height;

  const trainPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(h.trainLoss).toFixed(2)}`).join(' ');
  const valPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(h.validationLoss).toFixed(2)}`).join(' ');
  const last = history.at(-1)!;

  return (
    <div className="flex flex-col gap-1.5">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block h-16 w-full" role="img" aria-label={`Train loss ${last.trainLoss.toFixed(3)}, validation loss ${last.validationLoss.toFixed(3)} after ${history.length} epochs`}>
        <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line-faint)" strokeWidth={0.3} />
        <path d={trainPath} fill="none" stroke="var(--text-primary)" strokeWidth={0.7} />
        <path d={valPath} fill="none" stroke="var(--accent)" strokeWidth={0.9} />
      </svg>
      <div className="flex flex-wrap items-center gap-4">
        <span className="flex items-center gap-1.5">
          <svg width={12} height={10} aria-hidden><line x1={0} y1={5} x2={12} y2={5} stroke="var(--text-primary)" strokeWidth={1.4} /></svg>
          <span className="label">train loss</span>
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={12} height={10} aria-hidden><line x1={0} y1={5} x2={12} y2={5} stroke="var(--accent)" strokeWidth={1.6} /></svg>
          <span className="label">validation loss</span>
        </span>
        <span className="label ml-auto">val accuracy {(last.validationAccuracy * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

/* ── architecture editor ───────────────────────────────────────── */

function ArchitectureEditor({
  state,
  dispatch,
  disabled,
}: {
  state: NetworkBoundaryState;
  dispatch: (action: NetworkBoundaryAction) => void;
  disabled: boolean;
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
                disabled={disabled || units <= 1}
                onClick={() => setUnits(hidden.map((u, j) => (j === i ? u - 1 : u)))}
                className="min-h-0! px-2! py-1.5!"
              >
                <Minus size={12} strokeWidth={2} />
                <span className="sr-only">fewer units in layer {i + 1}</span>
              </Button>
              <span className="readout w-6 text-center text-sm text-primary">{units}</span>
              <Button
                disabled={disabled || units >= limits.maxUnits}
                onClick={() => setUnits(hidden.map((u, j) => (j === i ? u + 1 : u)))}
                className="min-h-0! px-2! py-1.5!"
              >
                <Plus size={12} strokeWidth={2} />
                <span className="sr-only">more units in layer {i + 1}</span>
              </Button>
            </div>
            <Button
              disabled={disabled}
              variant="danger"
              onClick={() => setUnits(hidden.filter((_, j) => j !== i))}
              className="min-h-0! px-2! py-1!"
            >
              <X size={11} strokeWidth={2} />
              <span className="sr-only">remove layer {i + 1}</span>
            </Button>
          </div>
        ))}

        <Button
          disabled={disabled || hidden.length >= limits.maxHidden}
          onClick={() => setUnits([...hidden, Math.min(4, limits.maxUnits)])}
        >
          <Plus size={13} strokeWidth={2} />
          layer
        </Button>
      </div>

      <p className="text-xs leading-relaxed text-muted">
        Changing the architecture rebuilds the network from scratch — training so far is discarded.
      </p>
    </div>
  );
}

/* ── hidden representations (level 3) ──────────────────────────── */

function HiddenLayers({ state }: { state: NetworkBoundaryState }) {
  const hiddenCount = state.architecture.length - 2;

  const layers = useMemo(() => {
    if (hiddenCount <= 0) return [];
    const sample: ToyPoint[] = [...state.trainSet, ...state.validationSet].slice(0, HIDDEN_SAMPLE_CAP);
    const perLayer: { x: number; y: number; label: 0 | 1 }[][] = Array.from({ length: hiddenCount }, () => []);
    for (const point of sample) {
      const rep = hiddenRepresentation(state, point.x);
      for (let l = 0; l < hiddenCount; l++) {
        const acts = rep[l] ?? [];
        perLayer[l]!.push({ x: acts[0] ?? 0, y: acts[1] ?? 0, label: point.label });
      }
    }
    return perLayer;
  }, [state, hiddenCount]);

  if (layers.length === 0) return null;

  return (
    <Panel label="hidden representations — first two units per layer">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {layers.map((points, i) => (
          <MiniScatter key={i} points={points} label={i === layers.length - 1 ? 'last hidden' : `layer ${i + 1}`} />
        ))}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Each panel is where that layer actually put every point, real forward passes projected onto its
        first two units. Watch a tangled input become separable one layer at a time.
      </p>
    </Panel>
  );
}

function MiniScatter({ points, label }: { points: { x: number; y: number; label: 0 | 1 }[]; label: string }) {
  const size = 84;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const lo = Math.min(...xs, ...ys, -0.1);
  const hi = Math.max(...xs, ...ys, 0.1);
  const pad = Math.max((hi - lo) * 0.1, 0.15);
  const domain: [number, number] = [lo - pad, hi + pad];
  const toScreen = (v: number, flip: boolean) =>
    flip ? size - ((v - domain[0]) / (domain[1] - domain[0])) * size : ((v - domain[0]) / (domain[1] - domain[0])) * size;

  return (
    <div className="flex shrink-0 flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="border border-line" style={{ borderRadius: 'var(--radius)' }} role="img" aria-label={`${label} representation`}>
        <rect x={0} y={0} width={size} height={size} fill="var(--surface-inset)" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={toScreen(p.x, false)}
            cy={toScreen(p.y, true)}
            r={1.3}
            fill={p.label === 1 ? 'var(--accent)' : 'none'}
            stroke={p.label === 1 ? 'none' : 'var(--text-secondary)'}
            strokeWidth={0.8}
          />
        ))}
      </svg>
      <span className="label">{label}</span>
    </div>
  );
}
