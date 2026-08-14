'use client';

/**
 * Chapter 2.3 — Gradient Descent.
 *
 * The surface is a real heightmap: every cell is the analytic loss at that
 * point (or the injected surface's, for the stretch level), computed the same
 * way the engine computes it — nothing here is a painted illustration. The
 * trajectory is `state.history`, one dot per real step.
 *
 * Model-free, same as the Perceptron and Loss Functions canvases: nothing for
 * `ModelGate` to gate, so it is skipped entirely.
 */
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { Check, ChevronsRight, RotateCcw, StepForward } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  surfaceValue,
  type GradientDescentAction,
  type GradientDescentConfig,
  type GradientDescentState,
} from '@/engines/gradientDescentEngine';
import { Button, Panel, Readout, Slider, Tag } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const PLOT = 100;
const GRID_RES = 30;

export function GradientDescentCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as GradientDescentConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<GradientDescentState | null>(null);

  // Synchronous: every surface here is analytic, computed from the level's
  // config. There is no download.
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: GradientDescentAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">seeding the surface</span>
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
  state: GradientDescentState;
  dispatch: (action: GradientDescentAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const resetRun = useCallback(() => dispatch({ type: 'RESET' }), [dispatch]);
  useRetrySignal(resetRun);

  const exhausted = state.steps >= state.config.maxSteps;
  const [x, y] = [state.position[0]!, state.position[1]!];
  const finite = Number.isFinite(x) && Number.isFinite(y);
  const currentLoss = state.history.at(-1)?.loss ?? 0;
  const currentGradientNorm = state.history.at(-1)?.gradientNorm ?? 0;

  const showMomentum = state.config.momentumRange !== undefined;
  const showNoise = (state.config.noiseRange?.[1] ?? 0) > 0;
  const isLiveNet = state.config.surface === 'liveNet';

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-3 lg:flex-row lg:items-start">
          <SurfacePlot state={state} />

          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <Panel label="descent">
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <Readout
                    label="loss"
                    value={finite ? currentLoss : NaN}
                    size="lg"
                    tone={!finite ? 'bad' : state.converged ? 'good' : 'accent'}
                  />
                  <Readout label="|gradient|" value={finite ? currentGradientNorm : NaN} size="sm" />
                  <Readout label="steps" value={`${state.steps} / ${state.config.maxSteps}`} size="sm" />
                </div>

                <div className="flex flex-wrap gap-x-6 gap-y-1">
                  <div className="flex items-baseline gap-2">
                    <span className="label">x</span>
                    <span className="readout text-sm text-primary">{finite ? x.toFixed(3) : '—'}</span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="label">y</span>
                    <span className="readout text-sm text-primary">{finite ? y.toFixed(3) : '—'}</span>
                  </div>
                </div>

                {!finite && <Tag tone="bad">diverged — numbers stopped being finite</Tag>}
                {finite && state.convergedAtStep !== null && (
                  <Tag tone="good">converged at step {state.convergedAtStep}</Tag>
                )}
                {finite && exhausted && !state.converged && (
                  <Tag tone="warn">step budget spent without converging</Tag>
                )}
              </div>
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
              <p className="mt-2 text-xs leading-relaxed text-muted">
                Too small and it crawls. Too large and each step overshoots the far side of the bowl —
                watch the trajectory bounce instead of settle.
              </p>
            </Panel>

            {showMomentum && (
              <Panel label="momentum">
                <Slider
                  label="β"
                  value={state.momentum}
                  min={state.config.momentumRange?.[0] ?? 0}
                  max={state.config.momentumRange?.[1] ?? 0.99}
                  step={0.01}
                  onChange={(value) => dispatch({ type: 'SET_MOMENTUM', value })}
                  format={(value) => value.toFixed(2)}
                />
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  Carries a fraction of the last step forward, which is what lets a small learning rate
                  keep moving through the flat axis of a ravine instead of crawling it.
                </p>
              </Panel>
            )}

            {showNoise && (
              <Panel label="gradient noise">
                <Slider
                  label="σ"
                  value={state.noiseScale}
                  min={state.config.noiseRange?.[0] ?? 0}
                  max={state.config.noiseRange?.[1] ?? 0.5}
                  step={0.01}
                  onChange={(value) => dispatch({ type: 'SET_NOISE', value })}
                  format={(value) => value.toFixed(2)}
                />
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  Jitters every step by a seeded random amount. Zero noise settles into whichever basin
                  is nearest the start — some noise is what lets a run climb back out of it.
                </p>
              </Panel>
            )}

            {isLiveNet && !state.surfaceFn && (
              <p className="text-xs leading-relaxed text-muted">
                Stretch level: this surface is meant to be sliced from the live-trained network in
                “Layers &amp; Forward Pass.” That chapter isn&rsquo;t wired up yet, so this is a flat
                placeholder rather than a fabricated one.
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        <Button onClick={() => dispatch({ type: 'STEP' })} disabled={exhausted}>
          <StepForward size={13} strokeWidth={2} />
          Step
        </Button>
        <Button onClick={() => dispatch({ type: 'RUN', steps: 10 })} disabled={exhausted}>
          <ChevronsRight size={13} strokeWidth={2} />
          ×10
        </Button>
        <Button onClick={() => dispatch({ type: 'RUN', steps: 50 })} disabled={exhausted}>
          <ChevronsRight size={13} strokeWidth={2} />
          ×50
        </Button>

        {exhausted && <span className="label">budget spent — reset to run again</span>}

        <Button variant="primary" className="ml-auto" onClick={() => onSubmit(evaluate(state))}>
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

/* ── the surface ────────────────────────────────────────────── */

function SurfacePlot({ state }: { state: GradientDescentState }) {
  const half = useMemo(() => domainHalfExtent(state.config), [state.config]);
  const toScreen = useCallback(
    (x: number, y: number): [number, number] => [
      ((x + half) / (2 * half)) * PLOT,
      ((half - y) / (2 * half)) * PLOT,
    ],
    [half]
  );

  const points = state.history.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const last = points.at(-1);
  const start = points[0];

  const path = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toScreen(p.x, p.y)[0].toFixed(2)} ${toScreen(p.x, p.y)[1].toFixed(2)}`)
    .join(' ');

  // A dot per step is too many past a few dozen; thin them out evenly instead.
  const dotStride = Math.max(1, Math.ceil(points.length / 60));
  const dots = points.filter((_, i) => i % dotStride === 0 || i === points.length - 1);

  return (
    <Panel label="loss surface" flush className="w-full shrink-0 lg:w-[420px]">
      <div className="p-2">
        <svg
          viewBox={`0 0 ${PLOT} ${PLOT}`}
          className="block w-full"
          role="img"
          aria-label={`Loss surface with the descent trajectory across ${points.length} steps`}
        >
          <defs>
            <clipPath id="gd-plot-clip">
              <rect x={0} y={0} width={PLOT} height={PLOT} />
            </clipPath>
          </defs>

          <HeightMap config={state.config} surfaceFn={state.surfaceFn} half={half} />

          <g clipPath="url(#gd-plot-clip)">
            {path && <path d={path} fill="none" stroke="var(--text-primary)" strokeWidth={0.7} opacity={0.85} />}

            {dots.map((p, i) => {
              const [sx, sy] = toScreen(p.x, p.y);
              return <circle key={i} cx={sx} cy={sy} r={0.9} fill="var(--text-primary)" opacity={0.6} />;
            })}

            {start && (
              <circle
                cx={toScreen(start.x, start.y)[0]}
                cy={toScreen(start.x, start.y)[1]}
                r={2.4}
                fill="none"
                stroke="var(--text-primary)"
                strokeWidth={0.8}
              />
            )}

            {last && (
              <circle
                cx={toScreen(last.x, last.y)[0]}
                cy={toScreen(last.x, last.y)[1]}
                r={2.4}
                fill="var(--accent)"
                stroke="var(--surface-base)"
                strokeWidth={0.6}
              />
            )}
          </g>
        </svg>

        <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-2">
          <Legend />
        </div>
      </div>
    </Panel>
  );
}

/**
 * The heightmap grid. Isolated and memoized on the level's own config, so a
 * per-step position update never recomputes 900 cells of real loss values —
 * only the trajectory layer above it re-renders.
 */
const HeightMap = memo(function HeightMap({
  config,
  surfaceFn,
  half,
}: {
  config: GradientDescentConfig;
  surfaceFn: GradientDescentState['surfaceFn'];
  half: number;
}) {
  const grid = useMemo(() => {
    const cell = (2 * half) / GRID_RES;
    const values: number[] = [];
    for (let row = 0; row < GRID_RES; row++) {
      for (let col = 0; col < GRID_RES; col++) {
        const x = -half + (col + 0.5) * cell;
        const y = half - (row + 0.5) * cell;
        const v = surfaceFn ? surfaceFn(x, y) : surfaceValue(config.surface, x, y);
        values.push(Number.isFinite(v) ? v : 0);
      }
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const flat = max - min < 1e-9;
    return { values, min, max, flat, cell };
  }, [config.surface, surfaceFn, half]);

  const bestIndex = useMemo(() => {
    if (grid.flat) return -1;
    let best = 0;
    for (let i = 1; i < grid.values.length; i++) if (grid.values[i]! < grid.values[best]!) best = i;
    return best;
  }, [grid]);

  return (
    <g>
      {grid.values.map((value, i) => {
        const row = Math.floor(i / GRID_RES);
        const col = i % GRID_RES;
        const x = (col / GRID_RES) * PLOT;
        const y = (row / GRID_RES) * PLOT;
        const w = PLOT / GRID_RES + 0.4;

        const t = grid.flat ? 0.5 : (grid.max - value) / (grid.max - grid.min);
        // Quantised into bands, so it reads as elevation contours rather than
        // a smooth gradient.
        const band = Math.round(t * 12) / 12;
        const mix = Math.max(4, Math.min(96, band * 92 + 4));

        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={w}
            height={w}
            fill={`color-mix(in oklab, var(--accent) ${mix.toFixed(0)}%, var(--surface-inset))`}
          />
        );
      })}

      {bestIndex >= 0 &&
        (() => {
          const row = Math.floor(bestIndex / GRID_RES);
          const col = bestIndex % GRID_RES;
          const cx = ((col + 0.5) / GRID_RES) * PLOT;
          const cy = ((row + 0.5) / GRID_RES) * PLOT;
          return (
            <g stroke="var(--text-primary)" strokeWidth={0.5} opacity={0.7}>
              <line x1={cx - 2.2} y1={cy} x2={cx + 2.2} y2={cy} />
              <line x1={cx} y1={cy - 2.2} x2={cx} y2={cy + 2.2} />
            </g>
          );
        })()}
    </g>
  );
});

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3} fill="none" stroke="var(--text-primary)" strokeWidth={1.2} />
        </svg>
        <span className="label">start</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3.5} fill="var(--accent)" />
        </svg>
        <span className="label">current</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <line x1={0} y1={5} x2={10} y2={5} stroke="var(--text-primary)" strokeWidth={1.5} />
          <line x1={5} y1={0} x2={5} y2={10} stroke="var(--text-primary)" strokeWidth={1.5} />
        </svg>
        <span className="label">lowest on grid</span>
      </span>
    </div>
  );
}

/** Square half-extent around the origin, sized from the level's own start point. */
function domainHalfExtent(config: GradientDescentConfig): number {
  const [sx, sy] = config.startPoint;
  const base = Math.max(Math.abs(sx ?? 0), Math.abs(sy ?? 0));
  return Math.max(3, base * 1.35);
}
