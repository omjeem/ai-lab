'use client';

/**
 * Chapter 2.4 — Overfitting & Regularization.
 *
 * Every control refits real ridge regression by Gaussian elimination on the
 * normal equations — there is no cached lookup table of "what degree 7 looks
 * like". Train and validation points are a genuinely disjoint split, which is
 * the only reason watching them diverge means anything.
 *
 * Model-free, same as the rest of World 2: nothing for `ModelGate` to gate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  type OverfitFitAction,
  type OverfitFitConfig,
  type OverfitFitState,
} from '@/engines/overfitFitEngine';
import { Button, Panel, Readout, Slider, Tag } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const PLOT = 100;

export function OverfitCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as OverfitFitConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<OverfitFitState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: OverfitFitAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">sampling the data</span>
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
  state: OverfitFitState;
  dispatch: (action: OverfitFitAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  const resetRun = useCallback(() => dispatch({ type: 'RESET' }), [dispatch]);
  useRetrySignal(resetRun);

  const gapMode = state.rules.passCriteria.metric === 'generalizationGap';
  const ceiling = state.config.maxValidationLoss;
  const overCeiling = ceiling !== undefined && state.validationLoss > ceiling;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-3 lg:flex-row lg:items-start">
          <FitPlot state={state} />

          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <Panel label="loss">
              <div className="flex flex-col gap-3">
                <LossBars state={state} ceiling={ceiling} />

                <div className="flex flex-wrap gap-x-6 gap-y-2">
                  <Readout label="train" value={state.trainLoss} size="sm" />
                  <Readout label="validation" value={state.validationLoss} size="sm" tone={overCeiling ? 'bad' : 'accent'} />
                  {gapMode && (
                    <Readout
                      label="gap"
                      value={result.value}
                      size="sm"
                      tone={result.passed ? 'good' : 'warn'}
                    />
                  )}
                </div>

                {overCeiling && (
                  <Tag tone="bad">
                    validation loss above {ceiling!.toFixed(2)} — a flat, useless fit does not count as
                    a small gap
                  </Tag>
                )}
              </div>
            </Panel>

            {state.config.degreeLocked ? (
              <Panel label="polynomial degree">
                <Tag tone="warn">locked at degree {state.degree}</Tag>
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  More capacity than this data deserves. The ridge penalty below is the only lever.
                </p>
              </Panel>
            ) : (
              <Panel label="polynomial degree">
                <Slider
                  label="degree"
                  value={state.degree}
                  min={state.config.degreeRange?.[0] ?? 1}
                  max={state.config.degreeRange?.[1] ?? 12}
                  step={1}
                  onChange={(value) => dispatch({ type: 'SET_DEGREE', value })}
                  format={(value) => value.toFixed(0)}
                />
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  More degree means more capacity to bend toward every point — including the noise.
                </p>
              </Panel>
            )}

            <Panel label="ridge penalty">
              <Slider
                label="λ"
                value={state.lambda}
                min={state.config.lambdaRange?.[0] ?? 0}
                max={state.config.lambdaRange?.[1] ?? 1}
                step={(state.config.lambdaRange?.[1] ?? 1) / 200}
                onChange={(value) => dispatch({ type: 'SET_LAMBDA', value })}
                format={(value) => value.toFixed(3)}
              />
              <p className="mt-2 text-xs leading-relaxed text-muted">
                Makes large coefficients expensive, so the fit has to justify complexity with real
                signal rather than chasing every point.
              </p>
            </Panel>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        <Button variant="primary" className="ml-auto" onClick={() => onSubmit(evaluate(state))}>
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

/* ── the fit ────────────────────────────────────────────────── */

function FitPlot({ state }: { state: OverfitFitState }) {
  const { xDomain, yDomain } = useMemo(() => computeDomain(state), [state]);

  const toScreen = useCallback(
    (x: number, y: number): [number, number] => [
      ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * PLOT,
      PLOT - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * PLOT,
    ],
    [xDomain, yDomain]
  );

  const path = state.curve
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toScreen(p[0], p[1])[0].toFixed(2)} ${toScreen(p[0], p[1])[1].toFixed(2)}`)
    .join(' ');

  return (
    <Panel label="the fit" flush className="w-full shrink-0 lg:w-[420px]">
      <div className="p-2">
        <svg
          viewBox={`0 0 ${PLOT} ${PLOT}`}
          className="block w-full"
          role="img"
          aria-label={`Degree ${state.degree} fit over ${state.trainSet.length} training and ${state.validationSet.length} validation points`}
        >
          <defs>
            <clipPath id="overfit-plot-clip">
              <rect x={0} y={0} width={PLOT} height={PLOT} />
            </clipPath>
          </defs>

          <rect x={0} y={0} width={PLOT} height={PLOT} fill="var(--surface-inset)" />

          <g clipPath="url(#overfit-plot-clip)">
            {path && <path d={path} fill="none" stroke="var(--accent)" strokeWidth={0.9} />}

            {state.validationSet.map(([x, y], i) => {
              const [sx, sy] = toScreen(x, y);
              return (
                <circle
                  key={`val-${i}`}
                  cx={sx}
                  cy={sy}
                  r={1.9}
                  fill="none"
                  stroke="var(--text-secondary)"
                  strokeWidth={1}
                />
              );
            })}

            {state.trainSet.map(([x, y], i) => {
              const [sx, sy] = toScreen(x, y);
              return <circle key={`train-${i}`} cx={sx} cy={sy} r={1.5} fill="var(--text-primary)" />;
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

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3} fill="var(--text-primary)" />
        </svg>
        <span className="label">train</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3} fill="none" stroke="var(--text-secondary)" strokeWidth={1.4} />
        </svg>
        <span className="label">validation</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={12} height={10} aria-hidden>
          <line x1={0} y1={5} x2={12} y2={5} stroke="var(--accent)" strokeWidth={1.5} />
        </svg>
        <span className="label">fit</span>
      </span>
    </div>
  );
}

/** Fixed x range (the engine samples in [-1, 1]); y from every plotted series. */
function computeDomain(state: OverfitFitState): { xDomain: [number, number]; yDomain: [number, number] } {
  const ys = [
    ...state.trainSet.map(([, y]) => y),
    ...state.validationSet.map(([, y]) => y),
    ...state.curve.map(([, y]) => y),
  ].filter((y) => Number.isFinite(y));

  const yLo = ys.length ? Math.min(...ys) : -1;
  const yHi = ys.length ? Math.max(...ys) : 1;
  const yPad = Math.max((yHi - yLo) * 0.12, 0.3);

  return { xDomain: [-1.08, 1.08], yDomain: [yLo - yPad, yHi + yPad] };
}

/* ── train vs validation, side by side ─────────────────────────── */

function LossBars({ state, ceiling }: { state: OverfitFitState; ceiling: number | undefined }) {
  const max = Math.max(state.trainLoss, state.validationLoss, ceiling ?? 0, 0.05) * 1.15;
  const width = 100;
  const height = 30;
  const barWidth = 30;
  const gap = 18;
  const trainH = (state.trainLoss / max) * (height - 4);
  const valH = (state.validationLoss / max) * (height - 4);
  const ceilingY = ceiling !== undefined ? height - (ceiling / max) * (height - 4) : null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="block h-20 w-full"
      role="img"
      aria-label={`Training loss ${state.trainLoss.toFixed(3)}, validation loss ${state.validationLoss.toFixed(3)}`}
    >
      <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line)" strokeWidth={0.3} />

      {ceilingY !== null && (
        <line
          x1={0}
          y1={ceilingY}
          x2={width}
          y2={ceilingY}
          stroke="var(--signal-warn)"
          strokeWidth={0.5}
          strokeDasharray="1.5 1.2"
        />
      )}

      <rect
        x={width / 2 - gap / 2 - barWidth}
        y={height - trainH}
        width={barWidth}
        height={trainH}
        fill="var(--text-primary)"
      />
      <rect
        x={width / 2 + gap / 2}
        y={height - valH}
        width={barWidth}
        height={valH}
        fill="var(--accent)"
      />
    </svg>
  );
}
