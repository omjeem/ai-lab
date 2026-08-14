'use client';

/**
 * Chapter 2.2 — Loss Functions.
 *
 * The line is dragged by its two ends, straight in the plot — no slider stands
 * between the player and the fit. Every residual gets a square built on its own
 * drop-line, side length equal to the drop's own pixel length, so the area really
 * does grow with the square of the error rather than just gesturing at it.
 *
 * Model-free, so there is nothing for `ModelGate` to gate. See PerceptronCanvas.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  type LossMinimizationAction,
  type LossMinimizationConfig,
  type LossMinimizationState,
  type LossType,
} from '@/engines/lossMinimizationEngine';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules, ScoreResult } from '@/types/game';

const PLOT = 100;

const LOSS_LABEL: Record<LossType, string> = {
  mse: 'mean squared error',
  mae: 'mean absolute error',
  crossEntropy: 'cross-entropy',
  hinge: 'hinge',
};

export function LossCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as LossMinimizationConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<LossMinimizationState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: LossMinimizationAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">setting up the data</span>
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
  state: LossMinimizationState;
  dispatch: (action: LossMinimizationAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Kept local rather than `state.status === 'complete'`: match-loss answers
  // bump the status back to `active` on every subsequent click, which would
  // un-reveal a just-submitted quiz. Same latent bug documented for canvases
  // 1–3; see README "Keeping levels honest".
  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'match-loss' ? (
          <MatchLossBoard state={state} dispatch={dispatch} revealed={revealed} />
        ) : (
          <MinimizeBoard state={state} dispatch={dispatch} result={result} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        {state.mode === 'match-loss' ? (
          <MatchLossSubmit
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

/* ── minimize mode ──────────────────────────────────────────── */

function MinimizeBoard({
  state,
  dispatch,
  result,
}: {
  state: LossMinimizationState;
  dispatch: (action: LossMinimizationAction) => void;
  result: ScoreResult;
}) {
  return (
    <div className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-3 lg:flex-row lg:items-start">
      <ScatterPlot state={state} dispatch={dispatch} />

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="loss">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout
                label={LOSS_LABEL[state.lossType]}
                value={result.value}
                size="lg"
                tone={result.passed ? 'good' : 'accent'}
              />
              <Tag tone="accent">this is what&rsquo;s scored</Tag>
            </div>

            <div className="flex flex-wrap gap-x-6 gap-y-2">
              <Readout
                label="mse"
                value={result.breakdown.mse ?? 0}
                size="sm"
                tone={state.lossType === 'mse' ? 'accent' : 'neutral'}
              />
              <Readout
                label="mae"
                value={result.breakdown.mae ?? 0}
                size="sm"
                tone={state.lossType === 'mae' ? 'accent' : 'neutral'}
              />
            </div>

            <div className="flex flex-wrap gap-x-6 gap-y-1">
              <div className="flex items-baseline gap-2">
                <span className="label">slope</span>
                <span className="readout text-sm text-primary">{state.slope.toFixed(3)}</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="label">intercept</span>
                <span className="readout text-sm text-primary">{state.intercept.toFixed(3)}</span>
              </div>
            </div>
          </div>
        </Panel>

        <p className="text-xs leading-relaxed text-muted">
          Drag either end of the line — up, down, with a mouse or the arrow keys once it&rsquo;s focused.
          Every point drops a line to the fit, and the square beside it is that residual squared, made
          visible: its side is exactly as long as the drop.
        </p>
      </div>
    </div>
  );
}

function ScatterPlot({
  state,
  dispatch,
}: {
  state: LossMinimizationState;
  dispatch: (action: LossMinimizationAction) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [focused, setFocused] = useState<0 | 1 | null>(null);

  const { xDomain, yDomain, handleX } = useMemo(() => computeDomain(state.config), [state.config]);

  const toScreen = useCallback(
    (x: number, y: number): [number, number] => [
      ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * PLOT,
      PLOT - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * PLOT,
    ],
    [xDomain, yDomain]
  );
  const fromScreenY = useCallback(
    (sy: number) => yDomain[0] + (1 - sy / PLOT) * (yDomain[1] - yDomain[0]),
    [yDomain]
  );

  const moveEndpoint = useCallback(
    (which: 0 | 1, newY: number) => {
      const other = which === 0 ? 1 : 0;
      const x0 = handleX[which];
      const x1 = handleX[other];
      const y1 = state.slope * x1 + state.intercept;
      const dx = x0 - x1;
      const newSlope = dx === 0 ? state.slope : (newY - y1) / dx;
      const newIntercept = newY - newSlope * x0;
      dispatch({ type: 'SET_SLOPE', value: newSlope });
      dispatch({ type: 'SET_INTERCEPT', value: newIntercept });
    },
    [handleX, state.slope, state.intercept, dispatch]
  );

  const dragTo = useCallback(
    (which: 0 | 1, clientY: number) => {
      const box = svgRef.current?.getBoundingClientRect();
      if (!box || box.height === 0) return;
      const fracY = (clientY - box.top) / box.height;
      moveEndpoint(which, fromScreenY(fracY * PLOT));
    },
    [fromScreenY, moveEndpoint]
  );

  const points = state.config.points ?? [];
  const lineAt = (x: number) => state.slope * x + state.intercept;

  return (
    <Panel label="the fit" flush className="w-full shrink-0 lg:w-[420px]">
      <div className="p-2">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${PLOT} ${PLOT}`}
          className="block w-full touch-none"
          role="img"
          aria-label={`${points.length} data points with a draggable line fitted to them`}
        >
          <defs>
            <clipPath id="loss-plot-clip">
              <rect x={0} y={0} width={PLOT} height={PLOT} />
            </clipPath>
          </defs>

          <rect x={0} y={0} width={PLOT} height={PLOT} fill="var(--surface-inset)" />

          {yDomain[0] < 0 && yDomain[1] > 0 && (
            <line
              x1={0}
              y1={toScreen(0, 0)[1]}
              x2={PLOT}
              y2={toScreen(0, 0)[1]}
              stroke="var(--line)"
              strokeWidth={0.4}
            />
          )}

          <g clipPath="url(#loss-plot-clip)">
            {/* Squares first, underneath everything else that has to stay legible. */}
            {points.map(([x, y], i) => {
              const predicted = lineAt(x);
              const [sx, sy] = toScreen(x, y);
              const [, spy] = toScreen(x, predicted);
              const side = Math.abs(spy - sy);
              if (side < 0.5) return null;
              return (
                <rect
                  key={`sq-${i}`}
                  x={sx}
                  y={Math.min(sy, spy)}
                  width={side}
                  height={side}
                  fill="var(--accent)"
                  opacity={0.14}
                  stroke="var(--accent)"
                  strokeOpacity={0.4}
                  strokeWidth={0.4}
                />
              );
            })}

            {points.map(([x, y], i) => {
              const predicted = lineAt(x);
              const [sx, sy] = toScreen(x, y);
              const [, spy] = toScreen(x, predicted);
              return (
                <line
                  key={`drop-${i}`}
                  x1={sx}
                  y1={sy}
                  x2={sx}
                  y2={spy}
                  stroke="var(--text-secondary)"
                  strokeWidth={0.5}
                  strokeDasharray="1.2 1"
                />
              );
            })}

            <line
              x1={toScreen(xDomain[0], lineAt(xDomain[0]))[0]}
              y1={toScreen(xDomain[0], lineAt(xDomain[0]))[1]}
              x2={toScreen(xDomain[1], lineAt(xDomain[1]))[0]}
              y2={toScreen(xDomain[1], lineAt(xDomain[1]))[1]}
              stroke="var(--accent)"
              strokeWidth={0.9}
            />

            {points.map(([x, y], i) => {
              const [sx, sy] = toScreen(x, y);
              return <circle key={`pt-${i}`} cx={sx} cy={sy} r={1.9} fill="var(--text-primary)" />;
            })}
          </g>

          {[0, 1].map((which) => {
            const index = which as 0 | 1;
            const x = handleX[index];
            const y = lineAt(x);
            const [sx, sy] = toScreen(x, y);
            return (
              <g
                key={index}
                tabIndex={0}
                role="slider"
                aria-label={`${index === 0 ? 'left' : 'right'} end of the line, currently at y = ${y.toFixed(2)}. Drag, or use arrow up and down.`}
                aria-valuemin={yDomain[0]}
                aria-valuemax={yDomain[1]}
                aria-valuenow={y}
                onFocus={() => setFocused(index)}
                onBlur={() => setFocused(null)}
                onKeyDown={(event) => {
                  const step = (yDomain[1] - yDomain[0]) * 0.03;
                  if (event.key === 'ArrowUp') {
                    event.preventDefault();
                    moveEndpoint(index, y + step);
                  } else if (event.key === 'ArrowDown') {
                    event.preventDefault();
                    moveEndpoint(index, y - step);
                  }
                }}
                onPointerDown={(event) => {
                  (event.currentTarget as Element).setPointerCapture(event.pointerId);
                  dragTo(index, event.clientY);
                }}
                onPointerMove={(event) => {
                  if (event.buttons !== 1) return;
                  dragTo(index, event.clientY);
                }}
                style={{ cursor: 'ns-resize', outline: 'none' }}
              >
                {focused === index && (
                  <circle cx={sx} cy={sy} r={4.2} fill="none" stroke="var(--accent)" strokeWidth={0.6} strokeDasharray="1.5 1" />
                )}
                <circle cx={sx} cy={sy} r={2.6} fill="var(--accent)" stroke="var(--surface-base)" strokeWidth={0.6} />
              </g>
            );
          })}
        </svg>
      </div>
    </Panel>
  );
}

/** Data-based viewport, generous enough that the configured range rarely clips. */
function computeDomain(config: LossMinimizationConfig): {
  xDomain: [number, number];
  yDomain: [number, number];
  handleX: [number, number];
} {
  const points = config.points ?? [];
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const xMin = xs.length ? Math.min(...xs) : 0;
  const xMax = xs.length ? Math.max(...xs) : 1;
  const xPad = Math.max((xMax - xMin) * 0.15, 0.4);
  const xDomain: [number, number] = [xMin - xPad, xMax + xPad];

  const [sMin, sMax] = config.slopeRange ?? [0, 0];
  const [iMin, iMax] = config.interceptRange ?? [0, 0];
  const candidateYs = [...ys];
  for (const s of [sMin, sMax]) {
    for (const i of [iMin, iMax]) {
      candidateYs.push(s * xMin + i, s * xMax + i);
    }
  }
  const yLo = candidateYs.length ? Math.min(...candidateYs) : 0;
  const yHi = candidateYs.length ? Math.max(...candidateYs) : 1;
  const yPad = Math.max((yHi - yLo) * 0.08, 0.5);

  return {
    xDomain,
    yDomain: [yLo - yPad, yHi + yPad],
    handleX: [xMin, xMax],
  };
}

/* ── match-loss mode ────────────────────────────────────────── */

function MatchLossBoard({
  state,
  dispatch,
  revealed,
}: {
  state: LossMinimizationState;
  dispatch: (action: LossMinimizationAction) => void;
  revealed: boolean;
}) {
  const scenarios = state.config.scenarios ?? [];
  const options = state.config.options ?? [];

  return (
    <div className="mx-auto my-auto flex w-full max-w-2xl flex-col gap-3">
      {scenarios.map((scenario) => {
        const picked = state.answers[scenario.id];
        const correct = picked === scenario.answer;
        return (
          <Panel
            key={scenario.id}
            label="scenario"
            actions={
              revealed ? (
                <Tag tone={correct ? 'good' : 'bad'}>
                  {correct ? 'correct' : `it was ${LOSS_LABEL[scenario.answer]}`}
                </Tag>
              ) : picked ? (
                <Tag tone="accent">answered</Tag>
              ) : null
            }
          >
            <div className="flex flex-col gap-3">
              <p className="text-sm leading-relaxed text-primary">{scenario.prompt}</p>
              <div className="flex flex-wrap gap-1.5">
                {options.map((option) => {
                  const chosen = picked === option;
                  const isAnswer = revealed && option === scenario.answer;
                  return (
                    <button
                      key={option}
                      type="button"
                      disabled={revealed}
                      onClick={() => dispatch({ type: 'ANSWER_SCENARIO', scenarioId: scenario.id, answer: option })}
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
                      {LOSS_LABEL[option]}
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

function MatchLossSubmit({
  state,
  revealed,
  onSubmit,
}: {
  state: LossMinimizationState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const scenarios = state.config.scenarios ?? [];
  const answered = scenarios.filter((s) => state.answers[s.id]).length;
  const blocked = answered < scenarios.length;

  return (
    <>
      {!revealed && blocked && (
        <span className="label">{scenarios.length - answered} scenario(s) left</span>
      )}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || revealed}
        title={blocked ? `${scenarios.length - answered} scenario(s) left` : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}
