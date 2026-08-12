'use client';

/**
 * Chapter 2.1 — The Perceptron.
 *
 * The plot is the model. The boundary drawn across it is `w·x + b = 0` for the
 * live weights, the shaded half is everything the perceptron currently calls
 * positive, and the ring marks the sample the rule is about to look at — which
 * matters, because the rule only fires when that sample is wrong. Stepping and
 * watching the line swing is the entire lesson.
 *
 * No download here: the model is forty numbers and a loop, built in the browser
 * from the level's seed. There is nothing for `ModelGate` to gate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, ChevronsRight, FastForward, RotateCcw, StepForward } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  predict,
  accuracyOf,
  type LinearClassifierAction,
  type LinearClassifierConfig,
  type LinearClassifierState,
  type Sample,
} from '@/engines/linearClassifierEngine';
import { Button, Panel, Readout, Slider, Tag } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

/** Half-width of the plotted region. The generators draw from [-1, 1]. */
const RANGE = 1.15;
/** User-space size of the plot's square viewBox. */
const PLOT = 100;

export function PerceptronCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as LinearClassifierConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<LinearClassifierState | null>(null);

  // Synchronous and cheap — the dataset is generated from the level's seed, so
  // this is the whole "model load" for World 2.
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: LinearClassifierAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">seeding the dataset</span>
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
  state: LinearClassifierState;
  dispatch: (action: LinearClassifierAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const resetRun = useCallback(() => dispatch({ type: 'RESET' }), [dispatch]);
  useRetrySignal(resetRun);

  const exhausted = state.steps >= state.config.maxSteps;
  // `learningRateRange` is what says this level is about the learning rate.
  const tunable = state.config.learningRateRange !== undefined;
  // `provesLimitation` is what says the boundary has to be placed by hand.
  const manual = state.config.provesLimitation === true;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-3 lg:flex-row lg:items-start">
          <ScatterPlot state={state} />

          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <Panel label="the model">
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <Readout
                    label="accuracy"
                    value={accuracyOfState(state)}
                    size="lg"
                    tone={state.converged ? 'good' : 'accent'}
                  />
                  <Readout label="updates" value={state.updates} size="sm" />
                  <Readout
                    label="steps"
                    value={`${state.steps} / ${state.config.maxSteps}`}
                    size="sm"
                  />
                </div>

                <WeightsReadout state={state} />

                {state.convergedAtStep !== null && (
                  <Tag tone="good">separated at step {state.convergedAtStep}</Tag>
                )}
                {exhausted && !state.converged && (
                  <Tag tone="warn">step budget spent without separating</Tag>
                )}
              </div>
            </Panel>

            {tunable && (
              <Panel label="learning rate">
                <Slider
                  label="η"
                  value={state.learningRate}
                  min={state.config.learningRateRange![0]}
                  max={state.config.learningRateRange![1]}
                  step={0.01}
                  onChange={(value) => dispatch({ type: 'SET_LEARNING_RATE', value })}
                  format={(value) => value.toFixed(2)}
                />
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  Every correction is scaled by η. Too small and the boundary crawls; too large and it
                  overshoots past the answer and has to come back.
                </p>
              </Panel>
            )}

            {manual && <ManualWeights state={state} dispatch={dispatch} />}

            {state.history.length > 1 && <AccuracyTrace state={state} />}
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
        <Button onClick={() => dispatch({ type: 'RUN', steps: 100 })} disabled={exhausted}>
          <FastForward size={13} strokeWidth={2} />
          ×100
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

/* ── the plot ───────────────────────────────────────────────── */

function ScatterPlot({ state }: { state: LinearClassifierState }) {
  const reduce = useReducedMotion();
  const [w0, w1] = [state.weights[0] ?? 0, state.weights[1] ?? 0];
  const bias = state.bias;
  const line = boundaryGeometry(w0, w1, bias);

  return (
    <Panel label="feature space" flush className="w-full shrink-0 lg:w-[420px]">
      <div className="p-2">
        <svg
          viewBox={`0 0 ${PLOT} ${PLOT}`}
          className="block w-full"
          role="img"
          aria-label={`${state.samples.length} samples in two dimensions, with the current decision boundary`}
        >
          <defs>
            <clipPath id="plot-clip">
              <rect x={0} y={0} width={PLOT} height={PLOT} />
            </clipPath>
          </defs>

          <rect x={0} y={0} width={PLOT} height={PLOT} fill="var(--surface-inset)" />

          {/* Axes through the origin, so the bias term is readable as an offset. */}
          <line x1={toScreen(-RANGE, 0)[0]} y1={toScreen(0, 0)[1]} x2={toScreen(RANGE, 0)[0]} y2={toScreen(0, 0)[1]} stroke="var(--line)" strokeWidth={0.4} />
          <line x1={toScreen(0, 0)[0]} y1={toScreen(0, -RANGE)[1]} x2={toScreen(0, 0)[0]} y2={toScreen(0, RANGE)[1]} stroke="var(--line)" strokeWidth={0.4} />

          <g clipPath="url(#plot-clip)">
            {/* Everything the perceptron currently calls +1. */}
            {line && (
              <motion.g
                initial={false}
                animate={{ rotate: line.angle, x: line.x, y: line.y }}
                transition={reduce ? { duration: 0 } : { duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                /**
                 * The pivot has to be this group's own local origin, which is the
                 * point on the boundary. Two things fight that: by default
                 * `origin` resolves against the element's bounding box — here the
                 * oversized half-plane rect — and Motion writes `transform-origin`
                 * itself, so setting it through `style` is silently overwritten.
                 * `transform-box: view-box` plus Motion's own origin props is the
                 * combination that lands on 0 0.
                 */
                style={{ transformBox: 'view-box', originX: 0, originY: 0 }}
              >
                <rect
                  x={-PLOT * 2}
                  y={0}
                  width={PLOT * 4}
                  height={PLOT * 2}
                  fill="var(--accent)"
                  opacity={0.09}
                />
                <line
                  x1={-PLOT * 2}
                  y1={0}
                  x2={PLOT * 2}
                  y2={0}
                  stroke="var(--accent)"
                  strokeWidth={0.9}
                />
              </motion.g>
            )}

            {state.samples.map((sample, index) => (
              <SamplePoint
                key={index}
                sample={sample}
                correct={predict(state.weights, state.bias, sample.x) === sample.label}
                isCursor={index === state.cursor}
              />
            ))}
          </g>
        </svg>

        <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-2">
          <Legend />
          <span className="label">next up: sample {state.cursor + 1}</span>
        </div>
      </div>
    </Panel>
  );
}

function SamplePoint({
  sample,
  correct,
  isCursor,
}: {
  sample: Sample;
  correct: boolean;
  isCursor: boolean;
}) {
  const [cx, cy] = toScreen(sample.x[0] ?? 0, sample.x[1] ?? 0);
  const positive = sample.label === 1;

  return (
    <g>
      {/* Class by shape as well as fill — the two must be tellable apart without
          relying on colour. */}
      {positive ? (
        <circle cx={cx} cy={cy} r={1.9} fill="var(--accent)" opacity={correct ? 1 : 0.55} />
      ) : (
        <rect
          x={cx - 1.7}
          y={cy - 1.7}
          width={3.4}
          height={3.4}
          fill="none"
          stroke="var(--text-secondary)"
          strokeWidth={0.9}
          opacity={correct ? 1 : 0.55}
        />
      )}

      {!correct && (
        <circle cx={cx} cy={cy} r={3.4} fill="none" stroke="var(--signal-bad)" strokeWidth={0.7} />
      )}
      {isCursor && (
        <circle cx={cx} cy={cy} r={5} fill="none" stroke="var(--text-primary)" strokeWidth={0.7} strokeDasharray="2 1.5" />
      )}
    </g>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={3.5} fill="var(--accent)" />
        </svg>
        <span className="label">label +1</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <rect x={1} y={1} width={8} height={8} fill="none" stroke="var(--text-secondary)" strokeWidth={1.5} />
        </svg>
        <span className="label">label −1</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width={10} height={10} aria-hidden>
          <circle cx={5} cy={5} r={4} fill="none" stroke="var(--signal-bad)" strokeWidth={1.2} />
        </svg>
        <span className="label">wrong</span>
      </span>
    </div>
  );
}

/* ── readouts and controls ──────────────────────────────────── */

function WeightsReadout({ state }: { state: LinearClassifierState }) {
  return (
    <dl className="flex flex-wrap gap-x-5 gap-y-1">
      {state.weights.map((weight, i) => (
        <div key={i} className="flex items-baseline gap-2">
          <dt className="label">w{i}</dt>
          <dd className="readout text-sm text-primary">{weight.toFixed(3)}</dd>
        </div>
      ))}
      <div className="flex items-baseline gap-2">
        <dt className="label">bias</dt>
        <dd className="readout text-sm text-primary">{state.bias.toFixed(3)}</dd>
      </div>
    </dl>
  );
}

/**
 * Hand placement of the boundary.
 *
 * Only offered where the level says `provesLimitation`. Everywhere else the
 * point is that the weights are found by the rule, and being able to type in
 * the answer would remove the chapter.
 */
function ManualWeights({
  state,
  dispatch,
}: {
  state: LinearClassifierState;
  dispatch: (action: LinearClassifierAction) => void;
}) {
  const set = (index: number, value: number) => {
    const weights = [...state.weights];
    weights[index] = value;
    dispatch({ type: 'SET_WEIGHTS', weights, bias: state.bias });
  };

  return (
    <Panel label="place the boundary by hand">
      <div className="flex flex-col gap-3">
        {state.weights.map((weight, i) => (
          <Slider
            key={i}
            label={`w${i}`}
            value={weight}
            min={-2}
            max={2}
            step={0.01}
            onChange={(value) => set(i, value)}
            format={(value) => value.toFixed(2)}
          />
        ))}
        <Slider
          label="bias"
          value={state.bias}
          min={-1.5}
          max={1.5}
          step={0.01}
          onChange={(value) => dispatch({ type: 'SET_WEIGHTS', weights: state.weights, bias: value })}
          format={(value) => value.toFixed(2)}
        />
        <p className="text-xs leading-relaxed text-muted">
          Training will not get there on this data, and no setting of these three numbers will either:
          one straight line has to give up a whole quadrant, so find the placement that gives up the
          least. Scaling all three together changes nothing — only the direction of (w0, w1) and its
          size relative to the bias move the line.
        </p>
      </div>
    </Panel>
  );
}

/** Accuracy against step. On XOR this is the chapter's punchline: it plateaus. */
function AccuracyTrace({ state }: { state: LinearClassifierState }) {
  const points = state.history;
  const width = 100;
  const height = 26;
  const last = points.at(-1)!;

  const path = points
    .map((entry, i) => {
      const x = points.length === 1 ? 0 : (i / (points.length - 1)) * width;
      const y = height - entry.accuracy * height;
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');

  return (
    <Panel label={`accuracy over ${points.length} steps`}>
      <div className="flex flex-col gap-1.5">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block h-16 w-full" role="img" aria-label={`Accuracy reached ${(last.accuracy * 100).toFixed(0)} percent`}>
          <line x1={0} y1={0} x2={width} y2={0} stroke="var(--line)" strokeWidth={0.3} />
          <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line)" strokeWidth={0.3} />
          <path d={path} fill="none" stroke="var(--accent)" strokeWidth={0.8} />
        </svg>
        <div className="flex items-baseline justify-between">
          <span className="label">0%</span>
          <span className="label">100%</span>
        </div>
      </div>
    </Panel>
  );
}

/* ── geometry ───────────────────────────────────────────────── */

function toScreen(x0: number, x1: number): [number, number] {
  return [((x0 + RANGE) / (2 * RANGE)) * PLOT, ((RANGE - x1) / (2 * RANGE)) * PLOT];
}

/**
 * Where to put the boundary, as a rotation about a point on the line.
 *
 * The half-plane and the line are drawn once in local coordinates — local +x
 * along the boundary, local +y into the positive side — and this places them.
 * SVG's rotate maps local +y to `(-sinθ, cosθ)`, which for `θ = atan2(-w0, -w1)`
 * is the screen direction of the weight vector: exactly the side the perceptron
 * calls +1.
 */
function boundaryGeometry(
  w0: number,
  w1: number,
  bias: number
): { x: number; y: number; angle: number } | null {
  const normSquared = w0 * w0 + w1 * w1;
  // An all-zero weight vector has no boundary to draw — every point scores the
  // same, which is exactly the untrained starting state of level 2.
  if (normSquared < 1e-12) return null;

  // Closest point on the line to the origin, in data space.
  const px = (-bias * w0) / normSquared;
  const py = (-bias * w1) / normSquared;
  const [sx, sy] = toScreen(px, py);

  return { x: sx, y: sy, angle: (Math.atan2(-w0, -w1) * 180) / Math.PI };
}

function accuracyOfState(state: LinearClassifierState): number {
  return accuracyOf(state.weights, state.bias, state.samples);
}
