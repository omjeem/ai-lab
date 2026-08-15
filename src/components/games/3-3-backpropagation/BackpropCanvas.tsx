'use client';

/**
 * Chapter 3.3 — Backpropagation.
 *
 * The network graph is real: nodes and edges come straight from the trained
 * `TinyNet`'s architecture, and every edge's thickness and colour is its real
 * gradient magnitude from `state.gradients` — the same backward pass the
 * engine scores against. Nothing here is a diagram of what backprop looks
 * like; it is backprop's own numbers laid out spatially.
 *
 * Model-free like the rest of World 3: the net trains synchronously from a
 * seed in `initState`, so there is nothing for `ModelGate` to gate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { ArrowDown, ArrowUp, Check, GripVertical, RotateCcw, Zap } from 'lucide-react';
import {
  initState,
  applyAction,
  evaluate,
  layerGradientMagnitude,
  type BackpropVisualAction,
  type BackpropVisualConfig,
  type BackpropVisualState,
} from '@/engines/backpropVisualEngine';
import type { Activation, EdgeGradient } from '@/models/tinyNetTrainer';
import { Button, Meter, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import { usePointerDrag } from '../usePointerDrag';
import { useDragReorder } from '../useDragReorder';
import type { GameComponentProps } from '../registry';
import type { EngineRules, ScoreResult } from '@/types/game';

const ACTIVATION_LABEL: Partial<Record<Activation, string>> = {
  sigmoid: 'sigmoid',
  tanh: 'tanh',
  relu: 'ReLU',
  leakyRelu: 'leaky ReLU',
  gelu: 'GELU',
};

export function BackpropCanvas({ level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as BackpropVisualConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<BackpropVisualState | null>(null);
  useEffect(() => setState(initState(config, rules)), [config, rules]);

  const dispatch = useCallback((action: BackpropVisualAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  if (!state) {
    return (
      <div className="grid-field flex flex-1 items-center justify-center">
        <span className="label">training the network</span>
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
  state: BackpropVisualState;
  dispatch: (action: BackpropVisualAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status === 'complete'`: every action bumps status back
  // to `active`, so a post-reveal answer change would silently un-reveal.
  const [revealed, setRevealed] = useState(false);
  const [pulseKey, setPulseKey] = useState(0);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const isQuiz = state.mode === 'predict-sign' || state.mode === 'rank-magnitude';

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    if (isQuiz) setRevealed(true);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'predict-sign' && (
          <SignBoard state={state} dispatch={dispatch} revealed={revealed} pulseKey={pulseKey} />
        )}
        {state.mode === 'rank-magnitude' && (
          <RankMagnitudeBoard state={state} dispatch={dispatch} revealed={revealed} pulseKey={pulseKey} />
        )}
        {state.mode === 'fix-vanishing' && (
          <VanishingBoard state={state} dispatch={dispatch} result={result} pulseKey={pulseKey} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Button onClick={() => setPulseKey((k) => k + 1)} title="Animate the signal travelling backwards through the graph">
          <Zap size={13} strokeWidth={2} />
          Run backward pass
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
  state: BackpropVisualState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = state.mode === 'predict-sign' && state.signRounds.some((r) => r.answer === null);
  const left = state.mode === 'predict-sign' ? state.signRounds.filter((r) => r.answer === null).length : 0;

  return (
    <>
      {!revealed && blocked && (
        <span className="label">{left} round{left === 1 ? '' : 's'} left</span>
      )}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button variant="primary" className="ml-auto" disabled={blocked || revealed} onClick={onSubmit}>
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

/* ── level 1: predict the sign ─────────────────────────────────── */

function SignBoard({
  state,
  dispatch,
  revealed,
  pulseKey,
}: {
  state: BackpropVisualState;
  dispatch: (action: BackpropVisualAction) => void;
  revealed: boolean;
  pulseKey: number;
}) {
  const [focused, setFocused] = useState<number | null>(null);

  const overlays = useMemo<EdgeVisual[]>(
    () =>
      state.signRounds.map((round, i) => ({
        layer: round.edge.layer,
        from: round.edge.from,
        to: round.edge.to,
        tone: revealed
          ? round.answer === round.trueDirection
            ? 'good'
            : 'bad'
          : focused === i
            ? 'focus'
            : 'pending',
        badge: String(i + 1),
      })),
    [state.signRounds, revealed, focused]
  );

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <Panel label="the network" flush className="w-full shrink-0 lg:w-[420px]">
        <div className="p-2">
          <NetworkGraph
            architecture={state.architecture}
            gradients={state.gradients}
            overlays={overlays}
            pulseKey={pulseKey}
            ariaLabel={`Network with ${state.signRounds.length} numbered edges to predict`}
          />
          <p className="mt-2 px-1 text-xs leading-relaxed text-muted">
            Numbered edges are the ones below. Line thickness is the real gradient magnitude from this
            batch&apos;s backward pass.
          </p>
        </div>
      </Panel>

      <Panel label="which way does each weight move?" className="min-w-0 flex-1">
        <div className="flex flex-col gap-2">
          {state.signRounds.map((round, i) => {
            const correct = round.answer === round.trueDirection;
            return (
              <div
                key={i}
                onMouseEnter={() => setFocused(i)}
                onFocus={() => setFocused(i)}
                onMouseLeave={() => setFocused((f) => (f === i ? null : f))}
                className={cx(
                  'flex flex-wrap items-center gap-3 border px-3 py-2 transition-colors',
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
                <span className="readout w-5 shrink-0 text-xs text-accent">{i + 1}</span>
                <span className="min-w-0 flex-1 font-mono text-xs text-secondary">
                  layer {round.edge.layer + 1} · unit {round.edge.from + 1} → unit {round.edge.to + 1}
                </span>
                <div className="flex shrink-0 items-center gap-1">
                  {(['increase', 'decrease'] as const).map((dir) => {
                    const chosen = round.answer === dir;
                    const isTruth = revealed && dir === round.trueDirection;
                    return (
                      <button
                        key={dir}
                        type="button"
                        disabled={revealed}
                        onClick={() => dispatch({ type: 'ANSWER_SIGN', roundIndex: i, value: dir })}
                        aria-pressed={chosen}
                        aria-label={`${dir} for edge ${i + 1}`}
                        className={cx(
                          'flex min-h-[36px] items-center gap-1 border px-2.5 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors disabled:cursor-default',
                          isTruth
                            ? 'border-good bg-good/10 text-good'
                            : chosen
                              ? revealed
                                ? 'border-bad bg-bad/10 text-bad'
                                : 'border-accent bg-accent-dim text-primary'
                              : 'border-line text-muted hover:border-accent'
                        )}
                        style={{ borderRadius: 'var(--radius)' }}
                      >
                        {dir === 'increase' ? (
                          <ArrowUp size={11} strokeWidth={2} />
                        ) : (
                          <ArrowDown size={11} strokeWidth={2} />
                        )}
                        {dir}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

/* ── level 2: rank by magnitude ────────────────────────────────── */

function RankMagnitudeBoard({
  state,
  dispatch,
  revealed,
  pulseKey,
}: {
  state: BackpropVisualState;
  dispatch: (action: BackpropVisualAction) => void;
  revealed: boolean;
  pulseKey: number;
}) {
  const [activeRound, setActiveRound] = useState(0);
  const round = state.rankRounds[activeRound];

  const truePosition = useMemo(() => {
    if (!round) return new Map<number, number>();
    const sorted = round.edges
      .map((edge, i) => ({ i, mag: Math.abs(edge.value) }))
      .sort((a, b) => b.mag - a.mag);
    return new Map(sorted.map((s, position) => [s.i, position]));
  }, [round]);

  const overlays = useMemo<EdgeVisual[]>(() => {
    if (!round) return [];
    return round.edges.map((edge, edgeIndex) => {
      const position = round.ordering.indexOf(edgeIndex);
      const matches = revealed && truePosition.get(edgeIndex) === position;
      return {
        layer: edge.layer,
        from: edge.from,
        to: edge.to,
        tone: revealed ? (matches ? 'good' : 'bad') : 'focus',
        badge: String(position + 1),
      };
    });
  }, [round, revealed, truePosition]);

  const move = useCallback(
    (from: number, to: number) => {
      if (!round || to < 0 || to >= round.ordering.length) return;
      const next = [...round.ordering];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved!);
      dispatch({ type: 'SET_ORDER', roundIndex: activeRound, ordering: next });
    },
    [round, activeRound, dispatch]
  );

  const reorder = useDragReorder(move);

  if (!round) return null;

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <Panel label="the network" flush className="w-full shrink-0 lg:w-[420px]">
        <div className="p-2">
          <NetworkGraph
            architecture={state.architecture}
            gradients={state.gradients}
            overlays={overlays}
            pulseKey={pulseKey}
            ariaLabel={`Network with the current round's ${round.edges.length} ranked edges`}
          />
          <p className="mt-2 px-1 text-xs leading-relaxed text-muted">
            Badges are your current ranking for this round, largest first. Line thickness is the real
            gradient magnitude.
          </p>
        </div>
      </Panel>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Round">
          {state.rankRounds.map((_, i) => (
            <button
              key={i}
              type="button"
              role="tab"
              aria-selected={activeRound === i}
              onClick={() => setActiveRound(i)}
              className={cx(
                'min-h-[36px] border px-3 py-1.5 font-mono text-xs transition-colors',
                activeRound === i
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-raised text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              round {i + 1}
            </button>
          ))}
        </div>

        <Panel label="order these edges, largest gradient magnitude first">
          <ol className="flex flex-col gap-1.5">
            {round.ordering.map((edgeIndex, position) => {
              const edge = round.edges[edgeIndex]!;
              const matches = revealed && truePosition.get(edgeIndex) === position;
              return (
                <EdgeRow
                  key={edgeIndex}
                  edge={edge}
                  position={position}
                  revealed={revealed}
                  matches={matches}
                  trueRank={truePosition.get(edgeIndex) ?? 0}
                  totalItems={round.ordering.length}
                  activeRound={activeRound}
                  dragging={reorder.dragIndex === position}
                  dropTarget={
                    reorder.overIndex === position &&
                    reorder.dragIndex !== null &&
                    reorder.dragIndex !== position
                  }
                  registerRef={reorder.registerItem(position)}
                  onGripDragStart={() => reorder.startDrag(position)}
                  onGripDragMove={reorder.dragTo}
                  onGripDragEnd={reorder.dropAt}
                  onMoveUp={() => move(position, position - 1)}
                  onMoveDown={() => move(position, position + 1)}
                />
              );
            })}
          </ol>
        </Panel>
      </div>
    </div>
  );
}

/**
 * One row of the ranking list.
 *
 * The grip glyph, not the whole row, is the drag target — the row also holds
 * the up/down buttons, and capturing the pointer at the row would swallow
 * their clicks too.
 */
function EdgeRow({
  edge,
  position,
  revealed,
  matches,
  trueRank,
  totalItems,
  activeRound,
  dragging,
  dropTarget,
  registerRef,
  onGripDragStart,
  onGripDragMove,
  onGripDragEnd,
  onMoveUp,
  onMoveDown,
}: {
  edge: EdgeGradient;
  position: number;
  revealed: boolean;
  matches: boolean;
  trueRank: number;
  totalItems: number;
  activeRound: number;
  dragging: boolean;
  dropTarget: boolean;
  registerRef: (element: HTMLElement | null) => void;
  onGripDragStart: () => void;
  onGripDragMove: (clientY: number) => void;
  onGripDragEnd: (clientY: number) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}) {
  const grip = usePointerDrag({
    onDragStart: onGripDragStart,
    onDragMove: (_x, y) => onGripDragMove(y),
    onDragEnd: (_x, y) => onGripDragEnd(y),
    threshold: 6,
  });

  return (
    <li
      ref={registerRef}
      className={cx(
        'flex items-center gap-3 border px-2 py-2 transition-opacity',
        dragging && 'opacity-40',
        dropTarget && 'border-accent',
        revealed
          ? matches
            ? 'border-good/40 bg-good/5'
            : 'border-bad/40 bg-bad/5'
          : 'border-line-strong bg-raised'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      <span className="readout w-5 shrink-0 text-xs text-muted">{position + 1}</span>
      {!revealed && (
        <span {...grip} style={{ ...grip.style, cursor: 'grab' }} className="text-muted active:cursor-grabbing">
          <GripVertical size={13} strokeWidth={1.75} aria-hidden />
        </span>
      )}
      <span className="min-w-0 flex-1 truncate font-mono text-xs text-primary">
        layer {edge.layer + 1} · unit {edge.from + 1} → unit {edge.to + 1}
      </span>
      {revealed ? (
        <span className="readout shrink-0 text-xs text-secondary">
          |{Math.abs(edge.value).toFixed(4)}| {matches ? 'exact' : `really #${trueRank + 1}`}
        </span>
      ) : (
        <span className="flex shrink-0 items-center gap-1">
          <MoveButton
            label={`Move edge up in round ${activeRound + 1}`}
            disabled={position === 0}
            onClick={onMoveUp}
          >
            <ArrowUp size={12} strokeWidth={2} />
          </MoveButton>
          <MoveButton
            label={`Move edge down in round ${activeRound + 1}`}
            disabled={position === totalItems - 1}
            onClick={onMoveDown}
          >
            <ArrowDown size={12} strokeWidth={2} />
          </MoveButton>
        </span>
      )}
    </li>
  );
}

function MoveButton({
  children,
  label,
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        'flex h-7 w-7 items-center justify-center border border-line text-muted transition-colors',
        'hover:border-accent hover:text-accent disabled:opacity-30 disabled:hover:border-line disabled:hover:text-muted'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {children}
    </button>
  );
}

/* ── level 3: fix the vanishing gradient ───────────────────────── */

function VanishingBoard({
  state,
  dispatch,
  result,
  pulseKey,
}: {
  state: BackpropVisualState;
  dispatch: (action: BackpropVisualAction) => void;
  result: ScoreResult;
  pulseKey: number;
}) {
  const lastLayer = state.net.weights.length - 1;
  const firstMag = result.breakdown.firstLayerMagnitude ?? 0;
  const lastMag = result.breakdown.lastLayerMagnitude ?? 0;
  const threshold = state.config.targetFirstLayerGradientRatio ?? state.rules.passCriteria.threshold;

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <Panel label="the network" flush className="w-full shrink-0 lg:w-[440px]">
        <div className="p-2">
          <NetworkGraph
            architecture={state.architecture}
            gradients={state.gradients}
            overlays={[]}
            pulseKey={pulseKey}
            ariaLabel={`Deep ${state.architecture.length - 1}-weight-layer network, gradient magnitude shown by edge thickness`}
          />
          <p className="mt-2 px-1 text-xs leading-relaxed text-muted">
            Line thickness is the real gradient magnitude, normalised across the whole net — watch how
            thin the edges nearest the input go.
          </p>
        </div>
      </Panel>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="first-layer / last-layer gradient ratio">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="ratio" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <Readout label="first layer |grad|" value={firstMag} size="sm" />
              <Readout label="last layer |grad|" value={lastMag} size="sm" />
            </div>
            <Meter value={result.value} max={Math.max(result.value, threshold) * 1.4} threshold={threshold} tone={result.passed ? 'good' : 'accent'} />
            <p className="text-xs leading-relaxed text-muted">
              How much of the output layer&apos;s gradient signal survives the trip back to the first layer.
              Needs to clear {threshold.toFixed(2)} to pass.
            </p>
          </div>
        </Panel>

        <Panel label="gradient magnitude by layer">
          <LayerMagnitudeTrace gradients={state.gradients} layerCount={lastLayer + 1} />
        </Panel>

        <Panel label="activation">
          <div className="flex flex-wrap gap-1.5">
            {(state.config.activationOptions ?? []).map((option) => {
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
      </div>
    </div>
  );
}

function LayerMagnitudeTrace({
  gradients,
  layerCount,
}: {
  gradients: readonly EdgeGradient[];
  layerCount: number;
}) {
  const magnitudes = useMemo(
    () => Array.from({ length: layerCount }, (_, l) => layerGradientMagnitude(gradients, l)),
    [gradients, layerCount]
  );
  const logs = magnitudes.map((m) => (m > 0 ? Math.log10(m) : -12));
  const hi = Math.max(...logs, -3);
  const lo = Math.min(...logs, hi - 6);

  return (
    <div className="flex flex-col gap-1.5">
      {magnitudes.map((magnitude, l) => {
        const t = hi === lo ? 0 : Math.max(0, Math.min(1, (logs[l]! - lo) / (hi - lo)));
        const edge = l === 0 ? ' (first)' : l === layerCount - 1 ? ' (last)' : '';
        return (
          <div key={l} className="flex items-center gap-2">
            <span className="label w-16 shrink-0">layer {l + 1}{edge}</span>
            <div className="h-2 flex-1 bg-inset">
              <div className="h-full bg-accent" style={{ width: `${(t * 100).toFixed(1)}%` }} />
            </div>
            <span className="readout w-16 shrink-0 text-right text-[10px] text-muted">
              {magnitude.toExponential(1)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ── the network graph ─────────────────────────────────────────── */

interface EdgeVisual {
  layer: number;
  from: number;
  to: number;
  tone: 'focus' | 'good' | 'bad' | 'pending';
  badge?: string;
}

const NODE_R = 4.2;
const LAYER_GAP = 68;
const NODE_GAP = 20;
const MARGIN = 24;
const PULSE_STEP = 0.4;

function edgeKey(e: { layer: number; from: number; to: number }): string {
  return `${e.layer}:${e.from}:${e.to}`;
}

function NetworkGraph({
  architecture,
  gradients,
  overlays,
  pulseKey,
  ariaLabel,
}: {
  architecture: number[];
  gradients: readonly EdgeGradient[];
  overlays: EdgeVisual[];
  pulseKey: number;
  ariaLabel: string;
}) {
  const reduce = useReducedMotion();
  const numLayers = architecture.length;
  const numWeightLayers = numLayers - 1;
  const maxNodes = Math.max(...architecture);
  const width = MARGIN * 2 + (numLayers - 1) * LAYER_GAP;
  const height = MARGIN * 2 + (maxNodes - 1) * NODE_GAP;

  const nodeXY = useCallback(
    (li: number, idx: number, count: number): [number, number] => {
      const x = MARGIN + li * LAYER_GAP;
      const y = height / 2 + (idx - (count - 1) / 2) * NODE_GAP;
      return [x, y];
    },
    [height]
  );

  const edges = useMemo(() => {
    const list: { key: string; layer: number; from: number; to: number; x1: number; y1: number; x2: number; y2: number }[] = [];
    for (let l = 0; l < numWeightLayers; l++) {
      const fromCount = architecture[l]!;
      const toCount = architecture[l + 1]!;
      for (let from = 0; from < fromCount; from++) {
        for (let to = 0; to < toCount; to++) {
          const [x1, y1] = nodeXY(l, from, fromCount);
          const [x2, y2] = nodeXY(l + 1, to, toCount);
          list.push({ key: edgeKey({ layer: l, from, to }), layer: l, from, to, x1, y1, x2, y2 });
        }
      }
    }
    return list;
  }, [architecture, numWeightLayers, nodeXY]);

  const magFor = useMemo(() => {
    const map = new Map<string, number>();
    for (const g of gradients) map.set(edgeKey(g), Math.abs(g.value));
    return map;
  }, [gradients]);
  const maxMag = useMemo(
    () => gradients.reduce((m, g) => Math.max(m, Math.abs(g.value)), 0) || 1,
    [gradients]
  );

  const overlayFor = useMemo(() => {
    const map = new Map<string, EdgeVisual>();
    for (const o of overlays) map.set(edgeKey(o), o);
    return map;
  }, [overlays]);

  const nodeOrder = useCallback((li: number) => numLayers - 1 - li, [numLayers]);
  const edgeOrder = useCallback((l: number) => nodeOrder(l + 1) + 0.5, [nodeOrder]);

  const toneColor: Record<EdgeVisual['tone'], string> = {
    focus: 'var(--accent)',
    good: 'var(--signal-good)',
    bad: 'var(--signal-bad)',
    pending: 'var(--text-secondary)',
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="block w-full" role="img" aria-label={ariaLabel}>
      <g>
        {edges.map((e) => {
          const norm = Math.min(1, (magFor.get(e.key) ?? 0) / maxMag);
          return (
            <line
              key={e.key}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke={`color-mix(in oklab, var(--accent) ${Math.round(norm * 85 + 6)}%, var(--line))`}
              strokeWidth={0.35 + norm * 2.3}
              strokeLinecap="round"
            />
          );
        })}
      </g>

      <g>
        {edges.map((e) => {
          const ov = overlayFor.get(e.key);
          if (!ov) return null;
          const color = toneColor[ov.tone];
          const mx = (e.x1 + e.x2) / 2;
          const my = (e.y1 + e.y2) / 2;
          return (
            <g key={`ov-${e.key}`}>
              <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={color} strokeWidth={1.8} strokeLinecap="round" opacity={0.85} />
              {ov.badge && (
                <g transform={`translate(${mx}, ${my})`}>
                  <circle r={5.5} fill="var(--surface-base)" stroke={color} strokeWidth={1} />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    style={{ fontSize: 5.5, fontFamily: 'var(--font-mono)', fill: color }}
                  >
                    {ov.badge}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </g>

      {pulseKey > 0 && (
        <g key={pulseKey}>
          {Array.from({ length: numWeightLayers }, (_, l) => l).map((l) => (
            <motion.g
              key={`pulse-edge-${l}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.95, 0] }}
              transition={
                reduce
                  ? { duration: 0.2, delay: 0 }
                  : { duration: 0.55, delay: edgeOrder(l) * PULSE_STEP, ease: 'easeOut' }
              }
            >
              {edges
                .filter((e) => e.layer === l)
                .map((e) => (
                  <line key={e.key} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="var(--accent)" strokeWidth={2.6} strokeLinecap="round" />
                ))}
            </motion.g>
          ))}
          {Array.from({ length: numLayers }, (_, li) => li).map((li) => {
            const count = architecture[li]!;
            return (
              <motion.g
                key={`pulse-node-${li}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 0] }}
                transition={
                  reduce
                    ? { duration: 0.2, delay: 0 }
                    : { duration: 0.55, delay: nodeOrder(li) * PULSE_STEP, ease: 'easeOut' }
                }
              >
                {Array.from({ length: count }, (_, idx) => {
                  const [x, y] = nodeXY(li, idx, count);
                  return <circle key={idx} cx={x} cy={y} r={NODE_R + 3} fill="none" stroke="var(--accent)" strokeWidth={1.4} />;
                })}
              </motion.g>
            );
          })}
        </g>
      )}

      <g>
        {architecture.map((count, li) =>
          Array.from({ length: count }, (_, idx) => {
            const [x, y] = nodeXY(li, idx, count);
            return (
              <circle
                key={`${li}-${idx}`}
                cx={x}
                cy={y}
                r={NODE_R}
                fill="var(--surface-raised)"
                stroke="var(--line-strong)"
                strokeWidth={1}
              />
            );
          })
        )}
      </g>
    </svg>
  );
}
