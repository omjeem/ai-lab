'use client';

/**
 * Chapter 1.5 — Probability Basics.
 *
 * The wheel is the chapter: its segments are the model's real softmaxed logits,
 * and spinning it runs genuine inverse-CDF sampling, so a 4% token really does
 * come up about one time in twenty-five. Watching an unlikely segment win is the
 * only way the point lands.
 *
 * Which means the wheel has to stay hidden until the player has committed. Every
 * level here asks you to predict something about the distribution, and all three
 * questions answer themselves the moment the segments are on screen. Level 2 is
 * the exception that proves it: the wheel is live from the start, but it draws
 * *your* allocation, not the model's.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, Dices, Lock, Plus, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ProbabilityWheelAction,
  type ProbabilityWheelConfig,
  type ProbabilityWheelState,
  type RoundState,
} from '@/engines/probabilityWheelEngine';
import { createRng, entropyBits, normalizeDistribution } from '@/engines/shared';
import { tinyCausalLM, preloadCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function ProbabilityCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as ProbabilityWheelConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<ProbabilityWheelState | null>(null);

  const load = useCallback(async () => {
    // The entropy level ships no prompts, so `prepare` runs no forward pass and
    // the gate would pass without fetching 92MB. Warm it explicitly.
    await preloadCausalLM();
    const prepared = await prepare(config, { causalLM: tinyCausalLM });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: ProbabilityWheelAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={CAUSAL_LM_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: ProbabilityWheelState;
  dispatch: (action: ProbabilityWheelAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  /**
   * Whether the model's distribution is on screen.
   *
   * Local rather than read off `state.status`: every round-level action goes
   * through the engine's `withRound`, which sets the status back to `active`.
   * Spinning the wheel would therefore un-reveal the wheel being spun.
   */
  const [revealed, setRevealed] = useState(false);

  // Bumping this remounts the mode board, clearing per-round UI state that a
  // reset has to clear too — locked-in estimates, half-typed prompts.
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
          {state.mode === 'predict-top1' && (
            <PredictBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'allocate-mass' && (
            <AllocateBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'estimate-entropy' && (
            <EntropyBoard state={state} dispatch={dispatch} revealed={revealed} />
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

/* ── level 1: call the top token ────────────────────────────── */

function PredictBoard({
  state,
  dispatch,
  revealed,
}: {
  state: ProbabilityWheelState;
  dispatch: (action: ProbabilityWheelAction) => void;
  revealed: boolean;
}) {
  return (
    <div className="flex flex-col gap-3">
      {state.rounds.map((round, index) => (
        <PredictRound
          key={round.prompt}
          round={round}
          revealed={revealed}
          onPick={(token) => dispatch({ type: 'PICK', roundIndex: index, token })}
          onSpin={() => dispatch({ type: 'SPIN', roundIndex: index })}
        />
      ))}
      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? 'Each segment is one of the model’s real top-k probabilities, renormalised over the k shown. Spin to sample from it for real.'
          : 'The candidates are the model’s own top-k, shuffled so their order says nothing. The probabilities stay hidden until you submit.'}
      </p>
    </div>
  );
}

function PredictRound({
  round,
  revealed,
  onPick,
  onSpin,
}: {
  round: RoundState;
  revealed: boolean;
  onPick: (token: string) => void;
  onSpin: () => void;
}) {
  // Seeded by the prompt so the order is stable across renders but carries no
  // hint — an unshuffled top-k would put the answer first every time.
  const shuffled = useMemo(() => shuffle(round.tokens, round.prompt), [round.tokens, round.prompt]);

  const topIndex = round.probs.indexOf(Math.max(...round.probs));
  const topToken = round.tokens[topIndex];
  const correct = round.pick === topToken;

  return (
    <Panel
      label="prompt"
      actions={
        revealed ? (
          <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : `really "${topToken}"`}</Tag>
        ) : round.pick ? (
          <Tag tone="accent">picked</Tag>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <PromptLine prompt={round.prompt} continuation={revealed ? topToken : undefined} />

        <div className="flex flex-wrap gap-1.5">
          {shuffled.map((token) => {
            const chosen = round.pick === token;
            const isTop = revealed && token === topToken;
            return (
              <button
                key={token}
                type="button"
                disabled={revealed}
                onClick={() => onPick(token)}
                aria-pressed={chosen}
                className={cx(
                  'min-h-[40px] border px-3 py-2 font-mono text-xs transition-colors disabled:cursor-default',
                  isTop
                    ? 'border-good bg-good/10 text-good'
                    : chosen
                      ? revealed
                        ? 'border-bad bg-bad/10 text-bad'
                        : 'border-accent bg-accent-dim text-primary'
                      : 'border-line-strong bg-raised text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {displayToken(token)}
                {revealed && (
                  <span className="readout ml-2 text-[10px] opacity-70">
                    {formatPercent(round.probs[round.tokens.indexOf(token)] ?? 0)}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {revealed && (
          <WheelPanel round={round} onSpin={onSpin} />
        )}
      </div>
    </Panel>
  );
}

/* ── level 2: allocate the mass ─────────────────────────────── */

function AllocateBoard({
  state,
  dispatch,
  revealed,
}: {
  state: ProbabilityWheelState;
  dispatch: (action: ProbabilityWheelAction) => void;
  revealed: boolean;
}) {
  const budget = state.config.budget ?? 100;

  return (
    <div className="flex flex-col gap-3">
      {state.rounds.map((round, index) => (
        <AllocateRound
          key={round.prompt}
          round={round}
          budget={budget}
          revealed={revealed}
          onAllocate={(tokenIndex, value) =>
            dispatch({ type: 'ALLOCATE', roundIndex: index, tokenIndex, value })
          }
          onSpin={() => dispatch({ type: 'SPIN', roundIndex: index })}
        />
      ))}

      {state.config.allowUserPrompt && !revealed && (
        <PromptAdder state={state} dispatch={dispatch} label="calibrate against your own prompt" />
      )}

      <p className="text-xs leading-relaxed text-muted">
        Only the shape is scored, not the total — your points are normalised before they are compared,
        so 10/20/30 and 100/200/300 are the same answer. Distance is measured as total variation: half
        the summed gap between your distribution and the model’s.
      </p>
    </div>
  );
}

function AllocateRound({
  round,
  budget,
  revealed,
  onAllocate,
  onSpin,
}: {
  round: RoundState;
  budget: number;
  revealed: boolean;
  onAllocate: (tokenIndex: number, value: number) => void;
  onSpin: () => void;
}) {
  const used = round.allocation.reduce((a, b) => a + b, 0);
  const mine = normalizeDistribution(round.allocation);
  const started = used > 0;

  return (
    <Panel
      label="prompt"
      actions={<span className="label">{used} of {budget} points</span>}
    >
      <div className="flex flex-col gap-3">
        <PromptLine prompt={round.prompt} />

        <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
          <div className="flex min-w-0 flex-1 flex-col gap-2.5">
            {round.tokens.map((token, i) => (
              <div key={token} className="flex flex-col gap-0.5">
                <Slider
                  label={displayTokenText(token)}
                  value={round.allocation[i] ?? 0}
                  min={0}
                  max={budget}
                  step={1}
                  disabled={revealed}
                  onChange={(value) => onAllocate(i, value)}
                  format={(value) => `${value}`}
                />
                <div className="flex items-baseline justify-between gap-3">
                  <span className="label text-[9px]">
                    yours {started ? formatPercent(mine[i] ?? 0) : '—'}
                  </span>
                  {revealed && (
                    <span className="readout text-[10px] text-accent">
                      model {formatPercent(round.probs[i] ?? 0)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Live shape of the allocation, so hedging *looks* like hedging. */}
          <div className="flex shrink-0 flex-col items-center gap-2">
            <Wheel
              segments={round.tokens.map((token, i) => ({
                token,
                probability: started ? (mine[i] ?? 0) : 1 / round.tokens.length,
              }))}
              size={132}
              dim={!started}
            />
            <span className="label">{started ? 'your shape' : 'allocate to shape it'}</span>
          </div>
        </div>

        {revealed && <WheelPanel round={round} onSpin={onSpin} />}
      </div>
    </Panel>
  );
}

/* ── level 3: read the entropy ──────────────────────────────── */

function EntropyBoard({
  state,
  dispatch,
  revealed,
}: {
  state: ProbabilityWheelState;
  dispatch: (action: ProbabilityWheelAction) => void;
  revealed: boolean;
}) {
  /**
   * Rounds whose estimate has been locked in.
   *
   * The engine has no per-round reveal — it does not need one — but the level
   * only means anything if the wheel stays hidden until the guess is committed,
   * one prompt at a time.
   */
  const [locked, setLocked] = useState<Set<number>>(new Set());

  const target = state.config.rounds ?? state.rounds.length;
  // Top-k renormalised, so the distribution cannot carry more than log2(k) bits.
  const ceiling = Math.log2(Math.max(2, state.config.topK));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <Readout label="rounds" value={`${state.rounds.length} / ${target}`} size="md" />
        <Readout label="locked in" value={`${locked.size} / ${target}`} size="sm" />
        <Readout label="entropy ceiling" value={ceiling} unit="bits" size="sm" tone="accent" />
      </div>

      {state.rounds.map((round, index) => (
        <EntropyRound
          key={`${round.prompt}-${index}`}
          round={round}
          ceiling={ceiling}
          locked={locked.has(index) || revealed}
          onEstimate={(value) => dispatch({ type: 'ESTIMATE_ENTROPY', roundIndex: index, value })}
          onLock={() => setLocked((prev) => new Set(prev).add(index))}
          onSpin={() => dispatch({ type: 'SPIN', roundIndex: index })}
        />
      ))}

      {state.rounds.length < target && !revealed && (
        <PromptAdder state={state} dispatch={dispatch} label={`prompt ${state.rounds.length + 1} of ${target}`} />
      )}

      <p className="text-xs leading-relaxed text-muted">
        Entropy is how spread out the distribution is: 0 bits means the model is certain, and{' '}
        {ceiling.toFixed(2)} bits — log₂ of the {state.config.topK} tokens shown — means all of them
        are equally likely. Every round you leave unanswered is scored at the maximum error, so a
        prompt you are unsure about still beats an empty one.
      </p>
    </div>
  );
}

function EntropyRound({
  round,
  ceiling,
  locked,
  onEstimate,
  onLock,
  onSpin,
}: {
  round: RoundState;
  ceiling: number;
  locked: boolean;
  onEstimate: (value: number) => void;
  onLock: () => void;
  onSpin: () => void;
}) {
  const error = round.entropyEstimate === null ? null : Math.abs(round.entropyEstimate - round.actualEntropyBits);

  return (
    <Panel
      label="prompt"
      actions={
        locked ? (
          <Tag tone={error !== null && error <= 0.45 ? 'good' : error !== null && error <= 1.2 ? 'warn' : 'bad'}>
            {error === null ? 'not estimated' : `off by ${error.toFixed(2)} bits`}
          </Tag>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <PromptLine prompt={round.prompt} />

        <Slider
          label="your entropy estimate"
          value={round.entropyEstimate ?? ceiling / 2}
          min={0}
          max={ceiling}
          step={0.01}
          disabled={locked}
          onChange={onEstimate}
          format={(value) => `${value.toFixed(2)} bits`}
        />

        {!locked ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={onLock} disabled={round.entropyEstimate === null}>
              <Lock size={13} strokeWidth={2} />
              Lock in and reveal
            </Button>
            <span className="label">
              {round.entropyEstimate === null ? 'move the slider first' : 'sharp is low, flat is high'}
            </span>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="you said" value={round.entropyEstimate ?? 0} unit="bits" size="sm" />
              <Readout
                label="actual"
                value={round.actualEntropyBits}
                unit="bits"
                size="md"
                tone="accent"
              />
            </div>
            <Meter value={round.actualEntropyBits} max={ceiling} label="how flat this distribution is" />
            <WheelPanel round={round} onSpin={onSpin} />
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ── adding a prompt of your own ────────────────────────────── */

function PromptAdder({
  state,
  dispatch,
  label,
}: {
  state: ProbabilityWheelState;
  dispatch: (action: ProbabilityWheelAction) => void;
  label: string;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    const prompt = draft.trim();
    if (!prompt || busy) return;
    setBusy(true);
    setError(null);
    try {
      const dist = await tinyCausalLM.nextTokenDistribution(prompt, state.config.topK);
      dispatch({
        type: 'ADD_ROUND',
        round: {
          prompt,
          tokens: dist.tokens,
          probs: dist.probs,
          actualEntropyBits: entropyBits(dist.probs),
        },
      });
      setDraft('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel label={label}>
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void run()}
            placeholder="type a prompt the model should continue"
            maxLength={200}
            disabled={busy}
            aria-label="Your prompt"
            className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-60"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={() => void run()} disabled={busy || draft.trim() === ''}>
            <Plus size={13} strokeWidth={2} />
            {busy ? 'running' : 'run it'}
          </Button>
        </div>
        {error && <p className="readout text-xs text-bad">the model could not run that — {error}</p>}
      </div>
    </Panel>
  );
}

/* ── the wheel ──────────────────────────────────────────────── */

function WheelPanel({ round, onSpin }: { round: RoundState; onSpin: () => void }) {
  const landedIndex = round.spinResult === null ? -1 : round.tokens.indexOf(round.spinResult);

  return (
    <div className="flex flex-col items-center gap-3 border-t border-line-faint pt-3 sm:flex-row sm:items-start">
      <Wheel
        segments={round.tokens.map((token, i) => ({ token, probability: round.probs[i] ?? 0 }))}
        size={168}
        landedIndex={landedIndex}
      />

      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <ol className="flex flex-col gap-1">
          {round.tokens.map((token, i) => (
            <li key={token} className="flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 shrink-0"
                style={{ background: segmentColour(i), borderRadius: '2px' }}
                aria-hidden
              />
              <span
                className={cx(
                  'min-w-0 flex-1 truncate font-mono text-xs',
                  i === landedIndex ? 'text-accent' : 'text-secondary'
                )}
              >
                {displayToken(token)}
              </span>
              <span className="readout shrink-0 text-xs text-primary">
                {formatPercent(round.probs[i] ?? 0)}
              </span>
            </li>
          ))}
        </ol>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={onSpin}>
            <Dices size={13} strokeWidth={2} />
            Spin
          </Button>
          {round.spinResult !== null && (
            <span className="text-xs text-secondary">
              landed on{' '}
              <span className="font-mono text-accent">{displayToken(round.spinResult)}</span> at{' '}
              {formatPercent(round.probs[landedIndex] ?? 0)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * A donut whose arc lengths are the probabilities.
 *
 * Spinning rotates the wheel so the sampled segment finishes under the pointer,
 * which is only honest because the sampling happened in the engine first — the
 * animation reports the draw, it does not decide it.
 */
function Wheel({
  segments,
  size,
  landedIndex = -1,
  dim,
}: {
  segments: { token: string; probability: number }[];
  size: number;
  landedIndex?: number;
  dim?: boolean;
}) {
  const reduce = useReducedMotion();
  const radius = size / 2;
  const inner = radius * 0.52;

  let cursor = 0;
  const arcs = segments.map((segment, i) => {
    const start = cursor * 360;
    cursor += segment.probability;
    const end = cursor * 360;
    return { ...segment, start, end, mid: (start + end) / 2, index: i };
  });

  // Bring the landed segment to the top, plus whole turns for the spin itself.
  const rotation = landedIndex >= 0 && arcs[landedIndex] ? 720 - arcs[landedIndex]!.mid : 0;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <motion.svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={segments
          .map((s) => `${s.token} ${formatPercent(s.probability)}`)
          .join(', ')}
        initial={false}
        animate={{ rotate: rotation }}
        transition={reduce ? { duration: 0 } : { duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
        style={{ opacity: dim ? 0.45 : 1 }}
      >
        {arcs.map((arc) => (
          <path
            key={arc.token}
            d={donutArc(radius, radius, radius - 1, inner, arc.start, arc.end)}
            fill={segmentColour(arc.index)}
            stroke="var(--surface-base)"
            strokeWidth={1}
            opacity={landedIndex >= 0 && arc.index !== landedIndex ? 0.35 : 1}
          />
        ))}
      </motion.svg>

      {/* Pointer stays put while the wheel turns under it. */}
      <span
        className="absolute left-1/2 top-0 -translate-x-1/2"
        style={{
          width: 0,
          height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderTop: '10px solid var(--text-primary)',
        }}
        aria-hidden
      />
    </div>
  );
}

/** SVG path for one donut segment. A full circle needs two arcs, not one. */
function donutArc(
  cx: number,
  cy: number,
  outer: number,
  inner: number,
  startDeg: number,
  endDeg: number
): string {
  const sweep = endDeg - startDeg;
  if (sweep <= 0) return '';

  // A 360° arc is degenerate — draw it as two halves instead.
  if (sweep >= 359.999) {
    return [
      donutArc(cx, cy, outer, inner, 0, 180),
      donutArc(cx, cy, outer, inner, 180, 360),
    ].join(' ');
  }

  const point = (radius: number, degrees: number) => {
    const radians = ((degrees - 90) * Math.PI) / 180;
    return [cx + radius * Math.cos(radians), cy + radius * Math.sin(radians)];
  };

  const [x1, y1] = point(outer, startDeg);
  const [x2, y2] = point(outer, endDeg);
  const [x3, y3] = point(inner, endDeg);
  const [x4, y4] = point(inner, startDeg);
  const large = sweep > 180 ? 1 : 0;

  return [
    `M ${x1} ${y1}`,
    `A ${outer} ${outer} 0 ${large} 1 ${x2} ${y2}`,
    `L ${x3} ${y3}`,
    `A ${inner} ${inner} 0 ${large} 0 ${x4} ${y4}`,
    'Z',
  ].join(' ');
}

/**
 * Segment fill.
 *
 * Stepped down from the world accent rather than a rainbow: the segments have
 * an order, and a hue per token would imply a meaning that is not there.
 */
function segmentColour(index: number): string {
  const mix = Math.max(12, 88 - index * 11);
  return `color-mix(in oklab, var(--accent) ${mix}%, var(--surface-inset))`;
}

/* ── shared bits ────────────────────────────────────────────── */

function PromptLine({ prompt, continuation }: { prompt: string; continuation?: string }) {
  return (
    <p
      className="border border-line bg-inset px-2 py-1.5 font-mono text-xs leading-relaxed text-primary"
      style={{ borderRadius: 'var(--radius)' }}
    >
      {prompt}
      {continuation !== undefined ? (
        <span className="text-accent">{displayToken(continuation)}</span>
      ) : (
        <span className="text-muted"> ▮</span>
      )}
    </p>
  );
}

/** Tokens carry their leading space; show it rather than letting it vanish. */
function displayToken(token: string): React.ReactNode {
  if (token.length === 0) return <span className="text-muted">∅</span>;
  return [...token].map((character, index) =>
    character === ' ' ? (
      <span key={index} className="text-muted/70">
        ␣
      </span>
    ) : character === '\n' ? (
      <span key={index} className="text-muted/70">
        ⏎
      </span>
    ) : (
      <span key={index}>{character}</span>
    )
  );
}

/** Plain-text form, for places that need a string rather than nodes. */
function displayTokenText(token: string): string {
  return token.replace(/ /g, '␣').replace(/\n/g, '⏎') || '∅';
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Deterministic Fisher–Yates, seeded by the prompt. */
function shuffle(items: readonly string[], seed: string): string[] {
  const rng = createRng(
    [...seed].reduce((hash, character) => (hash * 31 + character.charCodeAt(0)) | 0, 7)
  );
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: ProbabilityWheelState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = submitBlocked(state);
  return (
    <>
      {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
      {revealed && <Tag tone="accent">distribution revealed</Tag>}
      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || revealed}
        title={blocked ? submitHint(state) : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

function submitBlocked(state: ProbabilityWheelState): boolean {
  switch (state.mode) {
    case 'predict-top1':
      return state.rounds.some((round) => round.pick === null);
    case 'allocate-mass':
      return state.rounds.some((round) => round.allocation.reduce((a, b) => a + b, 0) <= 0);
    case 'estimate-entropy': {
      const target = state.config.rounds ?? state.rounds.length;
      return (
        state.rounds.length < target || state.rounds.some((round) => round.entropyEstimate === null)
      );
    }
  }
}

function submitHint(state: ProbabilityWheelState): string {
  switch (state.mode) {
    case 'predict-top1': {
      const left = state.rounds.filter((round) => round.pick === null).length;
      return `${left} prompt${left === 1 ? '' : 's'} left`;
    }
    case 'allocate-mass': {
      const left = state.rounds.filter(
        (round) => round.allocation.reduce((a, b) => a + b, 0) <= 0
      ).length;
      return `${left} prompt${left === 1 ? '' : 's'} unallocated`;
    }
    case 'estimate-entropy': {
      const target = state.config.rounds ?? state.rounds.length;
      if (state.rounds.length < target) return `add ${target - state.rounds.length} more prompt(s)`;
      const left = state.rounds.filter((round) => round.entropyEstimate === null).length;
      return `${left} estimate${left === 1 ? '' : 's'} left`;
    }
  }
}
