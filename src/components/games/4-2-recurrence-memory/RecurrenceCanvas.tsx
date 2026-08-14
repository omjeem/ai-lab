'use client';

/**
 * Chapter 4.2 — Why Recurrence Isn't Enough.
 *
 * A real char-RNN trains in-browser (`TinyRNN`), and every number here reads
 * off its actual hidden state — the decay curves are `measureDecay`'s real
 * cosine-similarity trace between two genuine forward passes, not an
 * animation of what forgetting would look like, and the recall bars are
 * `measureRecall`'s real per-gap accuracy from the trained network.
 *
 * `HiddenStateStrip` renders the literal hidden vector as cells: this is the
 * one chapter where watching that specific vector get overwritten step by
 * step *is* the lesson, so it is drawn directly rather than summarised.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, Play, RotateCcw, Search, Square, StepForward } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  measureDecay,
  type MemoryDecayAction,
  type MemoryDecayConfig,
  type MemoryDecayState,
} from '@/engines/memoryDecayEngine';
import { encodeChars } from '@/models/tinyRNNTrainer';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules, ScoreResult } from '@/types/game';

const TRAIN_CHUNK = 5;

export function RecurrenceCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as MemoryDecayConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<MemoryDecayState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { corpus: corpusLoader });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: MemoryDecayAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={game.modelRequirement.modelId}
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
  state: MemoryDecayState;
  dispatch: (action: MemoryDecayAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  // Local, not `state.status === 'complete'`: every action bumps status back
  // to `active`, so a post-reveal estimate change would silently un-reveal.
  const [revealed, setRevealed] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const maxEpochs = state.config.maxEpochs ?? state.config.epochs;
  const exhausted = state.epochsTrained >= maxEpochs;

  // Chunked, self-chaining training: each tick trains a short burst, commits,
  // and — because `epochsTrained` is a dependency — schedules the next tick,
  // so the loss curve visibly moves instead of freezing the tab.
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
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const isQuiz = state.mode === 'predict-decay';
  const isTrainable = state.mode === 'train' || state.mode === 'recall-task';

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    if (isQuiz) setRevealed(true);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        {state.mode === 'train' && <TrainBoard state={state} dispatch={dispatch} result={result} isTraining={isTraining} />}
        {state.mode === 'predict-decay' && <DecayBoard state={state} dispatch={dispatch} revealed={revealed} />}
        {state.mode === 'recall-task' && (
          <RecallBoard state={state} dispatch={dispatch} result={result} isTraining={isTraining} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>

        {isTrainable && (
          <>
            <Button onClick={() => dispatch({ type: 'TRAIN', epochs: TRAIN_CHUNK })} disabled={exhausted || isTraining}>
              <StepForward size={13} strokeWidth={2} />×{TRAIN_CHUNK}
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
            {state.mode === 'recall-task' && (
              <Button onClick={() => dispatch({ type: 'RUN_RECALL' })}>
                <Search size={13} strokeWidth={2} />
                Run recall
              </Button>
            )}
          </>
        )}

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
  state: MemoryDecayState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const left = state.rounds.filter((r) => r.estimate === null).length;
  const blocked = left > 0;

  return (
    <>
      {!revealed && blocked && <span className="label">{left} round{left === 1 ? '' : 's'} left</span>}
      {revealed && <Tag tone="accent">answers revealed</Tag>}
      <Button variant="primary" className="ml-auto" disabled={blocked || revealed} onClick={onSubmit}>
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

/* ── level 1: train the rnn ────────────────────────────────────── */

function TrainBoard({
  state,
  dispatch,
  result,
  isTraining,
}: {
  state: MemoryDecayState;
  dispatch: (action: MemoryDecayAction) => void;
  result: ScoreResult;
  isTraining: boolean;
}) {
  const probe = useMemo(() => state.encoded.slice(0, 40), [state.encoded]);
  const probeHidden = useMemo(() => state.rnn.forwardSequence(probe).hiddenStates.at(-1) ?? [], [state.rnn, probe]);

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[420px]">
        <Panel label="training loss">
          {state.lossHistory.length > 0 ? (
            <LineTrace values={state.lossHistory} ariaLabel="Training loss by epoch" />
          ) : (
            <p className="text-xs text-muted">Train to see the loss curve.</p>
          )}
        </Panel>
        <Panel label="hidden state — after reading the corpus' first 40 characters">
          <HiddenStateStrip values={probeHidden} />
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Every one of these {probeHidden.length} cells is being overwritten at every character. Watch it
            reorganise as training progresses.
          </p>
        </Panel>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Readout label="next-char accuracy" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
              <Readout label="epochs" value={`${state.epochsTrained} / ${state.config.maxEpochs ?? state.config.epochs}`} size="sm" />
            </div>
            <Meter
              value={result.value}
              max={Math.max(result.value, state.rules.passCriteria.threshold) * 1.4}
              threshold={state.rules.passCriteria.threshold}
              tone={result.passed ? 'good' : 'accent'}
            />
            {isTraining && <Tag tone="accent">training…</Tag>}
          </div>
        </Panel>

        <Panel label="hidden size">
          <Slider
            label="units"
            value={state.hiddenSize}
            min={state.config.hiddenSizeRange?.[0] ?? state.hiddenSize}
            max={state.config.hiddenSizeRange?.[1] ?? state.hiddenSize}
            step={1}
            disabled={isTraining}
            onChange={(value) => dispatch({ type: 'SET_HIDDEN_SIZE', value })}
            format={(value) => String(Math.round(value))}
          />
          {state.config.hiddenSizeRange && (
            <p className="mt-2 text-xs leading-relaxed text-muted">
              Changing this rebuilds the network from scratch — training so far is discarded.
            </p>
          )}
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
      </div>
    </div>
  );
}

/* ── level 2: watch it forget ──────────────────────────────────── */

function DecayBoard({
  state,
  dispatch,
  revealed,
}: {
  state: MemoryDecayState;
  dispatch: (action: MemoryDecayAction) => void;
  revealed: boolean;
}) {
  return (
    <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
      <p className="text-xs leading-relaxed text-muted">
        Each curve is the real cosine similarity between two hidden states from a trained network fed
        sequences that differ only in their first token. Estimate the distance at which that curve first
        crosses the {(state.config.decayThreshold ?? 0.99).toFixed(2)} threshold — the point the opening
        token is, for practical purposes, gone.
      </p>

      <Panel label="predict the decay distance">
        <div className="flex flex-col gap-4">
          {state.rounds.map((round, i) => {
            const off = round.estimate === null ? null : Math.abs(round.estimate - round.trueDecayStep);
            return (
              <div
                key={i}
                className={cx(
                  'flex flex-col gap-2 border p-3',
                  revealed
                    ? off !== null && off <= 2
                      ? 'border-good/40 bg-good/5'
                      : 'border-bad/40 bg-bad/5'
                    : 'border-line-strong bg-raised'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                <div className="flex items-center justify-between">
                  <span className="label">round {i + 1}</span>
                  {revealed && <Tag tone={off !== null && off <= 2 ? 'good' : 'bad'}>off by {off}</Tag>}
                </div>
                <SimilarityTrace
                  trace={round.similarityTrace}
                  estimate={round.estimate}
                  trueStep={revealed ? round.trueDecayStep : null}
                  ariaLabel={`Hidden-state similarity trace for round ${i + 1}`}
                />
                <Slider
                  label="distance estimate"
                  value={round.estimate ?? 0}
                  min={0}
                  max={state.config.sequenceLength}
                  step={1}
                  disabled={revealed}
                  onChange={(value) => dispatch({ type: 'ESTIMATE_DECAY', roundIndex: i, value })}
                  format={(value) => String(Math.round(value))}
                />
                {revealed && (
                  <span className="readout text-xs text-secondary">true distance: {round.trueDecayStep}</span>
                )}
              </div>
            );
          })}
        </div>
      </Panel>

      {state.config.allowUserInput && <ExplorerPanel state={state} />}
    </div>
  );
}

function ExplorerPanel({ state }: { state: MemoryDecayState }) {
  const defaultTail = useMemo(
    () => state.encoded.slice(0, 60).map((i) => state.vocab.chars[i] ?? '').join(''),
    [state.encoded, state.vocab]
  );
  const [tail, setTail] = useState(defaultTail);
  const [step, setStep] = useState(0);
  const [measured, setMeasured] = useState<{
    trace: number[];
    decayStep: number;
    hiddenA: number[][];
    hiddenB: number[][];
  } | null>(null);

  const markerA = state.config.probeToken || 'Q';
  const markerB = state.vocab.chars.find((c) => c !== markerA) ?? state.vocab.chars[0] ?? 'a';

  const measure = () => {
    const encodedTail = encodeChars(tail, state.vocab);
    const [markerAIndex] = encodeChars(markerA, state.vocab);
    const [markerBIndex] = encodeChars(markerB, state.vocab);
    const { trace, decayStep } = measureDecay(
      state.rnn,
      encodedTail,
      markerAIndex!,
      markerBIndex!,
      state.config.decayThreshold ?? 0.99
    );
    const hiddenA = state.rnn.forwardSequence([markerAIndex!, ...encodedTail]).hiddenStates;
    const hiddenB = state.rnn.forwardSequence([markerBIndex!, ...encodedTail]).hiddenStates;
    setMeasured({ trace, decayStep, hiddenA, hiddenB });
    setStep(0);
  };

  return (
    <Panel label="try your own sequence">
      <div className="flex flex-col gap-3">
        <p className="text-xs leading-relaxed text-muted">
          Compares two runs that differ only in the opening character — &quot;{markerA}&quot; against &quot;
          {markerB}&quot; — followed by whatever you type below, on the same trained network.
        </p>
        <div className="flex flex-wrap items-center gap-1.5">
          <input
            value={tail}
            onChange={(event) => setTail(event.target.value.slice(0, state.config.sequenceLength))}
            placeholder="text to follow the marker"
            aria-label="Text to follow the marker"
            className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={measure} disabled={tail.length === 0}>
            <Search size={13} strokeWidth={2} />
            Measure
          </Button>
        </div>

        {measured && (
          <div className="flex flex-col gap-3">
            <SimilarityTrace
              trace={measured.trace}
              trueStep={measured.decayStep}
              ariaLabel="Hidden-state similarity trace for your sequence"
            />
            <Slider
              label="step"
              value={step}
              min={0}
              max={Math.max(0, measured.hiddenA.length - 1)}
              step={1}
              onChange={(value) => setStep(Math.round(value))}
              format={(value) => String(Math.round(value))}
            />
            <HiddenStateStrip values={measured.hiddenA[step] ?? []} label={`starts with "${markerA}"`} />
            <HiddenStateStrip values={measured.hiddenB[step] ?? []} label={`starts with "${markerB}"`} />
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ── level 3: the recall task ──────────────────────────────────── */

function RecallBoard({
  state,
  dispatch,
  result,
  isTraining,
}: {
  state: MemoryDecayState;
  dispatch: (action: MemoryDecayAction) => void;
  result: ScoreResult;
  isTraining: boolean;
}) {
  const sorted = useMemo(() => [...state.recallResults].sort((a, b) => a.gap - b.gap), [state.recallResults]);

  return (
    <div className="mx-auto my-auto flex w-full max-w-5xl flex-col gap-3 lg:flex-row lg:items-start">
      <div className="flex w-full shrink-0 flex-col gap-3 lg:w-[440px]">
        <Panel label="recall accuracy by gap">
          {sorted.length > 0 ? (
            <RecallBars results={sorted} />
          ) : (
            <p className="text-xs text-muted">Run recall to measure accuracy at each gap length.</p>
          )}
        </Panel>
        <Panel label="score">
          <div className="flex flex-col gap-3">
            <Readout label="collapse score" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
            <div className="flex flex-wrap gap-4">
              <Readout label="nearest gap" value={result.breakdown.nearestGapAccuracy ?? 0} size="sm" />
              <Readout label="furthest gap" value={result.breakdown.furthestGapAccuracy ?? 0} size="sm" />
            </div>
            <Meter
              value={result.value}
              max={Math.max(result.value, state.rules.passCriteria.threshold) * 1.4}
              threshold={state.rules.passCriteria.threshold}
              tone={result.passed ? 'good' : 'accent'}
            />
          </div>
        </Panel>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <Panel label="training">
          <div className="flex flex-col gap-3">
            <Readout label="epochs" value={`${state.epochsTrained} / ${state.config.maxEpochs ?? state.config.epochs}`} size="sm" />
            {state.lossHistory.length > 0 && <LineTrace values={state.lossHistory} ariaLabel="Training loss by epoch" />}
            {isTraining && <Tag tone="accent">training…</Tag>}
          </div>
        </Panel>

        <Panel label="hidden size">
          <Slider
            label="units"
            value={state.hiddenSize}
            min={state.config.hiddenSizeRange?.[0] ?? state.hiddenSize}
            max={state.config.hiddenSizeRange?.[1] ?? state.hiddenSize}
            step={1}
            disabled={isTraining}
            onChange={(value) => dispatch({ type: 'SET_HIDDEN_SIZE', value })}
            format={(value) => String(Math.round(value))}
          />
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Changing this rebuilds the network from scratch — train again to re-measure recall.
          </p>
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
      </div>
    </div>
  );
}

function RecallBars({ results }: { results: { gap: number; accuracy: number }[] }) {
  return (
    <div className="flex items-end gap-2" role="img" aria-label="Recall accuracy at each gap length">
      {results.map((r) => (
        <div key={r.gap} className="flex flex-1 flex-col items-center gap-1">
          <span className="readout text-[10px] text-muted">{(r.accuracy * 100).toFixed(0)}%</span>
          <div className="relative h-20 w-full bg-inset">
            <div className="absolute inset-x-0 bottom-0 bg-accent" style={{ height: `${Math.max(2, r.accuracy * 100)}%` }} />
          </div>
          <span className="readout text-xs text-muted">{r.gap}</span>
        </div>
      ))}
    </div>
  );
}

/* ── shared: hidden-state strip ────────────────────────────────── */

function HiddenStateStrip({ values, label }: { values: number[]; label?: string }) {
  return (
    <div className="flex flex-col gap-1">
      {label && <span className="label">{label}</span>}
      <div className="flex flex-wrap gap-[2px]" role="img" aria-label={label ?? 'Hidden state vector'}>
        {values.map((v, i) => {
          const magnitude = Math.max(-1, Math.min(1, v));
          const color = magnitude >= 0 ? 'var(--accent)' : 'var(--signal-bad)';
          return (
            <div
              key={i}
              className="h-5 w-2.5 shrink-0"
              title={`unit ${i}: ${v.toFixed(3)}`}
              style={{
                background: `color-mix(in oklab, ${color} ${Math.round(Math.abs(magnitude) * 88 + 6)}%, var(--surface-inset))`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ── shared: line + similarity traces ──────────────────────────── */

function LineTrace({ values, ariaLabel }: { values: number[]; ariaLabel: string }) {
  const width = 100;
  const height = 28;
  const hi = Math.max(...values, 1e-6);
  const toX = (i: number) => (values.length <= 1 ? 0 : (i / (values.length - 1)) * width);
  const toY = (v: number) => height - Math.min(1, Math.max(0, v / hi)) * height;
  const path = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(v).toFixed(2)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block h-20 w-full" role="img" aria-label={ariaLabel}>
      <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line-faint)" strokeWidth={0.3} />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={0.8} />
    </svg>
  );
}

function SimilarityTrace({
  trace,
  estimate,
  trueStep,
  ariaLabel,
}: {
  trace: number[];
  estimate?: number | null;
  trueStep?: number | null;
  ariaLabel: string;
}) {
  const width = 100;
  const height = 32;
  const n = trace.length;
  const toX = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * width);
  const toY = (v: number) => height - Math.max(0, Math.min(1, v)) * height;
  const path = trace.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(2)} ${toY(v).toFixed(2)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block h-28 w-full" role="img" aria-label={ariaLabel}>
      <line x1={0} y1={height} x2={width} y2={height} stroke="var(--line-faint)" strokeWidth={0.3} />
      <path d={path} fill="none" stroke="var(--text-secondary)" strokeWidth={0.8} />
      {estimate != null && (
        <line x1={toX(estimate)} y1={0} x2={toX(estimate)} y2={height} stroke="var(--accent)" strokeWidth={0.9} strokeDasharray="2 1.5" />
      )}
      {trueStep != null && (
        <line x1={toX(trueStep)} y1={0} x2={toX(trueStep)} y2={height} stroke="var(--signal-good)" strokeWidth={1.1} />
      )}
    </svg>
  );
}
