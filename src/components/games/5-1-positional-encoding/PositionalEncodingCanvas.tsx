'use client';

/**
 * Chapter 5.1 — Positional Encoding.
 *
 * The sinusoidal formula is pure mathematics, computed live from
 * `positionalVector` — nothing here is a static illustration. The heatmap is
 * the dominant visual for all three levels: position down, dimension across,
 * exactly the image that makes "nearby positions look alike, distant ones
 * don't" legible at a glance. Level 3 layers real token embeddings on top,
 * so the encoding is shown as one ingredient of a real pipeline.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  positionalVector,
  encodingDistinctness,
  type PositionalEncodingAction,
  type PositionalEncodingConfig,
  type PositionalEncodingState,
} from '@/engines/positionalEncodingEngine';
import { embeddingModel, EMBEDDING_MODEL_ID } from '@/models/embeddingModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function PositionalEncodingCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as PositionalEncodingConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<PositionalEncodingState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { embedder: embeddingModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: PositionalEncodingAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  const needsModel = config.mode === 'combine-with-embeddings';

  // ModelGate only calls `load` when it has a real `modelId` to gate. Levels
  // 1 and 2 are pure maths with nothing to download, so they have to init
  // themselves rather than rely on the gate's own effect never firing.
  useEffect(() => {
    if (!needsModel) void load();
  }, [needsModel, load]);

  return (
    <ModelGate
      modelId={needsModel ? EMBEDDING_MODEL_ID : null}
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
  state: PositionalEncodingState;
  dispatch: (action: PositionalEncodingAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
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
          {state.mode === 'identify-position' && (
            <IdentifyPositionBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'tune-distinctness' && <TuneDistinctnessBoard state={state} dispatch={dispatch} />}
          {state.mode === 'combine-with-embeddings' && (
            <CombineBoard state={state} dispatch={dispatch} revealed={revealed} />
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

/* ── shared visuals ─────────────────────────────────────────── */

/** One row of an encoding — a single position's real vector, dimension across. */
function EncodingStripe({ values, height = 22 }: { values: number[]; height?: number }) {
  return (
    <div className="flex w-full overflow-hidden" style={{ height, borderRadius: '2px' }}>
      {values.map((v, i) => (
        <div
          key={i}
          className="min-w-px flex-1"
          style={{ background: `color-mix(in oklab, var(--accent) ${Math.round(((v + 1) / 2) * 100)}%, var(--surface-inset))` }}
        />
      ))}
    </div>
  );
}

/**
 * Position down, dimension across — the image that makes the scheme click.
 * Rendered on a canvas rather than one element per cell: level 2 can reach
 * 64 positions × 128 dimensions, redrawn on every slider tick.
 */
function EncodingHeatmap({ matrix, cellSize = 6 }: { matrix: number[][]; cellSize?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const width = cols * cellSize;
  const height = rows * cellSize;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const styles = getComputedStyle(canvas);
    const accent = styles.getPropertyValue('--accent').trim();
    const inset = styles.getPropertyValue('--surface-inset').trim();
    ctx.clearRect(0, 0, width, height);
    for (let i = 0; i < rows; i++) {
      const row = matrix[i]!;
      for (let j = 0; j < cols; j++) {
        const t = Math.round(((row[j]! + 1) / 2) * 100);
        ctx.fillStyle = `color-mix(in oklab, ${accent} ${t}%, ${inset})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
  }, [matrix, rows, cols, width, height, cellSize]);

  return (
    <Panel label="encoding — position down, dimension across" flush>
      <div className="overflow-x-auto p-2">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{
            width: '100%',
            height: 'auto',
            maxHeight: 400,
            objectFit: 'contain',
            imageRendering: 'pixelated',
          }}
          role="img"
          aria-label={`Positional encoding heatmap, ${rows} positions by ${cols} dimensions`}
        />
      </div>
    </Panel>
  );
}

function logSliderValue(value: number, min: number, max: number): number {
  const clamped = Math.max(min, Math.min(max, value));
  return (Math.log10(clamped) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
}

function logSliderToValue(t: number, min: number, max: number): number {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return Math.pow(10, logMin + t * (logMax - logMin));
}

/* ── level 1: read the signature ────────────────────────────── */

function IdentifyPositionBoard({
  state,
  dispatch,
  revealed,
}: {
  state: PositionalEncodingState;
  dispatch: (action: PositionalEncodingAction) => void;
  revealed: boolean;
}) {
  const [roundIndex, setRoundIndex] = useState(0);
  const round = state.rounds[roundIndex];

  if (!round) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <span className="label">
          round {roundIndex + 1} / {state.rounds.length}
        </span>
        <div className="flex items-center gap-1">
          {state.rounds.map((r, i) => {
            const correct = revealed && r.answer !== null ? r.answer === r.truePosition : null;
            return (
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
                    correct === true
                      ? 'var(--good)'
                      : correct === false
                        ? 'var(--bad)'
                        : r.answer !== null
                          ? 'var(--accent)'
                          : i === roundIndex
                            ? 'var(--text-secondary)'
                            : 'var(--line-strong)',
                }}
              />
            );
          })}
        </div>
      </div>

      <Panel label="mystery stripe — which position is this?">
        <EncodingStripe values={round.encoding} height={32} />
      </Panel>

      <div className="flex flex-col gap-2">
        {round.candidates.map((candidate) => {
          const vector = positionalVector(candidate, state.config.dModel, state.config.base);
          const picked = round.answer === candidate;
          const isTrue = candidate === round.truePosition;
          return (
            <button
              key={candidate}
              type="button"
              disabled={revealed}
              onClick={() => dispatch({ type: 'ANSWER_POSITION', roundIndex, value: candidate })}
              aria-pressed={picked}
              className={cx(
                'flex flex-col gap-1 border p-2 text-left transition-colors disabled:cursor-default',
                revealed
                  ? isTrue
                    ? 'border-good bg-good/10'
                    : picked
                      ? 'border-bad bg-bad/10'
                      : 'border-line-strong'
                  : picked
                    ? 'border-accent bg-accent-dim'
                    : 'border-line-strong hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              <span className="label">
                position {candidate}
                {revealed && isTrue && <span className="ml-2 text-good">correct</span>}
              </span>
              <EncodingStripe values={vector} />
            </button>
          );
        })}
      </div>

      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? `The real position was ${round.truePosition}. Low dimensions (left side of the stripe) oscillate fastest, so they disambiguate nearby candidates first.`
          : 'Click the candidate whose stripe matches the mystery one above.'}
      </p>
    </div>
  );
}

/* ── level 2: keep positions distinct ───────────────────────── */

function TuneDistinctnessBoard({
  state,
  dispatch,
}: {
  state: PositionalEncodingState;
  dispatch: (action: PositionalEncodingAction) => void;
}) {
  const matrix = useMemo(
    () => Array.from({ length: state.config.maxPosition }, (_, p) => positionalVector(p, state.dModel, state.base)),
    [state.config.maxPosition, state.dModel, state.base]
  );
  const value = encodingDistinctness(state.config.maxPosition, state.dModel, state.base);
  const [baseMin, baseMax] = state.config.baseRange ?? [2, 1e6];
  const cellSize = state.dModel <= 32 ? 8 : state.dModel <= 64 ? 5 : 3;

  return (
    <div className="flex flex-col gap-3">
      <EncodingHeatmap matrix={matrix} cellSize={cellSize} />

      <Meter value={value} max={1} threshold={state.rules.passCriteria.threshold} label="distinctness" tone="good" />
      <p className="readout text-sm text-accent">{value.toFixed(4)}</p>

      <div className="flex flex-col gap-1">
        <span className="label">dimensions (dModel)</span>
        <div className="flex flex-wrap gap-1.5">
          {state.config.dModelOptions?.map((d) => (
            <Button
              key={d}
              variant={state.dModel === d ? 'primary' : 'ghost'}
              onClick={() => dispatch({ type: 'SET_D_MODEL', value: d })}
            >
              {d}
            </Button>
          ))}
        </div>
      </div>

      <Slider
        label="base (log scale)"
        value={logSliderValue(state.base, baseMin, baseMax)}
        min={0}
        max={1}
        step={0.002}
        format={() => state.base.toFixed(0)}
        onChange={(t) => dispatch({ type: 'SET_BASE', value: logSliderToValue(t, baseMin, baseMax) })}
      />

      <p className="text-xs leading-relaxed text-muted">
        A small base gives every dimension a genuinely different frequency, so more dimensions keep helping. A large
        base makes the slow dimensions crawl so little over {state.config.maxPosition} positions that they stop
        adding anything — past a point, more dimensions can score worse, not better.
      </p>
    </div>
  );
}

/* ── level 3: word plus position ────────────────────────────── */

function findMovedWord(a: string, b: string): { word: string; posA: number; posB: number } | null {
  const wordsA = a.toLowerCase().split(/\s+/).filter(Boolean);
  const wordsB = b.toLowerCase().split(/\s+/).filter(Boolean);
  for (const word of new Set(wordsA)) {
    const posA = wordsA.indexOf(word);
    const posB = wordsB.indexOf(word);
    if (posB !== -1 && posA !== posB) return { word, posA, posB };
  }
  return null;
}

function CombineBoard({
  state,
  dispatch,
  revealed,
}: {
  state: PositionalEncodingState;
  dispatch: (action: PositionalEncodingAction) => void;
  revealed: boolean;
}) {
  const [pairIndex, setPairIndex] = useState(0);
  const pair = state.pairs[pairIndex];
  const [scaleMin, scaleMax] = state.config.scaleRange ?? [0, 2];

  const moved = useMemo(() => (pair ? findMovedWord(pair.a, pair.b) : null), [pair]);
  const movedVectors = useMemo(() => {
    if (!moved) return null;
    const embedding = state.vectors[moved.word];
    if (!embedding) return null;
    const dModel = embedding.length;
    const atA = embedding.map((v, i) => v + state.scale * positionalVector(moved.posA, dModel, state.base)[i]!);
    const atB = embedding.map((v, i) => v + state.scale * positionalVector(moved.posB, dModel, state.base)[i]!);
    return { atA, atB };
  }, [moved, state.vectors, state.scale, state.base]);

  if (!pair) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <span className="label">
          pair {pairIndex + 1} / {state.pairs.length}
        </span>
        <div className="flex items-center gap-1">
          {state.pairs.map((p, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setPairIndex(i)}
              aria-label={`Pair ${i + 1}`}
              aria-current={i === pairIndex}
              className="h-2 w-2"
              style={{
                borderRadius: '50%',
                background: p.estimate !== null ? 'var(--accent)' : i === pairIndex ? 'var(--text-secondary)' : 'var(--line-strong)',
              }}
            />
          ))}
        </div>
      </div>

      <Panel label="the pair — same words, different order">
        <p className="font-mono text-xs leading-relaxed text-primary">&quot;{pair.a}&quot;</p>
        <p className="font-mono text-xs leading-relaxed text-primary">&quot;{pair.b}&quot;</p>
      </Panel>

      {movedVectors && moved && (
        <Panel label={`"${moved.word}" at position ${moved.posA} vs position ${moved.posB}`}>
          <div className="flex flex-col gap-1.5">
            <EncodingStripe values={movedVectors.atA} />
            <EncodingStripe values={movedVectors.atB} />
          </div>
        </Panel>
      )}

      <Slider
        label="positional scale"
        value={state.scale}
        min={scaleMin}
        max={scaleMax}
        step={0.01}
        onChange={(value) => dispatch({ type: 'SET_SCALE', value })}
      />

      <Slider
        label="your estimate — divergence"
        value={pair.estimate ?? 0}
        min={0}
        max={1}
        step={0.001}
        disabled={revealed}
        onChange={(value) => dispatch({ type: 'ESTIMATE_DIVERGENCE', pairIndex, value })}
      />

      {revealed && (
        <p className="readout text-xs text-secondary">
          real divergence: <span className="text-accent">{pair.trueDivergence.toFixed(4)}</span> — yours:{' '}
          {(pair.estimate ?? 0).toFixed(4)}
        </p>
      )}

      <p className="text-xs leading-relaxed text-muted">
        All three real divergences stay under 0.09 — only one word moves, and cosine similarity is bounded, so the
        gap saturates quickly as the scale rises past about 1. The two rows above are the actual combined vector for
        the word that moved, at its two positions — the same word, genuinely two different vectors.
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
  state: PositionalEncodingState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked =
    state.mode === 'identify-position'
      ? state.rounds.some((r) => r.answer === null)
      : state.mode === 'combine-with-embeddings'
        ? state.pairs.some((p) => p.estimate === null)
        : false;

  const hint =
    state.mode === 'identify-position'
      ? `${state.rounds.filter((r) => r.answer === null).length} round(s) left`
      : state.mode === 'combine-with-embeddings'
        ? `${state.pairs.filter((p) => p.estimate === null).length} pair(s) left`
        : '';

  const hasReveal = state.mode !== 'tune-distinctness';

  return (
    <>
      {!revealed && blocked && <span className="label">{hint}</span>}
      {hasReveal && revealed && <Tag tone="accent">revealed</Tag>}
      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || (hasReveal && revealed)}
        title={blocked ? hint : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}
