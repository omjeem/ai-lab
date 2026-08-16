'use client';

/**
 * Chapter 5.2 — Self-Attention. The centrepiece.
 *
 * The attention matrix is the instrument: every cell is a real softmaxed
 * weight pulled from `Qdrant/all_miniLM_L6_v2_with_attentions`'s actual
 * forward pass — see `src/models/attentionModel.ts` for why that model and
 * not the one this chapter was originally scoped against. Nothing here is a
 * static illustration; changing the layer recomputes real cells.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import { Check, ChevronLeft, ChevronRight, Play, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  attentionRow,
  buildRound,
  editDistance,
  type AttentionGuessAction,
  type AttentionGuessConfig,
  type AttentionGuessState,
  type AttentionRound,
  type FlipAttempt,
} from '@/engines/attentionGuessEngine';
import type { AttentionResult } from '@/engines/deps';
import { attentionModel, ATTENTION_MODEL_ID } from '@/models/attentionModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Slider, Tag, cx } from '@/components/ui';
import { normalizeDistribution } from '@/engines/shared';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function SelfAttentionCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as AttentionGuessConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<AttentionGuessState | null>(null);
  const [results, setResults] = useState<AttentionResult[]>([]);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { attention: attentionModel });
    setResults(prepared.results);
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: AttentionGuessAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={ATTENTION_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} results={results} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  results,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: AttentionGuessState;
  results: AttentionResult[];
  dispatch: (action: AttentionGuessAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  /** Kept in local state, never `state.status` — every action resets status to `active`. */
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
        <div key={runId} className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-4">
          <Slider
            label={`layer (0–${state.config.layerRange?.[1] ?? state.layer})`}
            value={state.layer}
            min={state.config.layerRange?.[0] ?? state.layer}
            max={state.config.layerRange?.[1] ?? state.layer}
            step={1}
            onChange={(value) => dispatch({ type: 'SET_LAYER', value: Math.round(value) })}
            format={(value) => `${Math.round(value)}`}
          />

          {state.mode === 'guess-argmax' && (
            <GuessArgmaxBoard state={state} results={results} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'draw-distribution' && (
            <DrawDistributionBoard state={state} results={results} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'flip-reference' && (
            <FlipReferenceBoard state={state} baseline={results[0]} dispatch={dispatch} />
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {state.mode !== 'flip-reference' && (
          <Actions
            state={state}
            revealed={revealed}
            onSubmit={() => {
              onSubmit(evaluate(state));
              dispatch({ type: 'SUBMIT' });
              setRevealed(true);
            }}
          />
        )}
        {state.mode === 'flip-reference' && (
          <Button
            variant="primary"
            className="ml-auto"
            onClick={() => {
              onSubmit(evaluate(state));
              dispatch({ type: 'SUBMIT' });
            }}
          >
            <Check size={13} strokeWidth={2} />
            Submit
          </Button>
        )}
      </div>
    </div>
  );
}

/* ── level 1: where does it look? ───────────────────────────── */

function GuessArgmaxBoard({
  state,
  results,
  dispatch,
  revealed,
}: {
  state: AttentionGuessState;
  results: AttentionResult[];
  dispatch: (action: AttentionGuessAction) => void;
  revealed: boolean;
}) {
  const [roundIndex, setRoundIndex] = useState(0);
  const round = state.rounds[roundIndex];
  const result = results[roundIndex];

  const matrix = useMemo(() => {
    if (!result) return [];
    return result.tokens.map((_, q) => attentionRow(result, state.layer, q, state.config.headAggregation));
  }, [result, state.layer, state.config.headAggregation]);

  if (!round || !result) return null;

  return (
    <div className="flex flex-col gap-3">
      <RoundStepper
        rounds={state.rounds}
        index={roundIndex}
        onChange={setRoundIndex}
        revealed={revealed}
        isCorrect={(r) => isArgmaxCorrect(r)}
      />

      <PromptLine tokens={result.tokens} highlightIndex={round.queryIndex} />

      <AttentionMatrix
        tokens={result.tokens}
        matrix={matrix}
        queryIndex={round.queryIndex}
        maskRow={!revealed}
        selectedKeyIndex={round.guessIndex}
      />

      <div className="flex flex-wrap gap-1.5">
        {round.keyIndices.map((keyIndex, slot) => {
          const token = result.tokens[keyIndex]!;
          const picked = round.guessIndex === keyIndex;
          const weight = round.trueRow[slot] ?? 0;
          const isTop = revealed && weight === Math.max(...round.trueRow);
          return (
            <button
              key={keyIndex}
              type="button"
              disabled={revealed}
              onClick={() => dispatch({ type: 'GUESS', roundIndex, keyIndex })}
              aria-pressed={picked}
              className={cx(
                'min-h-[36px] border px-3 py-1.5 font-mono text-xs transition-colors disabled:cursor-default',
                isTop
                  ? 'border-good bg-good/10 text-good'
                  : picked
                    ? revealed
                      ? 'border-bad bg-bad/10 text-bad'
                      : 'border-accent bg-accent-dim text-primary'
                    : 'border-line-strong bg-raised text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {displayToken(token)}
              {revealed && <span className="readout ml-2 text-[10px] opacity-70">{formatPercent(weight)}</span>}
            </button>
          );
        })}
      </div>

      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? `Averaged over ${result.attention[state.layer]?.length ?? 12} heads at layer ${state.layer}, "${displayToken(
              result.tokens[round.queryIndex] ?? ''
            )}" attends most to the token highlighted green.`
          : `Pick which token "${displayToken(result.tokens[round.queryIndex] ?? '')}" (highlighted above) attends to most. That row of the matrix stays hidden until you submit.`}
      </p>
    </div>
  );
}

function isArgmaxCorrect(round: AttentionRound): boolean {
  if (round.guessIndex === null || round.trueRow.length === 0) return false;
  let best = 0;
  for (let i = 1; i < round.trueRow.length; i++) {
    if (round.trueRow[i]! > round.trueRow[best]!) best = i;
  }
  return round.keyIndices[best] === round.guessIndex;
}

/* ── level 2: draw the row ──────────────────────────────────── */

function DrawDistributionBoard({
  state,
  results,
  dispatch,
  revealed,
}: {
  state: AttentionGuessState;
  results: AttentionResult[];
  dispatch: (action: AttentionGuessAction) => void;
  revealed: boolean;
}) {
  const [roundIndex, setRoundIndex] = useState(0);
  const round = state.rounds[roundIndex];
  const result = results[roundIndex];
  const budget = state.config.budget ?? 100;

  const matrix = useMemo(() => {
    if (!result) return [];
    return result.tokens.map((_, q) => attentionRow(result, state.layer, q, state.config.headAggregation));
  }, [result, state.layer, state.config.headAggregation]);

  if (!round || !result) return null;

  const used = round.allocation.reduce((a, b) => a + b, 0);
  const mine = normalizeDistribution(round.allocation);

  return (
    <div className="flex flex-col gap-3">
      <RoundStepper
        rounds={state.rounds}
        index={roundIndex}
        onChange={setRoundIndex}
        revealed={revealed}
        isCorrect={null}
      />

      <PromptLine tokens={result.tokens} highlightIndex={round.queryIndex} />

      <AttentionMatrix
        tokens={result.tokens}
        matrix={matrix}
        queryIndex={round.queryIndex}
        maskRow={!revealed}
        selectedKeyIndex={null}
      />

      <Panel label={`allocate ${budget} units — ${used} used`}>
        <div className="flex flex-col gap-2.5">
          {round.keyIndices.map((keyIndex, slot) => (
            <div key={keyIndex} className="flex flex-col gap-0.5">
              <Slider
                label={displayToken(result.tokens[keyIndex]!)}
                value={round.allocation[slot] ?? 0}
                min={0}
                max={budget}
                step={1}
                disabled={revealed}
                onChange={(value) => dispatch({ type: 'ALLOCATE', roundIndex, keyIndex, value })}
                format={(value) => `${value}`}
              />
              <div className="flex items-baseline justify-between gap-3">
                <span className="label text-[9px]">yours {used > 0 ? formatPercent(mine[slot] ?? 0) : '—'}</span>
                {revealed && (
                  <span className="readout text-[10px] text-accent">
                    model {formatPercent(round.trueRow[slot] ?? 0)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <p className="text-xs leading-relaxed text-muted">
        Only the shape is scored — your points are normalised before comparison, so 10/20/30 and
        100/200/300 are the same answer. Distance is total variation: half the summed gap between your
        distribution and the model&apos;s real one.
      </p>
    </div>
  );
}

/* ── level 3: break the reference ───────────────────────────── */

function FlipReferenceBoard({
  state,
  baseline,
  dispatch,
}: {
  state: AttentionGuessState;
  baseline: AttentionResult | undefined;
  dispatch: (action: AttentionGuessAction) => void;
}) {
  const baselineRound = state.rounds[0];
  const [sentence, setSentence] = useState(state.config.sentences[0] ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<{ result: AttentionResult; round: AttentionRound } | null>(null);

  const maxEdit = state.config.maxEditDistance ?? Infinity;
  const maxAttempts = state.config.attempts ?? Infinity;
  const atLimit = state.attempts.length >= maxAttempts;

  // The displayed sentence is whichever was tried most recently, or the
  // baseline before any attempt — both computed the same way, at the
  // player's currently selected layer, so switching layers before a first
  // attempt still shows real (not stale) data.
  const activeResult = live?.result ?? baseline;
  const activeRound = live?.round ?? baselineRound;

  const matrix = useMemo(() => {
    if (!activeResult) return [];
    return activeResult.tokens.map((_, q) =>
      attentionRow(activeResult, state.layer, q, state.config.headAggregation)
    );
  }, [activeResult, state.layer, state.config.headAggregation]);

  const displayTokens = activeResult?.tokens ?? [];

  const tryEdit = async () => {
    const text = sentence.trim();
    if (!text || busy || atLimit) return;
    setBusy(true);
    setError(null);
    try {
      const result = await attentionModel.attention(text);
      const round = buildRound(state.config, result, state.layer);
      setLive({ result, round });
      dispatch({ type: 'SUBMIT_EDIT', sentence: text, attention: result });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  if (!baselineRound) return null;

  const distance = editDistance(state.config.sentences[0] ?? '', sentence);

  return (
    <div className="flex flex-col gap-3">
      <Panel label="baseline">
        <PromptLine tokens={baselineRound.tokens} highlightIndex={baselineRound.queryIndex} />
        <p className="mt-2 text-xs text-secondary">
          At layer {state.layer}, &quot;{displayToken(baselineRound.tokens[baselineRound.queryIndex] ?? '')}&quot;
          currently attends most to{' '}
          <span className="readout text-accent">{displayToken(state.baseline?.winnerToken ?? '')}</span>
          . Edit the sentence below and try it to see whether that changes.
        </p>
      </Panel>

      <div className="flex flex-wrap items-center gap-1.5">
        <input
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
          onKeyDown={(event) => event.key === 'Enter' && void tryEdit()}
          disabled={busy || atLimit}
          maxLength={220}
          aria-label="Your edited sentence"
          className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-60"
          style={{ borderRadius: 'var(--radius)' }}
        />
        <Button onClick={() => void tryEdit()} disabled={busy || atLimit || sentence.trim() === ''}>
          <Play size={13} strokeWidth={2} />
          {busy ? 'running' : 'try it'}
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <span className="label">
          edit distance {distance} {Number.isFinite(maxEdit) ? `/ ${maxEdit} max` : ''}
        </span>
        <span className="label">
          attempts {state.attempts.length} {Number.isFinite(maxAttempts) ? `/ ${maxAttempts}` : ''}
        </span>
      </div>
      {error && <p className="readout text-xs text-bad">the model could not run that — {error}</p>}

      {displayTokens.length > 0 && (
        <AttentionMatrix
          tokens={displayTokens}
          matrix={matrix}
          queryIndex={activeRound?.queryIndex ?? null}
          maskRow={false}
          selectedKeyIndex={null}
        />
      )}

      <AttemptLog attempts={state.attempts} bestScore={state.bestFlipScore} />

      <p className="text-xs leading-relaxed text-muted">
        Flipping is most of the credit; doing it with a small edit earns the rest. A flip needs the new
        winner to beat the runner-up by at least {(state.config.minFlipMargin ?? 0).toFixed(2)} of the row.
      </p>
    </div>
  );
}

function AttemptLog({ attempts, bestScore }: { attempts: FlipAttempt[]; bestScore: number }) {
  if (attempts.length === 0) return null;
  return (
    <Panel label="attempts" actions={<Tag tone={bestScore > 0 ? 'good' : 'neutral'}>best {bestScore.toFixed(2)}</Tag>}>
      <ol className="flex flex-col gap-1.5">
        {attempts.map((attempt, i) => (
          <li key={i} className="flex items-center gap-2 text-xs">
            <Tag tone={attempt.flipped ? 'good' : 'bad'}>{attempt.flipped ? 'flipped' : 'no flip'}</Tag>
            <span className="readout text-[10px] text-secondary">winner {displayToken(attempt.winnerToken)}</span>
            <span className="readout text-[10px] text-muted">dist {attempt.editDistance}</span>
            <span className="readout text-[10px] text-muted">margin {attempt.margin.toFixed(3)}</span>
            <span className="readout ml-auto text-[10px] text-accent">score {attempt.score.toFixed(2)}</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

/* ── shared visuals ─────────────────────────────────────────── */

function RoundStepper({
  rounds,
  index,
  onChange,
  revealed,
  isCorrect,
}: {
  rounds: AttentionRound[];
  index: number;
  onChange: (index: number) => void;
  revealed: boolean;
  isCorrect: ((round: AttentionRound) => boolean) | null;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Button onClick={() => onChange(Math.max(0, index - 1))} disabled={index === 0}>
        <ChevronLeft size={13} strokeWidth={2} />
        prev
      </Button>
      <div className="flex items-center gap-2">
        <span className="label">
          round {index + 1} / {rounds.length}
        </span>
        <div className="flex items-center gap-1">
          {rounds.map((round, i) => {
            const answered =
              round.guessIndex !== null || round.allocation.some((v) => v > 0);
            const correct = revealed && isCorrect ? isCorrect(round) : null;
            return (
              <button
                key={i}
                type="button"
                onClick={() => onChange(i)}
                aria-label={`Round ${i + 1}`}
                aria-current={i === index}
                className="h-2 w-2"
                style={{
                  borderRadius: '50%',
                  background:
                    correct === true
                      ? 'var(--good)'
                      : correct === false
                        ? 'var(--bad)'
                        : answered
                          ? 'var(--accent)'
                          : i === index
                            ? 'var(--text-secondary)'
                            : 'var(--line-strong)',
                }}
              />
            );
          })}
        </div>
      </div>
      <Button onClick={() => onChange(Math.min(rounds.length - 1, index + 1))} disabled={index === rounds.length - 1}>
        next
        <ChevronRight size={13} strokeWidth={2} />
      </Button>
    </div>
  );
}

/**
 * Tokens on both axes, cell intensity the real weight — the dominant visual
 * for every level. `matrix[query][key]` is expected to already be a real
 * softmaxed row (sums to ~1 across every key, specials included), scaled
 * per-row so the strongest cell in each row reads as fully saturated; that
 * is the standard convention for reading an attention heatmap, and it is
 * what makes a single row legible regardless of how the rest of the matrix
 * happens to be shaped.
 */
function AttentionMatrix({
  tokens,
  matrix,
  queryIndex,
  maskRow,
  selectedKeyIndex,
}: {
  tokens: string[];
  matrix: number[][];
  queryIndex: number | null;
  maskRow: boolean;
  selectedKeyIndex: number | null;
}) {
  const reduce = useReducedMotion();

  return (
    <Panel label="attention matrix" flush>
      <div className="overflow-x-auto p-2">
        <table className="border-collapse">
          <caption className="sr-only">
            Real attention weights between every pair of {tokens.length} tokens
          </caption>
          <thead>
            <tr>
              <th className="p-1" />
              {tokens.map((token, j) => (
                <th key={j} scope="col" className="px-[1px] pb-1 text-center font-mono text-[9px] font-normal text-muted">
                  {displayToken(token)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((rowToken, i) => {
              const isQueryRow = i === queryIndex;
              const row = matrix[i] ?? [];
              const rowMax = Math.max(...row, 1e-9);
              return (
                <tr key={i}>
                  <th
                    scope="row"
                    className={cx(
                      'whitespace-nowrap py-[1px] pr-2 text-right font-mono text-[10px] font-normal',
                      isQueryRow ? 'text-accent' : 'text-secondary'
                    )}
                  >
                    {displayToken(rowToken)}
                  </th>
                  {tokens.map((colToken, j) => {
                    const hide = isQueryRow && maskRow;
                    const value = row[j] ?? 0;
                    const intensity = hide ? 0 : Math.min(1, value / rowMax);
                    const selected = isQueryRow && selectedKeyIndex === j;
                    return (
                      <td key={j} className="p-[1px]">
                        <div
                          title={hide ? undefined : `${rowToken} → ${colToken}: ${(value * 100).toFixed(1)}%`}
                          className={cx('flex h-5 w-5 items-center justify-center', selected && 'ring-2 ring-accent')}
                          style={{
                            borderRadius: '2px',
                            background: hide
                              ? 'repeating-linear-gradient(45deg, var(--surface-inset), var(--surface-inset) 3px, var(--surface-raised) 3px, var(--surface-raised) 6px)'
                              : `color-mix(in oklab, var(--accent) ${Math.round(intensity * 82 + 6)}%, var(--surface-inset))`,
                            transition: reduce ? undefined : 'background 160ms ease-out',
                          }}
                        />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function PromptLine({ tokens, highlightIndex }: { tokens: string[]; highlightIndex: number }) {
  return (
    <p
      className="flex flex-wrap gap-1 border border-line bg-inset px-2 py-1.5 font-mono text-xs leading-relaxed"
      style={{ borderRadius: 'var(--radius)' }}
    >
      {tokens.map((token, i) => (
        <span key={i} className={i === highlightIndex ? 'bg-accent-dim px-1 text-accent' : 'text-primary'}>
          {displayToken(token)}
        </span>
      ))}
    </p>
  );
}

function displayToken(token: string): string {
  if (!token) return '∅';
  return token;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: AttentionGuessState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = submitBlocked(state);
  return (
    <>
      {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
      {revealed && <Tag tone="accent">revealed</Tag>}
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

function submitBlocked(state: AttentionGuessState): boolean {
  switch (state.mode) {
    case 'guess-argmax':
      return state.rounds.some((round) => round.guessIndex === null);
    case 'draw-distribution':
      return state.rounds.some((round) => round.allocation.reduce((a, b) => a + b, 0) <= 0);
    case 'flip-reference':
      return false;
  }
}

function submitHint(state: AttentionGuessState): string {
  switch (state.mode) {
    case 'guess-argmax': {
      const left = state.rounds.filter((round) => round.guessIndex === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
    case 'draw-distribution': {
      const left = state.rounds.filter((round) => round.allocation.reduce((a, b) => a + b, 0) <= 0).length;
      return `${left} round${left === 1 ? '' : 's'} unallocated`;
    }
    case 'flip-reference':
      return '';
  }
}
