'use client';

/**
 * Chapter 8.3 — Calibration & Hallucination.
 *
 * All three boards precompute every real decode and confidence value once in
 * `prepare()` (zero live model calls from player actions, unlike 8.2's
 * budget-subset board) — every round's real confidence is hidden until the
 * player commits a guess, so guessing is never a "read the biggest number"
 * exercise: level 1 hides real correctness behind a shown confidence value,
 * level 2 hides real confidence behind shown answer text, level 3 hides both
 * real confidence values behind a shown baseline answer.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw, X } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  realDropped,
  type BaselineResult,
  type CalibrationAction,
  type CalibrationConfig,
  type CalibrationState,
  type FramingRound,
  type PredictRound,
  type SpotRound,
} from '@/engines/calibrationEngine';
import { tinyCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const MODEL_SIZE_MB = 92;

export function CalibrationCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as CalibrationConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<CalibrationState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { corpus: corpusLoader, causalLM: tinyCausalLM });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: CalibrationAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={CAUSAL_LM_MODEL_ID}
      estimatedSizeMB={MODEL_SIZE_MB}
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
  state: CalibrationState;
  dispatch: (action: CalibrationAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    setRevealed(true);
  };

  const blocked = submitBlocked(state);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'predict-correctness' && <PredictBoard state={state} dispatch={dispatch} />}
          {state.mode === 'spot-hallucination' && <SpotBoard state={state} dispatch={dispatch} />}
          {state.mode === 'reduce-confidence' && <FramingBoard state={state} dispatch={dispatch} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
        {revealed && <Tag tone="accent">answers revealed</Tag>}
        <Button
          variant="primary"
          className="ml-auto"
          disabled={blocked || revealed}
          title={blocked ? submitHint(state) : undefined}
          onClick={submit}
        >
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

function submitBlocked(state: CalibrationState): boolean {
  switch (state.mode) {
    case 'predict-correctness':
      return state.predictRounds.some((r) => r.guess === null);
    case 'spot-hallucination':
      return state.spotRounds.some((r) => r.guessFactId === null);
    case 'reduce-confidence':
      return state.framingRounds.some((r) => r.guess === null);
  }
}

function submitHint(state: CalibrationState): string {
  switch (state.mode) {
    case 'predict-correctness': {
      const left = state.predictRounds.filter((r) => r.guess === null).length;
      return `${left} question${left === 1 ? '' : 's'} left`;
    }
    case 'spot-hallucination': {
      const left = state.spotRounds.filter((r) => r.guessFactId === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
    case 'reduce-confidence': {
      const left = state.framingRounds.filter((r) => r.guess === null).length;
      return `${left} question${left === 1 ? '' : 's'} left`;
    }
  }
}

function pct(value: number): string {
  return `${formatNumber(value * 100, 1)}%`;
}

/* ── level 1: predict correctness from confidence ────────────── */

function PredictBoard({
  state,
  dispatch,
}: {
  state: CalibrationState;
  dispatch: (action: CalibrationAction) => void;
}) {
  return (
    <Panel label="given its real confidence, will the model's real answer be correct?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.predictRounds.map((round, index) => (
          <PredictRow
            key={round.factId}
            round={round}
            onGuess={(guess) => dispatch({ type: 'GUESS_CORRECTNESS', roundIndex: index, guess })}
          />
        ))}
      </ol>
    </Panel>
  );
}

function PredictRow({ round, onGuess }: { round: PredictRound; onGuess: (guess: boolean) => void }) {
  const revealed = round.guess !== null;
  const correctGuess = revealed && round.guess === round.correct;

  return (
    <li
      className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-sm text-primary">{round.query}</p>
      <div className="flex items-center gap-3">
        <Readout label="real confidence" value={pct(round.confidence)} size="sm" />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'will be correct', guess: true },
            { label: 'will be wrong', guess: false },
          ] as const
        ).map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={round.guess === option.guess}
            disabled={revealed}
            onClick={() => onGuess(option.guess)}
            className={cx(
              'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
              round.guess === option.guess
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-transparent text-secondary hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {option.label}
          </button>
        ))}
        {revealed && (
          <Tag tone={correctGuess ? 'good' : 'bad'}>
            {correctGuess ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            really {round.correct ? 'correct' : 'wrong'}
          </Tag>
        )}
      </div>
      {revealed && <p className="readout text-[11px] text-muted">“{round.decodedText.trim()}”</p>}
    </li>
  );
}

/* ── level 2: spot the confident hallucination ───────────────── */

function SpotBoard({ state, dispatch }: { state: CalibrationState; dispatch: (action: CalibrationAction) => void }) {
  return (
    <div className="flex flex-col gap-4">
      {state.spotRounds.map((round, index) => (
        <SpotRoundPanel
          key={index}
          index={index}
          round={round}
          onGuess={(factId) => dispatch({ type: 'GUESS_HALLUCINATION', roundIndex: index, factId })}
        />
      ))}
    </div>
  );
}

function SpotRoundPanel({
  index,
  round,
  onGuess,
}: {
  index: number;
  round: SpotRound;
  onGuess: (factId: string) => void;
}) {
  const revealed = round.guessFactId !== null;
  const correctGuess = revealed && round.guessFactId === round.targetFactId;

  return (
    <Panel label={`round ${index + 1} — which real answer is wrong AND the most confident?`} flush>
      <div className="flex flex-col gap-2 p-2">
        {round.candidates.map((candidate) => {
          const isGuess = round.guessFactId === candidate.factId;
          const isTarget = revealed && candidate.factId === round.targetFactId;
          return (
            <button
              key={candidate.factId}
              type="button"
              aria-pressed={isGuess}
              disabled={revealed}
              onClick={() => onGuess(candidate.factId)}
              className={cx(
                'flex flex-col gap-1 border px-3 py-2 text-left transition-colors',
                isGuess && !revealed && 'border-accent bg-accent-dim',
                revealed && isTarget && 'border-good bg-good/10',
                revealed && isGuess && !isTarget && 'border-bad bg-bad/10',
                !isGuess && !(revealed && isTarget) && 'border-line-strong bg-transparent hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              <span className="text-xs text-secondary">{candidate.query}</span>
              <span className="readout text-xs text-primary">“{candidate.decodedText.trim()}”</span>
              {revealed && <CandidateReveal candidate={candidate} />}
            </button>
          );
        })}
        {revealed && (
          <Tag tone={correctGuess ? 'good' : 'bad'}>
            {correctGuess ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            {correctGuess ? 'found the real hallucination' : 'that one was not it'}
          </Tag>
        )}
      </div>
    </Panel>
  );
}

function CandidateReveal({ candidate }: { candidate: BaselineResult }) {
  return (
    <span className="flex items-center gap-2 text-[11px]">
      <Tag tone={candidate.correct ? 'good' : 'bad'}>{candidate.correct ? 'really correct' : 'really wrong'}</Tag>
      <span className="label">real confidence {pct(candidate.confidence)}</span>
    </span>
  );
}

/* ── level 3: can you talk it down? ──────────────────────────── */

function FramingBoard({
  state,
  dispatch,
}: {
  state: CalibrationState;
  dispatch: (action: CalibrationAction) => void;
}) {
  return (
    <Panel label="will the cautious framing really raise or drop its confidence?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.framingRounds.map((round, index) => (
          <FramingRow
            key={round.factId}
            round={round}
            onGuess={(guess) => dispatch({ type: 'GUESS_DELTA', roundIndex: index, guess })}
          />
        ))}
      </ol>
    </Panel>
  );
}

function FramingRow({ round, onGuess }: { round: FramingRound; onGuess: (guess: 'drop' | 'rise') => void }) {
  const revealed = round.guess !== null && round.framedConfidence !== null;
  const dropped = revealed && realDropped(round);
  const correctGuess = revealed && (round.guess === 'drop') === dropped;

  return (
    <li
      className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-sm text-primary">{round.query}</p>
      <p className="readout text-[11px] text-muted">baseline: “{round.decodedText.trim()}”</p>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'confidence will drop', guess: 'drop' as const },
            { label: 'confidence will rise', guess: 'rise' as const },
          ] as const
        ).map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={round.guess === option.guess}
            disabled={revealed}
            onClick={() => onGuess(option.guess)}
            className={cx(
              'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
              round.guess === option.guess
                ? 'border-accent bg-accent-dim text-primary'
                : 'border-line-strong bg-transparent text-secondary hover:border-accent'
            )}
            style={{ borderRadius: 'var(--radius)' }}
          >
            {option.label}
          </button>
        ))}
        {revealed && (
          <Tag tone={correctGuess ? 'good' : 'bad'}>
            {correctGuess ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            really {dropped ? 'dropped' : 'rose'}
          </Tag>
        )}
      </div>
      {revealed && (
        <div className="flex flex-wrap items-center gap-3">
          <Readout label="baseline" value={pct(round.confidence)} size="sm" />
          <Readout label="framed" value={pct(round.framedConfidence!)} size="sm" />
          <p className="readout text-[11px] text-muted">framed: “{round.framedDecodedText?.trim()}”</p>
        </div>
      )}
    </li>
  );
}
