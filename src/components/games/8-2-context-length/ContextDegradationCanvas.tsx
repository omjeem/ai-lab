'use client';

/**
 * Chapter 8.2 — Context Length & Degradation.
 *
 * Three boards, two different real models depending on the level: `tinyCausalLM`
 * for the needle-in-haystack and budget-subset modes, `attentionModel` for the
 * attention-dilution mode. `ModelGate` remounts fresh per level (`ChapterShell`
 * unmounts the canvas back to its concept screen between levels), so picking a
 * different real `modelId` per level's `mode` is safe — never two models
 * loading at once, unlike 8.1.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw, Search, X } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  realExtremeIndex,
  type ContextDegradationAction,
  type ContextDegradationConfig,
  type ContextDegradationState,
  type DilutionRound,
  type LengthRound,
} from '@/engines/contextDegradationEngine';
import { buildGroundedPrompt, greedyDecode } from '@/engines/groundedGenerationEngine';
import { tinyCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { attentionModel, ATTENTION_MODEL_ID } from '@/models/attentionModel';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const ATTENTION_SIZE_MB = 87;
const CAUSAL_LM_SIZE_MB = 92;

export function ContextDegradationCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as ContextDegradationConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<ContextDegradationState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, {
      corpus: corpusLoader,
      causalLM: tinyCausalLM,
      attention: attentionModel,
    });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: ContextDegradationAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  const usesAttention = config.mode === 'attention-dilution';

  return (
    <ModelGate
      modelId={usesAttention ? ATTENTION_MODEL_ID : CAUSAL_LM_MODEL_ID}
      estimatedSizeMB={usesAttention ? ATTENTION_SIZE_MB : CAUSAL_LM_SIZE_MB}
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
  state: ContextDegradationState;
  dispatch: (action: ContextDegradationAction) => void;
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
  const showFooter = state.mode !== 'budget-subset';

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'needle-haystack' && <LengthBoard state={state} dispatch={dispatch} />}
          {state.mode === 'attention-dilution' && <DilutionBoard state={state} dispatch={dispatch} />}
          {state.mode === 'budget-subset' && (
            <BudgetBoard state={state} dispatch={dispatch} result={result} onReset={resetRun} onSubmitFinal={submit} />
          )}
        </div>
      </div>

      {showFooter && (
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
      )}
    </div>
  );
}

function submitBlocked(state: ContextDegradationState): boolean {
  switch (state.mode) {
    case 'needle-haystack':
      return state.lengthRounds.some((r) => r.guess === null);
    case 'attention-dilution':
      return state.dilutionRounds.some((r) => r.guessIndex === null);
    case 'budget-subset':
      return true; // this board submits from its own "test" flow, not the shared footer
  }
}

function submitHint(state: ContextDegradationState): string {
  switch (state.mode) {
    case 'needle-haystack': {
      const left = state.lengthRounds.filter((r) => r.guess === null).length;
      return `${left} length${left === 1 ? '' : 's'} left`;
    }
    case 'attention-dilution': {
      const left = state.dilutionRounds.filter((r) => r.guessIndex === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
    case 'budget-subset':
      return '';
  }
}

/* ── level 1: needle in a haystack ─────────────────────────────── */

function LengthBoard({
  state,
  dispatch,
}: {
  state: ContextDegradationState;
  dispatch: (action: ContextDegradationAction) => void;
}) {
  return (
    <Panel label="will the real model still retrieve the fact at this length?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.lengthRounds.map((round, index) => (
          <LengthRow key={round.fillerWords} round={round} onGuess={(guess) => dispatch({ type: 'GUESS_LENGTH', roundIndex: index, guess })} />
        ))}
      </ol>
    </Panel>
  );
}

function LengthRow({ round, onGuess }: { round: LengthRound; onGuess: (guess: boolean) => void }) {
  const revealed = round.guess !== null;
  const correct = revealed && round.guess === round.passed;

  return (
    <li className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5" style={{ borderRadius: 'var(--radius)' }}>
      <p className="font-mono text-xs text-primary">
        ~{round.fillerWords.toLocaleString()} filler words before the question
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'still correct', guess: true },
            { label: 'fails here', guess: false },
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
          <Tag tone={correct ? 'good' : 'bad'}>
            {correct ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            {round.passed ? 'really correct' : 'really failed'}
          </Tag>
        )}
      </div>
      {revealed && <p className="readout text-[11px] text-muted">“{round.decodedText.trim()}”</p>}
    </li>
  );
}

/* ── level 2: attention dilution ───────────────────────────────── */

function DilutionBoard({
  state,
  dispatch,
}: {
  state: ContextDegradationState;
  dispatch: (action: ContextDegradationAction) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      {state.dilutionRounds.map((round, index) => (
        <DilutionRoundPanel
          key={round.sentence}
          index={index}
          round={round}
          onGuess={(tokenIndex) => dispatch({ type: 'GUESS_DILUTION', roundIndex: index, tokenIndex })}
        />
      ))}
    </div>
  );
}

function DilutionRoundPanel({
  index,
  round,
  onGuess,
}: {
  index: number;
  round: DilutionRound;
  onGuess: (tokenIndex: number) => void;
}) {
  const revealed = round.guessIndex !== null;
  const trueIndex = revealed ? realExtremeIndex(round) : null;
  const correct = revealed && round.guessIndex === trueIndex;
  const weightOf = (tokenIdx: number): number | null => {
    const slot = round.keyIndices.indexOf(tokenIdx);
    return slot === -1 ? null : (round.trueRow[slot] ?? null);
  };

  return (
    <Panel label={`round ${index + 1} — which token gets the ${round.target === 'min' ? 'least' : 'most'} real attention?`} flush>
      <div className="flex flex-col gap-2 p-2">
        <p className="font-mono text-xs text-secondary">{round.sentence}</p>
        <div className="flex flex-wrap gap-1.5">
          {round.tokens.map((token, tokenIdx) => {
            const selectable = round.keyIndices.includes(tokenIdx);
            if (!selectable) return null;
            const weight = weightOf(tokenIdx);
            const isTrue = revealed && tokenIdx === trueIndex;
            const isGuess = round.guessIndex === tokenIdx;
            return (
              <button
                key={tokenIdx}
                type="button"
                aria-pressed={isGuess}
                disabled={revealed}
                onClick={() => onGuess(tokenIdx)}
                className={cx(
                  'min-h-[32px] border px-2 py-1 font-mono text-xs transition-colors',
                  isGuess && !revealed && 'border-accent bg-accent-dim text-primary',
                  revealed && isTrue && 'border-good bg-good/10 text-good',
                  revealed && isGuess && !isTrue && 'border-bad bg-bad/10 text-bad',
                  !isGuess && !(revealed && isTrue) && 'border-line-strong bg-transparent text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {token}
                {revealed && weight !== null && <span className="ml-1 text-muted">{weight.toFixed(3)}</span>}
              </button>
            );
          })}
        </div>
        {revealed && (
          <Tag tone={correct ? 'good' : 'bad'}>
            {correct ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            real {round.target}: “{round.tokens[trueIndex!]}”
          </Tag>
        )}
      </div>
    </Panel>
  );
}

/* ── level 3: spend the budget ──────────────────────────────────── */

function BudgetBoard({
  state,
  dispatch,
  result,
  onReset,
  onSubmitFinal,
}: {
  state: ContextDegradationState;
  dispatch: (action: ContextDegradationAction) => void;
  result: ReturnType<typeof evaluate>;
  onReset: () => void;
  onSubmitFinal: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const test = async () => {
    if (!state.targetFact || !state.selectedDistractorId || busy) return;
    setBusy(true);
    setError(null);
    try {
      const distractor = state.distractorFacts.find((f) => f.id === state.selectedDistractorId)!;
      const targetText = state.targetFact.sentences.join(' ');
      const distractorText = distractor.sentences.join(' ');
      const contextText =
        state.order === 'target-first' ? `${targetText} ${distractorText}` : `${distractorText} ${targetText}`;
      const prompt = buildGroundedPrompt(contextText, state.targetFact.query);
      const decodedText = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens ?? 16);
      dispatch({ type: 'TEST_SUBSET', decodedText });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  if (!state.targetFact) return null;

  return (
    <div className="flex flex-col gap-4">
      <Panel label="target (always included)">
        <p className="text-sm text-primary">
          {state.targetFact.topic} — <span className="text-accent">{state.targetFact.query}</span>
        </p>
      </Panel>

      <Panel label="pick exactly one more, and where it goes" flush>
        <div className="flex flex-col gap-2 p-2">
          <div className="flex flex-wrap gap-2">
            {state.distractorFacts.map((fact) => (
              <button
                key={fact.id}
                type="button"
                aria-pressed={state.selectedDistractorId === fact.id}
                onClick={() => dispatch({ type: 'SELECT_DISTRACTOR', factId: fact.id })}
                className={cx(
                  'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                  state.selectedDistractorId === fact.id
                    ? 'border-accent bg-accent-dim text-primary'
                    : 'border-line-strong bg-transparent text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {fact.topic}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {(
              [
                { label: 'target first, then the extra fact', order: 'target-first' as const },
                { label: 'extra fact first, then the target', order: 'distractor-first' as const },
              ]
            ).map((option) => (
              <button
                key={option.order}
                type="button"
                aria-pressed={state.order === option.order}
                onClick={() => dispatch({ type: 'SET_ORDER', order: option.order })}
                className={cx(
                  'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs transition-colors',
                  state.order === option.order
                    ? 'border-accent bg-accent-dim text-primary'
                    : 'border-line-strong bg-transparent text-secondary hover:border-accent'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {option.label}
              </button>
            ))}
          </div>
          <Button onClick={() => void test()} disabled={busy || !state.selectedDistractorId}>
            <Search size={13} strokeWidth={2} />
            {busy ? 'running the real model…' : 'test this combination'}
          </Button>
          {error && <p className="readout text-xs text-bad">could not decode — {error}</p>}
        </div>
      </Panel>

      {state.attempts > 0 && (
        <Panel label="last real attempt">
          <div className="flex flex-col gap-3">
            <p className="readout text-xs text-secondary">“{state.lastDecodedText?.trim()}”</p>
            <Tag tone={state.solved ? 'good' : 'bad'}>
              {state.solved ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
              {state.solved ? 'answered correctly' : 'did not contain the real answer'}
            </Tag>
          </div>
        </Panel>
      )}

      <Panel label="score">
        <div className="flex items-end justify-between gap-4">
          <Readout label="attemptsToSolve" value={result.value} size="lg" tone={result.passed ? 'good' : 'accent'} />
          <Readout label="attempts" value={state.attempts} size="sm" />
        </div>
      </Panel>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onReset}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Button variant="primary" className="ml-auto" disabled={!state.solved} onClick={onSubmitFinal}>
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}
