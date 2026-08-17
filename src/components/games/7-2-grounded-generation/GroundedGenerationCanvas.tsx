'use client';

/**
 * Chapter 7.2 — Grounded Generation.
 *
 * Every completion on screen is a real greedy decode from the tiny causal LM,
 * read one true token at a time via `greedyDecode` — there is no simulated
 * text anywhere here. Level 1's two completions per round are computed once,
 * during `prepare()`, so the ModelGate's loading screen covers that cost.
 * Levels 2 and 3 decode on demand instead: the player commits to a passage or
 * a curated context, then the canvas runs one real decode and hands the
 * result back into the engine, the same "canvas supplies the real value, the
 * engine grades it" pattern every model-backed dependency in this app follows.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  buildGroundedPrompt,
  buildCuratedContextText,
  containsAnswer,
  type GroundedFact,
  type GroundedGenerationAction,
  type GroundedGenerationConfig,
  type GroundedGenerationState,
} from '@/engines/groundedGenerationEngine';
import { tinyCausalLM, preloadCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function GroundedGenerationCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as GroundedGenerationConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<GroundedGenerationState | null>(null);

  const load = useCallback(async () => {
    // pick-passage and curate-context decode on demand, so `prepare()` alone
    // would never trigger the download — warm the model explicitly, same fix
    // ProbabilityCanvas's entropy level needed for the same reason.
    await preloadCausalLM();
    const prepared = await prepare(config, { corpus: corpusLoader, causalLM: tinyCausalLM });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: GroundedGenerationAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={CAUSAL_LM_MODEL_ID}
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
  state: GroundedGenerationState;
  dispatch: (action: GroundedGenerationAction) => void;
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
          {state.mode === 'identify-grounded' && (
            <IdentifyBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'pick-passage' && <PickPassageBoard state={state} dispatch={dispatch} />}
          {state.mode === 'curate-context' && <CurateContextBoard state={state} dispatch={dispatch} result={result} />}
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

function submitBlocked(state: GroundedGenerationState): boolean {
  switch (state.mode) {
    case 'identify-grounded':
      return state.identifyRounds.some((r) => r.pick === null);
    case 'pick-passage':
      return state.pickRounds.some((r) => r.decodedText === null);
    case 'curate-context':
      return !state.solved;
  }
}

function submitHint(state: GroundedGenerationState): string {
  switch (state.mode) {
    case 'identify-grounded': {
      const left = state.identifyRounds.filter((r) => r.pick === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
    case 'pick-passage': {
      const left = state.pickRounds.filter((r) => r.decodedText === null).length;
      return `${left} round${left === 1 ? '' : 's'} left to test`;
    }
    case 'curate-context':
      return 'find a context that works';
  }
}

function factLookup(facts: readonly GroundedFact[], id: string): GroundedFact | undefined {
  return facts.find((f) => f.id === id);
}

/* ── level 1: spot the grounding ───────────────────────────────── */

function IdentifyBoard({
  state,
  dispatch,
  revealed,
}: {
  state: GroundedGenerationState;
  dispatch: (action: GroundedGenerationAction) => void;
  revealed: boolean;
}) {
  return (
    <div className="flex flex-col gap-3">
      {state.identifyRounds.map((round, index) => {
        const correct = round.pick === round.groundedSide;
        return (
          <Panel
            key={round.factId}
            label={round.query}
            actions={revealed ? <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : 'wrong'}</Tag> : undefined}
          >
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {(['left', 'right'] as const).map((side) => {
                const text = side === 'left' ? round.leftText : round.rightText;
                const isGrounded = revealed && round.groundedSide === side;
                const isChosen = round.pick === side;
                return (
                  <button
                    key={side}
                    type="button"
                    disabled={revealed}
                    aria-pressed={isChosen}
                    onClick={() => dispatch({ type: 'PICK_SIDE', roundIndex: index, side })}
                    className={cx(
                      'flex min-h-[64px] flex-col gap-1 border px-3 py-2 text-left font-mono text-xs transition-colors disabled:cursor-default',
                      revealed
                        ? isGrounded
                          ? 'border-good bg-good/10 text-good'
                          : isChosen
                            ? 'border-bad bg-bad/10 text-bad'
                            : 'border-line bg-raised text-secondary'
                        : isChosen
                          ? 'border-accent bg-accent-dim text-primary'
                          : 'border-line-strong bg-raised text-secondary hover:border-accent'
                    )}
                  >
                    <span className="label">{side}</span>
                    <span className="whitespace-pre-wrap text-primary">{text || '(empty)'}</span>
                    {revealed && isGrounded && <Tag tone="good">had the passage</Tag>}
                  </button>
                );
              })}
            </div>
          </Panel>
        );
      })}
    </div>
  );
}

/* ── level 2: pick the passage ─────────────────────────────────── */

function PickPassageBoard({
  state,
  dispatch,
}: {
  state: GroundedGenerationState;
  dispatch: (action: GroundedGenerationAction) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      {state.pickRounds.map((round, index) => {
        const target = factLookup(state.facts, round.targetFactId);
        return (
          <Panel key={round.targetFactId} label={round.query}>
            <div className="flex flex-col gap-2">
              {round.candidateFactIds.map((factId) => {
                const candidate = factLookup(state.facts, factId);
                if (!candidate) return null;
                const chosen = round.chosenFactId === factId;
                return (
                  <button
                    key={factId}
                    type="button"
                    aria-pressed={chosen}
                    onClick={() => dispatch({ type: 'CHOOSE_PASSAGE', roundIndex: index, factId })}
                    className={cx(
                      'flex flex-col gap-1 border px-3 py-2 text-left transition-colors',
                      chosen ? 'border-accent bg-accent-dim' : 'border-line-strong bg-raised hover:border-accent'
                    )}
                    style={{ borderRadius: 'var(--radius)' }}
                  >
                    <span className="font-mono text-xs text-primary">{candidate.topic}</span>
                    <span className="text-[11px] leading-relaxed text-secondary">
                      {candidate.sentences.join(' ')}
                    </span>
                  </button>
                );
              })}

              <PickTestRow state={state} round={round} roundIndex={index} target={target} dispatch={dispatch} />
            </div>
          </Panel>
        );
      })}
    </div>
  );
}

function PickTestRow({
  state,
  round,
  roundIndex,
  target,
  dispatch,
}: {
  state: GroundedGenerationState;
  round: GroundedGenerationState['pickRounds'][number];
  roundIndex: number;
  target: GroundedFact | undefined;
  dispatch: (action: GroundedGenerationAction) => void;
}) {
  const [busy, setBusy] = useState(false);

  const test = async () => {
    if (!round.chosenFactId || !target || busy) return;
    const candidate = factLookup(state.facts, round.chosenFactId);
    if (!candidate) return;
    setBusy(true);
    try {
      const prompt = buildGroundedPrompt(candidate.sentences.join(' '), target.query);
      const decodedText = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
      dispatch({ type: 'TEST_PASSAGE', roundIndex, decodedText });
    } finally {
      setBusy(false);
    }
  };

  const correct = round.decodedText !== null && !!target && containsAnswer(round.decodedText, target.answer);

  return (
    <div className="flex flex-col gap-2 border-t border-line-faint pt-2">
      <div className="flex items-center gap-2">
        <Button onClick={() => void test()} disabled={!round.chosenFactId || busy}>
          {busy ? 'running the real model…' : 'test this passage'}
        </Button>
        {round.decodedText !== null && <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'reached the fact' : 'missed it'}</Tag>}
      </div>
      {round.decodedText !== null && (
        <p className="readout whitespace-pre-wrap border border-line bg-inset p-2 text-xs text-secondary">
          {round.decodedText || '(no output)'}
        </p>
      )}
    </div>
  );
}

/* ── level 3: curate the context ───────────────────────────────── */

function CurateContextBoard({
  state,
  dispatch,
  result,
}: {
  state: GroundedGenerationState;
  dispatch: (action: GroundedGenerationAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [busy, setBusy] = useState(false);
  const target = factLookup(state.facts, state.targetFactId);
  const distractor = factLookup(state.facts, state.distractorFactId);

  const test = async () => {
    if (!target || !distractor || busy) return;
    setBusy(true);
    try {
      const context = buildCuratedContextText(target, distractor, state.order, state.truncateDistractor);
      const prompt = buildGroundedPrompt(context, target.query);
      const decodedText = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
      dispatch({ type: 'TEST_CONTEXT', decodedText });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel label="question">
        <p className="text-sm text-primary">{target?.query}</p>
        <p className="mt-1 text-xs text-muted">
          Must include a passage about <span className="text-secondary">{distractor?.topic}</span> alongside the real{' '}
          <span className="text-secondary">{target?.topic}</span> passage.
        </p>
      </Panel>

      <Panel label="order">
        <div className="flex gap-2">
          {(
            [
              { value: 'correct-first', label: `${target?.topic ?? 'target'} first` },
              { value: 'distractor-first', label: `${distractor?.topic ?? 'distractor'} first` },
            ] as const
          ).map((opt) => (
            <button
              key={opt.value}
              type="button"
              aria-pressed={state.order === opt.value}
              onClick={() => dispatch({ type: 'SET_ORDER', order: opt.value })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                state.order === opt.value
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </Panel>

      <Panel label="distractor length">
        <div className="flex gap-2">
          {[
            { value: false, label: 'full passage' },
            { value: true, label: 'first sentence only' },
          ].map((opt) => (
            <button
              key={String(opt.value)}
              type="button"
              aria-pressed={state.truncateDistractor === opt.value}
              onClick={() => dispatch({ type: 'SET_TRUNCATE', truncate: opt.value })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                state.truncateDistractor === opt.value
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex items-center gap-2">
        <Button variant="primary" onClick={() => void test()} disabled={busy}>
          {busy ? 'running the real model…' : 'test this context'}
        </Button>
        <Readout label="attempts" value={state.attempts} size="sm" />
        {state.solved && <Tag tone="good">solved on attempt {state.solvedAtAttempt}</Tag>}
      </div>

      {state.lastDecodedText !== null && (
        <Panel label="real model output">
          <p className="readout whitespace-pre-wrap text-xs text-secondary">{state.lastDecodedText || '(no output)'}</p>
        </Panel>
      )}

      {state.solved && (
        <Panel label="score">
          <Readout label="attemptsToSolve" value={result.value} size="lg" tone="good" />
        </Panel>
      )}
    </div>
  );
}
