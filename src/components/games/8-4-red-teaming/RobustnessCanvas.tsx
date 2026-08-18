'use client';

/**
 * Chapter 8.4 — Red-Teaming.
 *
 * All three boards precompute every real trial once in `prepare()` — no live
 * decode is triggered by a player action, the same fully-precomputed style
 * 8.3 uses. `find-attack`/`test-defense` reveal a candidate's real result the
 * moment it's tested (mirrors 8.2's curate-context "keep trying" loop, but
 * against already-real data rather than a fresh live decode per attempt).
 * `defense-transfer` hides the real result behind a guess instead, since
 * there's nothing left to "try" — only a prediction to make.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw, X } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type AttackRound,
  type DefenseRound,
  type RobustnessAction,
  type RobustnessConfig,
  type RobustnessState,
  type TransferRound,
} from '@/engines/robustnessEngine';
import { tinyCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const MODEL_SIZE_MB = 92;

export function RobustnessCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as RobustnessConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<RobustnessState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { corpus: corpusLoader, causalLM: tinyCausalLM });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: RobustnessAction) => {
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
  state: RobustnessState;
  dispatch: (action: RobustnessAction) => void;
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

  const isTransfer = state.mode === 'defense-transfer';
  const blocked = isTransfer ? state.transferRounds.some((r) => r.guess === null) : !state.solved;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'find-attack' && <AttackBoard state={state} dispatch={dispatch} />}
          {state.mode === 'test-defense' && <DefenseBoard state={state} dispatch={dispatch} />}
          {isTransfer && <TransferBoard state={state} dispatch={dispatch} />}

          {!isTransfer && (
            <Panel label="score">
              <div className="flex items-end justify-between gap-4">
                <Readout
                  label="attemptsToSolve"
                  value={result.value}
                  size="lg"
                  tone={result.passed ? 'good' : 'accent'}
                />
                <Readout label="attempts" value={state.attempts} size="sm" />
              </div>
            </Panel>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {!revealed && blocked && (
          <span className="label">{isTransfer ? 'predict every round first' : 'keep testing — none has worked yet'}</span>
        )}
        {revealed && <Tag tone="accent">answers revealed</Tag>}
        <Button
          variant="primary"
          className="ml-auto"
          disabled={blocked || revealed}
          title={blocked ? (isTransfer ? 'predict every round first' : 'find a real success first') : undefined}
          onClick={submit}
        >
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

/* ── level 1: find a working attack ──────────────────────────── */

function AttackBoard({ state, dispatch }: { state: RobustnessState; dispatch: (action: RobustnessAction) => void }) {
  return (
    <Panel label="the model's rule: “Answer in exactly one word. Do not explain.” — test each real follow-up" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.attackRounds.map((round) => (
          <AttackRow key={round.id} round={round} onTest={() => dispatch({ type: 'TEST_ATTACK', id: round.id })} />
        ))}
      </ol>
    </Panel>
  );
}

function AttackRow({ round, onTest }: { round: AttackRound; onTest: () => void }) {
  return (
    <li
      className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-sm text-primary">{round.label}</p>
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onTest} disabled={round.tested}>
          {round.tested ? 'tested' : 'test this attack'}
        </Button>
        {round.tested && (
          <Tag tone={round.violates ? 'good' : 'bad'}>
            {round.violates ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            {round.violates ? 'really broke the rule' : 'rule really held'}
          </Tag>
        )}
      </div>
      {round.tested && <p className="readout text-[11px] text-muted">“{round.decodedText.trim()}”</p>}
    </li>
  );
}

/* ── level 2: patch the break ────────────────────────────────── */

function DefenseBoard({ state, dispatch }: { state: RobustnessState; dispatch: (action: RobustnessAction) => void }) {
  return (
    <Panel label="test each real strengthened instruction against the same real roleplay attack" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.defenseRounds.map((round) => (
          <DefenseRow key={round.id} round={round} onTest={() => dispatch({ type: 'TEST_DEFENSE', id: round.id })} />
        ))}
      </ol>
    </Panel>
  );
}

function DefenseRow({ round, onTest }: { round: DefenseRound; onTest: () => void }) {
  return (
    <li
      className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-sm text-primary">{round.label}</p>
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onTest} disabled={round.tested}>
          {round.tested ? 'tested' : 'test this defense'}
        </Button>
        {round.tested && (
          <Tag tone={round.resists ? 'good' : 'bad'}>
            {round.resists ? <Check size={10} strokeWidth={2} /> : <X size={10} strokeWidth={2} />}
            {round.resists ? 'really resisted' : 'really broke anyway'}
          </Tag>
        )}
      </div>
      {round.tested && <p className="readout text-[11px] text-muted">“{round.decodedText.trim()}”</p>}
    </li>
  );
}

/* ── level 3: does the fix generalize? ───────────────────────── */

function TransferBoard({ state, dispatch }: { state: RobustnessState; dispatch: (action: RobustnessAction) => void }) {
  return (
    <Panel label="will the same winning defense hold against each of these other real attacks?" flush>
      <ol className="flex flex-col gap-2 p-2">
        {state.transferRounds.map((round, index) => (
          <TransferRow
            key={round.id}
            round={round}
            onGuess={(guess) => dispatch({ type: 'GUESS_TRANSFER', roundIndex: index, guess })}
          />
        ))}
      </ol>
    </Panel>
  );
}

function TransferRow({ round, onGuess }: { round: TransferRound; onGuess: (guess: boolean) => void }) {
  const revealed = round.guess !== null;
  const correctGuess = revealed && round.guess === round.resists;

  return (
    <li
      className="flex flex-col gap-2 border border-line-strong bg-raised px-3 py-2.5"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <p className="text-sm text-primary">{round.label}</p>
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { label: 'defense will hold', guess: true },
            { label: 'defense will fail', guess: false },
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
            really {round.resists ? 'held' : 'failed'}
          </Tag>
        )}
      </div>
      {revealed && <p className="readout text-[11px] text-muted">“{round.decodedText.trim()}”</p>}
    </li>
  );
}
