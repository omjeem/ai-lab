'use client';

/**
 * Chapter 5.5 — Full Transformer Assembly.
 *
 * One shared `StageRow` (grip + arrow-key drag, same pattern as every other
 * reorder canvas in this course) powers both the order-stages and fill-gaps
 * boards, since fill-gaps is really "insert into the tray, then reorder" —
 * inserting appends to the end and dragging is what carries a stage into its
 * real slot, so there is no second reorder implementation to keep in sync.
 *
 * The third level runs a real GPT-2 forward pass per round: the shuffled
 * choices, the true top token and every probability shown after a guess come
 * straight from `gpt2CausalLM`, not a canned distribution. Only the twelve
 * stage names and their order are ever "shown" without a live number behind
 * them — GPT-2's own ONNX export has no `attentions`/`hidden_states` output
 * (checked directly, same way A1 ruled it out for other models), so there is
 * no real per-stage tensor to visualise between tokenize and the final
 * distribution. The assembled pipeline is drawn as a static reference instead
 * of inventing numbers for stages nothing exposes.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Check,
  GripVertical,
  Plus,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  buildRound,
  type PredictionRound,
  type TransformerAssemblyAction,
  type TransformerAssemblyConfig,
  type TransformerAssemblyState,
} from '@/engines/transformerAssemblyEngine';
import { gpt2CausalLM, tokenizePrompt, GPT2_MODEL_ID } from '@/models/tinyCausalLM';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import { useDragReorder } from '../useDragReorder';
import { usePointerDrag } from '../usePointerDrag';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

// Residual-add and layer-norm each occur twice in the block. Two chips with
// the *same* label would be visually indistinguishable while dragging — a
// player could put a correct-looking sequence together and still fail the
// engine's exact per-slot check because the two identical-looking chips
// landed swapped. The A/B suffix is neutral (it doesn't say which position
// is which, so it gives nothing away) but makes the two chips trackable.
const STAGE_LABELS: Record<string, string> = {
  tokenize: 'Tokenize',
  embed: 'Embed',
  'positional-encoding': 'Positional Encoding',
  'qkv-projection': 'Q/K/V Projection',
  attention: 'Self-Attention',
  'residual-add-1': 'Residual Add A',
  'layer-norm-1': 'Layer Norm A',
  'feed-forward': 'Feed-Forward',
  'residual-add-2': 'Residual Add B',
  'layer-norm-2': 'Layer Norm B',
  unembed: 'Unembed',
  softmax: 'Softmax',
};

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage;
}

export function FullTransformerCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as TransformerAssemblyConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<TransformerAssemblyState | null>(null);
  const needsModel = config.mode === 'run-end-to-end';

  const load = useCallback(async () => {
    const prepared = await prepare(config, needsModel ? { causalLM: gpt2CausalLM } : {});
    setState(initState(config, rules, prepared));
  }, [config, rules, needsModel]);

  // Only the run-end-to-end level needs GPT-2. `ModelGate`'s own load effect
  // is gated on a truthy `modelId`, so a null one (order-stages / fill-gaps)
  // never calls `load` at all — this chapter is where that first mattered,
  // same root cause as 5-1-positional-encoding's mixed model/no-model levels.
  useEffect(() => {
    if (!needsModel) void load();
  }, [needsModel, load]);

  const dispatch = useCallback((action: TransformerAssemblyAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={needsModel ? GPT2_MODEL_ID : null}
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
  state: TransformerAssemblyState;
  dispatch: (action: TransformerAssemblyAction) => void;
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

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    setRevealed(true);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div key={runId} className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'order-stages' && (
            <OrderStagesBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'fill-gaps' && (
            <FillGapsBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'run-end-to-end' && <RunEndToEndBoard state={state} dispatch={dispatch} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Actions state={state} revealed={revealed} onSubmit={submit} />
      </div>
    </div>
  );
}

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: TransformerAssemblyState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = submitBlocked(state);
  return (
    <>
      {!revealed && blocked && <span className="label">run and answer every round first</span>}
      {revealed && <Tag tone="accent">answers revealed</Tag>}

      <Button
        variant="primary"
        className="ml-auto"
        disabled={blocked || revealed}
        title={blocked ? 'run and answer every round first' : undefined}
        onClick={onSubmit}
      >
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

function submitBlocked(state: TransformerAssemblyState): boolean {
  if (state.mode !== 'run-end-to-end') return false;
  const total = state.config.rounds ?? state.rounds.length;
  return state.rounds.length < total || state.rounds.some((r) => r.answer === null);
}

/* ── shared: one reorderable stage row ─────────────────────────
   Used by both order-stages (every stage always present) and
   fill-gaps (only the placed subset) — the caller supplies whichever
   "what's correct here" values apply. */

function StageRow({
  stage,
  index,
  revealed,
  correct,
  trueIndex,
  totalItems,
  critical,
  dragging,
  dropTarget,
  registerRef,
  onGripDragStart,
  onGripDragMove,
  onGripDragEnd,
  onMoveUp,
  onMoveDown,
  onRemove,
}: {
  stage: string;
  index: number;
  revealed: boolean;
  correct?: boolean;
  trueIndex?: number;
  totalItems: number;
  critical?: boolean;
  dragging: boolean;
  dropTarget: boolean;
  registerRef: (element: HTMLElement | null) => void;
  onGripDragStart: () => void;
  onGripDragMove: (clientY: number) => void;
  onGripDragEnd: (clientY: number) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onRemove?: () => void;
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
        revealed && correct === true
          ? 'border-good/40 bg-good/5'
          : revealed && correct === false
            ? 'border-bad/40 bg-bad/5'
            : 'border-line-strong bg-raised'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      <span className="readout w-5 shrink-0 text-xs text-muted">{index + 1}</span>

      {!revealed && (
        <span {...grip} style={{ ...grip.style, cursor: 'grab' }} className="text-muted active:cursor-grabbing">
          <GripVertical size={13} strokeWidth={1.75} aria-hidden />
        </span>
      )}

      <span className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-mono text-sm text-primary">{stageLabel(stage)}</span>
        {revealed && critical && <Tag tone="warn">critical</Tag>}
      </span>

      {revealed ? (
        <span className="label w-24 shrink-0 text-right">
          {correct ? 'correct spot' : `really #${(trueIndex ?? 0) + 1}`}
        </span>
      ) : (
        <span className="flex shrink-0 items-center gap-1">
          <MoveButton label={`Move ${stageLabel(stage)} up`} disabled={index === 0} onClick={onMoveUp}>
            <ArrowUp size={12} strokeWidth={2} />
          </MoveButton>
          <MoveButton
            label={`Move ${stageLabel(stage)} down`}
            disabled={index === totalItems - 1}
            onClick={onMoveDown}
          >
            <ArrowDown size={12} strokeWidth={2} />
          </MoveButton>
          {onRemove && (
            <MoveButton label={`Remove ${stageLabel(stage)} from the block`} disabled={false} onClick={onRemove}>
              <X size={12} strokeWidth={2} />
            </MoveButton>
          )}
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

/* ── level 1: order the pipeline ───────────────────────────────── */

function OrderStagesBoard({
  state,
  dispatch,
  revealed,
}: {
  state: TransformerAssemblyState;
  dispatch: (action: TransformerAssemblyAction) => void;
  revealed: boolean;
}) {
  const reorder = useDragReorder((from, to) => dispatch({ type: 'MOVE_STAGE', from, to }));

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs leading-relaxed text-muted">
        Drag the grip, or use the arrow buttons, to put all twelve stages in the order data actually
        flows through the block.
      </p>

      <Panel label={revealed ? 'your order against the real one' : 'first stage at the top'}>
        <ol className="flex flex-col gap-1.5">
          {state.arrangement.map((stage, index) => (
            <StageRow
              key={stage}
              stage={stage}
              index={index}
              revealed={revealed}
              correct={revealed ? state.correctOrder[index] === stage : undefined}
              trueIndex={revealed ? state.correctOrder.indexOf(stage) : undefined}
              totalItems={state.arrangement.length}
              dragging={reorder.dragIndex === index}
              dropTarget={
                reorder.overIndex === index && reorder.dragIndex !== null && reorder.dragIndex !== index
              }
              registerRef={reorder.registerItem(index)}
              onGripDragStart={() => reorder.startDrag(index)}
              onGripDragMove={reorder.dragTo}
              onGripDragEnd={reorder.dropAt}
              onMoveUp={() => dispatch({ type: 'MOVE_STAGE', from: index, to: index - 1 })}
              onMoveDown={() => dispatch({ type: 'MOVE_STAGE', from: index, to: index + 1 })}
            />
          ))}
        </ol>
      </Panel>
    </div>
  );
}

/* ── level 2: fill the gaps ─────────────────────────────────────── */

function FillGapsBoard({
  state,
  dispatch,
  revealed,
}: {
  state: TransformerAssemblyState;
  dispatch: (action: TransformerAssemblyAction) => void;
  revealed: boolean;
}) {
  const critical = useMemo(() => new Set(state.config.criticalStages ?? []), [state.config.criticalStages]);
  const reorder = useDragReorder((from, to) => dispatch({ type: 'MOVE_STAGE', from, to }));

  // What order the *placed* stages should be in — lets each row read "am I in
  // the right spot relative to the others placed so far", the same thing the
  // engine's pairwise ordering credit measures, without claiming an absolute
  // slot number a still-missing stage would shift.
  const expectedSubOrder = useMemo(
    () => state.correctOrder.filter((s) => state.arrangement.includes(s)),
    [state.correctOrder, state.arrangement]
  );

  const stillMissing = state.correctOrder.filter((s) => !state.arrangement.includes(s));

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs leading-relaxed text-muted">
        Click a tray stage to drop it back into the block, then drag it (or use the arrow buttons) into
        its real position.
      </p>

      {state.tray.length > 0 && (
        <Panel label="tray — pulled out of the block">
          <div className="flex flex-wrap gap-2">
            {state.tray.map((stage) => (
              <button
                key={stage}
                type="button"
                onClick={() =>
                  dispatch({ type: 'INSERT_STAGE', stage, position: state.arrangement.length })
                }
                className="inline-flex items-center gap-1 border border-dashed border-line-strong px-2.5 py-1.5 font-mono text-xs text-secondary transition-colors hover:border-accent hover:text-accent"
                style={{ borderRadius: 'var(--radius)' }}
              >
                <Plus size={11} strokeWidth={2} />
                {stageLabel(stage)}
              </button>
            ))}
          </div>
        </Panel>
      )}

      <Panel label={revealed ? 'your pipeline against the real one' : 'the block, as it stands'}>
        <ol className="flex flex-col gap-1.5">
          {state.arrangement.map((stage, index) => (
            <StageRow
              key={stage}
              stage={stage}
              index={index}
              revealed={revealed}
              correct={revealed ? expectedSubOrder[index] === stage : undefined}
              trueIndex={revealed ? state.correctOrder.indexOf(stage) : undefined}
              critical={critical.has(stage)}
              totalItems={state.arrangement.length}
              dragging={reorder.dragIndex === index}
              dropTarget={
                reorder.overIndex === index && reorder.dragIndex !== null && reorder.dragIndex !== index
              }
              registerRef={reorder.registerItem(index)}
              onGripDragStart={() => reorder.startDrag(index)}
              onGripDragMove={reorder.dragTo}
              onGripDragEnd={reorder.dropAt}
              onMoveUp={() => dispatch({ type: 'MOVE_STAGE', from: index, to: index - 1 })}
              onMoveDown={() => dispatch({ type: 'MOVE_STAGE', from: index, to: index + 1 })}
              onRemove={!revealed ? () => dispatch({ type: 'REMOVE_STAGE', position: index }) : undefined}
            />
          ))}
          {state.arrangement.length === 0 && (
            <li className="label px-2 py-4 text-center">nothing placed yet</li>
          )}
        </ol>
      </Panel>

      {revealed && stillMissing.length > 0 && (
        <Panel label="never made it back in">
          <div className="flex flex-wrap gap-2">
            {stillMissing.map((stage) => (
              <Tag key={stage} tone={critical.has(stage) ? 'bad' : 'warn'}>
                {stageLabel(stage)}
                {critical.has(stage) ? ' · critical' : ''}
              </Tag>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ── level 3: run it end to end ────────────────────────────────── */

function RunEndToEndBoard({
  state,
  dispatch,
}: {
  state: TransformerAssemblyState;
  dispatch: (action: TransformerAssemblyAction) => void;
}) {
  const [promptDraft, setPromptDraft] = useState('');
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const maxRounds = state.config.rounds ?? state.rounds.length;
  const canAddRound = Boolean(state.config.allowUserPrompt) && state.rounds.length < maxRounds;

  const runPrompt = async (prompt: string) => {
    setRunning(true);
    setRunError(null);
    try {
      const distribution = await gpt2CausalLM.nextTokenDistribution(prompt, state.config.topK ?? 5);
      const round = buildRound(prompt, distribution, state.rounds.length + 11);
      dispatch({ type: 'ADD_ROUND', round });
      setPromptDraft('');
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <PipelineStrip stages={state.correctOrder} />

      <p className="text-xs leading-relaxed text-muted">
        Every round below is a real GPT-2 forward pass. Pick which candidate you think it actually
        ranked highest, then the true distribution reveals underneath.
      </p>

      <div className="flex flex-col gap-3">
        {state.rounds.map((round, i) => (
          <RoundCard
            key={i}
            round={round}
            index={i}
            onAnswer={(token) => dispatch({ type: 'ANSWER_TOP_TOKEN', roundIndex: i, token })}
          />
        ))}
      </div>

      {canAddRound && (
        <Panel label={`add a round — ${state.rounds.length}/${maxRounds} run`}>
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              value={promptDraft}
              onChange={(event) => setPromptDraft(event.target.value)}
              placeholder="Type your own prompt…"
              aria-label="Your own prompt for the next round"
              className="min-w-0 flex-1 border border-line-strong bg-raised px-3 py-2 font-mono text-sm text-primary outline-none focus:border-accent"
              style={{ borderRadius: 'var(--radius)' }}
            />
            <Button
              variant="primary"
              disabled={running || promptDraft.trim().length === 0}
              onClick={() => void runPrompt(promptDraft.trim())}
            >
              <Sparkles size={13} strokeWidth={2} />
              {running ? 'Running…' : 'Run forward pass'}
            </Button>
          </div>
          {runError && <p className="mt-2 text-xs text-bad">{runError}</p>}
        </Panel>
      )}
    </div>
  );
}

function PipelineStrip({ stages }: { stages: string[] }) {
  return (
    <Panel label="the assembled block" flush>
      <div className="flex flex-wrap items-center gap-1.5 p-3">
        {stages.map((stage, i) => (
          <span key={stage} className="flex items-center gap-1.5">
            <span
              className="whitespace-nowrap border border-line-strong bg-raised px-2 py-1 font-mono text-[11px] text-primary"
              style={{ borderRadius: 'var(--radius)' }}
            >
              {stageLabel(stage)}
            </span>
            {i < stages.length - 1 && (
              <ArrowRight size={11} strokeWidth={2} className="shrink-0 text-muted" aria-hidden />
            )}
          </span>
        ))}
      </div>
    </Panel>
  );
}

function RoundCard({
  round,
  index,
  onAnswer,
}: {
  round: PredictionRound;
  index: number;
  onAnswer: (token: string) => void;
}) {
  const reduce = useReducedMotion();
  const [tokens, setTokens] = useState<string[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    void tokenizePrompt(round.prompt).then((result) => {
      if (!cancelled) setTokens(result.tokens);
    });
    return () => {
      cancelled = true;
    };
  }, [round.prompt]);

  const answered = round.answer !== null;
  const trueTop = round.distribution.tokens[0];
  const correct = answered && round.answer === trueTop;

  return (
    <Panel label={`round ${index + 1}`}>
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="label">tokenize</span>
          <div className="flex flex-wrap gap-1">
            {(tokens ?? ['…']).map((token, i) => (
              <span
                key={i}
                className="border border-line-strong bg-raised px-1.5 py-0.5 font-mono text-[11px] text-secondary"
                style={{ borderRadius: '3px' }}
              >
                {token}
              </span>
            ))}
          </div>
        </div>

        {!answered ? (
          <div className="flex flex-wrap gap-2">
            {round.choices.map((token) => (
              <button
                key={token}
                type="button"
                onClick={() => onAnswer(token)}
                className="border border-line-strong bg-raised px-3 py-2 font-mono text-sm text-primary transition-colors hover:border-accent hover:text-accent"
                style={{ borderRadius: 'var(--radius)' }}
              >
                {token}
              </button>
            ))}
          </div>
        ) : (
          <motion.div
            initial={reduce ? undefined : { opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: reduce ? 0 : 0.18 }}
            className="flex flex-col gap-2.5"
          >
            <Tag tone={correct ? 'good' : 'bad'}>
              {correct ? `correct — “${round.answer}” was the real top token` : `you picked “${round.answer}”`}
            </Tag>
            <ol className="flex flex-col gap-1.5">
              {round.distribution.tokens.map((token, i) => (
                <li
                  key={token}
                  className={cx('flex items-center gap-2.5', token !== trueTop && 'opacity-60')}
                >
                  <span
                    className={cx(
                      'w-20 shrink-0 truncate font-mono text-xs',
                      token === trueTop ? 'text-accent' : 'text-secondary'
                    )}
                  >
                    {token}
                  </span>
                  <span className="flex-1">
                    <Meter value={round.distribution.probs[i] ?? 0} tone="accent" />
                  </span>
                  <span className="readout w-14 shrink-0 text-right text-xs text-secondary">
                    {((round.distribution.probs[i] ?? 0) * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ol>
          </motion.div>
        )}
      </div>
    </Panel>
  );
}
