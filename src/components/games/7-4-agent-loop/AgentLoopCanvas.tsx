'use client';

/**
 * Chapter 7.4 — Agent Loop.
 *
 * Every tool call, real tool execution and final answer here is real. The
 * canvas orchestrates the full loop per round: decode the tool call (real),
 * parse it, run the real tool (`toolRuntime`), decode the final answer
 * (real) with the real tool result injected, and hand every real value back
 * into the engine — the same "canvas supplies the real value, the engine
 * grades it" pattern every model-backed chapter in this course follows.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  sampledDecode,
  buildSingleHopPrompt,
  buildToolSelectHopPrompt,
  buildFinalAnswerPrompt,
  parseAgentCall,
  containsAnswer,
  type AgentLoopAction,
  type AgentLoopConfig,
  type AgentLoopState,
  type BudgetTaskResult,
} from '@/engines/agentLoopEngine';
import { tinyCausalLM, preloadCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { toolRuntime } from '@/models/toolRuntime';
import { corpusLoader } from '@/models/corpusLoader';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function AgentLoopCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as AgentLoopConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<AgentLoopState | null>(null);

  const load = useCallback(async () => {
    await preloadCausalLM();
    const prepared = await prepare(config, { corpus: corpusLoader });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: AgentLoopAction) => {
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
  state: AgentLoopState;
  dispatch: (action: AgentLoopAction) => void;
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
          {state.mode === 'single-hop' && <SingleHopBoard state={state} dispatch={dispatch} />}
          {state.mode === 'tool-select-hop' && <ToolSelectHopBoard state={state} dispatch={dispatch} />}
          {state.mode === 'budget-tuning' && <BudgetTuningBoard state={state} dispatch={dispatch} result={result} />}
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

function submitBlocked(state: AgentLoopState): boolean {
  switch (state.mode) {
    case 'single-hop':
      return state.hopRounds.some((r) => !r.tested);
    case 'tool-select-hop':
      return state.selectRounds.some((r) => !r.tested);
    case 'budget-tuning':
      return state.budgetResults.length === 0;
  }
}

function submitHint(state: AgentLoopState): string {
  switch (state.mode) {
    case 'single-hop': {
      const left = state.hopRounds.filter((r) => !r.tested).length;
      return `${left} round${left === 1 ? '' : 's'} left to run`;
    }
    case 'tool-select-hop': {
      const left = state.selectRounds.filter((r) => !r.tested).length;
      return `${left} round${left === 1 ? '' : 's'} left to run`;
    }
    case 'budget-tuning':
      return 'run the batch first';
  }
}

/** Two animation frames, so a "busy" state change actually paints before the
 * WASM CPU backend's synchronous forward passes block the main thread. */
function yieldToPaint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
}

function seedFor(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) >>> 0;
  return h;
}

/* ── level 1: single hop ───────────────────────────────────────── */

function SingleHopBoard({ state, dispatch }: { state: AgentLoopState; dispatch: (action: AgentLoopAction) => void }) {
  return (
    <div className="flex flex-col gap-4">
      <Panel label="worked examples to include">
        <ExampleCountToggle exampleCount={state.exampleCount} dispatch={dispatch} />
      </Panel>

      {state.hopRounds.map((round, index) => (
        <HopRoundPanel key={round.question} state={state} round={round} roundIndex={index} dispatch={dispatch} />
      ))}
    </div>
  );
}

function ExampleCountToggle({
  exampleCount,
  dispatch,
}: {
  exampleCount: 0 | 1 | 2;
  dispatch: (action: AgentLoopAction) => void;
}) {
  return (
    <div className="flex gap-2">
      {([0, 1, 2] as const).map((count) => (
        <button
          key={count}
          type="button"
          aria-pressed={exampleCount === count}
          onClick={() => dispatch({ type: 'SET_EXAMPLE_COUNT', count })}
          className={cx(
            'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
            exampleCount === count
              ? 'border-accent bg-accent-dim text-primary'
              : 'border-line-strong bg-transparent text-secondary hover:border-accent'
          )}
          style={{ borderRadius: 'var(--radius)' }}
        >
          {count} example{count === 1 ? '' : 's'}
        </button>
      ))}
    </div>
  );
}

async function runSingleHop(
  question: string,
  exampleCount: 0 | 1 | 2,
  maxTokens: number,
  finalAnswerMaxTokens: number
): Promise<{ toolCallText: string; toolResult: string | null; finalAnswerText: string }> {
  const toolCallPrompt = buildSingleHopPrompt(question, exampleCount);
  const toolCallText = await greedyDecode(tinyCausalLM, toolCallPrompt, maxTokens);
  const call = parseAgentCall(toolCallText);
  if (!call) return { toolCallText, toolResult: null, finalAnswerText: '' };

  const expression = typeof call.args.expression === 'string' ? call.args.expression : '';
  const toolResult = await toolRuntime.run('calculator', { expression });
  if (!toolResult.ok) return { toolCallText: call.cleanJson, toolResult: null, finalAnswerText: '' };

  const finalPrompt = buildFinalAnswerPrompt(toolCallPrompt, call.cleanJson, toolResult.output);
  const finalAnswerText = await greedyDecode(tinyCausalLM, finalPrompt, finalAnswerMaxTokens);
  return { toolCallText: call.cleanJson, toolResult: toolResult.output, finalAnswerText };
}

function HopRoundPanel({
  state,
  round,
  roundIndex,
  dispatch,
}: {
  state: AgentLoopState;
  round: AgentLoopState['hopRounds'][number];
  roundIndex: number;
  dispatch: (action: AgentLoopAction) => void;
}) {
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (busy) return;
    setBusy(true);
    await yieldToPaint();
    try {
      const { toolCallText, toolResult, finalAnswerText } = await runSingleHop(
        round.question,
        state.exampleCount,
        state.config.maxTokens,
        state.config.finalAnswerMaxTokens
      );
      dispatch({ type: 'TEST_HOP', roundIndex, toolCallText, toolResult, finalAnswerText });
    } finally {
      setBusy(false);
    }
  };

  const correct = round.tested && !!round.finalAnswerText && containsAnswer(round.finalAnswerText, round.expectedAnswer);

  return (
    <Panel
      label={round.question}
      actions={round.tested ? <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : 'wrong'}</Tag> : undefined}
    >
      <div className="flex flex-col gap-2">
        <Button onClick={() => void run()} disabled={busy}>
          {busy ? 'running the real model…' : 'run this hop'}
        </Button>
        {round.tested && (
          <div className="flex flex-col gap-1 border-t border-line-faint pt-2 text-xs">
            <span className="text-muted">
              tool call: <span className="readout text-secondary">{round.toolCallText || '(none)'}</span>
            </span>
            <span className="text-muted">
              real tool result: <span className="readout text-secondary">{round.toolResult ?? '(tool failed)'}</span>
            </span>
            <span className="text-muted">
              final answer: <span className="readout text-secondary">{round.finalAnswerText || '(no output)'}</span>
            </span>
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ── level 2: tool select + hop ────────────────────────────────── */

function ToolSelectHopBoard({
  state,
  dispatch,
}: {
  state: AgentLoopState;
  dispatch: (action: AgentLoopAction) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      {state.selectRounds.map((round, index) => (
        <SelectHopRoundPanel key={round.text} state={state} round={round} roundIndex={index} dispatch={dispatch} />
      ))}
    </div>
  );
}

async function runSelectHop(
  instruction: string,
  lastTool: 'calculator' | 'lookup',
  maxTokens: number,
  finalAnswerMaxTokens: number
): Promise<{ toolCallText: string; pickedTool: string | null; toolResult: string | null; finalAnswerText: string }> {
  const toolCallPrompt = buildToolSelectHopPrompt(instruction, lastTool);
  const toolCallText = await greedyDecode(tinyCausalLM, toolCallPrompt, maxTokens);
  const call = parseAgentCall(toolCallText);
  if (!call) return { toolCallText, pickedTool: null, toolResult: null, finalAnswerText: '' };

  const toolResult = await toolRuntime.run(call.tool, call.args);
  if (!toolResult.ok) return { toolCallText: call.cleanJson, pickedTool: call.tool, toolResult: null, finalAnswerText: '' };

  const finalPrompt = buildFinalAnswerPrompt(toolCallPrompt, call.cleanJson, toolResult.output);
  const finalAnswerText = await greedyDecode(tinyCausalLM, finalPrompt, finalAnswerMaxTokens);
  return { toolCallText: call.cleanJson, pickedTool: call.tool, toolResult: toolResult.output, finalAnswerText };
}

function SelectHopRoundPanel({
  state,
  round,
  roundIndex,
  dispatch,
}: {
  state: AgentLoopState;
  round: AgentLoopState['selectRounds'][number];
  roundIndex: number;
  dispatch: (action: AgentLoopAction) => void;
}) {
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!round.lastOrder || busy) return;
    setBusy(true);
    await yieldToPaint();
    try {
      const { toolCallText, pickedTool, toolResult, finalAnswerText } = await runSelectHop(
        round.text,
        round.lastOrder,
        state.config.maxTokens,
        state.config.finalAnswerMaxTokens
      );
      dispatch({ type: 'TEST_SELECT_HOP', roundIndex, toolCallText, pickedTool, toolResult, finalAnswerText });
    } finally {
      setBusy(false);
    }
  };

  const correct =
    round.tested &&
    round.pickedTool === round.expectedTool &&
    !!round.finalAnswerText &&
    containsAnswer(round.finalAnswerText, round.expectedAnswer);

  return (
    <Panel
      label={round.text}
      actions={round.tested ? <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : 'wrong'}</Tag> : undefined}
    >
      <div className="flex flex-col gap-2">
        <span className="label">list and demonstrate this tool last</span>
        <div className="flex gap-2">
          {(['calculator', 'lookup'] as const).map((tool) => (
            <button
              key={tool}
              type="button"
              aria-pressed={round.lastOrder === tool}
              onClick={() => dispatch({ type: 'SET_TOOL_ORDER', roundIndex, lastTool: tool })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                round.lastOrder === tool
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {tool}
            </button>
          ))}
        </div>
        <Button onClick={() => void run()} disabled={!round.lastOrder || busy}>
          {busy ? 'running the real model…' : 'run this hop'}
        </Button>
        {round.tested && (
          <div className="flex flex-col gap-1 border-t border-line-faint pt-2 text-xs">
            <span className="text-muted">
              picked tool: <span className="readout text-secondary">{round.pickedTool ?? '(none)'}</span>
            </span>
            <span className="text-muted">
              real tool result: <span className="readout text-secondary">{round.toolResult ?? '(tool failed)'}</span>
            </span>
            <span className="text-muted">
              final answer: <span className="readout text-secondary">{round.finalAnswerText || '(no output)'}</span>
            </span>
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ── level 3: budget tuning ────────────────────────────────────── */

function BudgetTuningBoard({
  state,
  dispatch,
  result,
}: {
  state: AgentLoopState;
  dispatch: (action: AgentLoopAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [busy, setBusy] = useState(false);
  const maxBudget = state.config.retryBudgetMax ?? 3;
  const temperature = state.config.temperature ?? 0.7;
  const topK = state.config.sampleTopK ?? 20;

  const runBatch = async () => {
    if (busy) return;
    setBusy(true);
    await yieldToPaint();
    try {
      const results: BudgetTaskResult[] = [];
      for (const task of state.budgetTasks) {
        let solved = false;
        let attemptsUsed = 0;
        for (let attempt = 0; attempt < state.retryBudget; attempt++) {
          attemptsUsed++;
          const seed = seedFor(`${task.question}:${state.exampleCount}:${attempt}`);
          const toolCallPrompt = buildSingleHopPrompt(task.question, state.exampleCount);
          const toolCallText = await sampledDecode(tinyCausalLM, toolCallPrompt, state.config.maxTokens, {
            temperature,
            topK,
            seed,
          });
          const call = parseAgentCall(toolCallText);
          if (!call) continue;
          const expression = typeof call.args.expression === 'string' ? call.args.expression : '';
          const toolResult = await toolRuntime.run('calculator', { expression });
          if (!toolResult.ok) continue;
          const finalPrompt = buildFinalAnswerPrompt(toolCallPrompt, call.cleanJson, toolResult.output);
          const finalAnswerText = await sampledDecode(tinyCausalLM, finalPrompt, state.config.finalAnswerMaxTokens, {
            temperature,
            topK,
            seed: seed + 1000,
          });
          if (containsAnswer(finalAnswerText, task.expectedAnswer)) {
            solved = true;
            break;
          }
        }
        results.push({ question: task.question, solved, attemptsUsed });
      }
      dispatch({ type: 'RUN_BUDGET_TEST', results });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel label="worked examples to include">
        <ExampleCountToggle exampleCount={state.exampleCount} dispatch={dispatch} />
      </Panel>

      <Panel label="retry budget per task">
        <div className="flex gap-2">
          {Array.from({ length: maxBudget }, (_, i) => i + 1).map((budget) => (
            <button
              key={budget}
              type="button"
              aria-pressed={state.retryBudget === budget}
              onClick={() => dispatch({ type: 'SET_RETRY_BUDGET', budget })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                state.retryBudget === budget
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {budget} attempt{budget === 1 ? '' : 's'}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex items-center gap-2">
        <Button variant="primary" onClick={() => void runBatch()} disabled={busy}>
          {busy ? 'running the real model…' : `run all ${state.budgetTasks.length} tasks`}
        </Button>
        {state.budgetResults.length > 0 && (
          <Readout label="tasksSolvedRate" value={result.value} size="sm" tone={result.passed ? 'good' : 'bad'} />
        )}
      </div>

      {state.budgetResults.length > 0 && (
        <Panel label="real results">
          <div className="flex flex-col gap-2">
            {state.budgetResults.map((r, i) => (
              <div key={i} className="flex items-center gap-2 border-t border-line-faint pt-2 first:border-t-0 first:pt-0">
                <Tag tone={r.solved ? 'good' : 'bad'}>{r.solved ? 'solved' : 'not solved'}</Tag>
                <span className="text-xs text-secondary">{r.question}</span>
                <span className="ml-auto text-[11px] text-muted">{r.attemptsUsed} real attempt{r.attemptsUsed === 1 ? '' : 's'}</span>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
