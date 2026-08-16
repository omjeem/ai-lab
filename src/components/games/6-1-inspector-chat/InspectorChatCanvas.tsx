'use client';

/**
 * Chapter 6.1 — Inspector Chat, the capstone.
 *
 * Every response is rendered as a strip of real per-token chips (colour keyed
 * to that token's real probability), not a chat bubble — clicking one opens
 * the real alternatives the sampler was choosing between, straight off
 * `GenerationStep.alternatives`. Nothing here is a synthetic distribution.
 *
 * This chapter's local model needs WebGPU; `isWebLLMSupported()` is checked
 * before `ModelGate` is even allowed to attempt the ~880MB download, since a
 * doomed load without WebGPU fails with an internal WebLLM error, not a clean
 * "unsupported" message.
 *
 * The cloud model (level 3 only) returns no per-token detail at all — the
 * proxy route only ever gets back a finished string from Ollama Cloud, never
 * logprobs — so a cloud `GenerationTrace` always has `steps: []`. That is
 * real API behaviour, not a bug: `TraceStrip` says so explicitly rather than
 * rendering an empty strip that looks broken.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, Cloud, Play, RotateCcw, Send } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  cloudAvailable,
  type ChallengeState,
  type ComparisonRound,
  type InspectorChatAction,
  type InspectorChatConfig,
  type InspectorChatState,
} from '@/engines/inspectorChatEngine';
import type { GenerationStep, GenerationTrace } from '@/engines/deps';
import { getEngine, isWebLLMSupported, webllmCapstone, WEBLLM_MODEL_ID } from '@/models/webllmCapstone';
import { ollamaCloudClient } from '@/models/ollamaCloudClient';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, formatNumber, Meter, Panel, Slider, Tag, Toggle, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const UNSUPPORTED_MESSAGE =
  'This browser cannot run the local model — it requires WebGPU with a working GPU adapter. Try a recent desktop build of Chrome or Edge with hardware acceleration on. Worlds 1 through 5 still work fully without it.';

export function InspectorChatCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as InspectorChatConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<InspectorChatState | null>(null);

  const load = useCallback(async () => {
    if (!(await isWebLLMSupported())) throw new Error(UNSUPPORTED_MESSAGE);
    // Warm the engine explicitly so `ModelGate`'s progress bar reflects the
    // real ~880MB download on every level, not just find-fork (the only mode
    // `prepare()` itself touches the model for).
    await getEngine();
    await prepare(config, { chat: webllmCapstone });
    setState(initState(config, rules));
  }, [config, rules]);

  const dispatch = useCallback((action: InspectorChatAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={WEBLLM_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board
          state={state}
          dispatch={dispatch}
          fallbackMessage={game.modelRequirement.fallbackMessage}
          onScore={onScore}
          onSubmit={onSubmit}
        />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  fallbackMessage,
  onScore,
  onSubmit,
}: {
  state: InspectorChatState;
  dispatch: (action: InspectorChatAction) => void;
  fallbackMessage: string | null;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  // Real connectivity, not a simulated switch — Section 11's rule that the
  // cloud toggle must be genuinely disabled offline, not just styled that way.
  useEffect(() => {
    const sync = () => dispatch({ type: 'SET_ONLINE', value: navigator.onLine });
    sync();
    window.addEventListener('online', sync);
    window.addEventListener('offline', sync);
    return () => {
      window.removeEventListener('online', sync);
      window.removeEventListener('offline', sync);
    };
  }, [dispatch]);

  const resetRun = useCallback(() => dispatch({ type: 'RESET' }), [dispatch]);
  useRetrySignal(resetRun);

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'challenge-run' && <ChallengeRunBoard state={state} dispatch={dispatch} />}
          {state.mode === 'find-fork' && <FindForkBoard state={state} dispatch={dispatch} />}
          {state.mode === 'compare-scale' && (
            <CompareScaleBoard state={state} dispatch={dispatch} fallbackMessage={fallbackMessage} />
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        <Actions state={state} onSubmit={submit} />
      </div>
    </div>
  );
}

function Actions({ state, onSubmit }: { state: InspectorChatState; onSubmit: () => void }) {
  const blocked = submitBlocked(state);
  return (
    <>
      {blocked && <span className="label">{submitHint(state)}</span>}
      <Button variant="primary" className="ml-auto" disabled={blocked} onClick={onSubmit}>
        <Check size={13} strokeWidth={2} />
        Submit
      </Button>
    </>
  );
}

function submitBlocked(state: InspectorChatState): boolean {
  switch (state.mode) {
    case 'challenge-run':
      return state.transcript.length === 0;
    case 'find-fork': {
      const total = state.config.rounds ?? state.forkRounds.length;
      return state.forkRounds.length < total || state.forkRounds.some((r) => r.answer === null);
    }
    case 'compare-scale':
      return (
        !state.localTrace ||
        !state.cloudTrace ||
        state.comparisons.some((c) => c.prediction === null)
      );
  }
}

function submitHint(state: InspectorChatState): string {
  switch (state.mode) {
    case 'challenge-run':
      return 'generate at least one response first';
    case 'find-fork':
      return 'generate and answer every round first';
    case 'compare-scale':
      return 'run both models and predict every axis first';
  }
}

/* ── level 1: drive the distribution ───────────────────────────── */

function ChallengeRunBoard({
  state,
  dispatch,
}: {
  state: InspectorChatState;
  dispatch: (action: InspectorChatAction) => void;
}) {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (prompt: string) => {
    setGenerating(true);
    setError(null);
    try {
      const trace = await webllmCapstone.generateWithTrace(prompt, {
        maxTokens: state.config.maxTokens,
        temperature: state.temperature,
        topP: state.topP,
      });
      dispatch({ type: 'RECORD_RUN', prompt, trace, source: 'local' });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-2 sm:grid-cols-3">
        {state.challenges.map((challenge) => (
          <ChallengeCard key={challenge.id} challenge={challenge} />
        ))}
      </div>

      <PromptComposer state={state} dispatch={dispatch} onGenerate={handleGenerate} generating={generating} />
      {error && <p className="text-xs text-bad">{error}</p>}

      <div className="flex flex-col gap-3">
        {[...state.transcript].reverse().map((entry, i) => (
          <TraceStrip
            key={state.transcript.length - 1 - i}
            prompt={entry.prompt}
            trace={entry.trace}
            source={entry.source}
          />
        ))}
      </div>
    </div>
  );
}

function ChallengeCard({ challenge }: { challenge: ChallengeState }) {
  return (
    <Panel label={challenge.satisfied ? 'cleared' : 'open'}>
      <div className="flex flex-col gap-2">
        <p className="text-xs leading-relaxed text-secondary">{challenge.prompt}</p>
        <div className="flex items-center justify-between gap-2">
          <span className="label">{challenge.check}</span>
          <Tag tone={challenge.satisfied ? 'good' : 'neutral'}>
            {challenge.satisfied && <Check size={10} strokeWidth={2} />}
            {challenge.measured === null ? '—' : formatNumber(challenge.measured)}
            {' '}
            {challenge.comparator === 'gte' ? '≥' : '≤'} {challenge.target}
          </Tag>
        </div>
      </div>
    </Panel>
  );
}

/* ── level 2: find the fork ─────────────────────────────────────── */

function FindForkBoard({
  state,
  dispatch,
}: {
  state: InspectorChatState;
  dispatch: (action: InspectorChatAction) => void;
}) {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const maxRounds = state.config.rounds ?? state.forkRounds.length;
  const canGenerate = state.forkRounds.length < maxRounds;
  const currentIndex = state.forkRounds.length - 1;
  const currentRound = state.forkRounds[currentIndex] ?? null;
  const tolerance = state.config.positionTolerance ?? 0;

  const handleGenerate = async (prompt: string) => {
    setGenerating(true);
    setError(null);
    try {
      const trace = await webllmCapstone.generateWithTrace(prompt, {
        maxTokens: state.config.maxTokens,
        temperature: state.temperature,
        topP: state.topP,
      });
      dispatch({ type: 'RECORD_RUN', prompt, trace, source: 'local' });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs leading-relaxed text-muted">
        Round {Math.min(state.forkRounds.length + (canGenerate ? 1 : 0), maxRounds)} of {maxRounds}. Generate
        a response, then click the token where the model was genuinely least certain.
      </p>

      {canGenerate && (
        <PromptComposer state={state} dispatch={dispatch} onGenerate={handleGenerate} generating={generating} />
      )}
      {error && <p className="text-xs text-bad">{error}</p>}

      {currentRound && (
        <TraceStrip
          prompt={state.transcript[currentIndex]?.prompt ?? ''}
          trace={currentRound.trace}
          source="local"
          forkMode
          forkAnswer={currentRound.answer}
          forkTrueIndex={currentRound.answer !== null ? currentRound.trueForkIndex : null}
          onPickFork={(index) => dispatch({ type: 'ANSWER_FORK', roundIndex: currentIndex, index })}
        />
      )}

      {state.forkRounds.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {state.forkRounds.map((round, i) => {
            const hit = round.answer !== null && Math.abs(round.answer - round.trueForkIndex) <= tolerance;
            return (
              <Tag key={i} tone={round.answer === null ? 'neutral' : hit ? 'good' : 'bad'}>
                round {i + 1} · {round.answer === null ? 'unanswered' : hit ? 'correct' : 'missed'}
              </Tag>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── level 3: local versus cloud ────────────────────────────────── */

function CompareScaleBoard({
  state,
  dispatch,
  fallbackMessage,
}: {
  state: InspectorChatState;
  dispatch: (action: InspectorChatAction) => void;
  fallbackMessage: string | null;
}) {
  const [prompt, setPrompt] = useState('');
  const [runningLocal, setRunningLocal] = useState(false);
  const [runningCloud, setRunningCloud] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [minTemp, maxTemp] = state.config.temperatureRange ?? [0.1, 2];
  const [minTopP, maxTopP] = state.config.topPRange ?? [0.1, 1];

  const available = cloudAvailable(state);
  const trimmed = prompt.trim();

  const runLocal = async () => {
    if (!trimmed) return;
    setRunningLocal(true);
    setError(null);
    try {
      const trace = await webllmCapstone.generateWithTrace(trimmed, {
        maxTokens: state.config.maxTokens,
        temperature: state.temperature,
        topP: state.topP,
      });
      dispatch({ type: 'RECORD_RUN', prompt: trimmed, trace, source: 'local' });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunningLocal(false);
    }
  };

  const runCloud = async () => {
    if (!trimmed || !state.cloudEnabled) return;
    setRunningCloud(true);
    setError(null);
    try {
      const trace = await ollamaCloudClient.generateWithTrace(trimmed, {
        maxTokens: state.config.maxTokens,
        temperature: state.temperature,
        topP: state.topP,
      });
      dispatch({ type: 'RECORD_RUN', prompt: trimmed, trace, source: 'cloud' });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunningCloud(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel label="same prompt, both models">
        <div className="flex flex-col gap-3">
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Write a prompt…"
            rows={2}
            aria-label="Prompt for both models"
            className="w-full resize-none border border-line-strong bg-raised px-3 py-2 font-mono text-sm text-primary outline-none focus:border-accent"
            style={{ borderRadius: 'var(--radius)' }}
          />

          <div className="grid grid-cols-2 gap-3">
            <Slider
              label="temperature"
              value={state.temperature}
              min={minTemp}
              max={maxTemp}
              step={0.01}
              onChange={(value) => dispatch({ type: 'SET_TEMPERATURE', value })}
            />
            <Slider
              label="top-p"
              value={state.topP}
              min={minTopP}
              max={maxTopP}
              step={0.01}
              onChange={(value) => dispatch({ type: 'SET_TOP_P', value })}
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" disabled={runningLocal || !trimmed} onClick={() => void runLocal()}>
              <Play size={13} strokeWidth={2} />
              {runningLocal ? 'Running…' : 'Run local'}
            </Button>

            <Button
              variant="primary"
              disabled={runningCloud || !trimmed || !state.cloudEnabled}
              onClick={() => void runCloud()}
            >
              <Cloud size={13} strokeWidth={2} />
              {runningCloud ? 'Running…' : 'Run cloud'}
            </Button>

            <Toggle
              label="cloud escalation"
              checked={state.cloudEnabled}
              disabled={!available}
              onChange={(value) => dispatch({ type: 'TOGGLE_CLOUD', value })}
            />
          </div>

          {!available && fallbackMessage && <Tag tone="warn">{fallbackMessage}</Tag>}
          {error && <p className="text-xs text-bad">{error}</p>}
        </div>
      </Panel>

      <div className="grid gap-3 sm:grid-cols-2">
        {state.localTrace ? (
          <TraceStrip prompt={lastPromptFor(state, 'local')} trace={state.localTrace} source="local" />
        ) : (
          <Panel label="local">
            <p className="text-xs text-muted">Run the local model to see its trace here.</p>
          </Panel>
        )}
        {state.cloudTrace ? (
          <TraceStrip prompt={lastPromptFor(state, 'cloud')} trace={state.cloudTrace} source="cloud" />
        ) : (
          <Panel label="cloud">
            <p className="text-xs text-muted">Run the cloud model to see its trace here.</p>
          </Panel>
        )}
      </div>

      <Panel label="predict, then check">
        <div className="flex flex-col gap-2">
          {state.comparisons.map((comparison) => (
            <ComparisonRow key={comparison.axis} comparison={comparison} dispatch={dispatch} />
          ))}
        </div>
      </Panel>
    </div>
  );
}

/** The prompt behind whichever trace (local or cloud) is currently latest. */
function lastPromptFor(state: InspectorChatState, source: 'local' | 'cloud'): string {
  for (let i = state.transcript.length - 1; i >= 0; i--) {
    const entry = state.transcript[i];
    if (entry?.source === source) return entry.prompt;
  }
  return '';
}

function ComparisonRow({
  comparison,
  dispatch,
}: {
  comparison: ComparisonRound;
  dispatch: (action: InspectorChatAction) => void;
}) {
  const answered = comparison.trueAnswer !== null;
  const correct = answered && comparison.prediction === comparison.trueAnswer;

  return (
    <div
      className="flex flex-wrap items-center justify-between gap-2 border border-line-strong px-3 py-2"
      style={{ borderRadius: 'var(--radius)' }}
    >
      <span className="font-mono text-xs uppercase tracking-[0.1em] text-primary">{comparison.axis}</span>
      <div className="flex items-center gap-2">
        <Button
          variant={comparison.prediction === true ? 'primary' : 'ghost'}
          onClick={() => dispatch({ type: 'PREDICT_COMPARISON', axis: comparison.axis, value: true })}
        >
          cloud wins
        </Button>
        <Button
          variant={comparison.prediction === false ? 'primary' : 'ghost'}
          onClick={() => dispatch({ type: 'PREDICT_COMPARISON', axis: comparison.axis, value: false })}
        >
          local wins
        </Button>
        {answered && (
          <Tag tone={correct ? 'good' : 'bad'}>
            {comparison.trueAnswer ? 'cloud actually won' : 'local actually won'}
          </Tag>
        )}
      </div>
    </div>
  );
}

/* ── shared: prompt + sampling controls ────────────────────────── */

function PromptComposer({
  state,
  dispatch,
  onGenerate,
  generating,
}: {
  state: InspectorChatState;
  dispatch: (action: InspectorChatAction) => void;
  onGenerate: (prompt: string) => void;
  generating: boolean;
}) {
  const [prompt, setPrompt] = useState('');
  const [minTemp, maxTemp] = state.config.temperatureRange ?? [0.1, 2];
  const [minTopP, maxTopP] = state.config.topPRange ?? [0.1, 1];

  return (
    <Panel label="prompt">
      <div className="flex flex-col gap-3">
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Write a prompt…"
          rows={2}
          aria-label="Prompt for the local model"
          className="w-full resize-none border border-line-strong bg-raised px-3 py-2 font-mono text-sm text-primary outline-none focus:border-accent"
          style={{ borderRadius: 'var(--radius)' }}
        />

        <div className="grid grid-cols-2 gap-3">
          <Slider
            label="temperature"
            value={state.temperature}
            min={minTemp}
            max={maxTemp}
            step={0.01}
            onChange={(value) => dispatch({ type: 'SET_TEMPERATURE', value })}
          />
          <Slider
            label="top-p"
            value={state.topP}
            min={minTopP}
            max={maxTopP}
            step={0.01}
            onChange={(value) => dispatch({ type: 'SET_TOP_P', value })}
          />
        </div>

        <Button
          variant="primary"
          disabled={generating || prompt.trim().length === 0}
          onClick={() => {
            const trimmed = prompt.trim();
            if (!trimmed) return;
            onGenerate(trimmed);
            setPrompt('');
          }}
        >
          <Send size={13} strokeWidth={2} />
          {generating ? 'Generating…' : 'Generate'}
        </Button>
      </div>
    </Panel>
  );
}

/* ── shared: the transcript instrument ─────────────────────────── */

type ForkBadge = 'correct' | 'true-fork' | 'your-pick' | null;

function TraceStrip({
  prompt,
  trace,
  source,
  forkMode,
  forkAnswer,
  forkTrueIndex,
  onPickFork,
}: {
  prompt: string;
  trace: GenerationTrace;
  source: 'local' | 'cloud';
  forkMode?: boolean;
  forkAnswer?: number | null;
  forkTrueIndex?: number | null;
  onPickFork?: (index: number) => void;
}) {
  const reduce = useReducedMotion();
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const openStep = openIndex !== null ? (trace.steps[openIndex] ?? null) : null;

  return (
    <Panel label={prompt ? `${source} — “${prompt}”` : source}>
      <div className="flex flex-col gap-3">
        {trace.steps.length === 0 ? (
          <p className="text-xs leading-relaxed text-muted">
            {source === 'cloud'
              ? "The cloud endpoint doesn't return per-token detail — this trace is text only."
              : 'No tokens generated.'}
          </p>
        ) : (
          <div className="flex flex-wrap gap-1">
            {trace.steps.map((step, i) => {
              const badge: ForkBadge =
                forkAnswer === i && forkTrueIndex === i
                  ? 'correct'
                  : forkTrueIndex === i
                    ? 'true-fork'
                    : forkAnswer === i
                      ? 'your-pick'
                      : null;
              return (
                <TokenChip
                  key={i}
                  step={step}
                  selected={openIndex === i}
                  badge={badge}
                  onClick={() =>
                    forkMode && forkAnswer == null && onPickFork
                      ? onPickFork(i)
                      : setOpenIndex(openIndex === i ? null : i)
                  }
                />
              );
            })}
          </div>
        )}

        {trace.text && <p className="font-mono text-sm leading-relaxed text-primary">{trace.text}</p>}

        {openStep && (
          <motion.div
            initial={reduce ? undefined : { opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: reduce ? 0 : 0.18 }}
            className="flex flex-col gap-1.5 border-t border-line-faint pt-3"
          >
            <span className="label">
              alternatives at this step · entropy {openStep.entropyBits.toFixed(2)} bits
            </span>
            {openStep.alternatives.tokens.map((token, i) => (
              <div key={i} className="flex items-center gap-2.5">
                <span className="w-16 shrink-0 truncate font-mono text-xs text-secondary">{token}</span>
                <span className="flex-1">
                  <Meter value={openStep.alternatives.probs[i] ?? 0} tone="accent" />
                </span>
                <span className="readout w-12 shrink-0 text-right text-xs text-secondary">
                  {((openStep.alternatives.probs[i] ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </motion.div>
        )}
      </div>
    </Panel>
  );
}

function TokenChip({
  step,
  selected,
  badge,
  onClick,
}: {
  step: GenerationStep;
  selected: boolean;
  badge: ForkBadge;
  onClick: () => void;
}) {
  const label = `${step.token.trim() || step.token} — probability ${(step.probability * 100).toFixed(1)}%, entropy ${step.entropyBits.toFixed(2)} bits`;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cx(
        'border px-1.5 py-1 font-mono text-xs transition-colors',
        selected ? 'border-accent text-accent' : 'border-line-strong text-primary hover:border-accent',
        badge === 'correct' && 'border-good text-good',
        badge === 'true-fork' && 'border-good text-good',
        badge === 'your-pick' && 'border-warn text-warn'
      )}
      style={{
        borderRadius: '3px',
        background: `color-mix(in oklab, var(--accent) ${Math.round(step.probability * 35)}%, var(--bg-raised))`,
      }}
    >
      {step.token}
    </button>
  );
}
