'use client';

/**
 * Chapter 4.3 — Sampling Strategies.
 *
 * The model hands back one real distribution per prompt; every slider here
 * reshapes it, live, through the same temperature → top-k → top-p pipeline a
 * decoder actually runs. The bars are the whole chapter: nothing is sampled
 * or guessed at, the truncation cut is drawn exactly where `reshape()` put it.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import { Check, Plus, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  reshape,
  currentReshape,
  type SamplingAction,
  type SamplingConfig,
  type SamplingState,
  type ReshapeResult,
} from '@/engines/samplingEngine';
import { tinyCausalLM, preloadCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Slider, Tag, cx, formatNumber } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function SamplingCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as SamplingConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<SamplingState | null>(null);

  const load = useCallback(async () => {
    await preloadCausalLM();
    const prepared = await prepare(config, { causalLM: tinyCausalLM });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: SamplingAction) => {
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
  state: SamplingState;
  dispatch: (action: SamplingAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const resetRun = useCallback(() => dispatch({ type: 'RESET' }), [dispatch]);
  useRetrySignal(resetRun);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'entropy-target' && <EntropyTargetBoard state={state} dispatch={dispatch} />}
          {state.mode === 'compare-truncation' && <CompareTruncationBoard state={state} dispatch={dispatch} />}
          {state.mode === 'task-tuning' && <TaskTuningBoard state={state} dispatch={dispatch} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
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
      </div>
    </div>
  );
}

/* ── level 1: hit the entropy target ────────────────────────── */

function EntropyTargetBoard({
  state,
  dispatch,
}: {
  state: SamplingState;
  dispatch: (action: SamplingAction) => void;
}) {
  const result = currentReshape(state, 0);
  const target = state.config.targetEntropyBits ?? 0;
  const tolerance = state.config.toleranceBits ?? 0.25;
  const [tempMin, tempMax] = state.config.temperatureRange ?? [0.1, 2.5];
  const ceiling = Math.log2(Math.max(2, state.config.topK));
  const distribution = state.distributions[0];

  return (
    <div className="flex flex-col gap-4">
      {distribution && <PromptLine prompt={distribution.prompt} />}

      <Panel label="entropy">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <Readout
              label="reshaped entropy"
              value={result ? formatNumber(result.entropyBits, 2) : '—'}
              unit="bits"
              size="lg"
              tone="accent"
            />
            <Readout label="target" value={`${target.toFixed(2)} ± ${tolerance.toFixed(2)}`} unit="bits" size="sm" />
          </div>
          <EntropyGauge value={result?.entropyBits ?? 0} max={ceiling} target={target} tolerance={tolerance} />
        </div>
      </Panel>

      <Slider
        label="temperature"
        value={state.temperature}
        min={tempMin}
        max={tempMax}
        step={0.01}
        onChange={(value) => dispatch({ type: 'SET_TEMPERATURE', value })}
        format={(value) => value.toFixed(2)}
      />

      {result && <DistributionBars result={result} />}

      <p className="text-xs leading-relaxed text-muted">
        Temperature rescales the log-probabilities before the softmax. Below 1 it sharpens the
        distribution toward the model&apos;s favourite token; above 1 it flattens it toward uniform.
        The bars and the entropy readout are the same real distribution, just two views of it.
      </p>
    </div>
  );
}

/* ── level 2: top-k versus top-p ────────────────────────────── */

function CompareTruncationBoard({
  state,
  dispatch,
}: {
  state: SamplingState;
  dispatch: (action: SamplingAction) => void;
}) {
  const low = state.config.targetKeptMassLow ?? 0.75;
  const high = state.config.targetKeptMassHigh ?? 0.95;
  const [tempMin, tempMax] = state.config.temperatureRange ?? [0.1, 2];
  const [topKMin, topKMax] = state.config.topKRange ?? [1, state.config.topK];
  const [topPMin, topPMax] = state.config.topPRange ?? [0.1, 1];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Slider
          label="temperature"
          value={state.temperature}
          min={tempMin}
          max={tempMax}
          step={0.01}
          onChange={(value) => dispatch({ type: 'SET_TEMPERATURE', value })}
          format={(value) => value.toFixed(2)}
        />
        <Slider
          label="top-k"
          value={state.topK}
          min={topKMin}
          max={topKMax}
          step={1}
          onChange={(value) => dispatch({ type: 'SET_TOP_K', value: Math.round(value) })}
          format={(value) => `${Math.round(value)}`}
        />
        <Slider
          label="top-p"
          value={state.topP}
          min={topPMin}
          max={topPMax}
          step={0.01}
          onChange={(value) => dispatch({ type: 'SET_TOP_P', value })}
          format={(value) => value.toFixed(2)}
        />
      </div>

      <p className="text-xs leading-relaxed text-muted">
        One setting has to keep between {formatPercent(low)} and {formatPercent(high)} of the real
        probability mass on both prompts below — that band is the target zone on each meter.
      </p>

      {state.distributions.map((distribution, index) => {
        const result = currentReshape(state, index);
        if (!result) return null;
        const inBand = result.keptMass >= low && result.keptMass <= high;
        return (
          <Panel
            key={distribution.prompt}
            label="prompt"
            actions={
              <Tag tone={inBand ? 'good' : 'warn'}>
                {formatPercent(result.keptMass)} kept
              </Tag>
            }
          >
            <div className="flex flex-col gap-3">
              <PromptLine prompt={distribution.prompt} />
              <KeptMassGauge value={result.keptMass} low={low} high={high} />
              <DistributionBars result={result} />
            </div>
          </Panel>
        );
      })}
    </div>
  );
}

/* ── level 3: tune for the job ──────────────────────────────── */

function TaskTuningBoard({
  state,
  dispatch,
}: {
  state: SamplingState;
  dispatch: (action: SamplingAction) => void;
}) {
  const tasks = state.config.tasks ?? [];
  const activeTask = tasks.find((t) => t.id === state.activeTaskId) ?? tasks[0];
  const [tempMin, tempMax] = state.config.temperatureRange ?? [0.1, 2.5];
  const [topKMin, topKMax] = state.config.topKRange ?? [1, state.config.topK];
  const [topPMin, topPMax] = state.config.topPRange ?? [0.1, 1];

  const activeDistribution = activeTask
    ? state.distributions.find((d) => d.prompt === activeTask.prompt)
    : undefined;
  const activeResult = activeDistribution
    ? reshape(activeDistribution, {
        temperature: state.temperature,
        topK: state.topK,
        topP: state.topP,
      })
    : null;
  const ceiling = Math.log2(Math.max(2, state.config.topK));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="tasks">
        {tasks.map((task) => {
          const distribution = state.distributions.find((d) => d.prompt === task.prompt);
          const settings = state.taskSettings[task.id];
          const taskResult = distribution && settings ? reshape(distribution, settings) : null;
          const status = taskStatus(taskResult, task.targetEntropyBits, task.toleranceBits);
          const active = task.id === state.activeTaskId;
          return (
            <button
              key={task.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => dispatch({ type: 'SELECT_TASK', taskId: task.id })}
              className={cx(
                'flex items-center gap-2 border px-3 py-2 font-mono text-xs transition-colors',
                active
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-raised text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {task.id}
              <span
                className={cx(
                  'h-1.5 w-1.5 shrink-0',
                  status === 'pass' ? 'bg-good' : status === 'near' ? 'bg-warn' : 'bg-bad'
                )}
                style={{ borderRadius: '50%' }}
                aria-hidden
              />
            </button>
          );
        })}
      </div>

      {activeTask && activeDistribution && (
        <>
          <PromptLine prompt={activeDistribution.prompt} mono={activeTask.id === 'code'} />

          <Panel label="entropy">
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <Readout
                  label="reshaped entropy"
                  value={activeResult ? formatNumber(activeResult.entropyBits, 2) : '—'}
                  unit="bits"
                  size="lg"
                  tone="accent"
                />
                <Readout
                  label="target"
                  value={`${activeTask.targetEntropyBits.toFixed(2)} ± ${activeTask.toleranceBits.toFixed(2)}`}
                  unit="bits"
                  size="sm"
                />
              </div>
              <EntropyGauge
                value={activeResult?.entropyBits ?? 0}
                max={ceiling}
                target={activeTask.targetEntropyBits}
                tolerance={activeTask.toleranceBits}
              />
            </div>
          </Panel>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Slider
              label="temperature"
              value={state.temperature}
              min={tempMin}
              max={tempMax}
              step={0.01}
              onChange={(value) => dispatch({ type: 'SET_TEMPERATURE', value })}
              format={(value) => value.toFixed(2)}
            />
            <Slider
              label="top-k"
              value={state.topK}
              min={topKMin}
              max={topKMax}
              step={1}
              onChange={(value) => dispatch({ type: 'SET_TOP_K', value: Math.round(value) })}
              format={(value) => `${Math.round(value)}`}
            />
            <Slider
              label="top-p"
              value={state.topP}
              min={topPMin}
              max={topPMax}
              step={0.01}
              onChange={(value) => dispatch({ type: 'SET_TOP_P', value })}
              format={(value) => value.toFixed(2)}
            />
          </div>

          {activeResult && <DistributionBars result={activeResult} />}
        </>
      )}

      {state.config.allowUserPrompt && <PromptAdder state={state} dispatch={dispatch} />}

      <p className="text-xs leading-relaxed text-muted">
        Each task keeps its own settings — switching tasks restores whatever you last set for it. A
        factual completion wants a narrow, confident distribution; a creative one wants room to
        wander; code wants just enough freedom to pick a valid next token and no more.
      </p>
    </div>
  );
}

function taskStatus(
  result: ReshapeResult | null,
  target: number,
  tolerance: number
): 'pass' | 'near' | 'fail' {
  if (!result) return 'fail';
  const error = Math.abs(result.entropyBits - target);
  if (error <= tolerance) return 'pass';
  if (error <= tolerance * 2) return 'near';
  return 'fail';
}

function PromptAdder({
  state,
  dispatch,
}: {
  state: SamplingState;
  dispatch: (action: SamplingAction) => void;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ prompt: string; entropyBits: number } | null>(null);

  const run = async () => {
    const prompt = draft.trim();
    if (!prompt || busy) return;
    setBusy(true);
    setError(null);
    try {
      const dist = await tinyCausalLM.nextTokenDistribution(prompt, state.config.topK);
      dispatch({ type: 'ADD_DISTRIBUTION', distribution: { prompt, tokens: dist.tokens, probs: dist.probs } });
      const result = reshape(
        { prompt, tokens: dist.tokens, probs: dist.probs },
        { temperature: state.temperature, topK: state.topK, topP: state.topP }
      );
      setPreview({ prompt, entropyBits: result.entropyBits });
      setDraft('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel label="try your own prompt — not scored, just exploration">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void run()}
            placeholder="type a prompt to see its real reshaped entropy"
            maxLength={200}
            disabled={busy}
            aria-label="Your prompt"
            className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-60"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={() => void run()} disabled={busy || draft.trim() === ''}>
            <Plus size={13} strokeWidth={2} />
            {busy ? 'running' : 'run it'}
          </Button>
        </div>
        {error && <p className="readout text-xs text-bad">the model could not run that — {error}</p>}
        {preview && (
          <p className="text-xs text-secondary">
            at the current settings, “{preview.prompt}” reshapes to{' '}
            <span className="readout text-accent">{formatNumber(preview.entropyBits, 2)} bits</span> of entropy.
          </p>
        )}
      </div>
    </Panel>
  );
}

/* ── shared visuals ─────────────────────────────────────────── */

/** Horizontal ranked bars — kept candidates lit, discarded ones dimmed below a cut line. */
function DistributionBars({ result }: { result: ReshapeResult }) {
  const reduce = useReducedMotion();
  const cap = 30;
  const rows = result.rankedTokens.slice(0, cap);
  const overflow = result.rankedTokens.length - rows.length;
  const maxProb = Math.max(...result.rankedProbs, 1e-9);

  return (
    <div
      className="flex flex-col gap-1"
      role="img"
      aria-label={`Distribution over ${result.rankedTokens.length} candidates, ${result.keptCount} kept covering ${formatPercent(
        result.keptMass
      )} of the mass`}
    >
      {rows.map((token, i) => {
        const kept = i < result.keptCount;
        const prob = result.rankedProbs[i] ?? 0;
        const pct = (prob / maxProb) * 100;
        return (
          <div key={i} className="flex items-center gap-2">
            <span className="label w-5 shrink-0 text-right text-[9px]">{i + 1}</span>
            <span
              className={cx(
                'w-20 shrink-0 truncate font-mono text-xs',
                kept ? 'text-primary' : 'text-muted/60'
              )}
            >
              {displayToken(token)}
            </span>
            <div className="relative h-3.5 min-w-0 flex-1 bg-inset">
              <div
                className={cx('absolute inset-y-0 left-0', kept ? 'bg-accent' : 'bg-line-strong/50')}
                style={{
                  width: `${pct}%`,
                  transition: reduce ? undefined : 'width 160ms ease-out',
                }}
              />
            </div>
            <span
              className={cx('readout w-14 shrink-0 text-right text-xs', kept ? 'text-accent' : 'text-muted/60')}
            >
              {formatPercent(prob)}
            </span>
          </div>
        );
      })}

      {result.keptCount < rows.length && (
        <div className="flex items-center gap-2 border-t border-dashed border-line pt-1">
          <span className="w-5 shrink-0" />
          <span className="label text-[9px]">cut — {formatPercent(result.keptMass)} kept</span>
        </div>
      )}

      {overflow > 0 && <p className="label pl-7 text-[9px]">+{overflow} more discarded, not shown</p>}
    </div>
  );
}

/** A horizontal axis with the target band shaded and the current value marked. */
function EntropyGauge({
  value,
  max,
  target,
  tolerance,
}: {
  value: number;
  max: number;
  target: number;
  tolerance: number;
}) {
  const reduce = useReducedMotion();
  const safeMax = Math.max(max, 1e-6);
  const toPct = (v: number) => Math.max(0, Math.min(1, v / safeMax)) * 100;
  const bandStart = toPct(Math.max(0, target - tolerance));
  const bandEnd = toPct(Math.min(safeMax, target + tolerance));
  const markerPct = toPct(value);

  return (
    <div className="relative h-3 w-full bg-inset" role="img" aria-label={`${value.toFixed(2)} of ${max.toFixed(2)} bits, target ${target.toFixed(2)}`}>
      <div
        className="absolute inset-y-0 bg-good/25"
        style={{ left: `${bandStart}%`, width: `${Math.max(0, bandEnd - bandStart)}%` }}
        aria-hidden
      />
      <div
        className="absolute inset-y-[-3px] w-0.5 bg-accent"
        style={{
          left: `${markerPct}%`,
          transition: reduce ? undefined : 'left 160ms ease-out',
        }}
        aria-hidden
      />
    </div>
  );
}

/** Same band-plus-marker gauge, scaled 0..1 for a probability mass rather than bits. */
function KeptMassGauge({ value, low, high }: { value: number; low: number; high: number }) {
  const reduce = useReducedMotion();
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="relative h-3 w-full bg-inset" role="img" aria-label={`${formatPercent(value)} kept, target ${formatPercent(low)} to ${formatPercent(high)}`}>
      <div
        className="absolute inset-y-0 bg-good/25"
        style={{ left: `${low * 100}%`, width: `${Math.max(0, (high - low) * 100)}%` }}
        aria-hidden
      />
      <div
        className="absolute inset-y-[-3px] w-0.5 bg-accent"
        style={{ left: `${pct}%`, transition: reduce ? undefined : 'left 160ms ease-out' }}
        aria-hidden
      />
    </div>
  );
}

function PromptLine({ prompt, mono }: { prompt: string; mono?: boolean }) {
  return (
    <p
      className={cx(
        'border border-line bg-inset px-2 py-1.5 text-xs leading-relaxed text-primary',
        mono ? 'whitespace-pre-wrap font-mono' : 'font-mono'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {prompt}
      <span className="text-muted"> ▮</span>
    </p>
  );
}

/** Tokens carry their leading space; show it rather than letting it vanish. */
function displayToken(token: string): string {
  if (token.length === 0) return '∅';
  return token.replace(/ /g, '␣').replace(/\n/g, '⏎');
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
