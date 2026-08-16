'use client';

/**
 * Chapter 5.3 — Multi-Head Attention.
 *
 * The twelve small matrices are the instrument: every cell is a real
 * softmaxed weight from the same model 5.2 uses, at whichever layer the
 * slider is on. Nothing about head "roles" is authored — `profileHead`
 * measures each one from its own matrix, so relabelling the sentence or
 * moving the slider genuinely changes what the grid shows.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Check, Play, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type HeadBehaviour,
  type MultiHeadDetectiveAction,
  type MultiHeadDetectiveConfig,
  type MultiHeadDetectiveState,
} from '@/engines/multiHeadDetectiveEngine';
import { attentionModel, ATTENTION_MODEL_ID } from '@/models/attentionModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Slider, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const BEHAVIOUR_LABEL: Record<HeadBehaviour, string> = {
  'previous-token': 'prev',
  'next-token': 'next',
  self: 'self',
  delimiter: 'delim',
  diffuse: 'diffuse',
};

export function MultiHeadCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as MultiHeadDetectiveConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<MultiHeadDetectiveState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { attention: attentionModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const loadSentence = useCallback(
    async (sentence: string) => {
      const nextConfig: MultiHeadDetectiveConfig = { ...config, sentence };
      const prepared = await prepare(nextConfig, { attention: attentionModel });
      setState(initState(nextConfig, rules, prepared));
    },
    [config, rules]
  );

  const dispatch = useCallback((action: MultiHeadDetectiveAction) => {
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
        <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} loadSentence={loadSentence} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
  loadSentence,
}: {
  state: MultiHeadDetectiveState;
  dispatch: (action: MultiHeadDetectiveAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
  loadSentence: (sentence: string) => Promise<void>;
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  /** Local, never `state.status` — every action resets status back to `active`. */
  const [revealed, setRevealed] = useState(false);
  const [runId, setRunId] = useState(0);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
    setRunId((n) => n + 1);
  }, [dispatch]);

  useRetrySignal(resetRun);

  const [min, max] = state.config.layerRange ?? [state.layer, state.layer];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div key={runId} className="mx-auto my-auto flex w-full max-w-4xl flex-col gap-4">
          <Slider
            label={`layer (0–${max})`}
            value={state.layer}
            min={min}
            max={max}
            step={1}
            onChange={(value) => dispatch({ type: 'SET_LAYER', value: Math.round(value) })}
            format={(value) => `${Math.round(value)}`}
          />

          {state.mode === 'find-behaviour' && (
            <FindBehaviourBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'classify-all' && (
            <ClassifyAllBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'hunt-dependency' && (
            <HuntDependencyBoard state={state} dispatch={dispatch} loadSentence={loadSentence} />
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

/* ── shared: the twelve-matrix grid ─────────────────────────── */

function MiniMatrix({ matrix }: { matrix: number[][] }) {
  const n = matrix.length;
  if (n === 0) return null;
  return (
    <div
      className="grid aspect-square w-full"
      style={{ gridTemplateColumns: `repeat(${n}, 1fr)`, gridTemplateRows: `repeat(${n}, 1fr)`, gap: '1px' }}
    >
      {matrix.map((row, i) => {
        const rowMax = Math.max(...row, 1e-9);
        return row.map((value, j) => (
          <div
            key={`${i}-${j}`}
            style={{
              background: `color-mix(in oklab, var(--accent) ${Math.round(Math.min(1, value / rowMax) * 82 + 6)}%, var(--surface-inset))`,
            }}
          />
        ));
      })}
    </div>
  );
}

function HeadGrid({
  headCount,
  matrixFor,
  onSelect,
  disabled,
  tone,
  badge,
  ariaLabel,
}: {
  headCount: number;
  matrixFor: (head: number) => number[][];
  onSelect: (head: number) => void;
  disabled?: (head: number) => boolean;
  tone: (head: number) => 'neutral' | 'accent' | 'good' | 'bad';
  badge?: (head: number) => ReactNode;
  ariaLabel: (head: number) => string;
}) {
  const toneClass: Record<ReturnType<typeof tone>, string> = {
    neutral: 'border-line-strong hover:border-accent',
    accent: 'border-accent bg-accent-dim',
    good: 'border-good bg-good/10',
    bad: 'border-bad bg-bad/10',
  };
  return (
    <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
      {Array.from({ length: headCount }, (_, head) => head).map((head) => (
        <button
          key={head}
          type="button"
          onClick={() => onSelect(head)}
          disabled={disabled?.(head)}
          aria-label={ariaLabel(head)}
          className={cx(
            'flex flex-col items-center gap-1 border p-1.5 transition-colors',
            'disabled:cursor-not-allowed disabled:opacity-50',
            toneClass[tone(head)]
          )}
          style={{ borderRadius: 'var(--radius)' }}
        >
          <MiniMatrix matrix={matrixFor(head)} />
          <span className="label text-[9px]">H{head}</span>
          {badge?.(head)}
        </button>
      ))}
    </div>
  );
}

function LabeledMatrix({
  tokens,
  matrix,
  highlightFrom,
  highlightTo,
}: {
  tokens: string[];
  matrix: number[][];
  highlightFrom?: number | null;
  highlightTo?: number | null;
}) {
  return (
    <Panel label="enlarged" flush>
      <div className="overflow-x-auto p-2">
        <table className="border-collapse">
          <caption className="sr-only">Real attention weights between every pair of {tokens.length} tokens</caption>
          <thead>
            <tr>
              <th className="p-1" />
              {tokens.map((token, j) => (
                <th key={j} scope="col" className="px-[1px] pb-1 text-center font-mono text-[9px] font-normal text-muted">
                  {token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((rowToken, i) => {
              const row = matrix[i] ?? [];
              const rowMax = Math.max(...row, 1e-9);
              return (
                <tr key={i}>
                  <th
                    scope="row"
                    className={cx(
                      'whitespace-nowrap py-[1px] pr-2 text-right font-mono text-[10px] font-normal',
                      i === highlightFrom ? 'text-accent' : 'text-secondary'
                    )}
                  >
                    {rowToken}
                  </th>
                  {tokens.map((colToken, j) => {
                    const value = row[j] ?? 0;
                    const intensity = Math.min(1, value / rowMax);
                    const isPair = i === highlightFrom && j === highlightTo;
                    return (
                      <td key={j} className="p-[1px]">
                        <div
                          title={`${rowToken} → ${colToken}: ${(value * 100).toFixed(1)}%`}
                          className={cx('flex h-5 w-5 items-center justify-center', isPair && 'ring-2 ring-accent')}
                          style={{
                            borderRadius: '2px',
                            background: `color-mix(in oklab, var(--accent) ${Math.round(intensity * 82 + 6)}%, var(--surface-inset))`,
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

function PromptLine({
  tokens,
  onTokenClick,
  toneFor,
}: {
  tokens: string[];
  onTokenClick?: (index: number) => void;
  toneFor?: (index: number) => 'from' | 'to' | 'pending' | null;
}) {
  const toneClass = {
    from: 'bg-accent-dim px-1 text-accent',
    to: 'bg-good/20 px-1 text-good',
    pending: 'border border-accent px-1 text-accent',
  };
  return (
    <p
      className="flex flex-wrap gap-1 border border-line bg-inset px-2 py-1.5 font-mono text-xs leading-relaxed"
      style={{ borderRadius: 'var(--radius)' }}
    >
      {tokens.map((token, i) => {
        const tone = toneFor?.(i) ?? null;
        return onTokenClick ? (
          <button
            key={i}
            type="button"
            onClick={() => onTokenClick(i)}
            className={cx('rounded-sm text-primary hover:text-accent', tone && toneClass[tone])}
          >
            {token}
          </button>
        ) : (
          <span key={i} className="text-primary">
            {token}
          </span>
        );
      })}
    </p>
  );
}

/* ── level 1: spot the specialist ───────────────────────────── */

function FindBehaviourBoard({
  state,
  dispatch,
  revealed,
}: {
  state: MultiHeadDetectiveState;
  dispatch: (action: MultiHeadDetectiveAction) => void;
  revealed: boolean;
}) {
  const [roundIndex, setRoundIndex] = useState(0);
  const [focusHead, setFocusHead] = useState<number | null>(null);
  const nomination = state.nominations[roundIndex] ?? null;

  const matrixFor = useCallback(
    (head: number) => state.result.attention[state.layer]?.[head] ?? [],
    [state.result, state.layer]
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <span className="label">
          round {roundIndex + 1} / {state.nominations.length}
        </span>
        <div className="flex items-center gap-1">
          {state.nominations.map((nom, i) => {
            const correct = revealed && nom !== null ? state.targetHeads.includes(nom) : null;
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
                        : nom !== null
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

      <PromptLine tokens={state.tokens} />

      <Panel label={`find the ${state.config.targetBehaviour} head`}>
        <HeadGrid
          headCount={state.heads.length}
          matrixFor={matrixFor}
          disabled={() => revealed}
          onSelect={(head) => {
            setFocusHead(head);
            dispatch({ type: 'NOMINATE_HEAD', roundIndex, head });
          }}
          tone={(head) => {
            if (revealed) {
              if (state.targetHeads.includes(head)) return 'good';
              if (nomination === head) return 'bad';
              return 'neutral';
            }
            return nomination === head ? 'accent' : 'neutral';
          }}
          ariaLabel={(head) => `Head ${head}, nominate for round ${roundIndex + 1}`}
        />
      </Panel>

      {focusHead !== null && <LabeledMatrix tokens={state.tokens} matrix={matrixFor(focusHead)} />}

      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? `At layer ${state.layer}, the head${state.targetHeads.length === 1 ? '' : 's'} highlighted green ${
              state.targetHeads.length === 1 ? 'is' : 'are'
            } classified "${state.config.targetBehaviour}" — measured from its real matrix, not authored.`
          : `Click a head's matrix to nominate it for this round. Moving the layer slider re-measures every head and clears your nominations, since a head's behaviour only makes sense at the layer it was measured at.`}
      </p>
    </div>
  );
}

/* ── level 2: classify every head ───────────────────────────── */

function ClassifyAllBoard({
  state,
  dispatch,
  revealed,
}: {
  state: MultiHeadDetectiveState;
  dispatch: (action: MultiHeadDetectiveAction) => void;
  revealed: boolean;
}) {
  const [focusHead, setFocusHead] = useState<number>(0);
  const margin = state.config.confidenceMargin ?? 0;
  const focused = state.heads.find((h) => h.head === focusHead);

  const matrixFor = useCallback(
    (head: number) => state.result.attention[state.layer]?.[head] ?? [],
    [state.result, state.layer]
  );

  return (
    <div className="flex flex-col gap-3">
      <PromptLine tokens={state.tokens} />

      <Panel label="classify every head">
        <HeadGrid
          headCount={state.heads.length}
          matrixFor={matrixFor}
          onSelect={setFocusHead}
          tone={(head) => {
            const h = state.heads.find((entry) => entry.head === head);
            if (revealed && h && h.confidence >= margin) {
              return h.answer === h.behaviour ? 'good' : 'bad';
            }
            return head === focusHead ? 'accent' : 'neutral';
          }}
          badge={(head) => {
            const h = state.heads.find((entry) => entry.head === head);
            if (!h) return null;
            const ungraded = h.confidence < margin;
            return (
              <span className={cx('readout text-[8px]', ungraded ? 'text-muted' : 'text-accent')}>
                {h.answer ? BEHAVIOUR_LABEL[h.answer] : '—'}
                {ungraded && '*'}
              </span>
            );
          }}
          ariaLabel={(head) => `Head ${head}, select to label`}
        />
      </Panel>

      {focused && (
        <Panel
          label={`head ${focused.head}${focused.confidence < margin ? ' — below the confidence margin, not graded' : ''}`}
        >
          <div className="flex flex-wrap gap-1.5">
            {state.config.behaviours.map((behaviour) => (
              <Button
                key={behaviour}
                variant={focused.answer === behaviour ? 'primary' : 'ghost'}
                disabled={revealed}
                onClick={() => dispatch({ type: 'LABEL_HEAD', head: focused.head, behaviour })}
              >
                {behaviour}
              </Button>
            ))}
          </div>
          {revealed && (
            <p className="mt-2 text-xs text-secondary">
              real behaviour: <span className="readout text-accent">{focused.behaviour}</span> (confidence{' '}
              {focused.confidence.toFixed(2)})
            </p>
          )}
        </Panel>
      )}

      <LabeledMatrix tokens={state.tokens} matrix={matrixFor(focusHead)} />

      <p className="text-xs leading-relaxed text-muted">
        A delimiter head lights up a whole column at [CLS] or [SEP] regardless of the row. A diffuse head has no
        strong single pattern. Heads marked * fall below the confidence margin and aren&apos;t graded either way.
      </p>
    </div>
  );
}

/* ── level 3: hunt the dependency ───────────────────────────── */

function HuntDependencyBoard({
  state,
  dispatch,
  loadSentence,
}: {
  state: MultiHeadDetectiveState;
  dispatch: (action: MultiHeadDetectiveAction) => void;
  loadSentence: (sentence: string) => Promise<void>;
}) {
  const [pendingFrom, setPendingFrom] = useState<number | null>(null);
  const [focusHead, setFocusHead] = useState<number | null>(null);
  const [sentence, setSentence] = useState(state.config.sentence);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const maxAttempts = state.config.attempts ?? Infinity;
  const atLimit = state.attempts.length >= maxAttempts;
  const dep = state.dependency;

  const matrixFor = useCallback(
    (head: number) => state.result.attention[state.layer]?.[head] ?? [],
    [state.result, state.layer]
  );

  const handleTokenClick = (index: number) => {
    if (pendingFrom === null) {
      setPendingFrom(index);
      return;
    }
    if (pendingFrom === index) {
      setPendingFrom(null);
      return;
    }
    dispatch({ type: 'SET_DEPENDENCY', fromIndex: pendingFrom, toIndex: index });
    setPendingFrom(null);
  };

  const attemptFor = (head: number) =>
    [...state.attempts].reverse().find((a) => a.layer === state.layer && a.head === head) ?? null;

  const tryCustomSentence = async () => {
    const text = sentence.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    try {
      await loadSentence(text);
      setPendingFrom(null);
      setFocusHead(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      {state.config.allowUserSentence && (
        <div className="flex flex-wrap items-center gap-1.5">
          <input
            value={sentence}
            onChange={(event) => setSentence(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && void tryCustomSentence()}
            disabled={busy}
            maxLength={220}
            aria-label="Your sentence"
            className="min-w-0 flex-1 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent disabled:opacity-60"
            style={{ borderRadius: 'var(--radius)' }}
          />
          <Button onClick={() => void tryCustomSentence()} disabled={busy || sentence.trim() === ''}>
            <Play size={13} strokeWidth={2} />
            {busy ? 'running' : 'load sentence'}
          </Button>
        </div>
      )}
      {error && <p className="readout text-xs text-bad">the model could not run that — {error}</p>}

      <Panel label="pick two tokens: the dependency you want to trace">
        <PromptLine
          tokens={state.tokens}
          onTokenClick={handleTokenClick}
          toneFor={(i) => {
            if (i === pendingFrom) return 'pending';
            if (dep?.fromIndex === i) return 'from';
            if (dep?.toIndex === i) return 'to';
            return null;
          }}
        />
        <p className="mt-2 text-xs text-muted">
          {pendingFrom !== null
            ? `From "${state.tokens[pendingFrom]}" — click the token it should point to.`
            : dep
              ? `From "${state.tokens[dep.fromIndex]}" to "${state.tokens[dep.toIndex]}". Click a token to trace a different pair.`
              : 'Click a token to start, then click a second one to complete the pair.'}
        </p>
      </Panel>

      <Panel
        label={`probe a head (layer ${state.layer}) — ${state.attempts.length} / ${
          Number.isFinite(maxAttempts) ? maxAttempts : '∞'
        } attempts used`}
      >
        <HeadGrid
          headCount={12}
          matrixFor={matrixFor}
          disabled={() => !dep || atLimit}
          onSelect={(head) => {
            setFocusHead(head);
            if (!dep || atLimit) return;
            dispatch({ type: 'PROBE_HEAD', layer: state.layer, head });
          }}
          tone={(head) => (head === focusHead ? 'accent' : 'neutral')}
          badge={(head) => {
            const attempt = attemptFor(head);
            return attempt ? (
              <span className="readout text-[8px] text-accent">{(attempt.weight * 100).toFixed(0)}%</span>
            ) : null;
          }}
          ariaLabel={(head) => `Head ${head} at layer ${state.layer}, probe`}
        />
      </Panel>

      {focusHead !== null && (
        <LabeledMatrix
          tokens={state.tokens}
          matrix={matrixFor(focusHead)}
          highlightFrom={dep?.fromIndex}
          highlightTo={dep?.toIndex}
        />
      )}

      <AttemptLog attempts={state.attempts} bestWeight={state.bestDependencyWeight} />

      <p className="text-xs leading-relaxed text-muted">
        Attention isn&apos;t symmetric — probing A→B can be very different from B→A. The search spans every layer,
        not just the one shown, so move the slider between probes.
      </p>
    </div>
  );
}

function AttemptLog({
  attempts,
  bestWeight,
}: {
  attempts: { layer: number; head: number; weight: number }[];
  bestWeight: number;
}) {
  if (attempts.length === 0) return null;
  return (
    <Panel label="probes" actions={<Tag tone={bestWeight > 0 ? 'good' : 'neutral'}>best {bestWeight.toFixed(3)}</Tag>}>
      <ol className="flex flex-col gap-1.5">
        {attempts.map((attempt, i) => (
          <li key={i} className="flex items-center gap-2 text-xs">
            <span className="readout text-[10px] text-secondary">
              layer {attempt.layer} · head {attempt.head}
            </span>
            <span className="readout ml-auto text-[10px] text-accent">{(attempt.weight * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: MultiHeadDetectiveState;
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

function submitBlocked(state: MultiHeadDetectiveState): boolean {
  switch (state.mode) {
    case 'find-behaviour':
      return state.nominations.some((n) => n === null);
    case 'classify-all':
      return state.heads.some((h) => h.answer === null);
    case 'hunt-dependency':
      return false;
  }
}

function submitHint(state: MultiHeadDetectiveState): string {
  switch (state.mode) {
    case 'find-behaviour': {
      const left = state.nominations.filter((n) => n === null).length;
      return `${left} round${left === 1 ? '' : 's'} left`;
    }
    case 'classify-all': {
      const left = state.heads.filter((h) => h.answer === null).length;
      return `${left} head${left === 1 ? '' : 's'} left`;
    }
    case 'hunt-dependency':
      return '';
  }
}
