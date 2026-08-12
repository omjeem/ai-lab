'use client';

/**
 * Chapter 1.4 — Tokenization.
 *
 * The whole chapter is about pieces, so the pieces are the interface: every
 * token is a discrete chip, adjacent chips alternate tone so a boundary is
 * never ambiguous, and whitespace is drawn rather than left invisible — half
 * the surprises in this chapter are about spaces.
 *
 * Counting withholds the chips until the guess is in, for the same reason 1.3
 * withholds its matrix: showing them first turns the level into reading off an
 * answer. The merge puzzle and the fertility hunt are live, because there the
 * feedback *is* the lesson.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, CornerDownLeft, Merge as MergeIcon, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type PuzzleState,
  type SampleState,
  type TokenMergeAction,
  type TokenMergeConfig,
  type TokenMergeState,
} from '@/engines/tokenMergeEngine';
import { tokenizerModel, prettifyToken, TOKENIZER_MODEL_ID } from '@/models/tokenizerModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Meter, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function TokenizationCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as TokenMergeConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<TokenMergeState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { tokenizer: tokenizerModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: TokenMergeAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={TOKENIZER_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} dispatch={dispatch} onScore={onScore} onSubmit={onSubmit} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  onScore,
  onSubmit,
}: {
  state: TokenMergeState;
  dispatch: (action: TokenMergeAction) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  /**
   * Local rather than read off `state.status`: engine actions set the status
   * back to `active`, so logging another attempt after submitting would quietly
   * un-reveal the tokens.
   */
  const [revealed, setRevealed] = useState(false);
  const [runId, setRunId] = useState(0);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
    setRunId((n) => n + 1);
  }, [dispatch]);

  // A merge puzzle played to the end cannot be replayed without this.
  useRetrySignal(resetRun);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <div key={runId} className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'count-tokens' && (
            <CountBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'merge-puzzle' && (
            <MergeBoard state={state} dispatch={dispatch} revealed={revealed} />
          )}
          {state.mode === 'break-tokenizer' && <BreakBoard state={state} dispatch={dispatch} />}
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

/* ── level 1: count the tokens ──────────────────────────────── */

function CountBoard({
  state,
  dispatch,
  revealed,
}: {
  state: TokenMergeState;
  dispatch: (action: TokenMergeAction) => void;
  revealed: boolean;
}) {
  const tolerance = state.config.tolerance ?? 0;

  return (
    <div className="flex flex-col gap-3">
      {state.samples.map((sample, index) => (
        <SampleRow
          key={sample.text}
          sample={sample}
          tolerance={tolerance}
          revealed={revealed}
          onGuess={(value) => dispatch({ type: 'GUESS_COUNT', sampleIndex: index, value })}
        />
      ))}
      <p className="text-xs leading-relaxed text-muted">
        {revealed
          ? 'Every chip below is one token the real GPT-2 tokenizer produced. Spaces belong to the token that follows them.'
          : `Within ±${tolerance} counts as correct. The tokens stay hidden until you submit — otherwise there is nothing to estimate.`}
      </p>
    </div>
  );
}

function SampleRow({
  sample,
  tolerance,
  revealed,
  onGuess,
}: {
  sample: SampleState;
  tolerance: number;
  revealed: boolean;
  onGuess: (value: number) => void;
}) {
  const [draft, setDraft] = useState(sample.guess === null ? '' : String(sample.guess));
  const correct = sample.guess !== null && Math.abs(sample.guess - sample.count) <= tolerance;

  const commit = (raw: string) => {
    setDraft(raw);
    const value = Number.parseInt(raw, 10);
    if (Number.isInteger(value) && value >= 0) onGuess(value);
  };

  return (
    <Panel
      label={`sample · ${[...sample.text].length} characters`}
      actions={
        revealed ? (
          <Tag tone={correct ? 'good' : 'bad'}>
            {correct ? 'within tolerance' : `really ${sample.count}`}
          </Tag>
        ) : null
      }
    >
      <div className="flex flex-col gap-2.5">
        {/* The raw string, with its whitespace made visible before anything
            is tokenised — the spacing sample is meaningless otherwise. */}
        <pre className="overflow-x-auto whitespace-pre-wrap break-words border border-line bg-inset px-2 py-1.5 font-mono text-xs text-primary" style={{ borderRadius: 'var(--radius)' }}>
          {visibleWhitespace(sample.text)}
        </pre>

        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2">
            <span className="label">your count</span>
            <input
              type="number"
              min={0}
              max={999}
              inputMode="numeric"
              value={draft}
              disabled={revealed}
              onChange={(event) => commit(event.target.value)}
              aria-label={`Token count for ${sample.text}`}
              className="w-20 border border-line bg-inset px-2 py-1.5 font-mono text-xs text-primary focus:border-accent disabled:opacity-60"
              style={{ borderRadius: 'var(--radius)' }}
            />
          </label>

          {revealed && (
            <span className="readout text-xs text-secondary">
              actual {sample.count} · {(sample.count / [...sample.text].length).toFixed(2)} tokens
              per character
            </span>
          )}
        </div>

        {revealed && <TokenChips tokens={sample.tokens} />}
      </div>
    </Panel>
  );
}

/* ── level 2: the merge puzzle ──────────────────────────────── */

function MergeBoard({
  state,
  dispatch,
  revealed,
}: {
  state: TokenMergeState;
  dispatch: (action: TokenMergeAction) => void;
  revealed: boolean;
}) {
  /** Which position was just wrong, per puzzle — cleared on the next merge. */
  const [missed, setMissed] = useState<Record<number, number | null>>({});

  return (
    <div className="flex flex-col gap-3">
      {state.puzzles.map((puzzle, index) => (
        <PuzzleRow
          key={puzzle.word}
          puzzle={puzzle}
          revealed={revealed}
          missedAt={missed[index] ?? null}
          onMerge={(position) => {
            // The engine accepts a wrong merge and lets the board change, which
            // is what makes the mistake legible. Work out which it was here so
            // the player is told, without the next answer being given away.
            const expected = puzzle.merges[puzzle.stepIndex];
            const wrong =
              expected === undefined ||
              expected.left !== puzzle.symbols[position] ||
              expected.right !== puzzle.symbols[position + 1];
            setMissed((prev) => ({ ...prev, [index]: wrong ? position : null }));
            dispatch({ type: 'APPLY_MERGE', puzzleIndex: index, position });
          }}
        />
      ))}
      <p className="text-xs leading-relaxed text-muted">
        At each step the real tokenizer merges the adjacent pair with the lowest merge rank — the
        pair it saw most often while the vocabulary was being built. Lower rank means learned earlier.
      </p>
    </div>
  );
}

function PuzzleRow({
  puzzle,
  revealed,
  missedAt,
  onMerge,
}: {
  puzzle: PuzzleState;
  revealed: boolean;
  missedAt: number | null;
  onMerge: (position: number) => void;
}) {
  const reduce = useReducedMotion();
  const locked = puzzle.done || revealed;

  return (
    <Panel
      label={`"${puzzle.word}" · step ${Math.min(puzzle.stepIndex + 1, puzzle.merges.length)} of ${puzzle.merges.length}`}
      actions={
        <span className="flex items-center gap-2">
          {puzzle.attemptedSteps > 0 && (
            <span className="label">
              {puzzle.correctSteps}/{puzzle.attemptedSteps} right
            </span>
          )}
          {puzzle.done && (
            <Tag tone={puzzle.correctSteps === puzzle.merges.length ? 'good' : 'warn'}>
              {puzzle.correctSteps === puzzle.merges.length ? 'perfect' : 'finished'}
            </Tag>
          )}
        </span>
      }
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-1">
          {puzzle.symbols.map((symbol, i) => (
            <span key={`${symbol}-${i}`} className="flex items-center gap-1">
              <motion.span
                layout={!reduce}
                transition={reduce ? { duration: 0 } : { duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                className={cx(
                  'border px-2 py-1.5 font-mono text-sm',
                  missedAt !== null && (i === missedAt || i === missedAt + 1)
                    ? 'border-bad text-bad'
                    : 'border-line-strong bg-raised text-primary'
                )}
                style={{ borderRadius: 'var(--radius)' }}
              >
                {visibleWhitespace(prettifyToken(symbol))}
              </motion.span>

              {/* The join between two chips is the control — clicking it is
                  literally "merge these two". */}
              {i < puzzle.symbols.length - 1 && (
                <button
                  type="button"
                  disabled={locked}
                  onClick={() => onMerge(i)}
                  aria-label={`Merge "${puzzle.symbols[i]}" and "${puzzle.symbols[i + 1]}"`}
                  title="Merge this pair"
                  className={cx(
                    'flex h-6 w-5 items-center justify-center text-muted transition-colors',
                    !locked && 'hover:text-accent',
                    locked && 'opacity-30'
                  )}
                >
                  <MergeIcon size={12} strokeWidth={2} className="rotate-90" />
                </button>
              )}
            </span>
          ))}
        </div>

        {missedAt !== null && !puzzle.done && (
          <p className="text-xs text-bad">
            That pair is not the one the tokenizer merges next — but it is merged now, so the board
            has moved on.
          </p>
        )}

        {(puzzle.done || revealed) && (
          <ol className="flex flex-col gap-1 border-t border-line-faint pt-2">
            <li className="label">gpt-2&rsquo;s real merge order</li>
            {puzzle.merges.map((merge, i) => (
              <li key={`${merge.left}-${merge.right}-${i}`} className="flex items-baseline gap-2">
                <span className="readout w-4 text-[10px] text-muted">{i + 1}</span>
                <span className="font-mono text-xs text-secondary">
                  {visibleWhitespace(prettifyToken(merge.left))} +{' '}
                  {visibleWhitespace(prettifyToken(merge.right))}
                </span>
                <span className="readout ml-auto text-[10px] text-muted">rank {merge.rank}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </Panel>
  );
}

/* ── level 3: break the tokenizer ───────────────────────────── */

const EFFICIENT_RATIO = 0.25;
const SHATTERED_RATIO = 1;

function BreakBoard({
  state,
  dispatch,
}: {
  state: TokenMergeState;
  dispatch: (action: TokenMergeAction) => void;
}) {
  const [text, setText] = useState('');
  const [tokens, setTokens] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const request = useRef(0);

  const minChars = state.config.minChars ?? 0;
  const maxChars = state.config.maxChars ?? Infinity;
  const maxAttempts = state.config.attempts ?? Infinity;

  const charCount = [...text].length;
  const tooShort = charCount < minChars;
  const tooLong = charCount > maxChars;
  const spent = state.attempts.length >= maxAttempts;

  /** Live tokenisation, debounced — every keystroke is a real tokenizer call. */
  useEffect(() => {
    if (text.length === 0) {
      setTokens([]);
      return;
    }
    const ticket = ++request.current;
    setBusy(true);
    const timer = setTimeout(() => {
      void tokenizerModel
        .tokenize(text)
        .then((result) => {
          if (ticket === request.current) setTokens(result);
        })
        .finally(() => {
          if (ticket === request.current) setBusy(false);
        });
    }, 200);
    return () => clearTimeout(timer);
  }, [text]);

  const ratio = charCount === 0 ? 0 : tokens.length / charCount;
  const fertility = Math.min(
    1,
    Math.max(0, (ratio - EFFICIENT_RATIO) / (SHATTERED_RATIO - EFFICIENT_RATIO))
  );
  const submittable = !tooShort && !tooLong && !spent && !busy && tokens.length > 0;

  return (
    <div className="flex flex-col gap-4">
      <Panel label={`your string · ${charCount} of ${maxChars} characters`}>
        <div className="flex flex-col gap-3">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={2}
            placeholder="type anything — scripts, symbols, emoji, nonsense"
            aria-label="Text to tokenize"
            className="w-full resize-none border border-line bg-inset px-2 py-2 font-mono text-sm text-primary placeholder:text-muted focus:border-accent"
            style={{ borderRadius: 'var(--radius)' }}
          />

          <div className="flex flex-wrap items-end justify-between gap-4">
            <Readout label="tokens" value={tokens.length} size="md" />
            <Readout label="tokens per character" value={ratio} size="md" />
            <Readout
              label="fertility"
              value={fertility}
              size="md"
              tone={fertility >= 0.9 ? 'good' : fertility >= 0.5 ? 'accent' : 'neutral'}
            />
          </div>

          <Meter
            value={fertility}
            max={1}
            threshold={0.5}
            label={`0.25 tokens/char is fluent English · 1.0 is one token per character`}
          />

          {tokens.length > 0 && <TokenChips tokens={tokens} />}

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              disabled={!submittable}
              title={
                spent
                  ? 'No attempts left'
                  : tooShort
                    ? `At least ${minChars} characters`
                    : tooLong
                      ? `At most ${maxChars} characters`
                      : undefined
              }
              onClick={() =>
                dispatch({
                  type: 'SUBMIT_ATTEMPT',
                  attempt: { text, tokenCount: tokens.length, charCount },
                })
              }
            >
              <CornerDownLeft size={13} strokeWidth={2} />
              Log attempt
            </Button>

            <span className="label">
              {spent
                ? 'no attempts left'
                : tooShort
                  ? `${minChars - charCount} more characters`
                  : tooLong
                    ? `${charCount - maxChars} over the limit`
                    : `${maxAttempts - state.attempts.length} attempts left`}
            </span>
          </div>
        </div>
      </Panel>

      {state.attempts.length > 0 && (
        <Panel label="logged attempts">
          <ol className="flex flex-col gap-1.5">
            {state.attempts.map((attempt, index) => (
              <li key={`${attempt.text}-${index}`} className="flex items-center gap-3">
                <span className="readout w-4 shrink-0 text-[10px] text-muted">{index + 1}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-xs text-secondary">
                  {visibleWhitespace(attempt.text)}
                </span>
                <span className="readout shrink-0 text-xs text-primary">
                  {attempt.tokenCount}/{attempt.charCount} ={' '}
                  {(attempt.tokenCount / attempt.charCount).toFixed(2)}
                </span>
              </li>
            ))}
          </ol>
        </Panel>
      )}

      <p className="text-xs leading-relaxed text-muted">
        Only your best attempt counts. Fluent English costs about a quarter of a token per character;
        anything the vocabulary never learned falls back to raw bytes, and a single character can cost
        several tokens.
      </p>
    </div>
  );
}

/* ── tokens as pieces ───────────────────────────────────────── */

/**
 * The token strip.
 *
 * Adjacent chips alternate tone rather than being coloured by identity: the
 * only thing that has to be unambiguous is where one token stops and the next
 * begins, and a per-token hue would fight the world accent for no gain.
 */
function TokenChips({ tokens }: { tokens: string[] }) {
  return (
    <div className="flex flex-wrap items-center gap-1" role="list" aria-label={`${tokens.length} tokens`}>
      {tokens.map((token, index) => (
        <span
          key={`${token}-${index}`}
          role="listitem"
          title={`token ${index + 1}: ${JSON.stringify(token)}`}
          className="border border-line-strong px-1.5 py-1 font-mono text-xs text-primary"
          style={{
            borderRadius: 'var(--radius)',
            background: `color-mix(in oklab, var(--accent) ${index % 2 === 0 ? 26 : 12}%, var(--surface-inset))`,
          }}
        >
          {visibleWhitespace(prettifyToken(token))}
        </span>
      ))}
    </div>
  );
}

/**
 * Renders spaces and newlines as marks.
 *
 * Half this chapter's surprises are whitespace — a token that is nothing but a
 * space has to look like something.
 */
function visibleWhitespace(text: string): React.ReactNode {
  if (text.length === 0) return <span className="text-muted">∅</span>;

  return [...text].map((character, index) => {
    if (character === ' ') {
      return (
        <span key={index} className="text-muted/70">
          ␣
        </span>
      );
    }
    if (character === '\n') {
      return (
        <span key={index} className="text-muted/70">
          ⏎
        </span>
      );
    }
    return <span key={index}>{character}</span>;
  });
}

/* ── actions ────────────────────────────────────────────────── */

function Actions({
  state,
  revealed,
  onSubmit,
}: {
  state: TokenMergeState;
  revealed: boolean;
  onSubmit: () => void;
}) {
  const blocked = submitBlocked(state);
  return (
    <>
      {!revealed && blocked && <span className="label">{submitHint(state)}</span>}
      {revealed && <Tag tone="accent">tokens revealed</Tag>}
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

function submitBlocked(state: TokenMergeState): boolean {
  switch (state.mode) {
    case 'count-tokens':
      return state.samples.some((sample) => sample.guess === null);
    case 'merge-puzzle':
      return state.puzzles.some((puzzle) => !puzzle.done);
    case 'break-tokenizer':
      return state.attempts.length === 0;
  }
}

function submitHint(state: TokenMergeState): string {
  switch (state.mode) {
    case 'count-tokens': {
      const left = state.samples.filter((sample) => sample.guess === null).length;
      return `${left} estimate${left === 1 ? '' : 's'} left`;
    }
    case 'merge-puzzle': {
      const left = state.puzzles.filter((puzzle) => !puzzle.done).length;
      return `${left} word${left === 1 ? '' : 's'} left`;
    }
    case 'break-tokenizer':
      return 'log at least one attempt';
  }
}
