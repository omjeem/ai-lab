'use client';

/**
 * Chapter 1.1 — What is a Vector?
 *
 * The canvas is the interface. Words are embedded live by the real model,
 * projected to 2D, and dragged into place; the score comes from how well the
 * player's own layout separates the structure the model actually produced.
 *
 * Every drag target is keyboard-operable (Section 11.5) — select a word, move it
 * with the arrow keys. Dragging is the fast path, not the only path.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Check, Plus } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ClusterItem,
  type ClusterPlacementConfig,
  type ClusterPlacementState,
} from '@/engines/clusterPlacementEngine';
import { embeddingModel, EMBEDDING_MODEL_ID } from '@/models/embeddingModel';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Slider, cx } from '@/components/ui';
import { usePointerDrag } from '../usePointerDrag';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

const CANVAS = 100;

export function VectorCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as ClusterPlacementConfig;
  const rules: EngineRules = useMemo(
    () => ({
      passCriteria: level.passCriteria,
      starsRules: level.starsRules,
      xpReward: level.xpReward,
    }),
    [level]
  );

  const [state, setState] = useState<ClusterPlacementState | null>(null);

  const load = useCallback(async () => {
    const prepared = await prepare(config, { embedder: embeddingModel });
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  return (
    <ModelGate
      modelId={EMBEDDING_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board
          state={state}
          setState={setState}
          onScore={onScore}
          onSubmit={onSubmit}
          config={config}
        />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  setState,
  onScore,
  onSubmit,
  config,
}: {
  state: ClusterPlacementState;
  setState: (next: ClusterPlacementState) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
  config: ClusterPlacementConfig;
}) {
  const reduce = useReducedMotion();
  const surfaceRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [newWord, setNewWord] = useState('');
  const [busy, setBusy] = useState(false);
  const [ghost, setGhost] = useState<{ id: string; word: string; x: number; y: number } | null>(
    null
  );

  // The HUD tracks the score live, so every placement updates it.
  useEffect(() => onScore(evaluate(state)), [state, onScore]);

  const place = useCallback(
    (id: string, x: number, y: number) => setState(applyAction(state, { type: 'PLACE', id, x, y })),
    [state, setState]
  );

  const pointerPlace = useCallback(
    (id: string, clientX: number, clientY: number) => {
      const box = surfaceRef.current?.getBoundingClientRect();
      if (!box) return;
      place(id, ((clientX - box.left) / box.width) * CANVAS, ((clientY - box.top) / box.height) * CANVAS);
    },
    [place]
  );

  /** Arrow-key movement, the keyboard equivalent of dragging. */
  const nudge = useCallback(
    (item: ClusterItem, dx: number, dy: number) => {
      const step = 4;
      place(item.id, (item.placedX ?? CANVAS / 2) + dx * step, (item.placedY ?? CANVAS / 2) + dy * step);
    },
    [place]
  );

  const addWord = async () => {
    const word = newWord.trim().toLowerCase();
    if (!word || busy) return;
    setBusy(true);
    try {
      const prepared = await prepare({ ...config, words: [word] }, { embedder: embeddingModel });
      const item = prepared.items[0];
      if (item) setState(applyAction(state, { type: 'ADD_ITEM', item }));
      setNewWord('');
    } finally {
      setBusy(false);
    }
  };

  const unplaced = state.items.filter((i) => i.placedX === null);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* The plot is the dominant element, not a chart inside a card.
          A flex container here, not a plain block — `height: 100%` on the
          surface below never resolved against this div's own flex-computed
          height, since that height isn't "explicit" in the way percentage
          heights require. `flex-1` on the surface avoids that entirely.

          `min-h-60` (240px) is a real floor, not decoration: this div's
          flex-basis is 0%, so in a level whose controls below it are tall
          (guess-the-label's per-word buttons, one row per word) it is the
          first thing flexbox sacrifices — without a floor it had been
          measured shrinking to under 60px, no longer a usable drop target,
          while the controls below rendered at their full natural height. */}
      <div className="relative flex min-h-60 flex-1 flex-col p-3">
        <div
          ref={surfaceRef}
          className="grid-field relative min-h-0 w-full flex-1 border border-line bg-inset"
          style={{ borderRadius: 'var(--radius)' }}
        >
          {state.items
            .filter((item) => item.placedX !== null)
            .map((item) => (
              <PlacedWord
                key={item.id}
                item={item}
                reduce={reduce ?? false}
                selected={selected === item.id}
                onSelect={() => setSelected(selected === item.id ? null : item.id)}
                onNudge={(dx, dy) => nudge(item, dx, dy)}
                onMove={(x, y) => pointerPlace(item.id, x, y)}
              />
            ))}

          {ghost && (
            <div
              className="pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-1/2 border border-accent bg-accent-dim px-2 py-1 font-mono text-[11px] leading-none text-primary"
              style={{ left: ghost.x, top: ghost.y, borderRadius: 'var(--radius)' }}
            >
              {ghost.word}
            </div>
          )}

          {state.items.length > 0 && unplaced.length === state.items.length && (
            <p className="absolute inset-0 flex items-center justify-center text-center text-xs text-muted">
              Drag the words onto the field, or select one and use the arrow keys
            </p>
          )}
        </div>
      </div>

      {/* Controls sit under the instrument as a secondary layer.
          Extra horizontal padding, not just p-3's usual 12px: a chip that
          ends up flush against the screen edge on a narrow phone has its
          drag start inside iOS Safari's edge-swipe-back gesture zone, which
          wins over touch-action: none there — that gesture is recognised
          before the touch ever reaches page JS. */}
      <div className="flex flex-col gap-2 border-t border-line px-5 py-3">
        {unplaced.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {unplaced.map((item) => (
              <TrayChip
                key={item.id}
                item={item}
                onPlaceCenter={() => {
                  // Click-to-place drops it centre-field, then arrows refine it.
                  place(item.id, CANVAS / 2, CANVAS / 2);
                  setSelected(item.id);
                }}
                onGhostChange={(pos) => setGhost(pos ? { id: item.id, word: item.word, ...pos } : null)}
                onCommit={(x, y) => {
                  const box = surfaceRef.current?.getBoundingClientRect();
                  if (box && x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) {
                    pointerPlace(item.id, x, y);
                  }
                }}
              />
            ))}
          </div>
        )}

        {config.mode === 'guess-the-label' && <LabelControls state={state} setState={setState} />}
        {config.mode === 'drag-vector-magnitude' && (
          <MagnitudeControls state={state} setState={setState} />
        )}

        <div className="flex flex-wrap items-center gap-2">
          {config.allowUserWords && (
            <div className="flex items-center gap-1.5">
              <input
                value={newWord}
                onChange={(event) => setNewWord(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && void addWord()}
                placeholder="add any word"
                maxLength={40}
                className="w-36 border border-line bg-inset px-2 py-2 font-mono text-xs text-primary placeholder:text-muted focus:border-accent"
                style={{ borderRadius: 'var(--radius)' }}
              />
              <Button onClick={() => void addWord()} disabled={busy || newWord.trim().length === 0}>
                <Plus size={13} strokeWidth={2} />
                embed
              </Button>
            </div>
          )}

          <Button
            variant="primary"
            className="ml-auto"
            onClick={() => onSubmit(evaluate(state))}
          >
            <Check size={13} strokeWidth={2} />
            Submit layout
          </Button>
        </div>
      </div>
    </div>
  );
}

function PlacedWord({
  item,
  reduce,
  selected,
  onSelect,
  onNudge,
  onMove,
}: {
  item: ClusterItem;
  reduce: boolean;
  selected: boolean;
  onSelect: () => void;
  onNudge: (dx: number, dy: number) => void;
  onMove: (clientX: number, clientY: number) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const drag = usePointerDrag({
    onDragStart: () => setDragging(true),
    onDragMove: onMove,
    onDragEnd: () => setDragging(false),
  });

  return (
    <motion.button
      type="button"
      {...drag}
      onClick={onSelect}
      onKeyDown={(event) => {
        const map: Record<string, [number, number]> = {
          ArrowLeft: [-1, 0],
          ArrowRight: [1, 0],
          ArrowUp: [0, -1],
          ArrowDown: [0, 1],
        };
        const delta = map[event.key];
        if (!delta) return;
        event.preventDefault();
        onNudge(delta[0], delta[1]);
      }}
      aria-label={`${item.word}, at ${Math.round(item.placedX ?? 0)}, ${Math.round(item.placedY ?? 0)}. Arrow keys to move.`}
      className={cx(
        'absolute -translate-x-1/2 -translate-y-1/2 border px-2 py-1 font-mono text-[11px] leading-none',
        'cursor-grab active:cursor-grabbing',
        selected ? 'border-accent bg-accent-dim text-primary' : 'border-line-strong bg-raised text-secondary'
      )}
      style={{
        ...drag.style,
        left: `${item.placedX}%`,
        top: `${item.placedY}%`,
        borderRadius: 'var(--radius)',
      }}
      animate={{ left: `${item.placedX}%`, top: `${item.placedY}%` }}
      transition={dragging || reduce ? { duration: 0 } : { duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
    >
      {item.word}
    </motion.button>
  );
}

/**
 * A not-yet-placed word, in the tray under the plot.
 *
 * Dragging it doesn't place it until release — only then do we know whether
 * it landed on the plot at all — so a floating ghost (owned by `Board`,
 * since it has to render above the plot, not just this button) is the only
 * feedback during the drag itself.
 */
function TrayChip({
  item,
  onPlaceCenter,
  onGhostChange,
  onCommit,
}: {
  item: ClusterItem;
  onPlaceCenter: () => void;
  onGhostChange: (pos: { x: number; y: number } | null) => void;
  onCommit: (clientX: number, clientY: number) => void;
}) {
  const drag = usePointerDrag({
    onDragStart: (x, y) => onGhostChange({ x, y }),
    onDragMove: (x, y) => onGhostChange({ x, y }),
    onDragEnd: (x, y) => {
      onGhostChange(null);
      onCommit(x, y);
    },
  });

  return (
    <button
      type="button"
      {...drag}
      onClick={onPlaceCenter}
      className="border border-line-strong bg-raised px-2 py-1.5 font-mono text-xs text-primary transition-colors hover:border-accent"
      style={{ ...drag.style, borderRadius: 'var(--radius)' }}
    >
      {item.word}
    </button>
  );
}

function LabelControls({
  state,
  setState,
}: {
  state: ClusterPlacementState;
  setState: (next: ClusterPlacementState) => void;
}) {
  const labels = state.config.labels ?? [];
  return (
    <Panel label="assign labels">
      <div className="flex flex-col gap-1.5">
        {state.items.map((item) => (
          <div key={item.id} className="flex items-center gap-2">
            <span className="w-24 shrink-0 truncate font-mono text-xs text-primary">{item.word}</span>
            <div className="flex flex-wrap gap-1">
              {labels.map((label) => (
                <button
                  key={label}
                  type="button"
                  onClick={() =>
                    setState(applyAction(state, { type: 'ASSIGN_LABEL', id: item.id, label }))
                  }
                  className={cx(
                    'border px-1.5 py-1 font-mono text-[10px] uppercase tracking-wider',
                    item.assignedLabel === label
                      ? 'border-accent text-accent'
                      : 'border-line text-muted hover:border-line-strong'
                  )}
                  style={{ borderRadius: '2px' }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function MagnitudeControls({
  state,
  setState,
}: {
  state: ClusterPlacementState;
  setState: (next: ClusterPlacementState) => void;
}) {
  return (
    <Panel label="estimate magnitude">
      <div className="flex flex-col gap-3">
        {state.items.map((item) => (
          <Slider
            key={item.id}
            label={item.word}
            value={item.magnitudeGuess ?? 1}
            min={0}
            max={2}
            step={0.01}
            onChange={(value) =>
              setState(applyAction(state, { type: 'GUESS_MAGNITUDE', id: item.id, value }))
            }
          />
        ))}
      </div>
    </Panel>
  );
}
