import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type ClusterPlacementConfig,
} from '@/engines/clusterPlacementEngine';
import type { Embedder } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-1-fundamentals/1-1-vectors.json';

/**
 * Fake embedder with a planted structure: words starting with a/b/c land in
 * three well-separated regions. Nothing about the engine knows this — it has to
 * discover the grouping from the vectors, exactly as it does with real ones.
 */
const fakeEmbedder: Embedder = {
  async embed(texts: string[]) {
    return texts.map((t) => {
      const seed = t.charCodeAt(0);
      const group = seed % 3;
      const jitter = ((seed * 13) % 7) / 100;
      return [group * 10 + jitter, group * 10 - jitter, 1 + jitter];
    });
  },
};

const rulesFor = (levelIndex: number): EngineRules => {
  const level = game.levels[levelIndex]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};

const configFor = (levelIndex: number) =>
  game.levels[levelIndex]!.engineConfig as unknown as ClusterPlacementConfig;

describe('clusterPlacementEngine — prepare', () => {
  it('projects real embeddings to 2D and discovers clusters from them', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });

    expect(prepared.items).toHaveLength(config.words.length);
    for (const item of prepared.items) {
      expect(Number.isFinite(item.trueX)).toBe(true);
      expect(Number.isFinite(item.trueY)).toBe(true);
      expect(item.trueMagnitude).toBeGreaterThan(0);
      expect(item.trueCluster).toBeGreaterThanOrEqual(0);
      expect(item.trueCluster).toBeLessThan(config.clusterCount);
    }
  });

  it('groups words the fake model placed together into the same cluster', async () => {
    const prepared = await prepare(
      { ...configFor(0), words: ['apple', 'ant', 'bear', 'bull'], clusterCount: 2 },
      { embedder: fakeEmbedder }
    );
    const byWord = Object.fromEntries(prepared.items.map((i) => [i.word, i.trueCluster]));
    expect(byWord['apple']).toBe(byWord['ant']);
    expect(byWord['bear']).toBe(byWord['bull']);
    expect(byWord['apple']).not.toBe(byWord['bear']);
  });

  it('derives label truth by embedding the labels themselves', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const clusters = new Set(prepared.items.map((i) => i.trueCluster));
    for (const cluster of clusters) {
      expect(config.labels).toContain(prepared.labelTruth[cluster]);
    }
  });

  it('is deterministic across runs with the same input', async () => {
    const config = configFor(0);
    const a = await prepare(config, { embedder: fakeEmbedder });
    const b = await prepare(config, { embedder: fakeEmbedder });
    expect(a.items.map((i) => i.trueCluster)).toEqual(b.items.map((i) => i.trueCluster));
  });

  it('rejects an embedder that returns the wrong number of vectors', async () => {
    const broken: Embedder = { async embed() { return [[1, 2, 3]]; } };
    await expect(prepare(configFor(0), { embedder: broken })).rejects.toThrow();
  });

  it('handles an empty word list without crashing', async () => {
    const prepared = await prepare(
      { ...configFor(0), words: [] },
      { embedder: fakeEmbedder }
    );
    expect(prepared.items).toEqual([]);
  });
});

describe('clusterPlacementEngine — initState', () => {
  it('starts idle with nothing placed', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);

    expect(state.status).toBe('idle');
    expect(state.actionCount).toBe(0);
    expect(state.items).toHaveLength(config.words.length);
    expect(state.items.every((i) => i.placedX === null && i.placedY === null)).toBe(true);
    expect(state.items.every((i) => i.assignedLabel === null)).toBe(true);
  });

  it('carries the mode and rules through', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.mode).toBe('guess-the-label');
    expect(state.rules.xpReward).toBe(game.levels[1]!.xpReward);
  });
});

describe('clusterPlacementEngine — applyAction', () => {
  it('records a placement and moves to active', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    const id = state.items[0]!.id;

    state = applyAction(state, { type: 'PLACE', id, x: 10, y: 20 });
    expect(state.status).toBe('active');
    expect(state.actionCount).toBe(1);
    expect(state.items[0]!.placedX).toBe(10);
    expect(state.items[0]!.placedY).toBe(20);
  });

  it('does not mutate the previous state', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const before = initState(config, rulesFor(0), prepared);
    const after = applyAction(before, { type: 'PLACE', id: before.items[0]!.id, x: 5, y: 5 });

    expect(before.items[0]!.placedX).toBeNull();
    expect(after).not.toBe(before);
    expect(after.items).not.toBe(before.items);
  });

  it('clamps placements to the canvas instead of accepting off-board coordinates', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    const id = state.items[0]!.id;

    state = applyAction(state, { type: 'PLACE', id, x: -50, y: 9999 });
    expect(state.items[0]!.placedX).toBe(0);
    expect(state.items[0]!.placedY).toBe(config.canvas.height);
  });

  it('ignores actions targeting an unknown id', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    const next = applyAction(state, { type: 'PLACE', id: 'nope', x: 1, y: 1 });
    expect(next.items).toEqual(state.items);
    expect(next.actionCount).toBe(state.actionCount);
  });

  it('rejects a label that is not in the configured label set', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    const id = state.items[0]!.id;

    state = applyAction(state, { type: 'ASSIGN_LABEL', id, label: 'not-a-label' });
    expect(state.items[0]!.assignedLabel).toBeNull();
  });

  it('accepts a valid label assignment', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    const id = state.items[0]!.id;

    state = applyAction(state, { type: 'ASSIGN_LABEL', id, label: config.labels![0]! });
    expect(state.items[0]!.assignedLabel).toBe(config.labels![0]);
  });

  it('records magnitude guesses and refuses negative values', async () => {
    const config = configFor(2);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    const id = state.items[0]!.id;

    state = applyAction(state, { type: 'GUESS_MAGNITUDE', id, value: 2.5 });
    expect(state.items[0]!.magnitudeGuess).toBe(2.5);

    state = applyAction(state, { type: 'GUESS_MAGNITUDE', id, value: -1 });
    expect(state.items[0]!.magnitudeGuess).toBe(2.5);
  });

  it('adds a user-supplied word only when the level allows it', async () => {
    const openConfig = configFor(2);
    const preparedOpen = await prepare(openConfig, { embedder: fakeEmbedder });
    let open = initState(openConfig, rulesFor(2), preparedOpen);
    const added = (await prepare({ ...openConfig, words: ['zebra'] }, { embedder: fakeEmbedder }))
      .items[0]!;
    open = applyAction(open, { type: 'ADD_ITEM', item: added });
    expect(open.items.some((i) => i.word === 'zebra')).toBe(true);

    const closedConfig = configFor(0);
    const preparedClosed = await prepare(closedConfig, { embedder: fakeEmbedder });
    let closed = initState(closedConfig, rulesFor(0), preparedClosed);
    const count = closed.items.length;
    closed = applyAction(closed, { type: 'ADD_ITEM', item: added });
    expect(closed.items).toHaveLength(count);
  });

  it('resets placements back to the initial state', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'PLACE', id: state.items[0]!.id, x: 3, y: 4 });
    state = applyAction(state, { type: 'RESET' });

    expect(state.items.every((i) => i.placedX === null)).toBe(true);
    expect(state.status).toBe('idle');
  });

  it('marks the run complete on submit', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});

describe('clusterPlacementEngine — evaluate: place-and-cluster', () => {
  const config = { ...configFor(0), words: ['apple', 'ant', 'bear', 'bull'], clusterCount: 2 };

  it('scores a layout that mirrors the real structure highly', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);

    // Place each item near a corner chosen by its real cluster.
    for (const item of state.items) {
      const x = item.trueCluster === 0 ? 10 : 90;
      const y = item.trueCluster === 0 ? 10 : 90;
      state = applyAction(state, { type: 'PLACE', id: item.id, x, y });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('clusterSeparationScore');
    expect(result.value).toBeGreaterThan(0.88);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
    expect(result.xpEarned).toBe(game.levels[0]!.xpReward);
  });

  it('scores an interleaved layout poorly', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);

    state.items.forEach((item, i) => {
      state = applyAction(state, { type: 'PLACE', id: item.id, x: i % 2 === 0 ? 10 : 90, y: 50 });
    });

    const result = evaluate(state);
    expect(result.value).toBeLessThan(0.55);
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
    expect(result.xpEarned).toBe(0);
  });

  it('penalises leaving items unplaced', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let full = initState(config, rulesFor(0), prepared);
    let partial = initState(config, rulesFor(0), prepared);

    for (const item of full.items) {
      const x = item.trueCluster === 0 ? 10 : 90;
      full = applyAction(full, { type: 'PLACE', id: item.id, x, y: x });
    }
    for (const item of partial.items.slice(0, 2)) {
      const x = item.trueCluster === 0 ? 10 : 90;
      partial = applyAction(partial, { type: 'PLACE', id: item.id, x, y: x });
    }

    expect(evaluate(partial).value).toBeLessThan(evaluate(full).value);
  });

  it('scores zero when nothing has been placed', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('reports placement counts in the breakdown', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'PLACE', id: state.items[0]!.id, x: 1, y: 1 });

    const result = evaluate(state);
    expect(result.breakdown.placed).toBe(1);
    expect(result.breakdown.total).toBe(config.words.length);
  });
});

describe('clusterPlacementEngine — evaluate: guess-the-label', () => {
  const config = configFor(1);

  it('awards full accuracy when every label matches the model-derived truth', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    for (const item of state.items) {
      state = applyAction(state, {
        type: 'ASSIGN_LABEL',
        id: item.id,
        label: prepared.labelTruth[item.trueCluster]!,
      });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('labelAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('gives partial credit for partially correct labelling', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    state.items.forEach((item, i) => {
      const correct = prepared.labelTruth[item.trueCluster]!;
      const wrong = config.labels!.find((l) => l !== correct)!;
      state = applyAction(state, {
        type: 'ASSIGN_LABEL',
        id: item.id,
        label: i % 2 === 0 ? correct : wrong,
      });
    });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(0);
    expect(result.value).toBeLessThan(1);
  });

  it('scores unassigned items as wrong rather than skipping them', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    const first = state.items[0]!;
    state = applyAction(state, {
      type: 'ASSIGN_LABEL',
      id: first.id,
      label: prepared.labelTruth[first.trueCluster]!,
    });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(1 / state.items.length);
  });
});

describe('clusterPlacementEngine — evaluate: drag-vector-magnitude', () => {
  const config = configFor(2);

  it('scores an exact magnitude guess at 1', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    for (const item of state.items) {
      state = applyAction(state, {
        type: 'GUESS_MAGNITUDE',
        id: item.id,
        value: item.trueMagnitude,
      });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('magnitudeAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('falls off with relative error and floors at zero', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    for (const item of state.items) {
      state = applyAction(state, {
        type: 'GUESS_MAGNITUDE',
        id: item.id,
        value: item.trueMagnitude * 20,
      });
    }

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('treats an unanswered round as zero credit', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(0);
  });
});

describe('clusterPlacementEngine — level config coverage', () => {
  it('evaluates every level in the shipped chapter without throwing', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as ClusterPlacementConfig;
      const prepared = await prepare(config, { embedder: fakeEmbedder });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
