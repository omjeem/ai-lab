import { describe, it, expect } from 'vitest';
import {
  initState,
  applyAction,
  evaluate,
  measureCheck,
  measureAxis,
  findForkIndex,
  cloudAvailable,
  type InspectorChatConfig,
} from '@/engines/inspectorChatEngine';
import type { GenerationTrace } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-6-capstone/6-1-inspector-chat.json';

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as InspectorChatConfig;

/** Builds a trace with exactly the properties a test needs to assert on. */
function makeTrace(steps: { token: string; probability: number; entropyBits: number }[]): GenerationTrace {
  return {
    text: steps.map((s) => s.token).join(''),
    steps: steps.map((s) => ({
      ...s,
      alternatives: { tokens: [s.token, 'other'], probs: [s.probability, 1 - s.probability] },
    })),
  };
}

const confidentTrace = makeTrace([
  { token: 'Paris', probability: 0.95, entropyBits: 0.3 },
  { token: '.', probability: 0.9, entropyBits: 0.4 },
]);

const uncertainTrace = makeTrace([
  { token: 'Maybe', probability: 0.3, entropyBits: 3.5 },
  { token: ' the', probability: 0.6, entropyBits: 2.2 },
  { token: ' thing', probability: 0.15, entropyBits: 4.1 },
  { token: ' was', probability: 0.7, entropyBits: 2.5 },
]);

describe('trace measurements', () => {
  it('reads the first token probability', () => {
    expect(measureCheck('firstTokenProbability', confidentTrace)).toBeCloseTo(0.95);
  });

  it('reads the generated token count', () => {
    expect(measureCheck('generatedTokenCount', uncertainTrace)).toBe(4);
  });

  it('averages entropy across steps', () => {
    expect(measureCheck('meanEntropyBits', uncertainTrace)).toBeCloseTo((3.5 + 2.2 + 4.1 + 2.5) / 4);
  });

  it('returns zero for an empty trace rather than NaN', () => {
    const empty = makeTrace([]);
    expect(measureCheck('firstTokenProbability', empty)).toBe(0);
    expect(measureCheck('meanEntropyBits', empty)).toBe(0);
    expect(measureCheck('generatedTokenCount', empty)).toBe(0);
  });

  it('finds the least certain step as the fork', () => {
    expect(findForkIndex(uncertainTrace)).toBe(2);
    expect(findForkIndex(makeTrace([]))).toBe(-1);
  });

  it('measures each comparison axis from the trace', () => {
    expect(measureAxis('verbosity', uncertainTrace)).toBe(4);
    expect(measureAxis('confidence', confidentTrace)).toBeGreaterThan(
      measureAxis('confidence', uncertainTrace)
    );
    expect(measureAxis('decisiveness', confidentTrace)).toBeGreaterThan(
      measureAxis('decisiveness', uncertainTrace)
    );
    expect(measureAxis('diversity', uncertainTrace)).toBeCloseTo(1);
  });

  it('measures diversity as the distinct-token ratio', () => {
    const repetitive = makeTrace([
      { token: 'a', probability: 0.5, entropyBits: 1 },
      { token: 'a', probability: 0.5, entropyBits: 1 },
      { token: 'a', probability: 0.5, entropyBits: 1 },
      { token: 'b', probability: 0.5, entropyBits: 1 },
    ]);
    expect(measureAxis('diversity', repetitive)).toBeCloseTo(0.5);
  });
});

describe('inspectorChatEngine — challenge run', () => {
  const config = configFor(0);

  it('starts with every challenge unmet', () => {
    const state = initState(config, rulesFor(0));
    expect(state.challenges).toHaveLength(config.challenges!.length);
    expect(state.challenges.every((c) => !c.satisfied)).toBe(true);
    expect(evaluate(state).value).toBe(0);
  });

  it('marks a challenge satisfied from a real measurement', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'capital of france?', trace: confidentTrace, source: 'local' });

    const first = state.challenges.find((c) => c.check === 'firstTokenProbability')!;
    expect(first.measured).toBeCloseTo(0.95);
    expect(first.satisfied).toBe(true);

    // The same run also satisfies the short-answer challenge.
    expect(state.challenges.find((c) => c.check === 'generatedTokenCount')!.satisfied).toBe(true);
  });

  it('does not mark a challenge the measurement misses', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    expect(state.challenges.find((c) => c.check === 'meanEntropyBits')!.satisfied).toBe(false);
  });

  it('keeps a satisfied challenge satisfied across later runs', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p2', trace: uncertainTrace, source: 'local' });

    expect(state.challenges.find((c) => c.check === 'firstTokenProbability')!.satisfied).toBe(true);
    expect(state.challenges.find((c) => c.check === 'meanEntropyBits')!.satisfied).toBe(true);
  });

  it('reaches three stars once every challenge is met', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p2', trace: uncertainTrace, source: 'local' });

    const result = evaluate(state);
    expect(result.metric).toBe('challengesCompleted');
    expect(result.value).toBe(config.challenges!.length);
    expect(result.stars).toBe(3);
  });

  it('records the transcript of every run', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'first', trace: confidentTrace, source: 'local' });
    expect(state.transcript).toHaveLength(1);
    expect(state.transcript[0]!.prompt).toBe('first');
  });

  it('clamps sampling controls to their ranges', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'SET_TEMPERATURE', value: 99 });
    expect(state.temperature).toBe(config.temperatureRange![1]);
    state = applyAction(state, { type: 'SET_TOP_P', value: 99 });
    expect(state.topP).toBe(config.topPRange![1]);
  });
});

describe('inspectorChatEngine — find the fork', () => {
  const config = configFor(1);

  it('records a fork round per run, up to the configured count', () => {
    let state = initState(config, rulesFor(1));
    for (let i = 0; i < config.rounds! + 3; i++) {
      state = applyAction(state, { type: 'RECORD_RUN', prompt: `p${i}`, trace: uncertainTrace, source: 'local' });
    }
    expect(state.forkRounds).toHaveLength(config.rounds!);
  });

  it('derives the fork from the real per-step probabilities', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: uncertainTrace, source: 'local' });
    expect(state.forkRounds[0]!.trueForkIndex).toBe(2);
  });

  it('scores an exact identification, and one inside the tolerance', () => {
    let state = initState(config, rulesFor(1));
    for (let i = 0; i < config.rounds!; i++) {
      state = applyAction(state, { type: 'RECORD_RUN', prompt: `p${i}`, trace: uncertainTrace, source: 'local' });
    }
    state.forkRounds.forEach((round, i) => {
      // Off by one, which the configured tolerance allows.
      state = applyAction(state, { type: 'ANSWER_FORK', roundIndex: i, index: round.trueForkIndex + 1 });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('forkIdentificationAccuracy');
    expect(result.value).toBeCloseTo(1);
  });

  it('rejects an answer outside the trace', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: uncertainTrace, source: 'local' });
    state = applyAction(state, { type: 'ANSWER_FORK', roundIndex: 0, index: 99 });
    expect(state.forkRounds[0]!.answer).toBeNull();
  });

  it('counts rounds never played against the score', () => {
    const state = initState(config, rulesFor(1));
    expect(evaluate(state).value).toBe(0);
  });
});

describe('inspectorChatEngine — local versus cloud', () => {
  const config = configFor(2);

  it('starts with cloud escalation off', () => {
    const state = initState(config, rulesFor(2));
    expect(state.cloudEnabled).toBe(false);
    expect(cloudAvailable(state)).toBe(true);
  });

  it('refuses to enable cloud escalation while offline', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'SET_ONLINE', value: false });

    expect(cloudAvailable(state)).toBe(false);
    state = applyAction(state, { type: 'TOGGLE_CLOUD', value: true });
    expect(state.cloudEnabled).toBe(false);
  });

  it('switches cloud escalation off when connectivity is lost', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'TOGGLE_CLOUD', value: true });
    expect(state.cloudEnabled).toBe(true);

    state = applyAction(state, { type: 'SET_ONLINE', value: false });
    expect(state.cloudEnabled).toBe(false);
  });

  it('refuses to record a cloud run that escalation never authorised', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'cloud' });
    expect(state.transcript).toHaveLength(0);
    expect(state.cloudTrace).toBeNull();
  });

  it('leaves comparisons unanswerable until both models have run', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    expect(state.comparisons.every((c) => c.trueAnswer === null)).toBe(true);
    expect(evaluate(state).value).toBe(0);
  });

  it('measures every axis once both models have run', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'TOGGLE_CLOUD', value: true });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: uncertainTrace, source: 'cloud' });

    expect(state.comparisons.every((c) => c.trueAnswer !== null)).toBe(true);
    // The cloud trace here is longer, so it wins on verbosity.
    expect(state.comparisons.find((c) => c.axis === 'verbosity')!.trueAnswer).toBe(true);
    // …and less confident, so it loses there.
    expect(state.comparisons.find((c) => c.axis === 'confidence')!.trueAnswer).toBe(false);
  });

  it('scores all-correct predictions at 1', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'TOGGLE_CLOUD', value: true });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: uncertainTrace, source: 'cloud' });

    for (const comparison of state.comparisons) {
      state = applyAction(state, {
        type: 'PREDICT_COMPARISON',
        axis: comparison.axis,
        value: comparison.trueAnswer!,
      });
    }

    const result = evaluate(state);
    expect(result.metric).toBe('comparisonScore');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
    expect(result.breakdown.bothModelsRun).toBe(1);
  });

  it('scores all-wrong predictions at 0', () => {
    let state = initState(config, rulesFor(2));
    state = applyAction(state, { type: 'TOGGLE_CLOUD', value: true });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: confidentTrace, source: 'local' });
    state = applyAction(state, { type: 'RECORD_RUN', prompt: 'p', trace: uncertainTrace, source: 'cloud' });

    for (const comparison of state.comparisons) {
      state = applyAction(state, {
        type: 'PREDICT_COMPARISON',
        axis: comparison.axis,
        value: !comparison.trueAnswer,
      });
    }
    expect(evaluate(state).value).toBe(0);
  });

  it('ignores a prediction for an axis the level does not use', () => {
    let state = initState(config, rulesFor(2));
    // @ts-expect-error deliberately invalid axis
    state = applyAction(state, { type: 'PREDICT_COMPARISON', axis: 'factuality', value: true });
    expect(state.comparisons.every((c) => c.prediction === null)).toBe(true);
  });
});

describe('inspectorChatEngine — level config coverage', () => {
  it('handles every shipped level', () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as InspectorChatConfig;
      const state = initState(config, rulesFor(i));
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
