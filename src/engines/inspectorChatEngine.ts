/**
 * Chapter 6.1 — Inspector Chat.
 *
 * Section 3 of the spec lists no engine for the capstone, but rule 2 requires
 * every game's rules to live in JSON and be independently testable, so the
 * capstone gets one like every other chapter.
 *
 * Nothing here is judged by opinion. Every challenge is checked against a
 * measurement taken from the real generation trace — token counts, per-step
 * probabilities and entropies produced by the model that actually ran.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { ChatModelDep, GenerationTrace } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, mean } from './shared';

export type InspectorMode = 'challenge-run' | 'find-fork' | 'compare-scale';

export type ChallengeCheck = 'firstTokenProbability' | 'meanEntropyBits' | 'generatedTokenCount';

/** Axes with a measurable proxy in the trace — nothing scored on opinion. */
export type ComparisonAxis = 'verbosity' | 'confidence' | 'diversity' | 'decisiveness';

export interface Challenge {
  id: string;
  prompt: string;
  check: ChallengeCheck;
  comparator: 'gte' | 'lte';
  target: number;
}

export interface InspectorChatConfig {
  mode: InspectorMode;
  maxTokens: number;
  temperature: number;
  temperatureRange?: [number, number];
  topP: number;
  topPRange?: [number, number];
  challenges?: Challenge[];
  rounds?: number;
  positionTolerance?: number;
  cloudEscalation?: boolean;
  cloudRequiresOnline?: boolean;
  comparisonAxes?: ComparisonAxis[];
}

export interface ChallengeState extends Challenge {
  /** Measured value from the last run attempted against this challenge. */
  measured: number | null;
  satisfied: boolean;
}

export interface ForkRound {
  trace: GenerationTrace;
  /** Index of the step where the model was genuinely least certain. */
  trueForkIndex: number;
  answer: number | null;
}

export interface ComparisonRound {
  axis: ComparisonAxis;
  /** Player's call: does the larger cloud model score higher on this axis? */
  prediction: boolean | null;
  /** Measured answer, once both traces exist. */
  trueAnswer: boolean | null;
}

export interface InspectorChatState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: InspectorMode;
  config: InspectorChatConfig;
  temperature: number;
  topP: number;
  /** False disables cloud escalation entirely, per the offline fallback rule. */
  online: boolean;
  cloudEnabled: boolean;
  challenges: ChallengeState[];
  forkRounds: ForkRound[];
  comparisons: ComparisonRound[];
  localTrace: GenerationTrace | null;
  cloudTrace: GenerationTrace | null;
  transcript: { prompt: string; trace: GenerationTrace; source: 'local' | 'cloud' }[];
}

export type InspectorChatAction =
  | { type: 'SET_TEMPERATURE'; value: number }
  | { type: 'SET_TOP_P'; value: number }
  | { type: 'SET_ONLINE'; value: boolean }
  | { type: 'TOGGLE_CLOUD'; value: boolean }
  | { type: 'RECORD_RUN'; prompt: string; trace: GenerationTrace; source: 'local' | 'cloud' }
  | { type: 'ANSWER_FORK'; roundIndex: number; index: number }
  | { type: 'PREDICT_COMPARISON'; axis: ComparisonAxis; value: boolean }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Pulls the measurement a challenge asks for out of a real trace. */
export function measureCheck(check: ChallengeCheck, trace: GenerationTrace): number {
  switch (check) {
    case 'firstTokenProbability':
      return trace.steps[0]?.probability ?? 0;
    case 'meanEntropyBits':
      return trace.steps.length === 0 ? 0 : mean(trace.steps.map((s) => s.entropyBits));
    case 'generatedTokenCount':
      return trace.steps.length;
  }
}

/** Step where the model was least certain — the fork sampling decided. */
export function findForkIndex(trace: GenerationTrace): number {
  if (trace.steps.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < trace.steps.length; i++) {
    if (trace.steps[i]!.probability < trace.steps[best]!.probability) best = i;
  }
  return best;
}

/** Measurable proxy for each comparison axis. */
export function measureAxis(axis: ComparisonAxis, trace: GenerationTrace): number {
  switch (axis) {
    case 'verbosity':
      return trace.steps.length;
    case 'confidence':
      return trace.steps.length === 0 ? 0 : mean(trace.steps.map((s) => s.probability));
    case 'diversity': {
      if (trace.steps.length === 0) return 0;
      const distinct = new Set(trace.steps.map((s) => s.token)).size;
      return distinct / trace.steps.length;
    }
    case 'decisiveness':
      // Inverted entropy: lower spread means a more decisive model.
      return trace.steps.length === 0 ? 0 : -mean(trace.steps.map((s) => s.entropyBits));
  }
}

export async function prepare(
  config: InspectorChatConfig,
  deps: { chat?: ChatModelDep } = {}
): Promise<{ warmup: GenerationTrace | null }> {
  // The capstone is driven entirely by the player's own prompts, so there is
  // nothing to precompute beyond an optional readiness check.
  if (!deps.chat || config.mode !== 'find-fork') return { warmup: null };
  const trace = await deps.chat.generateWithTrace('Hello', {
    maxTokens: Math.min(config.maxTokens, 8),
    temperature: config.temperature,
    topP: config.topP,
  });
  return { warmup: trace };
}

export function initState(
  config: InspectorChatConfig,
  rules: EngineRules
): InspectorChatState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    temperature: config.temperature,
    topP: config.topP,
    online: true,
    cloudEnabled: false,
    challenges: (config.challenges ?? []).map((c) => ({ ...c, measured: null, satisfied: false })),
    forkRounds: [],
    comparisons: (config.comparisonAxes ?? []).map((axis) => ({
      axis,
      prediction: null,
      trueAnswer: null,
    })),
    localTrace: null,
    cloudTrace: null,
    transcript: [],
  };
}

function satisfies(challenge: Challenge, measured: number): boolean {
  return challenge.comparator === 'gte' ? measured >= challenge.target : measured <= challenge.target;
}

export function applyAction(
  state: InspectorChatState,
  action: InspectorChatAction
): InspectorChatState {
  const bump = (next: Partial<InspectorChatState>): InspectorChatState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_TEMPERATURE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.temperatureRange ?? [0.01, 2];
      return bump({ temperature: clamp(action.value, min, max) });
    }

    case 'SET_TOP_P': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.topPRange ?? [0.01, 1];
      return bump({ topP: clamp(action.value, min, max) });
    }

    case 'SET_ONLINE':
      // Going offline must switch cloud escalation off, not merely grey it out.
      return bump({
        online: action.value,
        cloudEnabled: action.value ? state.cloudEnabled : false,
      });

    case 'TOGGLE_CLOUD': {
      if (!state.config.cloudEscalation) return state;
      if (action.value && state.config.cloudRequiresOnline && !state.online) return state;
      return bump({ cloudEnabled: action.value });
    }

    case 'RECORD_RUN': {
      // The cloud can only have produced this if escalation was actually on.
      if (action.source === 'cloud' && !state.cloudEnabled) return state;

      const challenges = state.challenges.map((challenge) => {
        const measured = measureCheck(challenge.check, action.trace);
        const nowSatisfied = satisfies(challenge, measured);
        // Once cleared, a challenge stays cleared — later runs cannot undo it.
        return challenge.satisfied
          ? challenge
          : { ...challenge, measured, satisfied: nowSatisfied };
      });

      const forkRounds =
        state.mode === 'find-fork' && state.forkRounds.length < (state.config.rounds ?? 0)
          ? [
              ...state.forkRounds,
              { trace: action.trace, trueForkIndex: findForkIndex(action.trace), answer: null },
            ]
          : state.forkRounds;

      const localTrace = action.source === 'local' ? action.trace : state.localTrace;
      const cloudTrace = action.source === 'cloud' ? action.trace : state.cloudTrace;

      // Comparison answers only become knowable once both models have run.
      const comparisons =
        localTrace && cloudTrace
          ? state.comparisons.map((c) => ({
              ...c,
              trueAnswer: measureAxis(c.axis, cloudTrace) > measureAxis(c.axis, localTrace),
            }))
          : state.comparisons;

      return bump({
        challenges,
        forkRounds,
        localTrace,
        cloudTrace,
        comparisons,
        transcript: [
          ...state.transcript,
          { prompt: action.prompt, trace: action.trace, source: action.source },
        ],
      });
    }

    case 'ANSWER_FORK': {
      const round = state.forkRounds[action.roundIndex];
      if (!round) return state;
      if (action.index < 0 || action.index >= round.trace.steps.length) return state;
      const forkRounds = [...state.forkRounds];
      forkRounds[action.roundIndex] = { ...round, answer: action.index };
      return bump({ forkRounds });
    }

    case 'PREDICT_COMPARISON': {
      const index = state.comparisons.findIndex((c) => c.axis === action.axis);
      if (index === -1) return state;
      const comparisons = [...state.comparisons];
      comparisons[index] = { ...comparisons[index]!, prediction: action.value };
      return bump({ comparisons });
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Whether the cloud toggle should be offered at all, given connectivity. */
export function cloudAvailable(state: InspectorChatState): boolean {
  if (!state.config.cloudEscalation) return false;
  return !state.config.cloudRequiresOnline || state.online;
}

export function evaluate(state: InspectorChatState): ScoreResult {
  switch (state.mode) {
    case 'challenge-run': {
      const completed = state.challenges.filter((c) => c.satisfied).length;
      return scoreLevel({
        metric: 'challengesCompleted',
        value: completed,
        rules: state.rules,
        breakdown: { completed, total: state.challenges.length, runs: state.transcript.length },
      });
    }

    case 'find-fork': {
      const expected = state.config.rounds ?? state.forkRounds.length;
      const total = Math.max(expected, state.forkRounds.length);
      if (total === 0) {
        return scoreLevel({ metric: 'forkIdentificationAccuracy', value: 0, rules: state.rules });
      }
      const tolerance = state.config.positionTolerance ?? 0;
      const correct = state.forkRounds.filter(
        (r) => r.answer !== null && Math.abs(r.answer - r.trueForkIndex) <= tolerance
      ).length;

      return scoreLevel({
        metric: 'forkIdentificationAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total, tolerance },
      });
    }

    case 'compare-scale': {
      const total = state.comparisons.length;
      if (total === 0) {
        return scoreLevel({ metric: 'comparisonScore', value: 0, rules: state.rules });
      }
      // Unanswerable until both models have run, which is the point of the level.
      const correct = state.comparisons.filter(
        (c) => c.trueAnswer !== null && c.prediction !== null && c.prediction === c.trueAnswer
      ).length;

      return scoreLevel({
        metric: 'comparisonScore',
        value: correct / total,
        rules: state.rules,
        breakdown: {
          correct,
          total,
          bothModelsRun: state.localTrace && state.cloudTrace ? 1 : 0,
        },
      });
    }
  }
}
