/**
 * Chapter 4.3 — Sampling Strategies.
 *
 * Reshapes a real next-token distribution pulled from the model. Temperature,
 * top-k and top-p are applied exactly as a decoder applies them, so the entropy
 * readout responds the way it would in a real generation loop.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, entropyBits, mean } from './shared';

export type SamplingMode = 'entropy-target' | 'compare-truncation' | 'task-tuning';

export interface SamplingTask {
  id: string;
  prompt: string;
  targetEntropyBits: number;
  toleranceBits: number;
}

export interface SamplingConfig {
  mode: SamplingMode;
  prompt?: string;
  prompts?: string[];
  tasks?: SamplingTask[];
  topK: number;
  topKRange?: [number, number];
  topP?: number;
  topPRange?: [number, number];
  temperature: number;
  temperatureRange?: [number, number];
  targetEntropyBits?: number;
  toleranceBits?: number;
  targetKeptMassLow?: number;
  targetKeptMassHigh?: number;
  allowUserPrompt?: boolean;
}

export interface PromptDistribution {
  prompt: string;
  tokens: string[];
  /** The model's real probabilities, before any reshaping. */
  probs: number[];
}

export interface PreparedSamplingData {
  distributions: PromptDistribution[];
}

export interface TaskSettings {
  temperature: number;
  topK: number;
  topP: number;
}

export interface SamplingState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: SamplingMode;
  config: SamplingConfig;
  distributions: PromptDistribution[];
  temperature: number;
  topK: number;
  topP: number;
  /** Per-task settings for the multi-task level. */
  taskSettings: Record<string, TaskSettings>;
  activeTaskId: string | null;
}

export type SamplingAction =
  | { type: 'SET_TEMPERATURE'; value: number }
  | { type: 'SET_TOP_K'; value: number }
  | { type: 'SET_TOP_P'; value: number }
  | { type: 'SELECT_TASK'; taskId: string }
  | { type: 'ADD_DISTRIBUTION'; distribution: PromptDistribution }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

export interface ReshapeResult {
  tokens: string[];
  probs: number[];
  /** Probability mass the truncation kept, before renormalising. */
  keptMass: number;
  keptCount: number;
  entropyBits: number;
  /**
   * The full tempered pool, ranked, before either truncation — including the
   * tokens top-k/top-p discarded. Un-renormalised, so its values are directly
   * comparable to `keptMass`. Display-only: scoring is entirely off `probs`
   * and `keptMass` above, this is what lets a canvas draw the cut honestly
   * rather than guessing where it fell.
   */
  rankedTokens: string[];
  rankedProbs: number[];
}

/**
 * Applies temperature, then top-k, then top-p, in the order a decoder does.
 *
 * Temperature is applied to the log-probabilities, which is equivalent to
 * scaling the original logits — the model's probabilities are all we are given.
 */
export function reshape(
  distribution: PromptDistribution,
  settings: { temperature: number; topK: number; topP: number }
): ReshapeResult {
  const { tokens, probs } = distribution;
  if (probs.length === 0) {
    return {
      tokens: [],
      probs: [],
      keptMass: 0,
      keptCount: 0,
      entropyBits: 0,
      rankedTokens: [],
      rankedProbs: [],
    };
  }

  const temperature = Math.max(settings.temperature, 1e-6);
  const scaled = probs.map((p) => Math.log(Math.max(p, 1e-12)) / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  const tempered = exps.map((e) => e / total);

  // Rank once, then apply both truncations against that ranking.
  const ranked = tempered
    .map((p, i) => ({ index: i, p }))
    .sort((a, b) => b.p - a.p);

  const k = Math.max(1, Math.min(settings.topK, ranked.length));
  const byTopK = ranked.slice(0, k);

  const kept: typeof byTopK = [];
  let cumulative = 0;
  for (const entry of byTopK) {
    kept.push(entry);
    cumulative += entry.p;
    // Top-p keeps the smallest prefix reaching the mass threshold, so the cut
    // adapts to how confident the model is at this position.
    if (cumulative >= settings.topP) break;
  }

  const keptMass = kept.reduce((a, e) => a + e.p, 0);
  const renormalised = kept.map((e) => e.p / keptMass);

  return {
    tokens: kept.map((e) => tokens[e.index]!),
    probs: renormalised,
    keptMass,
    keptCount: kept.length,
    entropyBits: entropyBits(renormalised),
    rankedTokens: ranked.map((e) => tokens[e.index]!),
    rankedProbs: ranked.map((e) => e.p),
  };
}

export async function prepare(
  config: SamplingConfig,
  deps: { causalLM: CausalLMDep }
): Promise<PreparedSamplingData> {
  const prompts = [
    ...(config.prompt ? [config.prompt] : []),
    ...(config.prompts ?? []),
    ...(config.tasks ?? []).map((t) => t.prompt),
  ];

  const distributions: PromptDistribution[] = [];
  for (const prompt of prompts) {
    const dist = await deps.causalLM.nextTokenDistribution(prompt, config.topK);
    if (dist.tokens.length !== dist.probs.length) {
      throw new Error(
        `Model returned ${dist.tokens.length} tokens and ${dist.probs.length} probabilities`
      );
    }
    distributions.push({ prompt, tokens: dist.tokens, probs: dist.probs });
  }

  return { distributions };
}

export function initState(
  config: SamplingConfig,
  rules: EngineRules,
  prepared: PreparedSamplingData
): SamplingState {
  const defaults: TaskSettings = {
    temperature: config.temperature,
    topK: config.topK,
    topP: config.topP ?? 1,
  };

  const taskSettings: Record<string, TaskSettings> = {};
  for (const task of config.tasks ?? []) taskSettings[task.id] = { ...defaults };

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    distributions: prepared.distributions.map((d) => ({ ...d })),
    ...defaults,
    taskSettings,
    activeTaskId: config.tasks?.[0]?.id ?? null,
  };
}

export function applyAction(state: SamplingState, action: SamplingAction): SamplingState {
  /** Writes a control change to the active task, or to the shared settings. */
  const setControl = (patch: Partial<TaskSettings>): SamplingState => {
    const next = { ...state, ...patch, status: 'active' as const, actionCount: state.actionCount + 1 };
    if (state.activeTaskId && state.taskSettings[state.activeTaskId]) {
      next.taskSettings = {
        ...state.taskSettings,
        [state.activeTaskId]: { ...state.taskSettings[state.activeTaskId]!, ...patch },
      };
    }
    return next;
  };

  switch (action.type) {
    case 'SET_TEMPERATURE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.temperatureRange ?? [0.01, 5];
      return setControl({ temperature: clamp(action.value, min, max) });
    }

    case 'SET_TOP_K': {
      if (!Number.isInteger(action.value)) return state;
      const [min, max] = state.config.topKRange ?? [1, state.config.topK];
      return setControl({ topK: Math.round(clamp(action.value, min, max)) });
    }

    case 'SET_TOP_P': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.topPRange ?? [0.01, 1];
      return setControl({ topP: clamp(action.value, min, max) });
    }

    case 'SELECT_TASK': {
      const settings = state.taskSettings[action.taskId];
      if (!settings) return state;
      // Switching tasks restores that task's own controls.
      return {
        ...state,
        activeTaskId: action.taskId,
        ...settings,
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'ADD_DISTRIBUTION': {
      if (!state.config.allowUserPrompt) return state;
      return {
        ...state,
        distributions: [...state.distributions, { ...action.distribution }],
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'RESET':
      return initState(state.config, state.rules, { distributions: state.distributions });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Current reshaped view of a prompt's distribution — what the UI renders. */
export function currentReshape(state: SamplingState, promptIndex = 0): ReshapeResult | null {
  const distribution = state.distributions[promptIndex];
  if (!distribution) return null;
  return reshape(distribution, {
    temperature: state.temperature,
    topK: state.topK,
    topP: state.topP,
  });
}

export function evaluate(state: SamplingState): ScoreResult {
  switch (state.mode) {
    case 'entropy-target': {
      const result = currentReshape(state);
      if (!result) {
        return scoreLevel({ metric: 'entropyError', value: Infinity, rules: state.rules });
      }
      const target = state.config.targetEntropyBits ?? 0;
      const error = Math.abs(result.entropyBits - target);
      return scoreLevel({
        metric: 'entropyError',
        value: error,
        rules: state.rules,
        breakdown: {
          entropyBits: result.entropyBits,
          target,
          temperature: state.temperature,
          keptCount: result.keptCount,
        },
      });
    }

    case 'compare-truncation': {
      if (state.distributions.length === 0) {
        return scoreLevel({ metric: 'truncationBalanceScore', value: 0, rules: state.rules });
      }
      const low = state.config.targetKeptMassLow ?? 0.75;
      const high = state.config.targetKeptMassHigh ?? 0.95;

      // One setting has to behave on both a confident and an uncertain prompt,
      // which is exactly the argument for top-p over a fixed top-k.
      const scores = state.distributions.map((distribution) => {
        const result = reshape(distribution, {
          temperature: state.temperature,
          topK: state.topK,
          topP: state.topP,
        });
        const mass = result.keptMass;
        if (mass >= low && mass <= high) return 1;
        const distance = mass < low ? low - mass : mass - high;
        return clamp(1 - distance / low, 0, 1);
      });

      return scoreLevel({
        metric: 'truncationBalanceScore',
        value: mean(scores),
        rules: state.rules,
        breakdown: { prompts: scores.length, worst: Math.min(...scores) },
      });
    }

    case 'task-tuning': {
      const tasks = state.config.tasks ?? [];
      if (tasks.length === 0) {
        return scoreLevel({ metric: 'taskTuningScore', value: 0, rules: state.rules });
      }

      const scores = tasks.map((task) => {
        const distribution = state.distributions.find((d) => d.prompt === task.prompt);
        if (!distribution) return 0;
        const settings = state.taskSettings[task.id] ?? {
          temperature: state.temperature,
          topK: state.topK,
          topP: state.topP,
        };
        const result = reshape(distribution, settings);
        const error = Math.abs(result.entropyBits - task.targetEntropyBits);
        // Full credit inside the tolerance band, tapering outside it.
        return error <= task.toleranceBits
          ? 1
          : clamp(1 - (error - task.toleranceBits) / Math.max(task.targetEntropyBits, 1), 0, 1);
      });

      return scoreLevel({
        metric: 'taskTuningScore',
        value: mean(scores),
        rules: state.rules,
        breakdown: { tasks: tasks.length, worst: Math.min(...scores) },
      });
    }
  }
}
