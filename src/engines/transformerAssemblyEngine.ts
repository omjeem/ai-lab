/**
 * Chapter 5.5 — Full Transformer Assembly.
 *
 * The first two levels are about the shape of the block. The last runs a real
 * forward pass through a causal LM and asks the player to call the top token
 * from the stage outputs before the final softmax is revealed.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep, TokenDistribution } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, createRng } from './shared';

export type AssemblyMode = 'order-stages' | 'fill-gaps' | 'run-end-to-end';

export interface TransformerAssemblyConfig {
  mode: AssemblyMode;
  stages: string[];
  removedStages?: string[];
  criticalStages?: string[];
  shuffleSeed?: number;
  allowPartialCredit?: boolean;
  prompt?: string;
  layer?: number;
  topK?: number;
  rounds?: number;
  allowUserPrompt?: boolean;
}

export interface PredictionRound {
  prompt: string;
  /** The model's real top-k for this prompt. */
  distribution: TokenDistribution;
  /** Shuffled choices shown to the player. */
  choices: string[];
  answer: string | null;
}

export interface PreparedAssemblyData {
  rounds: { prompt: string; distribution: TokenDistribution }[];
}

export interface TransformerAssemblyState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: AssemblyMode;
  config: TransformerAssemblyConfig;
  /** Correct pipeline order, straight from config. */
  correctOrder: string[];
  /** The player's current arrangement. */
  arrangement: string[];
  /** Stages still waiting to be placed, for the fill-gaps level. */
  tray: string[];
  rounds: PredictionRound[];
}

export type TransformerAssemblyAction =
  | { type: 'SET_ARRANGEMENT'; stages: string[] }
  | { type: 'MOVE_STAGE'; from: number; to: number }
  | { type: 'INSERT_STAGE'; stage: string; position: number }
  | { type: 'REMOVE_STAGE'; position: number }
  | { type: 'ANSWER_TOP_TOKEN'; roundIndex: number; token: string }
  | { type: 'ADD_ROUND'; round: PredictionRound }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

function shuffle<T>(items: readonly T[], seed: number): T[] {
  const rng = createRng(seed);
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

export async function prepare(
  config: TransformerAssemblyConfig,
  deps: { causalLM?: CausalLMDep } = {}
): Promise<PreparedAssemblyData> {
  if (config.mode !== 'run-end-to-end' || !deps.causalLM || !config.prompt) {
    return { rounds: [] };
  }
  const distribution = await deps.causalLM.nextTokenDistribution(config.prompt, config.topK ?? 5);
  if (distribution.tokens.length !== distribution.probs.length) {
    throw new Error('Model returned mismatched token and probability arrays');
  }
  return { rounds: [{ prompt: config.prompt, distribution }] };
}

export function buildRound(
  prompt: string,
  distribution: TokenDistribution,
  seed: number
): PredictionRound {
  return {
    prompt,
    distribution,
    // Shuffled so the display order does not give away the ranking.
    choices: shuffle(distribution.tokens, seed),
    answer: null,
  };
}

export function initState(
  config: TransformerAssemblyConfig,
  rules: EngineRules,
  prepared: PreparedAssemblyData = { rounds: [] }
): TransformerAssemblyState {
  const correctOrder = [...config.stages];
  const seed = config.shuffleSeed ?? 1;

  let arrangement: string[];
  let tray: string[];

  if (config.mode === 'fill-gaps') {
    const removed = new Set(config.removedStages ?? []);
    arrangement = correctOrder.filter((s) => !removed.has(s));
    tray = shuffle([...removed], seed);
  } else if (config.mode === 'order-stages') {
    arrangement = shuffle(correctOrder, seed);
    tray = [];
  } else {
    arrangement = [...correctOrder];
    tray = [];
  }

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    correctOrder,
    arrangement,
    tray,
    rounds: prepared.rounds.map((r, i) => buildRound(r.prompt, r.distribution, seed + i)),
  };
}

export function applyAction(
  state: TransformerAssemblyState,
  action: TransformerAssemblyAction
): TransformerAssemblyState {
  const bump = (next: Partial<TransformerAssemblyState>): TransformerAssemblyState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_ARRANGEMENT': {
      // Must stay a rearrangement of what is currently placed.
      const a = [...state.arrangement].sort();
      const b = [...action.stages].sort();
      if (a.length !== b.length || a.some((s, i) => s !== b[i])) return state;
      return bump({ arrangement: [...action.stages] });
    }

    case 'MOVE_STAGE': {
      const { from, to } = action;
      if (from < 0 || from >= state.arrangement.length) return state;
      if (to < 0 || to >= state.arrangement.length) return state;
      const arrangement = [...state.arrangement];
      const [moved] = arrangement.splice(from, 1);
      arrangement.splice(to, 0, moved!);
      return bump({ arrangement });
    }

    case 'INSERT_STAGE': {
      if (!state.tray.includes(action.stage)) return state;
      const position = clamp(action.position, 0, state.arrangement.length);
      const arrangement = [...state.arrangement];
      arrangement.splice(position, 0, action.stage);
      return bump({
        arrangement,
        tray: state.tray.filter((s) => s !== action.stage),
      });
    }

    case 'REMOVE_STAGE': {
      if (action.position < 0 || action.position >= state.arrangement.length) return state;
      const arrangement = [...state.arrangement];
      const [removed] = arrangement.splice(action.position, 1);
      return bump({ arrangement, tray: [...state.tray, removed!] });
    }

    case 'ANSWER_TOP_TOKEN': {
      const round = state.rounds[action.roundIndex];
      if (!round || !round.choices.includes(action.token)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, answer: action.token };
      return bump({ rounds });
    }

    case 'ADD_ROUND': {
      if (!state.config.allowUserPrompt) return state;
      const max = state.config.rounds ?? Infinity;
      if (state.rounds.length >= max) return state;
      return bump({ rounds: [...state.rounds, action.round] });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        rounds: state.rounds.map((r) => ({ prompt: r.prompt, distribution: r.distribution })),
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/**
 * How much of the correct ordering the arrangement preserves.
 *
 * Uses ordered pairs rather than exact positions, so a single misplaced stage
 * costs a little rather than shifting everything after it to zero.
 */
export function orderingAccuracy(arrangement: readonly string[], correct: readonly string[]): number {
  const position = new Map(correct.map((s, i) => [s, i]));
  const indices = arrangement.map((s) => position.get(s)).filter((i): i is number => i !== undefined);
  if (indices.length < 2) return indices.length === correct.length ? 1 : 0;

  let ordered = 0;
  let pairs = 0;
  for (let i = 0; i < indices.length; i++) {
    for (let j = i + 1; j < indices.length; j++) {
      pairs++;
      if (indices[i]! < indices[j]!) ordered++;
    }
  }
  return pairs === 0 ? 0 : ordered / pairs;
}

export function evaluate(state: TransformerAssemblyState): ScoreResult {
  switch (state.mode) {
    case 'order-stages': {
      const exact =
        state.arrangement.length === state.correctOrder.length &&
        state.arrangement.every((s, i) => s === state.correctOrder[i]);
      const value = exact
        ? 1
        : state.config.allowPartialCredit
          ? orderingAccuracy(state.arrangement, state.correctOrder)
          : 0;

      return scoreLevel({
        metric: 'assemblyAccuracy',
        value,
        rules: state.rules,
        breakdown: { exact: exact ? 1 : 0, stages: state.correctOrder.length },
      });
    }

    case 'fill-gaps': {
      const critical = new Set(state.config.criticalStages ?? []);
      const placed = new Set(state.arrangement);
      const missing = state.correctOrder.filter((s) => !placed.has(s));

      // Critical stages count double: leaving out the softmax breaks the block
      // in a way that leaving out a second layer norm does not.
      const weightOf = (stage: string) => (critical.has(stage) ? 2 : 1);
      const totalWeight = state.correctOrder.reduce((n, s) => n + weightOf(s), 0);
      const missingWeight = missing.reduce((n, s) => n + weightOf(s), 0);

      const completeness = totalWeight === 0 ? 0 : 1 - missingWeight / totalWeight;
      // Restoring the stages in the wrong order is only partial credit.
      const value = completeness * orderingAccuracy(state.arrangement, state.correctOrder);

      return scoreLevel({
        metric: 'pipelineCompleteness',
        value,
        rules: state.rules,
        breakdown: {
          completeness,
          missing: missing.length,
          criticalMissing: missing.filter((s) => critical.has(s)).length,
        },
      });
    }

    case 'run-end-to-end': {
      const expected = state.config.rounds ?? state.rounds.length;
      const total = Math.max(expected, state.rounds.length);
      if (total === 0) {
        return scoreLevel({ metric: 'endToEndScore', value: 0, rules: state.rules });
      }

      let correct = 0;
      for (let i = 0; i < total; i++) {
        const round = state.rounds[i];
        if (!round || round.answer === null) continue;
        const probs = round.distribution.probs;
        let best = 0;
        for (let k = 1; k < probs.length; k++) if (probs[k]! > probs[best]!) best = k;
        if (round.distribution.tokens[best] === round.answer) correct++;
      }

      // The assembled pipeline still has to be right, so ordering gates the score.
      const pipeline = orderingAccuracy(state.arrangement, state.correctOrder);
      return scoreLevel({
        metric: 'endToEndScore',
        value: (correct / total) * pipeline,
        rules: state.rules,
        breakdown: { correct, total, pipeline },
      });
    }
  }
}
