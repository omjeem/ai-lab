/**
 * Chapter 1.5 — Probability Basics.
 *
 * The wheel's segments are the model's real top-k next-token probabilities.
 * Spinning samples from that distribution for real, so an unlikely token
 * genuinely does come up occasionally.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, entropyBits, normalizeDistribution, totalVariationDistance } from './shared';

export type ProbabilityWheelMode = 'predict-top1' | 'allocate-mass' | 'estimate-entropy';

export interface ProbabilityWheelConfig {
  mode: ProbabilityWheelMode;
  prompts: string[];
  topK: number;
  budget?: number;
  rounds?: number;
  maxEntropyBits?: number;
  allowUserPrompt: boolean;
}

export interface PreparedRound {
  prompt: string;
  tokens: string[];
  probs: number[];
  actualEntropyBits: number;
}

export interface PreparedProbabilityData {
  rounds: PreparedRound[];
}

export interface RoundState extends PreparedRound {
  pick: string | null;
  allocation: number[];
  entropyEstimate: number | null;
  spinResult: string | null;
}

export interface ProbabilityWheelState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: ProbabilityWheelMode;
  config: ProbabilityWheelConfig;
  rounds: RoundState[];
}

export type ProbabilityWheelAction =
  | { type: 'PICK'; roundIndex: number; token: string }
  | { type: 'ALLOCATE'; roundIndex: number; tokenIndex: number; value: number }
  | { type: 'ESTIMATE_ENTROPY'; roundIndex: number; value: number }
  | { type: 'ADD_ROUND'; round: PreparedRound }
  | { type: 'SPIN'; roundIndex: number; random?: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

const DEFAULT_MAX_ENTROPY_BITS = 6;

export async function prepare(
  config: ProbabilityWheelConfig,
  deps: { causalLM: CausalLMDep }
): Promise<PreparedProbabilityData> {
  const rounds: PreparedRound[] = [];
  for (const prompt of config.prompts) {
    const dist = await deps.causalLM.nextTokenDistribution(prompt, config.topK);
    if (dist.tokens.length !== dist.probs.length) {
      throw new Error(
        `Model returned ${dist.tokens.length} tokens and ${dist.probs.length} probabilities`
      );
    }
    rounds.push({
      prompt,
      tokens: dist.tokens,
      probs: dist.probs,
      actualEntropyBits: entropyBits(dist.probs),
    });
  }
  return { rounds };
}

export function initState(
  config: ProbabilityWheelConfig,
  rules: EngineRules,
  prepared: PreparedProbabilityData
): ProbabilityWheelState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    rounds: prepared.rounds.map((round) => ({
      ...round,
      pick: null,
      allocation: new Array<number>(round.tokens.length).fill(0),
      entropyEstimate: null,
      spinResult: null,
    })),
  };
}

export function applyAction(
  state: ProbabilityWheelState,
  action: ProbabilityWheelAction
): ProbabilityWheelState {
  const withRound = (index: number, update: (r: RoundState) => RoundState): ProbabilityWheelState => {
    const round = state.rounds[index];
    if (!round) return state;
    const rounds = [...state.rounds];
    rounds[index] = update(round);
    return { ...state, rounds, status: 'active', actionCount: state.actionCount + 1 };
  };

  switch (action.type) {
    case 'PICK':
      return withRound(action.roundIndex, (round) =>
        round.tokens.includes(action.token) ? { ...round, pick: action.token } : round
      );

    case 'ALLOCATE':
      return withRound(action.roundIndex, (round) => {
        if (action.tokenIndex < 0 || action.tokenIndex >= round.allocation.length) return round;
        if (!Number.isFinite(action.value) || action.value < 0) return round;
        const budget = state.config.budget ?? Infinity;
        const allocation = [...round.allocation];
        allocation[action.tokenIndex] = Math.min(action.value, budget);
        return { ...round, allocation };
      });

    case 'ESTIMATE_ENTROPY':
      return withRound(action.roundIndex, (round) => {
        if (!Number.isFinite(action.value)) return round;
        const max = state.config.maxEntropyBits ?? DEFAULT_MAX_ENTROPY_BITS;
        return { ...round, entropyEstimate: clamp(action.value, 0, max) };
      });

    case 'ADD_ROUND': {
      if (!state.config.allowUserPrompt) return state;
      const max = state.config.rounds ?? Infinity;
      if (state.rounds.length >= max) return state;
      return {
        ...state,
        status: 'active',
        actionCount: state.actionCount + 1,
        rounds: [
          ...state.rounds,
          {
            ...action.round,
            pick: null,
            allocation: new Array<number>(action.round.tokens.length).fill(0),
            entropyEstimate: null,
            spinResult: null,
          },
        ],
      };
    }

    case 'SPIN':
      return withRound(action.roundIndex, (round) => {
        // Real inverse-CDF sampling over the model's own probabilities.
        const r = action.random ?? Math.random();
        let cumulative = 0;
        let landed = round.tokens.at(-1) ?? null;
        for (let i = 0; i < round.probs.length; i++) {
          cumulative += round.probs[i]!;
          if (r < cumulative) {
            landed = round.tokens[i]!;
            break;
          }
        }
        return { ...round, spinResult: landed };
      });

    case 'RESET':
      return initState(state.config, state.rules, {
        rounds: state.rounds.map(({ prompt, tokens, probs, actualEntropyBits }) => ({
          prompt,
          tokens,
          probs,
          actualEntropyBits,
        })),
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: ProbabilityWheelState): ScoreResult {
  switch (state.mode) {
    case 'predict-top1': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'predictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.rounds.filter((round) => {
        if (round.pick === null) return false;
        const topIndex = round.probs.indexOf(Math.max(...round.probs));
        return round.tokens[topIndex] === round.pick;
      }).length;

      return scoreLevel({
        metric: 'predictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'allocate-mass': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'calibrationError', value: 1, rules: state.rules });
      }
      let sum = 0;
      let answered = 0;
      for (const round of state.rounds) {
        const allocated = round.allocation.reduce((a, b) => a + b, 0);
        if (allocated <= 0) {
          // Declining to allocate is the worst possible calibration, not a free pass.
          sum += 1;
          continue;
        }
        answered++;
        sum += totalVariationDistance(normalizeDistribution(round.allocation), round.probs);
      }

      return scoreLevel({
        metric: 'calibrationError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }

    case 'estimate-entropy': {
      const max = state.config.maxEntropyBits ?? DEFAULT_MAX_ENTROPY_BITS;
      // Denominator is the level's round count, so skipping rounds cannot help.
      const total = Math.max(state.config.rounds ?? state.rounds.length, state.rounds.length);
      if (total === 0) {
        return scoreLevel({ metric: 'entropyEstimateError', value: max, rules: state.rules });
      }

      let sum = 0;
      let answered = 0;
      for (let i = 0; i < total; i++) {
        const round = state.rounds[i];
        if (!round || round.entropyEstimate === null) {
          sum += max;
          continue;
        }
        answered++;
        sum += Math.abs(round.entropyEstimate - round.actualEntropyBits);
      }

      return scoreLevel({
        metric: 'entropyEstimateError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }
  }
}
