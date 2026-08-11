/**
 * Chapter 5.4 — Residuals & Layer Norm.
 *
 * Traces are built from the model's real per-layer hidden states. Toggling the
 * residual path off does not fabricate a hypothetical network: it recomputes the
 * trace from the same measured activations using only each block's own
 * contribution — the difference between consecutive layers — which is exactly
 * what the block would emit without the skip connection carrying its input
 * forward.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { HiddenStateDep, HiddenStateResult } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, mean, norm, standardDeviation, subtractVectors } from './shared';

export type ResidualMode = 'predict-trace' | 'toggle-stability' | 'minimise-drift';

export interface ResidualToggleConfig {
  mode: ResidualMode;
  sentence: string;
  layerCount: number;
  residual: boolean;
  layerNorm: boolean;
  residualScale?: number;
  residualScaleRange?: [number, number];
  normEpsilon?: number;
  normEpsilonRange?: [number, number];
  minRepresentationVariance?: number;
  stableBand?: [number, number];
  traceMetric: 'l2norm';
  rounds?: number;
  allowUserSentence?: boolean;
}

export interface PreparedHiddenStates {
  result: HiddenStateResult;
}

export interface TraceRound {
  layer: number;
  trueValue: number;
  estimate: number | null;
}

export interface ResidualToggleState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: ResidualMode;
  config: ResidualToggleConfig;
  residual: boolean;
  layerNorm: boolean;
  residualScale: number;
  normEpsilon: number;
  result: HiddenStateResult;
  /** Current per-layer trace under the active toggles. */
  trace: number[];
  /** Trace with both mechanisms on, for comparison. */
  referenceTrace: number[];
  rounds: TraceRound[];
}

export type ResidualToggleAction =
  | { type: 'SET_RESIDUAL'; value: boolean }
  | { type: 'SET_LAYER_NORM'; value: boolean }
  | { type: 'SET_RESIDUAL_SCALE'; value: number }
  | { type: 'SET_NORM_EPSILON'; value: number }
  | { type: 'ESTIMATE_TRACE'; roundIndex: number; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Mean L2 norm across the token positions of one layer. */
function layerMagnitude(layer: readonly number[][]): number {
  if (layer.length === 0) return 0;
  return mean(layer.map((token) => norm(token)));
}

/** Mean per-token variance — how much representation survives normalisation. */
function layerVariance(layer: readonly number[][]): number {
  if (layer.length === 0) return 0;
  return mean(layer.map((token) => standardDeviation(token) ** 2));
}

export interface TraceOptions {
  residual: boolean;
  layerNorm: boolean;
  residualScale: number;
  normEpsilon: number;
}

/**
 * Recomputes the per-layer magnitude trace under the chosen configuration.
 *
 * With the residual on, each layer's state is what the model actually produced.
 * With it off, only the block's own delta remains, which is what makes deep
 * stacks drift. Layer norm rescales each layer to unit RMS before measuring.
 */
export function computeTrace(
  result: HiddenStateResult,
  options: TraceOptions
): { trace: number[]; variances: number[] } {
  const layers = result.hiddenStates;
  const trace: number[] = [];
  const variances: number[] = [];

  for (let l = 0; l < layers.length; l++) {
    const current = layers[l]!;
    const previous = l > 0 ? layers[l - 1]! : null;

    const effective = options.residual
      ? current.map((token) => token.map((v) => v * options.residualScale))
      : // Without the skip connection only this block's contribution carries on.
        current.map((token, i) =>
          previous && previous[i] ? subtractVectors(token, previous[i]!) : token
        );

    const normalised = options.layerNorm
      ? effective.map((token) => {
          const rms = Math.sqrt(mean(token.map((v) => v * v)) + options.normEpsilon);
          return token.map((v) => v / rms);
        })
      : effective;

    trace.push(layerMagnitude(normalised));
    variances.push(layerVariance(normalised));
  }

  return { trace, variances };
}

export async function prepare(
  config: ResidualToggleConfig,
  deps: { hiddenStates: HiddenStateDep }
): Promise<PreparedHiddenStates> {
  const result = await deps.hiddenStates.hiddenStates(config.sentence);
  if (result.hiddenStates.length === 0) {
    throw new Error(`No hidden states returned for "${config.sentence}"`);
  }
  return { result };
}

export function initState(
  config: ResidualToggleConfig,
  rules: EngineRules,
  prepared: PreparedHiddenStates
): ResidualToggleState {
  const options: TraceOptions = {
    residual: config.residual,
    layerNorm: config.layerNorm,
    residualScale: config.residualScale ?? 1,
    normEpsilon: config.normEpsilon ?? 1e-5,
  };

  const { trace } = computeTrace(prepared.result, options);
  const { trace: referenceTrace } = computeTrace(prepared.result, {
    ...options,
    residual: true,
    layerNorm: true,
  });

  const rounds: TraceRound[] =
    config.mode === 'predict-trace'
      ? Array.from({ length: Math.min(config.rounds ?? 0, trace.length) }, (_, i) => {
          // Space the asked-about layers across the stack.
          const layer = Math.floor(((i + 1) * (trace.length - 1)) / ((config.rounds ?? 1) + 1));
          return { layer, trueValue: trace[layer] ?? 0, estimate: null };
        })
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    residual: options.residual,
    layerNorm: options.layerNorm,
    residualScale: options.residualScale,
    normEpsilon: options.normEpsilon,
    result: prepared.result,
    trace,
    referenceTrace,
    rounds,
  };
}

function recompute(state: ResidualToggleState, patch: Partial<TraceOptions>): ResidualToggleState {
  const options: TraceOptions = {
    residual: patch.residual ?? state.residual,
    layerNorm: patch.layerNorm ?? state.layerNorm,
    residualScale: patch.residualScale ?? state.residualScale,
    normEpsilon: patch.normEpsilon ?? state.normEpsilon,
  };
  const { trace } = computeTrace(state.result, options);
  return {
    ...state,
    ...options,
    trace,
    status: 'active',
    actionCount: state.actionCount + 1,
  };
}

export function applyAction(
  state: ResidualToggleState,
  action: ResidualToggleAction
): ResidualToggleState {
  switch (action.type) {
    case 'SET_RESIDUAL':
      return recompute(state, { residual: action.value });

    case 'SET_LAYER_NORM':
      return recompute(state, { layerNorm: action.value });

    case 'SET_RESIDUAL_SCALE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.residualScaleRange ?? [0, 2];
      return recompute(state, { residualScale: clamp(action.value, min, max) });
    }

    case 'SET_NORM_EPSILON': {
      if (!Number.isFinite(action.value) || action.value <= 0) return state;
      const [min, max] = state.config.normEpsilonRange ?? [1e-7, 0.1];
      return recompute(state, { normEpsilon: clamp(action.value, min, max) });
    }

    case 'ESTIMATE_TRACE': {
      const round = state.rounds[action.roundIndex];
      if (!round || !Number.isFinite(action.value) || action.value < 0) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, estimate: action.value };
      return { ...state, rounds, status: 'active', actionCount: state.actionCount + 1 };
    }

    case 'RESET':
      return initState(state.config, state.rules, { result: state.result });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: ResidualToggleState): ScoreResult {
  const { trace } = state;

  switch (state.mode) {
    case 'predict-trace': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'traceEstimateError', value: 1, rules: state.rules });
      }
      // Errors are relative to the trace's own scale, so the level does not
      // depend on the model's absolute activation magnitudes.
      const scale = Math.max(mean(trace), 1e-6);
      let answered = 0;
      let sum = 0;
      for (const round of state.rounds) {
        if (round.estimate === null) {
          sum += 1;
          continue;
        }
        answered++;
        sum += Math.abs(round.estimate - round.trueValue) / scale;
      }
      return scoreLevel({
        metric: 'traceEstimateError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total, scale },
      });
    }

    case 'toggle-stability': {
      const [low, high] = state.config.stableBand ?? [0.5, 2];
      if (trace.length === 0) {
        return scoreLevel({ metric: 'stabilityScore', value: 0, rules: state.rules });
      }
      // Compare each layer against the stack's own mean, so stability means
      // "the scale is not running away with depth".
      const reference = Math.max(mean(trace), 1e-9);
      const inBand = trace.filter((v) => v / reference >= low && v / reference <= high).length;
      return scoreLevel({
        metric: 'stabilityScore',
        value: inBand / trace.length,
        rules: state.rules,
        breakdown: {
          inBand,
          layers: trace.length,
          residual: state.residual ? 1 : 0,
          layerNorm: state.layerNorm ? 1 : 0,
        },
      });
    }

    case 'minimise-drift': {
      if (trace.length < 2) {
        return scoreLevel({ metric: 'activationDrift', value: 1, rules: state.rules });
      }
      const steps: number[] = [];
      for (let i = 1; i < trace.length; i++) {
        const previous = Math.max(trace[i - 1]!, 1e-9);
        steps.push(Math.abs(trace[i]! - previous) / previous);
      }
      const drift = mean(steps);

      // Flattening the representation would trivially minimise drift, so a
      // collapsed representation is charged back against the score.
      const { variances } = computeTrace(state.result, {
        residual: state.residual,
        layerNorm: state.layerNorm,
        residualScale: state.residualScale,
        normEpsilon: state.normEpsilon,
      });
      const variance = mean(variances);
      const floor = state.config.minRepresentationVariance ?? 0;
      const penalty = variance < floor ? (floor - variance) / Math.max(floor, 1e-9) : 0;

      return scoreLevel({
        metric: 'activationDrift',
        value: drift + penalty,
        rules: state.rules,
        breakdown: { drift, variance, floor, collapsePenalty: penalty },
      });
    }
  }
}
