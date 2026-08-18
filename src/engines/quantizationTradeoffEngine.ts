/**
 * Chapter 8.1 — Quantization Tradeoffs.
 *
 * Both "precisions" here are the same model (SmolLM2-135M-Instruct, the model
 * `tinyCausalLM` already ships) loaded twice at different `dtype`s — resolves
 * plan-docs A4 via the cheaper path it flagged: `transformers.js`'s `dtype`
 * option on `from_pretrained` covers this with zero new Hub repos, confirmed
 * directly (see `src/models/quantizationModel.ts`). Every divergence, every
 * timing number and every surprisal value below comes from a real forward
 * pass through one of the two injected `PrecisionModelDep`s — nothing here is
 * an authored "quantization is worse" assumption. It measurably is not,
 * always: `compare-outputs` finds only some prompts diverge at all, and
 * `compare-speed`'s real per-token inference latency has no consistent
 * winner at this model's tiny size, only its real download size and load
 * time do. `compare-confidence` is the one place quantization consistently
 * costs something real — but only once the leading, purely-formatting token
 * of the answer's tokenization is excluded from the average (see
 * `contentTokenMeanBits` below); including it swamps the actual numeric-digit
 * signal with a near-universal "does it emit a factual-sounding token first"
 * effect that has nothing to do with the answer's digits.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CorpusDep, PrecisionModelDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { mean } from './shared';

export type QuantizationMode = 'compare-outputs' | 'compare-speed' | 'compare-confidence';

export interface QuantizationTradeoffConfig {
  mode: QuantizationMode;
  /** compare-outputs: fixed prompt set. compare-speed: single probe prompt. */
  prompts?: string[];
  topK?: number;
  /** compare-confidence: corpus + which of its facts to compare. */
  corpus?: string;
  factIds?: string[];
}

export interface QuantizationFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

interface RawCorpus {
  facts: QuantizationFact[];
}

export interface DivergenceRound {
  prompt: string;
  referenceTop: string;
  referenceProb: number;
  quantizedTop: string;
  quantizedProb: number;
  /** Real: do the two precisions' actual top tokens differ. */
  diverges: boolean;
  pick: boolean | null;
}

export interface SpeedMeasurement {
  referenceSizeMB: number;
  quantizedSizeMB: number;
  referenceLoadMs: number;
  quantizedLoadMs: number;
  referenceInferenceMs: number;
  quantizedInferenceMs: number;
}

export type SpeedPredictionField = 'smallerIsQuantized' | 'fasterLoadIsQuantized' | 'fasterInferenceIsQuantized';

export interface SpeedPredictions {
  smallerIsQuantized: boolean | null;
  fasterLoadIsQuantized: boolean | null;
  fasterInferenceIsQuantized: boolean | null;
}

export interface ConfidenceRound {
  factId: string;
  topic: string;
  query: string;
  answer: string;
  referenceMeanBits: number;
  quantizedMeanBits: number;
  /** Real: fp32's content-token surprisal is lower (more confident) than q8's. */
  referenceMoreConfident: boolean;
  pick: 'reference' | 'quantized' | null;
}

export interface PreparedQuantizationData {
  divergenceRounds: DivergenceRound[];
  speedMeasurement: SpeedMeasurement | null;
  confidenceRounds: ConfidenceRound[];
}

export interface QuantizationTradeoffState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: QuantizationMode;
  config: QuantizationTradeoffConfig;

  divergenceRounds: DivergenceRound[];
  speedMeasurement: SpeedMeasurement | null;
  speedPredictions: SpeedPredictions;
  confidenceRounds: ConfidenceRound[];
}

export type QuantizationTradeoffAction =
  | { type: 'PICK_DIVERGENCE'; roundIndex: number; guess: boolean }
  | { type: 'PREDICT_SPEED'; field: SpeedPredictionField; value: boolean }
  | { type: 'PICK_CONFIDENCE'; roundIndex: number; pick: 'reference' | 'quantized' }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/**
 * Mean surprisal over every token except the first.
 *
 * The first token of a tokenized continuation is almost always a generic
 * leading-space/formatting token shared by nearly every answer regardless of
 * its content — measured directly (plan-docs D, 8.1 findings): including it
 * makes the metric mostly measure "does this precision start an answer the
 * same way", not whether it knows the actual digits. Falls back to the full
 * mean for a single-token continuation, where there is no separate content
 * token to isolate.
 */
function contentTokenMeanBits(bits: readonly number[]): number {
  if (bits.length <= 1) return mean(bits);
  return mean(bits.slice(1));
}

function buildPrompt(fact: QuantizationFact): string {
  return `Context: ${fact.sentences.join(' ')}\nQuestion: ${fact.query}\nAnswer:`;
}

export async function prepare(
  config: QuantizationTradeoffConfig,
  deps: { reference: PrecisionModelDep; quantized: PrecisionModelDep; corpus?: CorpusDep }
): Promise<PreparedQuantizationData> {
  const divergenceRounds: DivergenceRound[] = [];
  let speedMeasurement: SpeedMeasurement | null = null;
  const confidenceRounds: ConfidenceRound[] = [];

  if (config.mode === 'compare-outputs') {
    const topK = config.topK ?? 5;
    for (const prompt of config.prompts ?? []) {
      const [refResult, quantResult] = await Promise.all([
        deps.reference.run(prompt, topK),
        deps.quantized.run(prompt, topK),
      ]);
      const referenceTop = refResult.topK.tokens[0] ?? '';
      const quantizedTop = quantResult.topK.tokens[0] ?? '';
      divergenceRounds.push({
        prompt,
        referenceTop,
        referenceProb: refResult.topK.probs[0] ?? 0,
        quantizedTop,
        quantizedProb: quantResult.topK.probs[0] ?? 0,
        diverges: referenceTop !== quantizedTop,
        pick: null,
      });
    }
  }

  if (config.mode === 'compare-speed') {
    const probePrompt = config.prompts?.[0] ?? '';
    const [referenceLoadMs, quantizedLoadMs] = await Promise.all([
      deps.reference.ensureLoaded(),
      deps.quantized.ensureLoaded(),
    ]);
    const [refRun, quantRun] = await Promise.all([
      deps.reference.run(probePrompt, 1),
      deps.quantized.run(probePrompt, 1),
    ]);
    speedMeasurement = {
      referenceSizeMB: deps.reference.sizeMB,
      quantizedSizeMB: deps.quantized.sizeMB,
      referenceLoadMs,
      quantizedLoadMs,
      referenceInferenceMs: refRun.inferenceTimeMs,
      quantizedInferenceMs: quantRun.inferenceTimeMs,
    };
  }

  if (config.mode === 'compare-confidence') {
    if (!deps.corpus) throw new Error('compare-confidence mode requires a corpus dependency');
    const raw = await deps.corpus.load(config.corpus ?? '');
    const parsed = JSON.parse(raw) as RawCorpus;

    for (const fact of parsed.facts) {
      const passage = fact.sentences.join(' ');
      if (!passage.includes(fact.answer)) {
        throw new Error(`Fact "${fact.id}"'s answer "${fact.answer}" does not appear in its own passage`);
      }
    }

    const factById = new Map(parsed.facts.map((f) => [f.id, f]));
    for (const factId of config.factIds ?? []) {
      const fact = factById.get(factId);
      if (!fact) continue;
      const prompt = buildPrompt(fact);
      const continuation = ` ${fact.answer}`;
      const [refBits, quantBits] = await Promise.all([
        deps.reference.continuationSurprisal(prompt, continuation),
        deps.quantized.continuationSurprisal(prompt, continuation),
      ]);
      const referenceMeanBits = contentTokenMeanBits(refBits);
      const quantizedMeanBits = contentTokenMeanBits(quantBits);
      confidenceRounds.push({
        factId: fact.id,
        topic: fact.topic,
        query: fact.query,
        answer: fact.answer,
        referenceMeanBits,
        quantizedMeanBits,
        referenceMoreConfident: referenceMeanBits < quantizedMeanBits,
        pick: null,
      });
    }
  }

  return { divergenceRounds, speedMeasurement, confidenceRounds };
}

export function initState(
  config: QuantizationTradeoffConfig,
  rules: EngineRules,
  prepared: PreparedQuantizationData
): QuantizationTradeoffState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    divergenceRounds: prepared.divergenceRounds.map((r) => ({ ...r, pick: null })),
    speedMeasurement: prepared.speedMeasurement,
    speedPredictions: { smallerIsQuantized: null, fasterLoadIsQuantized: null, fasterInferenceIsQuantized: null },
    confidenceRounds: prepared.confidenceRounds.map((r) => ({ ...r, pick: null })),
  };
}

export function applyAction(
  state: QuantizationTradeoffState,
  action: QuantizationTradeoffAction
): QuantizationTradeoffState {
  const bump = (next: Partial<QuantizationTradeoffState>): QuantizationTradeoffState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'PICK_DIVERGENCE': {
      const round = state.divergenceRounds[action.roundIndex];
      if (!round) return state;
      const divergenceRounds = [...state.divergenceRounds];
      divergenceRounds[action.roundIndex] = { ...round, pick: action.guess };
      return bump({ divergenceRounds });
    }

    case 'PREDICT_SPEED':
      return bump({ speedPredictions: { ...state.speedPredictions, [action.field]: action.value } });

    case 'PICK_CONFIDENCE': {
      const round = state.confidenceRounds[action.roundIndex];
      if (!round) return state;
      const confidenceRounds = [...state.confidenceRounds];
      confidenceRounds[action.roundIndex] = { ...round, pick: action.pick };
      return bump({ confidenceRounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        divergenceRounds: state.divergenceRounds,
        speedMeasurement: state.speedMeasurement,
        confidenceRounds: state.confidenceRounds,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: QuantizationTradeoffState): ScoreResult {
  switch (state.mode) {
    case 'compare-outputs': {
      const total = state.divergenceRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'divergenceAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.divergenceRounds.filter((r) => r.pick !== null && r.pick === r.diverges).length;
      return scoreLevel({
        metric: 'divergenceAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'compare-speed': {
      if (!state.speedMeasurement) {
        return scoreLevel({ metric: 'tradeoffPredictionAccuracy', value: 0, rules: state.rules });
      }
      const m = state.speedMeasurement;
      const truth: SpeedPredictions = {
        smallerIsQuantized: m.quantizedSizeMB < m.referenceSizeMB,
        fasterLoadIsQuantized: m.quantizedLoadMs < m.referenceLoadMs,
        fasterInferenceIsQuantized: m.quantizedInferenceMs < m.referenceInferenceMs,
      };
      const fields: SpeedPredictionField[] = ['smallerIsQuantized', 'fasterLoadIsQuantized', 'fasterInferenceIsQuantized'];
      const answered = fields.filter((f) => state.speedPredictions[f] !== null);
      const correct = fields.filter(
        (f) => state.speedPredictions[f] !== null && state.speedPredictions[f] === truth[f]
      ).length;
      if (answered.length < fields.length) {
        return scoreLevel({
          metric: 'tradeoffPredictionAccuracy',
          value: 0,
          rules: state.rules,
          breakdown: { correct, total: fields.length, answered: answered.length },
        });
      }
      return scoreLevel({
        metric: 'tradeoffPredictionAccuracy',
        value: correct / fields.length,
        rules: state.rules,
        breakdown: { correct, total: fields.length, answered: answered.length },
      });
    }

    case 'compare-confidence': {
      const total = state.confidenceRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'confidencePredictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.confidenceRounds.filter(
        (r) => r.pick !== null && r.pick === (r.referenceMoreConfident ? 'reference' : 'quantized')
      ).length;
      return scoreLevel({
        metric: 'confidencePredictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }
  }
}
