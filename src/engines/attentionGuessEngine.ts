/**
 * Chapter 5.2 — Self-Attention. The centrepiece.
 *
 * Every weight scored against here is extracted from a real transformer's
 * attention tensor. Nothing is approximated: the argmax the player guesses is
 * the model's actual argmax, and the distribution they draw is compared against
 * the model's actual softmaxed row.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { AttentionDep, AttentionResult } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, normalizeDistribution, totalVariationDistance } from './shared';

export type AttentionMode = 'guess-argmax' | 'draw-distribution' | 'flip-reference';
export type HeadAggregation = 'mean' | 'max';

export interface AttentionGuessConfig {
  mode: AttentionMode;
  sentences: string[];
  layer: number;
  layerRange?: [number, number];
  headAggregation: HeadAggregation;
  excludeSpecialTokens: boolean;
  rounds?: number;
  budget?: number;
  allowUserSentence?: boolean;
  queryToken?: string;
  maxEditDistance?: number;
  attempts?: number;
  minFlipMargin?: number;
}

export interface AttentionRound {
  sentence: string;
  tokens: string[];
  /** Index of the token whose attention row is under inspection. */
  queryIndex: number;
  /** Candidate key indices, special tokens already removed if configured. */
  keyIndices: number[];
  /** The model's real attention row over those keys, renormalised. */
  trueRow: number[];
  guessIndex: number | null;
  allocation: number[];
}

export interface FlipAttempt {
  sentence: string;
  editDistance: number;
  winnerIndex: number;
  margin: number;
  flipped: boolean;
  score: number;
}

export interface PreparedAttentionData {
  results: AttentionResult[];
}

export interface AttentionGuessState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: AttentionMode;
  config: AttentionGuessConfig;
  layer: number;
  rounds: AttentionRound[];
  /** Baseline for the flip level: which token originally wins. */
  baseline: { sentence: string; winnerIndex: number; candidates: number[] } | null;
  attempts: FlipAttempt[];
  bestFlipScore: number;
}

export type AttentionGuessAction =
  | { type: 'GUESS'; roundIndex: number; keyIndex: number }
  | { type: 'ALLOCATE'; roundIndex: number; keyIndex: number; value: number }
  | { type: 'SET_LAYER'; value: number }
  | { type: 'ADD_ROUND'; round: AttentionRound }
  | { type: 'SUBMIT_EDIT'; sentence: string; attention: AttentionResult }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/** Tokens the model adds that are not part of the sentence. */
const SPECIAL_TOKENS = new Set(['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']);

export function isSpecialToken(token: string): boolean {
  return SPECIAL_TOKENS.has(token);
}

/**
 * Collapses the head dimension of a real attention tensor into one row.
 *
 * `attention` is [layer][head][query][key]; this returns the aggregated
 * distribution over keys for one query position.
 */
export function attentionRow(
  result: AttentionResult,
  layer: number,
  queryIndex: number,
  aggregation: HeadAggregation
): number[] {
  const layerTensor = result.attention[layer];
  if (!layerTensor || layerTensor.length === 0) return [];

  const heads = layerTensor
    .map((head) => head[queryIndex])
    .filter((row): row is number[] => Array.isArray(row));
  if (heads.length === 0) return [];

  const width = heads[0]!.length;
  return Array.from({ length: width }, (_, key) => {
    const values = heads.map((row) => row[key] ?? 0);
    return aggregation === 'max'
      ? Math.max(...values)
      : values.reduce((a, b) => a + b, 0) / values.length;
  });
}

/** Picks the query position: the configured token, else the last real token. */
function findQueryIndex(tokens: readonly string[], queryToken: string | undefined): number {
  if (queryToken) {
    const index = tokens.findIndex((t) => t.toLowerCase() === queryToken.toLowerCase());
    if (index !== -1) return index;
  }
  // "it" is the interesting query in every reference-resolution sentence here.
  const pronoun = tokens.findIndex((t) => t.toLowerCase() === 'it');
  if (pronoun !== -1) return pronoun;

  for (let i = tokens.length - 1; i >= 0; i--) {
    if (!isSpecialToken(tokens[i]!)) return i;
  }
  return 0;
}

export function buildRound(
  config: AttentionGuessConfig,
  result: AttentionResult,
  layer: number
): AttentionRound {
  const { tokens } = result;
  const queryIndex = findQueryIndex(tokens, config.queryToken);

  const keyIndices = tokens
    .map((token, i) => ({ token, i }))
    .filter(({ token, i }) => {
      if (i === queryIndex) return false;
      return !(config.excludeSpecialTokens && isSpecialToken(token));
    })
    .map(({ i }) => i);

  const fullRow = attentionRow(result, layer, queryIndex, config.headAggregation);
  // Renormalise over the candidate keys so the row remains a distribution after
  // dropping the query itself and any special tokens.
  const trueRow = normalizeDistribution(keyIndices.map((i) => fullRow[i] ?? 0));

  return {
    sentence: result.tokens.join(' '),
    tokens,
    queryIndex,
    keyIndices,
    trueRow,
    guessIndex: null,
    allocation: new Array<number>(keyIndices.length).fill(0),
  };
}

export async function prepare(
  config: AttentionGuessConfig,
  deps: { attention: AttentionDep }
): Promise<PreparedAttentionData> {
  const results: AttentionResult[] = [];
  for (const sentence of config.sentences) {
    const result = await deps.attention.attention(sentence);
    if (result.attention.length === 0) {
      throw new Error(`No attention returned for "${sentence}"`);
    }
    results.push(result);
  }
  return { results };
}

/** Levenshtein distance, used to keep the reference-flip edit minimal. */
export function editDistance(a: string, b: string): number {
  const previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = previous[0]!;
    previous[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const temp = previous[j]!;
      previous[j] = Math.min(
        previous[j]! + 1,
        previous[j - 1]! + 1,
        diagonal + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      diagonal = temp;
    }
  }
  return previous[b.length]!;
}

export function initState(
  config: AttentionGuessConfig,
  rules: EngineRules,
  prepared: PreparedAttentionData
): AttentionGuessState {
  const layer = config.layer;
  const limit = config.rounds ?? prepared.results.length;
  const rounds = prepared.results.slice(0, limit).map((result) => buildRound(config, result, layer));

  // The flip level needs a baseline: which token currently wins, and the two
  // strongest candidates it could flip between.
  let baseline: AttentionGuessState['baseline'] = null;
  if (config.mode === 'flip-reference' && rounds[0]) {
    const round = rounds[0];
    const ranked = round.trueRow
      .map((weight, i) => ({ weight, index: round.keyIndices[i]! }))
      .sort((a, b) => b.weight - a.weight);
    baseline = {
      sentence: config.sentences[0] ?? '',
      winnerIndex: ranked[0]?.index ?? -1,
      candidates: ranked.slice(0, 2).map((r) => r.index),
    };
  }

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    layer,
    rounds,
    baseline,
    attempts: [],
    bestFlipScore: 0,
  };
}

export function applyAction(
  state: AttentionGuessState,
  action: AttentionGuessAction
): AttentionGuessState {
  const bump = (next: Partial<AttentionGuessState>): AttentionGuessState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'GUESS': {
      const round = state.rounds[action.roundIndex];
      if (!round || !round.keyIndices.includes(action.keyIndex)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, guessIndex: action.keyIndex };
      return bump({ rounds });
    }

    case 'ALLOCATE': {
      const round = state.rounds[action.roundIndex];
      if (!round) return state;
      const slot = round.keyIndices.indexOf(action.keyIndex);
      if (slot === -1) return state;
      if (!Number.isFinite(action.value) || action.value < 0) return state;

      const allocation = [...round.allocation];
      allocation[slot] = Math.min(action.value, state.config.budget ?? Infinity);
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = { ...round, allocation };
      return bump({ rounds });
    }

    case 'SET_LAYER': {
      const [min, max] = state.config.layerRange ?? [state.config.layer, state.config.layer];
      if (!Number.isInteger(action.value)) return state;
      return bump({ layer: Math.round(clamp(action.value, min, max)) });
    }

    case 'ADD_ROUND':
      if (!state.config.allowUserSentence) return state;
      return bump({ rounds: [...state.rounds, action.round] });

    case 'SUBMIT_EDIT': {
      if (state.mode !== 'flip-reference' || !state.baseline) return state;
      const max = state.config.attempts ?? Infinity;
      if (state.attempts.length >= max) return state;

      const round = buildRound(state.config, action.attention, state.layer);
      const ranked = round.trueRow
        .map((weight, i) => ({ weight, index: round.keyIndices[i]! }))
        .sort((a, b) => b.weight - a.weight);

      const winnerIndex = ranked[0]?.index ?? -1;
      const margin = (ranked[0]?.weight ?? 0) - (ranked[1]?.weight ?? 0);
      const distance = editDistance(state.baseline.sentence, action.sentence);
      const minMargin = state.config.minFlipMargin ?? 0;
      const maxEdit = state.config.maxEditDistance ?? Infinity;

      const flipped = winnerIndex !== state.baseline.winnerIndex && margin >= minMargin;
      // Flipping is most of the credit; doing it with a small edit earns the rest.
      const editCredit = maxEdit === Infinity ? 1 : clamp(1 - distance / maxEdit, 0, 1);
      const score = flipped ? 0.6 + 0.4 * editCredit : 0;

      return bump({
        attempts: [
          ...state.attempts,
          { sentence: action.sentence, editDistance: distance, winnerIndex, margin, flipped, score },
        ],
        bestFlipScore: Math.max(state.bestFlipScore, score),
      });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        results: state.rounds.map((r) => ({ tokens: r.tokens, attention: [] })),
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: AttentionGuessState): ScoreResult {
  switch (state.mode) {
    case 'guess-argmax': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'attentionGuessAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.rounds.filter((round) => {
        if (round.guessIndex === null || round.trueRow.length === 0) return false;
        let best = 0;
        for (let i = 1; i < round.trueRow.length; i++) {
          if (round.trueRow[i]! > round.trueRow[best]!) best = i;
        }
        return round.keyIndices[best] === round.guessIndex;
      }).length;

      return scoreLevel({
        metric: 'attentionGuessAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total, layer: state.layer },
      });
    }

    case 'draw-distribution': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'attentionDistributionError', value: 1, rules: state.rules });
      }
      let answered = 0;
      let sum = 0;
      for (const round of state.rounds) {
        const allocated = round.allocation.reduce((a, b) => a + b, 0);
        if (allocated <= 0 || round.trueRow.length === 0) {
          // Not answering is the worst possible distribution, not a free pass.
          sum += 1;
          continue;
        }
        answered++;
        sum += totalVariationDistance(normalizeDistribution(round.allocation), round.trueRow);
      }

      return scoreLevel({
        metric: 'attentionDistributionError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total, layer: state.layer },
      });
    }

    case 'flip-reference':
      return scoreLevel({
        metric: 'referenceFlipScore',
        value: state.bestFlipScore,
        rules: state.rules,
        breakdown: {
          attempts: state.attempts.length,
          flips: state.attempts.filter((a) => a.flipped).length,
        },
      });
  }
}
