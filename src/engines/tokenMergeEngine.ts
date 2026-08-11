/**
 * Chapter 1.4 — Tokenization.
 *
 * Counts come from the real tokenizer. The merge puzzle is scored against the
 * genuine BPE merge order, replayed from the tokenizer's own merge-rank table
 * rather than an authored sequence.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { TokenizerDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp } from './shared';

export type TokenMergeMode = 'count-tokens' | 'merge-puzzle' | 'break-tokenizer';

export interface TokenMergeConfig {
  mode: TokenMergeMode;
  samples?: string[];
  tolerance?: number;
  words?: string[];
  maxMergesPerWord?: number;
  maxChars?: number;
  minChars?: number;
  attempts?: number;
  allowUserWords?: boolean;
}

export interface Merge {
  left: string;
  right: string;
  rank: number;
}

export interface PreparedSample {
  text: string;
  tokens: string[];
  count: number;
}

export interface PreparedPuzzle {
  word: string;
  symbols: string[];
  merges: Merge[];
}

export interface PreparedTokenData {
  samples: PreparedSample[];
  puzzles: PreparedPuzzle[];
}

export interface SampleState extends PreparedSample {
  guess: number | null;
}

export interface PuzzleState {
  word: string;
  symbols: string[];
  merges: Merge[];
  stepIndex: number;
  correctSteps: number;
  attemptedSteps: number;
  done: boolean;
}

export interface BreakAttempt {
  text: string;
  tokenCount: number;
  charCount: number;
}

export interface TokenMergeState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: TokenMergeMode;
  config: TokenMergeConfig;
  samples: SampleState[];
  puzzles: PuzzleState[];
  attempts: BreakAttempt[];
  bestFertility: number;
}

export type TokenMergeAction =
  | { type: 'GUESS_COUNT'; sampleIndex: number; value: number }
  | { type: 'APPLY_MERGE'; puzzleIndex: number; position: number }
  | { type: 'SUBMIT_ATTEMPT'; attempt: BreakAttempt }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/**
 * Tokens-per-character bounds used to normalise the "break the tokenizer"
 * score: a well-handled English string sits near 0.25, byte-level fallback
 * (one token per character) sits at 1.
 */
const EFFICIENT_RATIO = 0.25;
const SHATTERED_RATIO = 1;

export async function prepare(
  config: TokenMergeConfig,
  deps: { tokenizer: TokenizerDep }
): Promise<PreparedTokenData> {
  const samples: PreparedSample[] = [];
  for (const text of config.samples ?? []) {
    const tokens = await deps.tokenizer.tokenize(text);
    samples.push({ text, tokens, count: tokens.length });
  }

  const puzzles: PreparedPuzzle[] = [];
  if ((config.words ?? []).length > 0) {
    const ranks = await deps.tokenizer.mergeRanks();
    for (const word of config.words!) {
      puzzles.push(buildPuzzle(word, ranks, config.maxMergesPerWord ?? Infinity));
    }
  }

  return { samples, puzzles };
}

/**
 * Replays BPE from single characters: at each step the adjacent pair with the
 * lowest merge rank wins, which is exactly the order the real tokenizer uses.
 */
function buildPuzzle(
  word: string,
  ranks: ReadonlyMap<string, number>,
  maxMerges: number
): PreparedPuzzle {
  const symbols = [...word];
  const working = [...symbols];
  const merges: Merge[] = [];

  while (merges.length < maxMerges && working.length > 1) {
    let bestIndex = -1;
    let bestRank = Infinity;
    for (let i = 0; i < working.length - 1; i++) {
      const rank = ranks.get(`${working[i]} ${working[i + 1]}`);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex === -1) break;

    merges.push({ left: working[bestIndex]!, right: working[bestIndex + 1]!, rank: bestRank });
    working.splice(bestIndex, 2, working[bestIndex]! + working[bestIndex + 1]!);
  }

  return { word, symbols, merges };
}

export function initState(
  config: TokenMergeConfig,
  rules: EngineRules,
  prepared: PreparedTokenData
): TokenMergeState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    samples: prepared.samples.map((s) => ({ ...s, guess: null })),
    puzzles: prepared.puzzles.map((p) => ({
      word: p.word,
      symbols: [...p.symbols],
      merges: p.merges,
      stepIndex: 0,
      correctSteps: 0,
      attemptedSteps: 0,
      done: p.merges.length === 0,
    })),
    attempts: [],
    bestFertility: 0,
  };
}

export function applyAction(state: TokenMergeState, action: TokenMergeAction): TokenMergeState {
  const bump = (next: Partial<TokenMergeState>): TokenMergeState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'GUESS_COUNT': {
      const sample = state.samples[action.sampleIndex];
      if (!sample) return state;
      // Token counts are whole and non-negative; anything else is a UI bug.
      if (!Number.isInteger(action.value) || action.value < 0) return state;

      const samples = [...state.samples];
      samples[action.sampleIndex] = { ...sample, guess: action.value };
      return bump({ samples });
    }

    case 'APPLY_MERGE': {
      const puzzle = state.puzzles[action.puzzleIndex];
      if (!puzzle || puzzle.done) return state;
      if (action.position < 0 || action.position >= puzzle.symbols.length - 1) return state;

      const expected = puzzle.merges[puzzle.stepIndex];
      const left = puzzle.symbols[action.position]!;
      const right = puzzle.symbols[action.position + 1]!;
      const isCorrect = expected !== undefined && expected.left === left && expected.right === right;

      const symbols = [...puzzle.symbols];
      symbols.splice(action.position, 2, left + right);

      const puzzles = [...state.puzzles];
      puzzles[action.puzzleIndex] = {
        ...puzzle,
        symbols,
        // Only a correct merge advances the expected step; a wrong one still
        // changes the board, which is what makes the mistake visible.
        stepIndex: isCorrect ? puzzle.stepIndex + 1 : puzzle.stepIndex,
        correctSteps: puzzle.correctSteps + (isCorrect ? 1 : 0),
        attemptedSteps: puzzle.attemptedSteps + 1,
        done: (isCorrect ? puzzle.stepIndex + 1 : puzzle.stepIndex) >= puzzle.merges.length
          || symbols.length === 1,
      };
      return bump({ puzzles });
    }

    case 'SUBMIT_ATTEMPT': {
      const max = state.config.attempts ?? Infinity;
      if (state.attempts.length >= max) return state;

      const { text, tokenCount, charCount } = action.attempt;
      if (charCount <= 0 || tokenCount < 0) return state;
      if (state.config.minChars !== undefined && charCount < state.config.minChars) return state;
      if (state.config.maxChars !== undefined && charCount > state.config.maxChars) return state;

      const ratio = tokenCount / charCount;
      const fertility = clamp(
        (ratio - EFFICIENT_RATIO) / (SHATTERED_RATIO - EFFICIENT_RATIO),
        0,
        1
      );

      return bump({
        attempts: [...state.attempts, { text, tokenCount, charCount }],
        bestFertility: Math.max(state.bestFertility, fertility),
      });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        samples: state.samples.map(({ text, tokens, count }) => ({ text, tokens, count })),
        puzzles: state.puzzles.map((p) => ({
          word: p.word,
          symbols: [...p.word],
          merges: p.merges,
        })),
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: TokenMergeState): ScoreResult {
  switch (state.mode) {
    case 'count-tokens': {
      const total = state.samples.length;
      if (total === 0) {
        return scoreLevel({ metric: 'tokenCountAccuracy', value: 0, rules: state.rules });
      }
      const tolerance = state.config.tolerance ?? 0;
      const correct = state.samples.filter(
        (s) => s.guess !== null && Math.abs(s.guess - s.count) <= tolerance
      ).length;

      return scoreLevel({
        metric: 'tokenCountAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'merge-puzzle': {
      const totalSteps = state.puzzles.reduce((n, p) => n + p.merges.length, 0);
      if (totalSteps === 0) {
        return scoreLevel({ metric: 'mergeOrderAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.puzzles.reduce((n, p) => n + p.correctSteps, 0);
      const attempted = state.puzzles.reduce((n, p) => n + p.attemptedSteps, 0);

      return scoreLevel({
        metric: 'mergeOrderAccuracy',
        value: correct / totalSteps,
        rules: state.rules,
        breakdown: { correct, attempted, totalSteps },
      });
    }

    case 'break-tokenizer':
      return scoreLevel({
        metric: 'fertilityScore',
        value: state.bestFertility,
        rules: state.rules,
        breakdown: { attempts: state.attempts.length },
      });
  }
}
