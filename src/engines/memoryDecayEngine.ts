/**
 * Chapter 4.2 — Why Recurrence Isn't Enough.
 *
 * This chapter exists to show a real limitation, so nothing here is scripted.
 * A genuine char-level RNN is trained in the browser, and the decay curve is
 * measured by running two sequences that differ only in one early token and
 * watching the network's actual hidden states converge as it overwrites them.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { TinyRNN, buildCharVocab, encodeChars, type CharVocab } from '@/models/tinyRNNTrainer';
import type { CorpusDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { clamp, cosineSimilarity, createRng } from './shared';

export type MemoryDecayMode = 'train' | 'predict-decay' | 'recall-task';

export interface MemoryDecayConfig {
  mode: MemoryDecayMode;
  corpus: string;
  hiddenSize: number;
  hiddenSizeRange?: [number, number];
  sequenceLength: number;
  learningRate: number;
  learningRateRange?: [number, number];
  epochs: number;
  maxEpochs?: number;
  seed: number;
  vocabLimit: number;
  probeToken?: string;
  decayThreshold?: number;
  rounds?: number;
  allowUserInput?: boolean;
  gapLengths?: number[];
  trials?: number;
}

export interface PreparedRnnCorpus {
  text: string;
}

export interface DecayRound {
  /** Distance at which the state stops distinguishing the early token. */
  trueDecayStep: number;
  estimate: number | null;
  /** Measured state similarity at each step, for the plot. */
  similarityTrace: number[];
}

export interface RecallResult {
  gap: number;
  accuracy: number;
}

export interface MemoryDecayState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: MemoryDecayMode;
  config: MemoryDecayConfig;
  hiddenSize: number;
  learningRate: number;
  epochsTrained: number;
  vocab: CharVocab;
  encoded: number[];
  rnn: TinyRNN;
  lossHistory: number[];
  rounds: DecayRound[];
  recallResults: RecallResult[];
}

export type MemoryDecayAction =
  | { type: 'TRAIN'; epochs: number }
  | { type: 'SET_HIDDEN_SIZE'; value: number }
  | { type: 'SET_LEARNING_RATE'; value: number }
  | { type: 'ESTIMATE_DECAY'; roundIndex: number; value: number }
  | { type: 'RUN_RECALL' }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

export async function prepare(
  config: MemoryDecayConfig,
  deps: { corpus: CorpusDep }
): Promise<PreparedRnnCorpus> {
  // The recall task is a controlled probe, so its data is generated rather than
  // drawn from prose — the gap length has to be exact for the measurement.
  if (config.corpus === 'synthetic-recall') {
    return { text: generateRecallCorpus(config) };
  }
  const text = await deps.corpus.load(config.corpus);
  if (text.length === 0) throw new Error(`Corpus "${config.corpus}" is empty`);
  return { text };
}

function generateRecallCorpus(config: MemoryDecayConfig): string {
  const rng = createRng(config.seed);
  const filler = 'abcdefgh';
  const markers = 'XY';
  let out = '';

  for (let i = 0; i < 200; i++) {
    const marker = markers[Math.floor(rng() * markers.length)]!;
    const gap = (config.gapLengths ?? [10])[i % (config.gapLengths ?? [10]).length]!;
    let middle = '';
    for (let j = 0; j < gap; j++) middle += filler[Math.floor(rng() * filler.length)]!;
    // The marker must be recalled after the gap to predict the closing token.
    out += `${marker}${middle}${marker}`;
  }
  return out;
}

function trainCopy(
  rnn: TinyRNN,
  encoded: readonly number[],
  sequenceLength: number,
  learningRate: number,
  epochs: number,
  startEpoch: number
): { rnn: TinyRNN; losses: number[] } {
  const copy = rnn.clone();
  const losses: number[] = [];

  for (let e = 0; e < epochs; e++) {
    // Walk the corpus in windows, so each epoch sees the whole text.
    let total = 0;
    let windows = 0;
    const offset = ((startEpoch + e) * 7) % Math.max(1, sequenceLength);

    for (let start = offset; start + sequenceLength + 1 < encoded.length; start += sequenceLength) {
      const inputs = encoded.slice(start, start + sequenceLength);
      const targets = encoded.slice(start + 1, start + sequenceLength + 1);
      total += copy.trainSequence(inputs, targets, learningRate);
      windows++;
    }
    losses.push(windows === 0 ? 0 : total / windows);
  }

  return { rnn: copy, losses };
}

/**
 * Measures how long one early token stays visible in the hidden state.
 *
 * Runs two sequences identical except for their first token and records the
 * cosine similarity of the two real hidden states at each step. `threshold` is
 * the similarity at which the difference counts as erased — it has to be close
 * to 1, since two unrelated hidden states routinely sit above 0.5 by chance.
 */
export function measureDecay(
  rnn: TinyRNN,
  baseSequence: readonly number[],
  markerA: number,
  markerB: number,
  threshold: number
): { trace: number[]; decayStep: number } {
  const withA = rnn.forwardSequence([markerA, ...baseSequence]).hiddenStates;
  const withB = rnn.forwardSequence([markerB, ...baseSequence]).hiddenStates;

  const trace: number[] = [];
  let decayStep = baseSequence.length;

  for (let step = 1; step < withA.length; step++) {
    const similarity = cosineSimilarity(withA[step]!, withB[step]!);
    trace.push(similarity);
    // Reported as distance from the distinguishing token, not absolute step, so
    // it never exceeds the sequence the player is shown.
    const distance = step - 1;
    if (similarity >= threshold && decayStep === baseSequence.length) {
      decayStep = distance;
    }
  }

  return { trace, decayStep };
}

function buildDecayRounds(
  config: MemoryDecayConfig,
  rnn: TinyRNN,
  encoded: readonly number[],
  vocabSize: number
): DecayRound[] {
  const wanted = config.rounds ?? 0;
  if (wanted === 0 || encoded.length < config.sequenceLength) return [];

  const rng = createRng(config.seed * 13 + 1);
  const threshold = config.decayThreshold ?? 0.99;

  return Array.from({ length: wanted }, (_, r) => {
    const start = Math.floor(rng() * Math.max(1, encoded.length - config.sequenceLength - 1));
    const base = encoded.slice(start, start + config.sequenceLength);
    const markerA = r % vocabSize;
    const markerB = (r + Math.floor(vocabSize / 2)) % vocabSize;
    const { trace, decayStep } = measureDecay(rnn, base, markerA, markerB, threshold);
    return { trueDecayStep: decayStep, estimate: null, similarityTrace: trace };
  });
}

export function initState(
  config: MemoryDecayConfig,
  rules: EngineRules,
  prepared: PreparedRnnCorpus
): MemoryDecayState {
  const vocab = buildCharVocab(prepared.text, config.vocabLimit);
  const encoded = encodeChars(prepared.text, vocab);
  const rnn = new TinyRNN({
    hiddenSize: config.hiddenSize,
    vocabSize: vocab.chars.length,
    seed: config.seed,
  });

  // Levels that inspect a trained network train it up front; the training level
  // starts untrained so the player can watch it learn.
  const warmup = config.mode === 'train' ? 0 : config.epochs;
  const trained =
    warmup > 0
      ? trainCopy(rnn, encoded, config.sequenceLength, config.learningRate, warmup, 0)
      : { rnn, losses: [] };

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    hiddenSize: config.hiddenSize,
    learningRate: config.learningRate,
    epochsTrained: warmup,
    vocab,
    encoded,
    rnn: trained.rnn,
    lossHistory: trained.losses,
    rounds:
      config.mode === 'predict-decay'
        ? buildDecayRounds(config, trained.rnn, encoded, vocab.chars.length)
        : [],
    recallResults: [],
  };
}

/** Real recall accuracy at each gap length, measured on the trained network. */
export function measureRecall(state: MemoryDecayState): RecallResult[] {
  const gaps = state.config.gapLengths ?? [];
  const trials = state.config.trials ?? 20;
  const vocabSize = state.vocab.chars.length;
  const rng = createRng(state.config.seed * 97 + 5);

  return gaps.map((gap) => {
    let correct = 0;
    for (let trial = 0; trial < trials; trial++) {
      const marker = Math.floor(rng() * vocabSize);
      const filler = Array.from({ length: gap }, () => Math.floor(rng() * vocabSize));
      const distribution = state.rnn.predictNext([marker, ...filler]);

      let best = 0;
      for (let v = 1; v < distribution.length; v++) {
        if (distribution[v]! > distribution[best]!) best = v;
      }
      if (best === marker) correct++;
    }
    return { gap, accuracy: correct / trials };
  });
}

export function applyAction(
  state: MemoryDecayState,
  action: MemoryDecayAction
): MemoryDecayState {
  switch (action.type) {
    case 'TRAIN': {
      const max = state.config.maxEpochs ?? state.config.epochs;
      const epochs = Math.max(0, Math.min(action.epochs, max - state.epochsTrained));
      if (epochs === 0) return state;

      const { rnn, losses } = trainCopy(
        state.rnn,
        state.encoded,
        state.config.sequenceLength,
        state.learningRate,
        epochs,
        state.epochsTrained
      );

      return {
        ...state,
        rnn,
        epochsTrained: state.epochsTrained + epochs,
        lossHistory: [...state.lossHistory, ...losses],
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_HIDDEN_SIZE': {
      const range = state.config.hiddenSizeRange;
      if (!range || !Number.isFinite(action.value)) return state;
      const hiddenSize = Math.round(clamp(action.value, range[0], range[1]));
      // Resizing the state means starting over — the weights no longer fit.
      return {
        ...state,
        hiddenSize,
        epochsTrained: 0,
        lossHistory: [],
        recallResults: [],
        rnn: new TinyRNN({
          hiddenSize,
          vocabSize: state.vocab.chars.length,
          seed: state.config.seed,
        }),
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_LEARNING_RATE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.learningRateRange ?? [0, Infinity];
      return {
        ...state,
        learningRate: clamp(action.value, min, max),
        actionCount: state.actionCount + 1,
      };
    }

    case 'ESTIMATE_DECAY': {
      const round = state.rounds[action.roundIndex];
      if (!round || !Number.isFinite(action.value) || action.value < 0) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = {
        ...round,
        estimate: Math.round(clamp(action.value, 0, state.config.sequenceLength)),
      };
      return { ...state, rounds, status: 'active', actionCount: state.actionCount + 1 };
    }

    case 'RUN_RECALL':
      return {
        ...state,
        recallResults: measureRecall(state),
        status: 'active',
        actionCount: state.actionCount + 1,
      };

    case 'RESET':
      return initState(state.config, state.rules, {
        text: state.vocab.chars.length > 0 ? decodeAll(state) : '',
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

function decodeAll(state: MemoryDecayState): string {
  return state.encoded.map((i) => state.vocab.chars[i] ?? '').join('');
}

/** Next-character accuracy over the corpus, from the real trained network. */
export function characterAccuracy(state: MemoryDecayState): number {
  const window = Math.min(state.config.sequenceLength * 4, state.encoded.length - 1);
  if (window <= 1) return 0;
  const inputs = state.encoded.slice(0, window);
  const targets = state.encoded.slice(1, window + 1);
  return state.rnn.accuracy(inputs, targets);
}

export function evaluate(state: MemoryDecayState): ScoreResult {
  switch (state.mode) {
    case 'train': {
      const accuracy = characterAccuracy(state);
      return scoreLevel({
        metric: 'characterAccuracy',
        value: accuracy,
        rules: state.rules,
        breakdown: {
          epochs: state.epochsTrained,
          hiddenSize: state.hiddenSize,
          finalLoss: state.lossHistory.at(-1) ?? 0,
        },
      });
    }

    case 'predict-decay': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'decayPredictionError', value: Infinity, rules: state.rules });
      }
      let answered = 0;
      let sum = 0;
      for (const round of state.rounds) {
        if (round.estimate === null) {
          // Skipping costs the whole sequence length, the largest possible miss.
          sum += state.config.sequenceLength;
          continue;
        }
        answered++;
        sum += Math.abs(round.estimate - round.trueDecayStep);
      }
      return scoreLevel({
        metric: 'decayPredictionError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }

    case 'recall-task': {
      const results = state.recallResults;
      if (results.length < 2) {
        return scoreLevel({ metric: 'recallCollapseScore', value: 0, rules: state.rules });
      }
      const sorted = [...results].sort((a, b) => a.gap - b.gap);
      const nearest = sorted[0]!.accuracy;
      const furthest = sorted.at(-1)!.accuracy;

      // The point of the level is demonstrating the collapse: credit is for
      // showing recall that works up close and fails at distance.
      const collapse = Math.max(0, nearest - furthest);
      const value = clamp(collapse * 0.5 + nearest * 0.5, 0, 1);

      return scoreLevel({
        metric: 'recallCollapseScore',
        value,
        rules: state.rules,
        breakdown: {
          nearestGapAccuracy: nearest,
          furthestGapAccuracy: furthest,
          collapse,
          hiddenSize: state.hiddenSize,
        },
      });
    }
  }
}
