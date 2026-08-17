/**
 * Chapter 7.1 — Retrieval.
 *
 * Every "right answer" comes from real cosine similarity over the live
 * embedder, against a bundled fact corpus (`public/corpora/retrieval-facts.json`,
 * loaded through `CorpusDep` the same way the n-gram chapter loads its text) —
 * never an authored ranking. `prepare()` validates that each fact's `answer`
 * literally appears in its own passage, so the corpus can't drift from its own
 * stated facts.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CorpusDep, Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { cosineSimilarity, spearmanCorrelation } from './shared';

export type RetrievalMode = 'rank' | 'break-retriever' | 'chunking';

export interface RetrievalRankConfig {
  mode: RetrievalMode;
  corpus: string;
  /** rank mode: the anchor query, and which facts (in initial order) to rank. */
  query?: string;
  factIds?: string[];
  /** break-retriever mode: which fact the player's query must surface top-1. */
  targetFactId?: string;
  /** chunking mode: which combined documents to build rounds from. */
  documentIds?: string[];
}

export interface RetrievalFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

export interface RetrievalDocument {
  id: string;
  factIds: string[];
}

interface RawCorpus {
  facts: RetrievalFact[];
  documents: RetrievalDocument[];
}

export interface PreparedRetrievalData {
  facts: RetrievalFact[];
  documents: RetrievalDocument[];
  /** Whole-passage embedding per fact. */
  factVectors: Record<string, number[]>;
  /** Per-sentence embeddings per fact, same order as `fact.sentences`. */
  sentenceVectors: Record<string, number[][]>;
  /** Whole-document embedding per combined document (its facts' sentences, joined). */
  documentVectors: Record<string, number[]>;
  /** Each fact's own headline query, embedded. */
  factQueryVectors: Record<string, number[]>;
  /** rank mode's anchor query, embedded — null outside rank mode. */
  rankQueryVector: number[] | null;
}

export interface ChunkRound {
  factId: string;
  docId: string;
  strategy: 'paragraph' | 'sentence' | null;
}

export interface RetrievalRankState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: RetrievalMode;
  config: RetrievalRankConfig;
  facts: RetrievalFact[];
  documents: RetrievalDocument[];
  factVectors: Record<string, number[]>;
  sentenceVectors: Record<string, number[][]>;
  documentVectors: Record<string, number[]>;
  factQueryVectors: Record<string, number[]>;

  // rank mode
  query: string;
  queryVector: number[];
  ordering: string[];
  trueOrder: string[];

  // break-retriever mode
  targetFactId: string;
  submittedQuery: string | null;
  submittedVector: number[] | null;

  // chunking mode
  chunkRounds: ChunkRound[];
}

export type RetrievalRankAction =
  | { type: 'SET_ORDER'; ordering: string[] }
  | { type: 'MOVE_ITEM'; from: number; to: number }
  | { type: 'SUBMIT_QUERY'; query: string; vector: number[] }
  | { type: 'SET_STRATEGY'; roundIndex: number; strategy: 'paragraph' | 'sentence' }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

function docTextFor(doc: RetrievalDocument, facts: readonly RetrievalFact[]): string {
  return doc.factIds
    .flatMap((fid) => facts.find((f) => f.id === fid)?.sentences ?? [])
    .join(' ');
}

export async function prepare(
  config: RetrievalRankConfig,
  deps: { corpus: CorpusDep; embedder: Embedder }
): Promise<PreparedRetrievalData> {
  const raw = await deps.corpus.load(config.corpus);
  const parsed = JSON.parse(raw) as RawCorpus;

  for (const fact of parsed.facts) {
    const passage = fact.sentences.join(' ');
    if (!passage.includes(fact.answer)) {
      throw new Error(`Fact "${fact.id}"'s answer "${fact.answer}" does not appear in its own passage`);
    }
  }

  const passages = parsed.facts.map((f) => f.sentences.join(' '));
  const allSentences = parsed.facts.flatMap((f) => f.sentences);
  const factQueries = parsed.facts.map((f) => f.query);
  const docTexts = parsed.documents.map((d) => docTextFor(d, parsed.facts));

  const [passageVecs, sentenceVecsFlat, queryVecs, docVecs, rankQueryVecs] = await Promise.all([
    deps.embedder.embed(passages),
    deps.embedder.embed(allSentences),
    deps.embedder.embed(factQueries),
    deps.embedder.embed(docTexts),
    config.mode === 'rank' && config.query ? deps.embedder.embed([config.query]) : Promise.resolve(null),
  ]);

  const factVectors: Record<string, number[]> = {};
  const sentenceVectors: Record<string, number[][]> = {};
  const factQueryVectors: Record<string, number[]> = {};
  let cursor = 0;
  parsed.facts.forEach((fact, i) => {
    factVectors[fact.id] = passageVecs[i]!;
    sentenceVectors[fact.id] = fact.sentences.map(() => sentenceVecsFlat[cursor++]!);
    factQueryVectors[fact.id] = queryVecs[i]!;
  });

  const documentVectors: Record<string, number[]> = {};
  parsed.documents.forEach((doc, i) => {
    documentVectors[doc.id] = docVecs[i]!;
  });

  return {
    facts: parsed.facts,
    documents: parsed.documents,
    factVectors,
    sentenceVectors,
    documentVectors,
    factQueryVectors,
    rankQueryVector: rankQueryVecs?.[0] ?? null,
  };
}

function isPermutation(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  const counts = new Map<string, number>();
  for (const x of a) counts.set(x, (counts.get(x) ?? 0) + 1);
  for (const x of b) {
    const n = counts.get(x);
    if (n === undefined || n === 0) return false;
    counts.set(x, n - 1);
  }
  return true;
}

export function initState(
  config: RetrievalRankConfig,
  rules: EngineRules,
  prepared: PreparedRetrievalData
): RetrievalRankState {
  const factIds = config.factIds ?? prepared.facts.map((f) => f.id);
  const queryVector = prepared.rankQueryVector ?? [];

  const trueOrder =
    config.mode === 'rank'
      ? [...factIds].sort(
          (a, b) =>
            cosineSimilarity(queryVector, prepared.factVectors[b] ?? []) -
            cosineSimilarity(queryVector, prepared.factVectors[a] ?? [])
        )
      : [];

  const documents =
    config.mode === 'chunking'
      ? prepared.documents.filter((d) => (config.documentIds ?? prepared.documents.map((x) => x.id)).includes(d.id))
      : prepared.documents;

  const chunkRounds: ChunkRound[] =
    config.mode === 'chunking'
      ? documents.flatMap((doc) => doc.factIds.map((factId) => ({ factId, docId: doc.id, strategy: null })))
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    facts: prepared.facts,
    documents,
    factVectors: prepared.factVectors,
    sentenceVectors: prepared.sentenceVectors,
    documentVectors: prepared.documentVectors,
    factQueryVectors: prepared.factQueryVectors,

    query: config.query ?? '',
    queryVector,
    ordering: [...factIds],
    trueOrder,

    targetFactId: config.targetFactId ?? '',
    submittedQuery: null,
    submittedVector: null,

    chunkRounds,
  };
}

export function applyAction(
  state: RetrievalRankState,
  action: RetrievalRankAction
): RetrievalRankState {
  const bump = (next: Partial<RetrievalRankState>): RetrievalRankState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_ORDER': {
      if (!isPermutation(state.ordering, action.ordering)) return state;
      return bump({ ordering: [...action.ordering] });
    }

    case 'MOVE_ITEM': {
      const { from, to } = action;
      if (from < 0 || from >= state.ordering.length) return state;
      if (to < 0 || to >= state.ordering.length) return state;
      const ordering = [...state.ordering];
      const [moved] = ordering.splice(from, 1);
      ordering.splice(to, 0, moved!);
      return bump({ ordering });
    }

    case 'SUBMIT_QUERY':
      return bump({ submittedQuery: action.query, submittedVector: action.vector });

    case 'SET_STRATEGY': {
      const round = state.chunkRounds[action.roundIndex];
      if (!round) return state;
      const chunkRounds = [...state.chunkRounds];
      chunkRounds[action.roundIndex] = { ...round, strategy: action.strategy };
      return bump({ chunkRounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        facts: state.facts,
        documents: state.documents,
        factVectors: state.factVectors,
        sentenceVectors: state.sentenceVectors,
        documentVectors: state.documentVectors,
        factQueryVectors: state.factQueryVectors,
        rankQueryVector: state.mode === 'rank' ? state.queryVector : null,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/** Fact whose whole-passage vector is closest to `vector`. */
function topFactByPassage(vector: readonly number[], facts: readonly RetrievalFact[], factVectors: Record<string, number[]>): string {
  let best = facts[0]!.id;
  let bestSim = -Infinity;
  for (const fact of facts) {
    const sim = cosineSimilarity(vector, factVectors[fact.id] ?? []);
    if (sim > bestSim) {
      bestSim = sim;
      best = fact.id;
    }
  }
  return best;
}

/**
 * Real precision for one chunking round under whichever strategy it currently
 * holds (or 0 if unanswered) — relevant sentences / retrieved sentences.
 * Exported so a canvas can show live per-round feedback the moment a
 * strategy is picked, not just the aggregate from `evaluate()`.
 */
export function roundPrecision(state: RetrievalRankState, round: ChunkRound): number {
  return round.strategy ? chunkPrecision(state, round) : 0;
}

function chunkPrecision(state: RetrievalRankState, round: ChunkRound): number {
  const doc = state.documents.find((d) => d.id === round.docId);
  if (!doc) return 0;
  const queryVector = state.factQueryVectors[round.factId] ?? [];
  const targetFact = state.facts.find((f) => f.id === round.factId);
  if (!targetFact) return 0;

  if (round.strategy === 'paragraph') {
    let best = state.documents[0]!.id;
    let bestSim = -Infinity;
    for (const candidate of state.documents) {
      const sim = cosineSimilarity(queryVector, state.documentVectors[candidate.id] ?? []);
      if (sim > bestSim) {
        bestSim = sim;
        best = candidate.id;
      }
    }
    if (best !== doc.id) return 0;
    const totalSentences = doc.factIds.reduce(
      (sum, fid) => sum + (state.facts.find((f) => f.id === fid)?.sentences.length ?? 0),
      0
    );
    return totalSentences === 0 ? 0 : targetFact.sentences.length / totalSentences;
  }

  if (round.strategy === 'sentence') {
    // Searches every sentence chunk across every configured document, not
    // just this round's own doc — a real retriever doesn't know in advance
    // which document a query's answer lives in.
    const candidateFactIds = state.documents.flatMap((d) => d.factIds);
    let bestFactId = candidateFactIds[0]!;
    let bestSim = -Infinity;
    for (const fid of candidateFactIds) {
      const vectors = state.sentenceVectors[fid] ?? [];
      for (const sv of vectors) {
        const sim = cosineSimilarity(queryVector, sv);
        if (sim > bestSim) {
          bestSim = sim;
          bestFactId = fid;
        }
      }
    }
    return bestFactId === round.factId ? 1 : 0;
  }

  return 0;
}

export function evaluate(state: RetrievalRankState): ScoreResult {
  switch (state.mode) {
    case 'rank': {
      if (state.trueOrder.length < 2) {
        return scoreLevel({ metric: 'rankCorrelation', value: 0, rules: state.rules });
      }
      const truePosition = new Map(state.trueOrder.map((id, i) => [id, i]));
      const userRanks = state.ordering.map((_, i) => i);
      const trueRanks = state.ordering.map((id) => truePosition.get(id) ?? 0);

      return scoreLevel({
        metric: 'rankCorrelation',
        value: spearmanCorrelation(userRanks, trueRanks),
        rules: state.rules,
        breakdown: { items: state.ordering.length },
      });
    }

    case 'break-retriever': {
      if (!state.submittedVector) {
        return scoreLevel({ metric: 'retrievalMargin', value: -1, rules: state.rules });
      }
      const targetSim = cosineSimilarity(state.submittedVector, state.factVectors[state.targetFactId] ?? []);
      const others = state.facts.filter((f) => f.id !== state.targetFactId);
      const bestOther = Math.max(
        -Infinity,
        ...others.map((f) => cosineSimilarity(state.submittedVector!, state.factVectors[f.id] ?? []))
      );
      const topFact = topFactByPassage(state.submittedVector, state.facts, state.factVectors);

      return scoreLevel({
        metric: 'retrievalMargin',
        value: targetSim - bestOther,
        rules: state.rules,
        breakdown: { targetSimilarity: targetSim, topRetrieved: topFact === state.targetFactId ? 1 : 0 },
      });
    }

    case 'chunking': {
      const total = state.chunkRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'retrievalPrecision', value: 0, rules: state.rules });
      }
      const precisions = state.chunkRounds.map((round) => (round.strategy ? chunkPrecision(state, round) : 0));
      const answered = state.chunkRounds.filter((r) => r.strategy !== null).length;
      const value = precisions.reduce((a, b) => a + b, 0) / total;

      return scoreLevel({
        metric: 'retrievalPrecision',
        value,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }
  }
}
