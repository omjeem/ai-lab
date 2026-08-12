import { describe, it, expect } from 'vitest';
import {
  selectVocabularyWords,
  nearestInIndex,
  type VocabularyIndex,
} from '@/models/embeddingVocabulary';

/**
 * A WordPiece vocabulary as the tokenizer actually stores it: special tokens,
 * continuations, punctuation and short fragments mixed in with real words, keyed
 * by token id.
 */
const RAW_VOCAB: Record<string, number> = {
  '[PAD]': 0,
  '[UNK]': 100,
  '[CLS]': 101,
  a: 103,
  of: 104,
  the: 110,
  king: 120,
  '##ing': 121,
  queen: 130,
  '.': 131,
  '1990': 132,
  woman: 140,
  Paris: 141,
  man: 150,
  'co-op': 151,
  swimming: 160,
};

/** Builds an index whose rows are already unit length, as the real one is. */
function buildIndex(rows: Record<string, number[]>): VocabularyIndex {
  const words = Object.keys(rows);
  const dims = rows[words[0]!]!.length;
  const matrix = new Float32Array(words.length * dims);

  words.forEach((word, row) => {
    const vector = rows[word]!;
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0)) || 1;
    vector.forEach((v, i) => {
      matrix[row * dims + i] = v / norm;
    });
  });

  return { words, matrix, dims };
}

describe('embeddingVocabulary — selectVocabularyWords', () => {
  it('keeps whole lowercase words and drops everything else', () => {
    const words = selectVocabularyWords(RAW_VOCAB, 100);

    expect(words).toContain('king');
    expect(words).toContain('queen');
    expect(words).toContain('swimming');

    // Special tokens, continuations, punctuation, digits, capitals and words
    // under three letters are all noise in a neighbour list.
    expect(words).not.toContain('[PAD]');
    expect(words).not.toContain('[UNK]');
    expect(words).not.toContain('##ing');
    expect(words).not.toContain('.');
    expect(words).not.toContain('1990');
    expect(words).not.toContain('Paris');
    expect(words).not.toContain('co-op');
    expect(words).not.toContain('a');
    expect(words).not.toContain('of');
  });

  it('orders by token id, which is the model’s own frequency order', () => {
    const words = selectVocabularyWords(RAW_VOCAB, 100);
    expect(words).toEqual(['the', 'king', 'queen', 'woman', 'man', 'swimming']);
  });

  it('takes the most frequent words when limited', () => {
    expect(selectVocabularyWords(RAW_VOCAB, 3)).toEqual(['the', 'king', 'queen']);
  });

  it('returns nothing for a zero or negative limit', () => {
    expect(selectVocabularyWords(RAW_VOCAB, 0)).toEqual([]);
    expect(selectVocabularyWords(RAW_VOCAB, -5)).toEqual([]);
  });

  it('survives a vocabulary with no usable words', () => {
    expect(selectVocabularyWords({ '[PAD]': 0, '##s': 1, '!': 2 }, 10)).toEqual([]);
  });
});

describe('embeddingVocabulary — nearestInIndex', () => {
  const index = buildIndex({
    king: [1, 1, 0],
    queen: [1, 0, 0],
    man: [0, 1, 0.6],
    woman: [0, 0.05, 0],
    river: [0, 0, 1],
    engine: [0, -0.2, 1],
  });

  it('ranks by cosine similarity, nearest first', () => {
    const near = nearestInIndex(index, [1, 0, 0], 3);

    expect(near[0]!.word).toBe('queen');
    expect(near[0]!.similarity).toBeCloseTo(1);
    expect(near.map((n) => n.similarity)).toEqual(
      [...near.map((n) => n.similarity)].sort((a, b) => b - a)
    );
  });

  it('returns exactly k results', () => {
    expect(nearestInIndex(index, [1, 1, 0], 2)).toHaveLength(2);
    expect(nearestInIndex(index, [1, 1, 0], 4)).toHaveLength(4);
  });

  it('never returns more than the vocabulary holds', () => {
    expect(nearestInIndex(index, [1, 0, 0], 50)).toHaveLength(6);
  });

  it('excludes the words asked for, which is what makes an analogy resolve', () => {
    // A result vector sits closest to the terms that built it, so the standard
    // analogy protocol drops them before reading the neighbour off.
    const query = [1, 1, 0]; // king's own vector
    expect(nearestInIndex(index, query, 1)[0]!.word).toBe('king');

    const excluded = nearestInIndex(index, query, 1, new Set(['king', 'man', 'woman']));
    expect(excluded[0]!.word).toBe('queen');
  });

  it('is unaffected by the query’s magnitude', () => {
    const small = nearestInIndex(index, [1, 0, 0], 3);
    const large = nearestInIndex(index, [50, 0, 0], 3);
    expect(large.map((n) => n.word)).toEqual(small.map((n) => n.word));
    expect(large[0]!.similarity).toBeCloseTo(small[0]!.similarity);
  });

  it('reports negative similarity rather than clamping it', () => {
    const near = nearestInIndex(index, [0, -1, 0], 6);
    const king = near.find((n) => n.word === 'king');
    expect(king!.similarity).toBeLessThan(0);
  });

  it('returns nothing for a zero query, a wrong-width query or k <= 0', () => {
    expect(nearestInIndex(index, [0, 0, 0], 3)).toEqual([]);
    expect(nearestInIndex(index, [1, 0], 3)).toEqual([]);
    expect(nearestInIndex(index, [1, 0, 0], 0)).toEqual([]);
  });

  it('returns nothing for an empty index', () => {
    const empty: VocabularyIndex = { words: [], matrix: new Float32Array(0), dims: 3 };
    expect(nearestInIndex(empty, [1, 0, 0], 5)).toEqual([]);
  });

  it('handles every word being excluded', () => {
    expect(nearestInIndex(index, [1, 0, 0], 3, new Set(index.words))).toEqual([]);
  });
});
