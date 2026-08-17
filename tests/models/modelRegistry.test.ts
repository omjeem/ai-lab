import { describe, it, expect, beforeEach } from 'vitest';
import {
  kindForModel,
  registerChapterModel,
  getModelEntry,
  listDownloadableModels,
  totalDownloadSizeMB,
  chaptersUsingModel,
  resetRegistry,
} from '@/models/modelRegistry';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import type { GameDefinition } from '@/types/game';

const GAMES_DIR = path.resolve(__dirname, '../../data/games');

function loadAllGames(): GameDefinition[] {
  const games: GameDefinition[] = [];
  for (const entry of readdirSync(GAMES_DIR)) {
    const dir = path.join(GAMES_DIR, entry);
    if (!statSync(dir).isDirectory()) continue;
    for (const file of readdirSync(dir)) {
      if (file.endsWith('.json')) {
        games.push(JSON.parse(readFileSync(path.join(dir, file), 'utf8')));
      }
    }
  }
  return games;
}

describe('kindForModel', () => {
  it('routes each model id to the wrapper that services it', () => {
    expect(kindForModel('Xenova/all-MiniLM-L6-v2')).toBe('embedding');
    expect(kindForModel('Xenova/distilbert-base-uncased')).toBe('attention');
    expect(kindForModel('Xenova/gpt2')).toBe('tokenizer');
    expect(kindForModel('HuggingFaceTB/SmolLM2-135M-Instruct')).toBe('causal-lm');
    expect(kindForModel('Llama-3.2-1B-Instruct-q4f16_1-MLC')).toBe('webllm');
  });

  it('recognises the locally trained models, which download nothing', () => {
    expect(kindForModel('local:tiny-net')).toBe('local-net');
    expect(kindForModel('local:char-rnn')).toBe('local-rnn');
    expect(kindForModel('local:ngram-corpus')).toBe('local-corpus');
  });

  it('reports none for a chapter with no model', () => {
    expect(kindForModel(null)).toBe('none');
  });
});

describe('registry', () => {
  beforeEach(() => resetRegistry());

  it('knows every chapter in the manifest before anything is registered', () => {
    expect(getModelEntry('1-1-vectors')).not.toBeNull();
    expect(getModelEntry('6-1-inspector-chat')).not.toBeNull();
    expect(getModelEntry('does-not-exist')).toBeNull();
  });

  it('records what a chapter declares', () => {
    registerChapterModel('1-1-vectors', 'Xenova/all-MiniLM-L6-v2', 23, 'browser-light');
    const entry = getModelEntry('1-1-vectors')!;

    expect(entry.modelId).toBe('Xenova/all-MiniLM-L6-v2');
    expect(entry.estimatedSizeMB).toBe(23);
    expect(entry.kind).toBe('embedding');
    expect(entry.tier).toBe('browser-light');
  });

  it('deduplicates models shared across chapters', () => {
    registerChapterModel('1-1-vectors', 'Xenova/all-MiniLM-L6-v2', 23, 'browser-light');
    registerChapterModel('1-2-vector-arithmetic', 'Xenova/all-MiniLM-L6-v2', 23, 'browser-light');

    const downloads = listDownloadableModels();
    expect(downloads).toHaveLength(1);
    expect(downloads[0]!.chapters).toEqual(['1-1-vectors', '1-2-vector-arithmetic']);
    expect(totalDownloadSizeMB()).toBe(23);
  });

  it('excludes locally trained models from the download list', () => {
    registerChapterModel('2-1-perceptron', 'local:perceptron-2d', 0, 'browser-heavy');
    expect(listDownloadableModels()).toHaveLength(0);
    expect(totalDownloadSizeMB()).toBe(0);
  });

  it('lists downloads smallest first, so the cheapest unlock is obvious', () => {
    registerChapterModel('5-5-full-transformer', 'Xenova/gpt2', 124, 'browser-heavy');
    registerChapterModel('1-1-vectors', 'Xenova/all-MiniLM-L6-v2', 23, 'browser-light');
    registerChapterModel('5-2-self-attention', 'Xenova/distilbert-base-uncased', 67, 'browser-heavy');

    expect(listDownloadableModels().map((m) => m.estimatedSizeMB)).toEqual([23, 67, 124]);
  });

  it('answers which chapters a given model unlocks', () => {
    registerChapterModel('5-2-self-attention', 'Xenova/distilbert-base-uncased', 67, 'browser-heavy');
    registerChapterModel('5-3-multi-head-attention', 'Xenova/distilbert-base-uncased', 67, 'browser-heavy');
    expect(chaptersUsingModel('Xenova/distilbert-base-uncased')).toEqual([
      '5-2-self-attention',
      '5-3-multi-head-attention',
    ]);
  });
});

describe('registry against the shipped curriculum', () => {
  beforeEach(() => resetRegistry());

  it('accepts every chapter definition on disk', () => {
    const games = loadAllGames();
    expect(games).toHaveLength(23);

    for (const game of games) {
      registerChapterModel(
        game.id,
        game.modelRequirement.modelId,
        game.modelRequirement.estimatedSizeMB,
        game.modelRequirement.tier
      );
      const entry = getModelEntry(game.id);
      expect(entry).not.toBeNull();
      expect(entry!.tier).toBe(game.modelRequirement.tier);
    }
  });

  it('gives every model-backed chapter a wrapper to load it', () => {
    for (const game of loadAllGames()) {
      registerChapterModel(
        game.id,
        game.modelRequirement.modelId,
        game.modelRequirement.estimatedSizeMB,
        game.modelRequirement.tier
      );
      const entry = getModelEntry(game.id)!;
      if (game.modelRequirement.tier === 'none') {
        expect(entry.kind).toBe('none');
      } else {
        expect(entry.kind).not.toBe('none');
      }
    }
  });

  it('reports a download budget matching the distinct models the course needs', () => {
    for (const game of loadAllGames()) {
      registerChapterModel(
        game.id,
        game.modelRequirement.modelId,
        game.modelRequirement.estimatedSizeMB,
        game.modelRequirement.tier
      );
    }

    const downloads = listDownloadableModels();
    const ids = downloads.map((d) => d.modelId);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain('Xenova/all-MiniLM-L6-v2');
    expect(ids).toContain('Qdrant/all_miniLM_L6_v2_with_attentions');
    expect(ids).toContain('yaww85/all-MiniLM-L6-v2-hidden-states-exposed-v1');
    expect(totalDownloadSizeMB()).toBeGreaterThan(0);
  });
});
