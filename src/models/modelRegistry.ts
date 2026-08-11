/**
 * The one place that knows which model each chapter needs.
 *
 * Built from the curriculum JSON rather than a hand-kept list, so a chapter can
 * never disagree with its own definition about what it downloads.
 */
import manifestJson from '@data/games/curriculum-manifest.json';
import type { CurriculumManifest, ModelTier } from '@/types/game';

/** Which wrapper in this folder services a given model. */
export type ModelKind =
  | 'embedding'
  | 'tokenizer'
  | 'causal-lm'
  | 'attention'
  | 'webllm'
  | 'local-net'
  | 'local-rnn'
  | 'local-corpus'
  | 'none';

export interface ModelEntry {
  chapterId: string;
  tier: ModelTier;
  modelId: string | null;
  estimatedSizeMB: number;
  kind: ModelKind;
}

const manifest = manifestJson as CurriculumManifest;

/**
 * Chapter → model. Sizes and ids come from each chapter's own definition; the
 * kind is derived from the model id, so adding a chapter that reuses an
 * existing model needs no change here.
 */
export function kindForModel(modelId: string | null): ModelKind {
  if (!modelId) return 'none';
  if (modelId.startsWith('local:')) {
    if (modelId.includes('rnn')) return 'local-rnn';
    if (modelId.includes('ngram') || modelId.includes('corpus')) return 'local-corpus';
    return 'local-net';
  }
  if (modelId.includes('MiniLM')) return 'embedding';
  if (modelId.includes('distilbert')) return 'attention';
  if (modelId.includes('MLC')) return 'webllm';
  if (modelId.includes('gpt2')) return 'tokenizer';
  return 'causal-lm';
}

/** Model ids used by the app, deduplicated, with their download sizes. */
export interface DownloadableModel {
  modelId: string;
  kind: ModelKind;
  estimatedSizeMB: number;
  /** Chapters that need it, so the UI can say what a download unlocks. */
  chapters: string[];
}

const REGISTRY = new Map<string, ModelEntry>();

for (const world of manifest.worlds) {
  for (const chapter of world.chapters) {
    REGISTRY.set(chapter.id, {
      chapterId: chapter.id,
      tier: chapter.tier,
      modelId: null,
      estimatedSizeMB: 0,
      kind: 'none',
    });
  }
}

/**
 * Fills in the model details a chapter declares.
 *
 * The manifest carries the tier but not the model id, so the chapter loader
 * calls this once it has read the full definition. Keeps the registry the single
 * source of truth without duplicating every field into the manifest.
 */
export function registerChapterModel(
  chapterId: string,
  modelId: string | null,
  estimatedSizeMB: number | null,
  tier: ModelTier
): void {
  REGISTRY.set(chapterId, {
    chapterId,
    tier,
    modelId,
    estimatedSizeMB: estimatedSizeMB ?? 0,
    kind: kindForModel(modelId),
  });
}

export function getModelEntry(chapterId: string): ModelEntry | null {
  return REGISTRY.get(chapterId) ?? null;
}

export function listDownloadableModels(): DownloadableModel[] {
  const byModel = new Map<string, DownloadableModel>();

  for (const entry of REGISTRY.values()) {
    // Locally trained models and pure-maths chapters download nothing.
    if (!entry.modelId || entry.modelId.startsWith('local:')) continue;

    const existing = byModel.get(entry.modelId);
    if (existing) {
      existing.chapters.push(entry.chapterId);
      continue;
    }
    byModel.set(entry.modelId, {
      modelId: entry.modelId,
      kind: entry.kind,
      estimatedSizeMB: entry.estimatedSizeMB,
      chapters: [entry.chapterId],
    });
  }

  return [...byModel.values()].sort((a, b) => a.estimatedSizeMB - b.estimatedSizeMB);
}

/** Total megabytes to have every browser-tier chapter available offline. */
export function totalDownloadSizeMB(): number {
  return listDownloadableModels().reduce((sum, model) => sum + model.estimatedSizeMB, 0);
}

export function chaptersUsingModel(modelId: string): string[] {
  return [...REGISTRY.values()].filter((e) => e.modelId === modelId).map((e) => e.chapterId);
}

/** Test seam. */
export function resetRegistry(): void {
  for (const [id, entry] of REGISTRY) {
    REGISTRY.set(id, { ...entry, modelId: null, estimatedSizeMB: 0, kind: 'none' });
  }
}
