/**
 * Curriculum access and the unlock resolver.
 *
 * Section 7 asks for a generic unlock resolver rather than per-chapter
 * conditionals: everything here is driven by `unlockRequires` in the chapter
 * JSON plus the manifest's ordering, so adding a chapter never means editing
 * unlock logic.
 */
import manifestJson from '@data/games/curriculum-manifest.json';
import type {
  CurriculumManifest,
  GameDefinition,
  ManifestChapter,
  ManifestWorld,
  Rank,
} from '@/types/game';

/* Statically imported so the whole curriculum ships in the bundle and the map
   renders offline without a fetch. */
import g11 from '@data/games/world-1-fundamentals/1-1-vectors.json';
import g12 from '@data/games/world-1-fundamentals/1-2-vector-arithmetic.json';
import g13 from '@data/games/world-1-fundamentals/1-3-similarity-distance.json';
import g14 from '@data/games/world-1-fundamentals/1-4-tokenization.json';
import g15 from '@data/games/world-1-fundamentals/1-5-probability.json';
import g21 from '@data/games/world-2-classical-ml/2-1-perceptron.json';
import g22 from '@data/games/world-2-classical-ml/2-2-loss-functions.json';
import g23 from '@data/games/world-2-classical-ml/2-3-gradient-descent.json';
import g24 from '@data/games/world-2-classical-ml/2-4-overfitting.json';
import g31 from '@data/games/world-3-neural-networks/3-1-neurons-activations.json';
import g32 from '@data/games/world-3-neural-networks/3-2-layers-forward-pass.json';
import g33 from '@data/games/world-3-neural-networks/3-3-backpropagation.json';
import g34 from '@data/games/world-3-neural-networks/3-4-training-dynamics.json';
import g41 from '@data/games/world-4-sequence-models/4-1-ngrams.json';
import g42 from '@data/games/world-4-sequence-models/4-2-recurrence-memory.json';
import g43 from '@data/games/world-4-sequence-models/4-3-sampling-strategies.json';
import g51 from '@data/games/world-5-transformers/5-1-positional-encoding.json';
import g52 from '@data/games/world-5-transformers/5-2-self-attention.json';
import g53 from '@data/games/world-5-transformers/5-3-multi-head-attention.json';
import g54 from '@data/games/world-5-transformers/5-4-residuals-layernorm.json';
import g55 from '@data/games/world-5-transformers/5-5-full-transformer.json';
import g61 from '@data/games/world-6-capstone/6-1-inspector-chat.json';
import g71 from '@data/games/world-7-grounding-tools/7-1-retrieval.json';
import g72 from '@data/games/world-7-grounding-tools/7-2-grounded-generation.json';

export const manifest = manifestJson as CurriculumManifest;

const GAMES = [
  g11, g12, g13, g14, g15,
  g21, g22, g23, g24,
  g31, g32, g33, g34,
  g41, g42, g43,
  g51, g52, g53, g54, g55,
  g61,
  g71, g72,
] as unknown as GameDefinition[];

const BY_ID = new Map(GAMES.map((game) => [game.id, game]));

export function allGames(): GameDefinition[] {
  return GAMES;
}

export function getGame(chapterId: string): GameDefinition | null {
  return BY_ID.get(chapterId) ?? null;
}

export function getWorld(worldNumber: number): ManifestWorld | null {
  return manifest.worlds.find((w) => w.world === worldNumber) ?? null;
}

export function getManifestChapter(chapterId: string): ManifestChapter | null {
  for (const world of manifest.worlds) {
    const chapter = world.chapters.find((c) => c.id === chapterId);
    if (chapter) return chapter;
  }
  return null;
}

export function worldOfChapter(chapterId: string): ManifestWorld | null {
  return manifest.worlds.find((w) => w.chapters.some((c) => c.id === chapterId)) ?? null;
}

/** Every chapter in play order, flattened across worlds. */
export function orderedChapters(): ManifestChapter[] {
  return manifest.worlds.flatMap((world) =>
    [...world.chapters].sort((a, b) => a.order - b.order)
  );
}

export function nextChapterAfter(chapterId: string): ManifestChapter | null {
  const ordered = orderedChapters();
  const index = ordered.findIndex((c) => c.id === chapterId);
  return index === -1 ? null : (ordered[index + 1] ?? null);
}

/* ────────────────────────────────────────────────────────────
   Unlock resolution
   ──────────────────────────────────────────────────────────── */

export type ChapterStatus = 'locked' | 'available' | 'completed';

export interface ChapterState {
  id: string;
  status: ChapterStatus;
  /** Requirements not yet met, for the "needs X first" copy. */
  missing: string[];
  stars: number;
}

export interface CompletionRecord {
  /** Highest stars earned per chapter. Absent means never completed. */
  [chapterId: string]: number;
}

/**
 * Resolves the whole unlock graph in one pass.
 *
 * A chapter is available once every id in its `unlockRequires` has been
 * completed. Nothing here knows about specific chapters.
 */
export function resolveUnlocks(completed: CompletionRecord): Map<string, ChapterState> {
  const states = new Map<string, ChapterState>();

  for (const game of GAMES) {
    const stars = completed[game.id];
    const isCompleted = stars !== undefined;
    const missing = game.unlockRequires.filter((req) => completed[req] === undefined);

    states.set(game.id, {
      id: game.id,
      status: isCompleted ? 'completed' : missing.length === 0 ? 'available' : 'locked',
      missing,
      stars: stars ?? 0,
    });
  }

  return states;
}

export function isChapterUnlocked(chapterId: string, completed: CompletionRecord): boolean {
  const game = getGame(chapterId);
  if (!game) return false;
  return game.unlockRequires.every((req) => completed[req] !== undefined);
}

/** The chapter a returning player should be dropped into. */
export function nextPlayableChapter(completed: CompletionRecord): ManifestChapter | null {
  const states = resolveUnlocks(completed);
  for (const chapter of orderedChapters()) {
    if (states.get(chapter.id)?.status === 'available') return chapter;
  }
  return null;
}

/* ────────────────────────────────────────────────────────────
   Progress summary
   ──────────────────────────────────────────────────────────── */

export interface ProgressSummary {
  xp: number;
  rank: Rank;
  nextRank: Rank | null;
  xpToNextRank: number;
  chaptersCompleted: number;
  chaptersTotal: number;
  starsEarned: number;
  starsPossible: number;
}

export function summarise(completed: CompletionRecord, xp: number): ProgressSummary {
  const ranks = [...manifest.ranks].sort((a, b) => a.minXp - b.minXp);

  let rank = ranks[0]!;
  let nextRank: Rank | null = null;
  for (const candidate of ranks) {
    if (xp >= candidate.minXp) rank = candidate;
    else {
      nextRank = candidate;
      break;
    }
  }

  const starsEarned = Object.values(completed).reduce((a, b) => a + b, 0);

  return {
    xp,
    rank,
    nextRank,
    xpToNextRank: nextRank ? nextRank.minXp - xp : 0,
    chaptersCompleted: Object.keys(completed).length,
    chaptersTotal: GAMES.length,
    starsEarned,
    // Three stars per chapter is the ceiling.
    starsPossible: GAMES.length * 3,
  };
}
