'use client';

/**
 * Durable player progress.
 *
 * Persisted to IndexedDB rather than localStorage (Section 7) so there is one
 * local data layer shared with the activity queue and model records. Hydration
 * is synchronous-feeling: the store reads at startup and the map renders from
 * whatever is there, with no spinner for progress.
 */
import { create } from 'zustand';
import { persist, createJSONStorage, type StateStorage } from 'zustand/middleware';
import { getDb, PROGRESS_STORE } from '@/models/modelCache';
import type { CompletionRecord } from '@/lib/curriculum';
import { GAME_SCHEMA_VERSION } from '@/types/game';

export interface LevelResult {
  levelId: string;
  stars: number;
  metric: string;
  value: number;
  xpEarned: number;
  attempts: number;
  completedAt: number;
}

export interface ChapterProgress {
  chapterId: string;
  /** Best stars across every attempt. */
  stars: number;
  xpEarned: number;
  levels: Record<string, LevelResult>;
  completedAt: number | null;
  /** Schema version the progress was earned under, for future migrations. */
  schemaVersion: number;
}

interface GameProgressState {
  userId: string | null;
  displayName: string | null;
  xp: number;
  chapters: Record<string, ChapterProgress>;
  soundEnabled: boolean;
  hydrated: boolean;

  setIdentity: (userId: string, displayName: string) => void;
  recordLevel: (chapterId: string, result: LevelResult) => void;
  completeChapter: (chapterId: string, stars: number) => void;
  setSoundEnabled: (enabled: boolean) => void;
  reset: () => void;
  markHydrated: () => void;

  completionRecord: () => CompletionRecord;
  chapterProgress: (chapterId: string) => ChapterProgress | null;
}

/** IndexedDB adapter shaped for zustand's persist middleware. */
const indexedDbStorage: StateStorage = {
  async getItem(name) {
    try {
      const db = await getDb();
      return ((await db.get(PROGRESS_STORE, name)) as string | undefined) ?? null;
    } catch {
      return null;
    }
  },
  async setItem(name, value) {
    try {
      const db = await getDb();
      await db.put(PROGRESS_STORE, value, name);
    } catch {
      // Losing a write costs this session's progress, never the running game.
    }
  },
  async removeItem(name) {
    try {
      const db = await getDb();
      await db.delete(PROGRESS_STORE, name);
    } catch {
      // Nothing to remove if the database was never reachable.
    }
  },
};

const emptyChapter = (chapterId: string): ChapterProgress => ({
  chapterId,
  stars: 0,
  xpEarned: 0,
  levels: {},
  completedAt: null,
  schemaVersion: GAME_SCHEMA_VERSION,
});

export const useGameProgressStore = create<GameProgressState>()(
  persist(
    (set, get) => ({
      userId: null,
      displayName: null,
      xp: 0,
      chapters: {},
      soundEnabled: false,
      hydrated: false,

      setIdentity: (userId, displayName) => set({ userId, displayName }),

      recordLevel: (chapterId, result) =>
        set((state) => {
          const chapter = state.chapters[chapterId] ?? emptyChapter(chapterId);
          const previous = chapter.levels[result.levelId];

          // Only the improvement is paid out, so replaying a level cannot farm XP.
          const previousXp = previous?.xpEarned ?? 0;
          const xpDelta = Math.max(0, result.xpEarned - previousXp);
          const keepBest = previous && previous.stars > result.stars ? previous : result;

          return {
            xp: state.xp + xpDelta,
            chapters: {
              ...state.chapters,
              [chapterId]: {
                ...chapter,
                xpEarned: chapter.xpEarned + xpDelta,
                levels: { ...chapter.levels, [result.levelId]: keepBest },
              },
            },
          };
        }),

      completeChapter: (chapterId, stars) =>
        set((state) => {
          const chapter = state.chapters[chapterId] ?? emptyChapter(chapterId);
          return {
            chapters: {
              ...state.chapters,
              [chapterId]: {
                ...chapter,
                stars: Math.max(chapter.stars, stars),
                completedAt: chapter.completedAt ?? Date.now(),
              },
            },
          };
        }),

      setSoundEnabled: (soundEnabled) => set({ soundEnabled }),

      reset: () => set({ xp: 0, chapters: {} }),

      markHydrated: () => set({ hydrated: true }),

      completionRecord: () => {
        const record: CompletionRecord = {};
        for (const chapter of Object.values(get().chapters)) {
          if (chapter.completedAt !== null) record[chapter.chapterId] = chapter.stars;
        }
        return record;
      },

      chapterProgress: (chapterId) => get().chapters[chapterId] ?? null,
    }),
    {
      name: 'game-progress',
      storage: createJSONStorage(() => indexedDbStorage),
      partialize: (state) => ({
        userId: state.userId,
        displayName: state.displayName,
        xp: state.xp,
        chapters: state.chapters,
        soundEnabled: state.soundEnabled,
      }),
      onRehydrateStorage: () => (state) => {
        // Marks the point from which the map can trust what it renders. Runs
        // even when nothing was stored, so a first-time player is not stuck
        // behind a loading state that never resolves.
        state?.markHydrated();
      },
    }
  )
);
