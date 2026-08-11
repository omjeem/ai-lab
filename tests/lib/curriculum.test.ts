import { describe, it, expect } from 'vitest';
import {
  allGames,
  getGame,
  getWorld,
  worldOfChapter,
  orderedChapters,
  nextChapterAfter,
  resolveUnlocks,
  isChapterUnlocked,
  nextPlayableChapter,
  summarise,
  manifest,
  type CompletionRecord,
} from '@/lib/curriculum';

describe('curriculum access', () => {
  it('exposes every chapter', () => {
    expect(allGames()).toHaveLength(22);
    expect(getGame('1-1-vectors')?.chapterTitle).toBe('What is a Vector?');
    expect(getGame('nope')).toBeNull();
  });

  it('finds the world a chapter belongs to', () => {
    expect(worldOfChapter('5-2-self-attention')?.world).toBe(5);
    expect(getWorld(3)?.title).toBe('Neural Networks');
    expect(getWorld(99)).toBeNull();
  });

  it('orders chapters for play', () => {
    const ordered = orderedChapters();
    expect(ordered).toHaveLength(22);
    expect(ordered[0]!.id).toBe('1-1-vectors');
    expect(ordered.at(-1)!.id).toBe('6-1-inspector-chat');
  });

  it('walks to the next chapter, across a world boundary', () => {
    expect(nextChapterAfter('1-1-vectors')?.id).toBe('1-2-vector-arithmetic');
    expect(nextChapterAfter('1-5-probability')?.id).toBe('2-1-perceptron');
    expect(nextChapterAfter('6-1-inspector-chat')).toBeNull();
  });
});

describe('unlock resolution', () => {
  it('opens only the first chapter to a new player', () => {
    const states = resolveUnlocks({});
    expect(states.get('1-1-vectors')!.status).toBe('available');

    const available = [...states.values()].filter((s) => s.status === 'available');
    expect(available).toHaveLength(1);
  });

  it('opens the next chapter once its requirement is met', () => {
    const states = resolveUnlocks({ '1-1-vectors': 3 });
    expect(states.get('1-1-vectors')!.status).toBe('completed');
    expect(states.get('1-2-vector-arithmetic')!.status).toBe('available');
    expect(states.get('1-3-similarity-distance')!.status).toBe('locked');
  });

  it('reports which requirements are still missing', () => {
    const state = resolveUnlocks({}).get('1-3-similarity-distance')!;
    expect(state.missing).toEqual(['1-2-vector-arithmetic']);
  });

  it('carries the earned star rating through', () => {
    expect(resolveUnlocks({ '1-1-vectors': 2 }).get('1-1-vectors')!.stars).toBe(2);
    expect(resolveUnlocks({}).get('1-1-vectors')!.stars).toBe(0);
  });

  it('treats a one-star clear as unlocking, not just three stars', () => {
    expect(isChapterUnlocked('1-2-vector-arithmetic', { '1-1-vectors': 1 })).toBe(true);
  });

  it('refuses to unlock a chapter for an unknown id', () => {
    expect(isChapterUnlocked('nope', {})).toBe(false);
  });

  it('opens the whole course once everything is complete', () => {
    const completed: CompletionRecord = {};
    for (const game of allGames()) completed[game.id] = 3;

    const states = resolveUnlocks(completed);
    expect([...states.values()].every((s) => s.status === 'completed')).toBe(true);
  });

  it('resolves a chain from an arbitrary mid-course position', () => {
    const completed: CompletionRecord = {};
    for (const chapter of orderedChapters().slice(0, 10)) completed[chapter.id] = 2;

    const states = resolveUnlocks(completed);
    expect(states.get(orderedChapters()[10]!.id)!.status).toBe('available');
    expect(states.get(orderedChapters()[11]!.id)!.status).toBe('locked');
  });
});

describe('nextPlayableChapter', () => {
  it('sends a new player to the first chapter', () => {
    expect(nextPlayableChapter({})?.id).toBe('1-1-vectors');
  });

  it('sends a returning player to where they left off', () => {
    const completed: CompletionRecord = { '1-1-vectors': 3, '1-2-vector-arithmetic': 2 };
    expect(nextPlayableChapter(completed)?.id).toBe('1-3-similarity-distance');
  });

  it('returns nothing once the course is finished', () => {
    const completed: CompletionRecord = {};
    for (const game of allGames()) completed[game.id] = 3;
    expect(nextPlayableChapter(completed)).toBeNull();
  });
});

describe('progress summary', () => {
  it('starts a new player at the first rank', () => {
    const summary = summarise({}, 0);
    expect(summary.rank.title).toBe('Gradient Novice');
    expect(summary.chaptersCompleted).toBe(0);
    expect(summary.starsEarned).toBe(0);
    expect(summary.starsPossible).toBe(66);
  });

  it('promotes through the ranks with XP', () => {
    expect(summarise({}, 250).rank.title).toBe('Vector Wrangler');
    expect(summarise({}, 249).rank.title).toBe('Gradient Novice');
  });

  it('reports the XP still needed for the next rank', () => {
    const summary = summarise({}, 100);
    expect(summary.nextRank?.title).toBe('Vector Wrangler');
    expect(summary.xpToNextRank).toBe(150);
  });

  it('reports no next rank at the top', () => {
    const top = manifest.ranks.reduce((a, b) => (b.minXp > a.minXp ? b : a));
    const summary = summarise({}, top.minXp + 1000);
    expect(summary.rank.title).toBe(top.title);
    expect(summary.nextRank).toBeNull();
    expect(summary.xpToNextRank).toBe(0);
  });

  it('totals stars across completed chapters', () => {
    const summary = summarise({ '1-1-vectors': 3, '1-2-vector-arithmetic': 2 }, 250);
    expect(summary.starsEarned).toBe(5);
    expect(summary.chaptersCompleted).toBe(2);
  });

  it('makes the top rank reachable by completing the course', () => {
    const totalXp = allGames().reduce((sum, game) => sum + game.xpReward, 0);
    const top = manifest.ranks.reduce((a, b) => (b.minXp > a.minXp ? b : a));
    expect(totalXp).toBeGreaterThanOrEqual(top.minXp);
  });
});
