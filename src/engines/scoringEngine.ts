/**
 * Shared XP and star-rating logic.
 *
 * Every engine funnels its raw metric through here, so "what counts as a pass"
 * and "what earns three stars" are defined once and read from level JSON —
 * never reimplemented per chapter.
 */
import type {
  Comparator,
  EngineRules,
  PassCriteria,
  Rank,
  ScoreResult,
  StarRule,
} from '@/types/game';

/** Tolerance for `eq` comparisons, so float noise cannot fail an exact match. */
const EQ_EPSILON = 1e-9;

export function meetsCriteria(value: number, criteria: PassCriteria): boolean {
  if (Number.isNaN(value)) return false;
  switch (criteria.comparator) {
    case 'gte':
      return value >= criteria.threshold;
    case 'lte':
      return value <= criteria.threshold;
    case 'eq':
      return Math.abs(value - criteria.threshold) <= EQ_EPSILON;
  }
}

/**
 * Highest star band the metric reaches.
 *
 * Bands are read in the comparator's direction: under `gte` a higher value is
 * better, under `lte` a lower one is. Rules may be declared in any order and
 * may skip a star value.
 */
export function starsFor(
  value: number,
  rules: readonly StarRule[],
  comparator: Comparator
): 0 | 1 | 2 | 3 {
  if (Number.isNaN(value)) return 0;

  let best = 0;
  for (const rule of rules) {
    const reached =
      comparator === 'gte'
        ? value >= rule.threshold
        : comparator === 'lte'
          ? value <= rule.threshold
          : Math.abs(value - rule.threshold) <= EQ_EPSILON;
    if (reached && rule.stars > best) best = rule.stars;
  }
  return best as 0 | 1 | 2 | 3;
}

export interface ScoreLevelParams {
  metric: string;
  value: number;
  rules: EngineRules;
  breakdown?: Record<string, number>;
}

export function scoreLevel({ metric, value, rules, breakdown }: ScoreLevelParams): ScoreResult {
  const passed = meetsCriteria(value, rules.passCriteria);
  // Stars are gated behind the pass so a loose star band in JSON can never
  // hand out a rating for a run that did not clear the bar.
  const stars = passed ? starsFor(value, rules.starsRules, rules.passCriteria.comparator) : 0;

  return {
    metric,
    value,
    passed,
    stars,
    xpEarned: passed ? rules.xpReward : 0,
    breakdown: breakdown ?? {},
  };
}

/** Highest rank whose XP requirement the player has met. */
export function rankForXp(xp: number, ranks: readonly Rank[]): Rank {
  const sorted = [...ranks].sort((a, b) => a.minXp - b.minXp);
  let current = sorted[0]!;
  for (const rank of sorted) {
    if (xp >= rank.minXp) current = rank;
    else break;
  }
  return current;
}

/** XP still needed for the next rank, and that rank — null once at the top. */
export function nextRank(xp: number, ranks: readonly Rank[]): { rank: Rank; xpRemaining: number } | null {
  const sorted = [...ranks].sort((a, b) => a.minXp - b.minXp);
  for (const rank of sorted) {
    if (xp < rank.minXp) return { rank, xpRemaining: rank.minXp - xp };
  }
  return null;
}
