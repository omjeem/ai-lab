import { describe, it, expect } from 'vitest';
import { meetsCriteria, starsFor, scoreLevel, rankForXp } from '@/engines/scoringEngine';
import type { EngineRules, PassCriteria, StarRule, Rank } from '@/types/game';

const gteRules: EngineRules = {
  passCriteria: { metric: 'accuracy', threshold: 0.6, comparator: 'gte' },
  starsRules: [
    { threshold: 0.6, stars: 1 },
    { threshold: 0.8, stars: 2 },
    { threshold: 0.95, stars: 3 },
  ],
  xpReward: 40,
};

const lteRules: EngineRules = {
  passCriteria: { metric: 'error', threshold: 0.3, comparator: 'lte' },
  starsRules: [
    { threshold: 0.3, stars: 1 },
    { threshold: 0.2, stars: 2 },
    { threshold: 0.1, stars: 3 },
  ],
  xpReward: 50,
};

describe('meetsCriteria', () => {
  it('handles gte at, above and below the threshold', () => {
    const c: PassCriteria = { metric: 'm', threshold: 0.6, comparator: 'gte' };
    expect(meetsCriteria(0.6, c)).toBe(true);
    expect(meetsCriteria(0.61, c)).toBe(true);
    expect(meetsCriteria(0.59, c)).toBe(false);
  });

  it('handles lte at, below and above the threshold', () => {
    const c: PassCriteria = { metric: 'm', threshold: 0.3, comparator: 'lte' };
    expect(meetsCriteria(0.3, c)).toBe(true);
    expect(meetsCriteria(0.29, c)).toBe(true);
    expect(meetsCriteria(0.31, c)).toBe(false);
  });

  it('handles eq with a tolerance for float noise', () => {
    const c: PassCriteria = { metric: 'm', threshold: 3, comparator: 'eq' };
    expect(meetsCriteria(3, c)).toBe(true);
    expect(meetsCriteria(3 + 1e-12, c)).toBe(true);
    expect(meetsCriteria(3.01, c)).toBe(false);
  });

  it('treats a non-finite metric as a failure', () => {
    const c: PassCriteria = { metric: 'm', threshold: 0.6, comparator: 'gte' };
    expect(meetsCriteria(NaN, c)).toBe(false);
    expect(meetsCriteria(Infinity, c)).toBe(true);
    expect(meetsCriteria(-Infinity, c)).toBe(false);
  });

  it('accepts negative thresholds, which loss-style metrics rely on', () => {
    const c: PassCriteria = { metric: 'loss', threshold: -1.5, comparator: 'lte' };
    expect(meetsCriteria(-1.8, c)).toBe(true);
    expect(meetsCriteria(-1.2, c)).toBe(false);
  });
});

describe('starsFor', () => {
  it('awards the highest band reached under gte', () => {
    expect(starsFor(0.55, gteRules.starsRules, 'gte')).toBe(0);
    expect(starsFor(0.6, gteRules.starsRules, 'gte')).toBe(1);
    expect(starsFor(0.79, gteRules.starsRules, 'gte')).toBe(1);
    expect(starsFor(0.8, gteRules.starsRules, 'gte')).toBe(2);
    expect(starsFor(1, gteRules.starsRules, 'gte')).toBe(3);
  });

  it('awards the highest band reached under lte, where lower is better', () => {
    expect(starsFor(0.4, lteRules.starsRules, 'lte')).toBe(0);
    expect(starsFor(0.3, lteRules.starsRules, 'lte')).toBe(1);
    expect(starsFor(0.15, lteRules.starsRules, 'lte')).toBe(2);
    expect(starsFor(0.02, lteRules.starsRules, 'lte')).toBe(3);
  });

  it('reads rules correctly when they are declared out of order', () => {
    const shuffled: StarRule[] = [
      { threshold: 0.95, stars: 3 },
      { threshold: 0.6, stars: 1 },
      { threshold: 0.8, stars: 2 },
    ];
    expect(starsFor(0.85, shuffled, 'gte')).toBe(2);
  });

  it('supports sparse bands that skip a star value', () => {
    const sparse: StarRule[] = [
      { threshold: 0.66, stars: 1 },
      { threshold: 0.99, stars: 3 },
    ];
    expect(starsFor(0.7, sparse, 'gte')).toBe(1);
    expect(starsFor(1, sparse, 'gte')).toBe(3);
  });

  it('awards no stars for a non-finite metric', () => {
    expect(starsFor(NaN, gteRules.starsRules, 'gte')).toBe(0);
  });
});

describe('scoreLevel', () => {
  it('pays full XP on a pass and reports the band reached', () => {
    const result = scoreLevel({ metric: 'accuracy', value: 0.82, rules: gteRules });
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(2);
    expect(result.xpEarned).toBe(40);
    expect(result.metric).toBe('accuracy');
    expect(result.value).toBe(0.82);
  });

  it('pays nothing and awards no stars on a fail', () => {
    const result = scoreLevel({ metric: 'accuracy', value: 0.4, rules: gteRules });
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
    expect(result.xpEarned).toBe(0);
  });

  it('never awards stars to a run that failed the pass criteria', () => {
    // A star band looser than the pass threshold must not leak through.
    const inconsistent: EngineRules = {
      passCriteria: { metric: 'm', threshold: 0.9, comparator: 'gte' },
      starsRules: [
        { threshold: 0.1, stars: 1 },
        { threshold: 0.95, stars: 3 },
      ],
      xpReward: 10,
    };
    const result = scoreLevel({ metric: 'm', value: 0.5, rules: inconsistent });
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
  });

  it('passes the breakdown through untouched', () => {
    const result = scoreLevel({
      metric: 'error',
      value: 0.12,
      rules: lteRules,
      breakdown: { attempts: 3, elapsedMs: 4200 },
    });
    expect(result.breakdown).toEqual({ attempts: 3, elapsedMs: 4200 });
    expect(result.stars).toBe(2);
  });

  it('defaults the breakdown to an empty object', () => {
    expect(scoreLevel({ metric: 'm', value: 1, rules: gteRules }).breakdown).toEqual({});
  });

  it('coerces a NaN metric into a clean failure instead of propagating it', () => {
    const result = scoreLevel({ metric: 'm', value: NaN, rules: gteRules });
    expect(result.passed).toBe(false);
    expect(result.xpEarned).toBe(0);
    expect(Number.isNaN(result.value)).toBe(true);
  });
});

describe('rankForXp', () => {
  const ranks: Rank[] = [
    { title: 'Gradient Novice', minXp: 0 },
    { title: 'Vector Wrangler', minXp: 250 },
    { title: 'Backprop Adept', minXp: 1000 },
  ];

  it('returns the highest rank whose requirement is met', () => {
    expect(rankForXp(0, ranks).title).toBe('Gradient Novice');
    expect(rankForXp(249, ranks).title).toBe('Gradient Novice');
    expect(rankForXp(250, ranks).title).toBe('Vector Wrangler');
    expect(rankForXp(999, ranks).title).toBe('Vector Wrangler');
    expect(rankForXp(50_000, ranks).title).toBe('Backprop Adept');
  });

  it('floors negative XP at the first rank', () => {
    expect(rankForXp(-10, ranks).title).toBe('Gradient Novice');
  });

  it('reads ranks declared out of order', () => {
    const shuffled: Rank[] = [ranks[2]!, ranks[0]!, ranks[1]!];
    expect(rankForXp(300, shuffled).title).toBe('Vector Wrangler');
  });
});
