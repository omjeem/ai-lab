import { z } from 'zod';

/**
 * Schema version for every game definition on disk.
 *
 * Bump this whenever the shape changes in a way that would make previously
 * saved progress ambiguous. Saved progress records the version it was earned
 * under so old runs can be migrated rather than silently misread.
 */
export const GAME_SCHEMA_VERSION = 1;

/** Where a chapter's inference runs. */
export const modelTierSchema = z.enum([
  'none', // pure math / JS, no model involved
  'browser-light', // transformers.js, < 100MB
  'browser-heavy', // larger transformers.js model, WebLLM, or a live-trained net
  'cloud', // Ollama Cloud via the API proxy
]);
export type ModelTier = z.infer<typeof modelTierSchema>;

export const difficultySchema = z.enum(['easy', 'medium', 'hard']);
export type Difficulty = z.infer<typeof difficultySchema>;

export const comparatorSchema = z.enum(['gte', 'lte', 'eq']);
export type Comparator = z.infer<typeof comparatorSchema>;

export const passCriteriaSchema = z.object({
  /** Name of the metric the engine's evaluate() reports, e.g. "clusterSeparationScore". */
  metric: z.string().min(1),
  threshold: z.number(),
  comparator: comparatorSchema,
});
export type PassCriteria = z.infer<typeof passCriteriaSchema>;

/**
 * Score band -> stars. Interpreted in the direction of the level's comparator:
 * for "gte" levels a higher metric earns more stars, for "lte" a lower one does.
 */
export const starRuleSchema = z.object({
  threshold: z.number(),
  stars: z.number().int().min(1).max(3),
});
export type StarRule = z.infer<typeof starRuleSchema>;

export const gameLevelSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  difficulty: difficultySchema,
  instructions: z.string().min(1),
  /** Maps to exactly one module in /src/engines. */
  engineType: z.string().min(1),
  /** Level-specific parameters the engine consumes. Shape is engine-defined. */
  engineConfig: z.record(z.string(), z.unknown()),
  passCriteria: passCriteriaSchema,
  starsRules: z.array(starRuleSchema).min(1),
  /** Portion of the chapter's XP awarded for clearing this level. */
  xpReward: z.number().int().nonnegative(),
  /** Stretch levels that do not gate chapter completion. */
  optional: z.boolean().optional(),
  /**
   * Progressive hints, revealed one at a time by `ChapterShell`'s hint panel.
   * Optional so chapters without a canvas yet can stay silent on it, but every
   * built chapter should carry 2–3: the first names the approach or direction
   * to try, the middle one(s) go deeper into why, and the last gives the
   * concrete answer or a verified worked example — never a vague restatement.
   * Never computed or fabricated; write these against the level's own real
   * config and, where the mechanic allows it, verify them by actually running
   * the engine before shipping.
   */
  hints: z.array(z.string().min(1)).min(1).max(6).optional(),
});
export type GameLevel = z.infer<typeof gameLevelSchema>;

export const modelRequirementSchema = z.object({
  tier: modelTierSchema,
  /** e.g. "Xenova/all-MiniLM-L6-v2", an Ollama model id, or null for tier "none". */
  modelId: z.string().nullable(),
  estimatedSizeMB: z.number().nonnegative().nullable(),
  /** Shown when tier is "cloud" and the device is offline. */
  fallbackMessage: z.string().nullable(),
  /** Shown when a browser-tier model fails to download or initialise. */
  loadFailureMessage: z.string().nullable(),
});
export type ModelRequirement = z.infer<typeof modelRequirementSchema>;

export const conceptSchema = z.object({
  /** 3-5 sentences, shown before play. */
  shortExplanation: z.string().min(1),
  /** Shown on chapter completion. */
  ahaMoment: z.string().min(1),
});

export const gameDefinitionSchema = z
  .object({
    schemaVersion: z.literal(GAME_SCHEMA_VERSION),
    id: z.string().min(1),
    world: z.number().int().min(1).max(6),
    worldTitle: z.string().min(1),
    chapterTitle: z.string().min(1),
    order: z.number().int().min(1),
    concept: conceptSchema,
    modelRequirement: modelRequirementSchema,
    xpReward: z.number().int().positive(),
    /** Chapter ids that must be completed before this one unlocks. */
    unlockRequires: z.array(z.string()),
    levels: z.array(gameLevelSchema).min(1),
  })
  .superRefine((game, ctx) => {
    const ids = new Set<string>();
    for (const level of game.levels) {
      if (ids.has(level.id)) {
        ctx.addIssue({
          code: 'custom',
          message: `Duplicate level id "${level.id}"`,
          path: ['levels'],
        });
      }
      ids.add(level.id);
    }

    // A chapter's XP must be exactly what its required levels pay out, so the
    // map's "chapter worth N XP" label can never drift from what play awards.
    const requiredXp = game.levels
      .filter((l) => !l.optional)
      .reduce((sum, l) => sum + l.xpReward, 0);
    if (requiredXp !== game.xpReward) {
      ctx.addIssue({
        code: 'custom',
        message: `xpReward (${game.xpReward}) must equal the sum of non-optional level xpReward (${requiredXp})`,
        path: ['xpReward'],
      });
    }

    if (game.modelRequirement.tier === 'none') {
      if (game.modelRequirement.modelId !== null) {
        ctx.addIssue({
          code: 'custom',
          message: 'tier "none" must have a null modelId',
          path: ['modelRequirement', 'modelId'],
        });
      }
    } else if (!game.modelRequirement.modelId) {
      ctx.addIssue({
        code: 'custom',
        message: `tier "${game.modelRequirement.tier}" requires a modelId`,
        path: ['modelRequirement', 'modelId'],
      });
    }

    if (game.modelRequirement.tier === 'cloud' && !game.modelRequirement.fallbackMessage) {
      ctx.addIssue({
        code: 'custom',
        message: 'cloud-tier chapters must define a fallbackMessage for offline use',
        path: ['modelRequirement', 'fallbackMessage'],
      });
    }

    const browserTier =
      game.modelRequirement.tier === 'browser-light' ||
      game.modelRequirement.tier === 'browser-heavy';
    if (browserTier && !game.modelRequirement.loadFailureMessage) {
      ctx.addIssue({
        code: 'custom',
        message: 'browser-tier chapters must define a loadFailureMessage for download failures',
        path: ['modelRequirement', 'loadFailureMessage'],
      });
    }

    for (const [i, level] of game.levels.entries()) {
      const sorted = [...level.starsRules].sort((a, b) => a.stars - b.stars);
      const starValues = sorted.map((r) => r.stars);
      if (new Set(starValues).size !== starValues.length) {
        ctx.addIssue({
          code: 'custom',
          message: `Level "${level.id}" repeats a star value`,
          path: ['levels', i, 'starsRules'],
        });
      }
      // Thresholds must get stricter as stars increase, in the comparator's direction.
      for (let j = 1; j < sorted.length; j++) {
        const prev = sorted[j - 1]!;
        const cur = sorted[j]!;
        const stricter =
          level.passCriteria.comparator === 'lte'
            ? cur.threshold < prev.threshold
            : cur.threshold > prev.threshold;
        if (level.passCriteria.comparator !== 'eq' && !stricter) {
          ctx.addIssue({
            code: 'custom',
            message: `Level "${level.id}": star thresholds must tighten as stars increase (comparator "${level.passCriteria.comparator}")`,
            path: ['levels', i, 'starsRules'],
          });
        }
      }
    }
  });

export type GameDefinition = z.infer<typeof gameDefinitionSchema>;

/* ────────────────────────────────────────────────────────────
   Curriculum manifest
   ──────────────────────────────────────────────────────────── */

export const manifestChapterSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  order: z.number().int().min(1),
  file: z.string().min(1),
  tier: modelTierSchema,
  xpReward: z.number().int().positive(),
  unlockRequires: z.array(z.string()),
});
export type ManifestChapter = z.infer<typeof manifestChapterSchema>;

export const manifestWorldSchema = z.object({
  world: z.number().int().min(1).max(6),
  title: z.string().min(1),
  subtitle: z.string().min(1),
  /** CSS custom-property hue driving this world's accent across its chapters. */
  accent: z.string().regex(/^#[0-9a-fA-F]{6}$/),
  directory: z.string().min(1),
  chapters: z.array(manifestChapterSchema).min(1),
});
export type ManifestWorld = z.infer<typeof manifestWorldSchema>;

export const rankSchema = z.object({
  title: z.string().min(1),
  minXp: z.number().int().nonnegative(),
});
export type Rank = z.infer<typeof rankSchema>;

export const curriculumManifestSchema = z.object({
  schemaVersion: z.literal(GAME_SCHEMA_VERSION),
  worlds: z.array(manifestWorldSchema).min(1),
  ranks: z.array(rankSchema).min(1),
});
export type CurriculumManifest = z.infer<typeof curriculumManifestSchema>;

/* ────────────────────────────────────────────────────────────
   Engine contracts
   ──────────────────────────────────────────────────────────── */

/** The subset of a level an engine needs in order to score itself. */
export interface EngineRules {
  passCriteria: PassCriteria;
  starsRules: StarRule[];
  xpReward: number;
}

export interface ScoreResult {
  metric: string;
  value: number;
  passed: boolean;
  stars: 0 | 1 | 2 | 3;
  xpEarned: number;
  /** Per-metric detail the UI can surface without recomputing anything. */
  breakdown: Record<string, number>;
}

/** Status shared by every engine state. */
export type EngineStatus = 'idle' | 'active' | 'complete';

export interface BaseEngineState {
  rules: EngineRules;
  status: EngineStatus;
  /** Every action applied, in order — powers undo, replay and attempt counting. */
  actionCount: number;
}
