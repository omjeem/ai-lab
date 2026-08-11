/**
 * Validates every game definition and the curriculum manifest.
 *
 * Runs as a pre-step for `dev` and `build`, so a malformed level config can
 * never reach the browser. Checks each file against the Zod schema, then the
 * cross-file invariants the schema alone cannot see: manifest/definition
 * agreement, unlock-graph integrity, and engine coverage.
 *
 *   pnpm validate:games
 */
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import {
  gameDefinitionSchema,
  curriculumManifestSchema,
  type GameDefinition,
  type CurriculumManifest,
} from '../src/types/game';

const ROOT = path.resolve(__dirname, '..');
const GAMES_DIR = path.join(ROOT, 'data', 'games');
const ENGINES_DIR = path.join(ROOT, 'src', 'engines');
const MANIFEST_PATH = path.join(GAMES_DIR, 'curriculum-manifest.json');

const errors: string[] = [];
const warnings: string[] = [];

function fail(file: string, message: string) {
  errors.push(`  ${file}\n    ${message}`);
}

function readJson(filePath: string): unknown {
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch (err) {
    fail(path.relative(ROOT, filePath), `Invalid JSON — ${(err as Error).message}`);
    return null;
  }
}

/* ── collect game files ─────────────────────────────────────── */

function collectGameFiles(): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(GAMES_DIR)) {
    const full = path.join(GAMES_DIR, entry);
    if (!statSync(full).isDirectory()) continue;
    for (const file of readdirSync(full)) {
      if (file.endsWith('.json')) files.push(path.join(full, file));
    }
  }
  return files.sort();
}

const gameFiles = collectGameFiles();
const games = new Map<string, GameDefinition>();

for (const filePath of gameFiles) {
  const rel = path.relative(ROOT, filePath);
  const raw = readJson(filePath);
  if (raw === null) continue;

  const parsed = gameDefinitionSchema.safeParse(raw);
  if (!parsed.success) {
    for (const issue of parsed.error.issues) {
      fail(rel, `${issue.path.join('.') || '(root)'}: ${issue.message}`);
    }
    continue;
  }

  const game = parsed.data;
  const expectedName = `${game.id}.json`;
  if (path.basename(filePath) !== expectedName) {
    fail(rel, `Filename must match the chapter id (expected ${expectedName})`);
  }
  if (games.has(game.id)) {
    fail(rel, `Duplicate chapter id "${game.id}"`);
  }
  games.set(game.id, game);
}

/* ── manifest ───────────────────────────────────────────────── */

let manifest: CurriculumManifest | null = null;
if (!existsSync(MANIFEST_PATH)) {
  errors.push(`  data/games/curriculum-manifest.json\n    File is missing`);
} else {
  const raw = readJson(MANIFEST_PATH);
  const parsed = curriculumManifestSchema.safeParse(raw);
  if (!parsed.success) {
    for (const issue of parsed.error.issues) {
      fail('data/games/curriculum-manifest.json', `${issue.path.join('.')}: ${issue.message}`);
    }
  } else {
    manifest = parsed.data;
  }
}

/* ── cross-file invariants ──────────────────────────────────── */

if (manifest) {
  const M = 'data/games/curriculum-manifest.json';
  const listed = new Set<string>();

  for (const world of manifest.worlds) {
    const orders = world.chapters.map((c) => c.order);
    if (new Set(orders).size !== orders.length) {
      fail(M, `World ${world.world} has duplicate chapter order values`);
    }

    for (const chapter of world.chapters) {
      listed.add(chapter.id);
      const game = games.get(chapter.id);
      if (!game) {
        fail(M, `Chapter "${chapter.id}" is listed but has no game definition`);
        continue;
      }
      if (!existsSync(path.join(GAMES_DIR, chapter.file))) {
        fail(M, `Chapter "${chapter.id}" points at a missing file: ${chapter.file}`);
      }
      if (game.world !== world.world) {
        fail(M, `Chapter "${chapter.id}" is under world ${world.world} but declares world ${game.world}`);
      }
      if (game.order !== chapter.order) {
        fail(M, `Chapter "${chapter.id}" order ${chapter.order} disagrees with definition order ${game.order}`);
      }
      if (game.xpReward !== chapter.xpReward) {
        fail(M, `Chapter "${chapter.id}" xpReward ${chapter.xpReward} disagrees with definition ${game.xpReward}`);
      }
      if (game.modelRequirement.tier !== chapter.tier) {
        fail(M, `Chapter "${chapter.id}" tier "${chapter.tier}" disagrees with definition "${game.modelRequirement.tier}"`);
      }
      if (game.chapterTitle !== chapter.title) {
        fail(M, `Chapter "${chapter.id}" title "${chapter.title}" disagrees with definition "${game.chapterTitle}"`);
      }
      const a = [...game.unlockRequires].sort().join(',');
      const b = [...chapter.unlockRequires].sort().join(',');
      if (a !== b) {
        fail(M, `Chapter "${chapter.id}" unlockRequires disagree between manifest and definition`);
      }
    }
  }

  for (const id of games.keys()) {
    if (!listed.has(id)) fail(M, `Game "${id}" exists on disk but is not listed in the manifest`);
  }

  // Ranks must start at zero and increase, or XP will map to no title.
  const ranks = manifest.ranks;
  if (ranks[0]!.minXp !== 0) {
    fail(M, 'The first rank must start at 0 XP');
  }
  for (let i = 1; i < ranks.length; i++) {
    if (ranks[i]!.minXp <= ranks[i - 1]!.minXp) {
      fail(M, `Rank "${ranks[i]!.title}" does not require more XP than the rank before it`);
    }
  }
}

/* ── unlock graph ───────────────────────────────────────────── */

for (const game of games.values()) {
  for (const req of game.unlockRequires) {
    if (!games.has(req)) {
      fail(`${game.id}.json`, `unlockRequires references unknown chapter "${req}"`);
    }
  }
}

// Cycle detection — a cycle would make a chapter permanently unreachable.
{
  const state = new Map<string, 'visiting' | 'done'>();
  const walk = (id: string, trail: string[]): void => {
    const seen = state.get(id);
    if (seen === 'done') return;
    if (seen === 'visiting') {
      fail('unlock graph', `Cycle detected: ${[...trail, id].join(' → ')}`);
      return;
    }
    state.set(id, 'visiting');
    for (const req of games.get(id)?.unlockRequires ?? []) {
      if (games.has(req)) walk(req, [...trail, id]);
    }
    state.set(id, 'done');
  };
  for (const id of games.keys()) walk(id, []);

  const roots = [...games.values()].filter((g) => g.unlockRequires.length === 0);
  if (roots.length === 0) {
    fail('unlock graph', 'No chapter has an empty unlockRequires — nothing can ever be played first');
  }
}

/* ── engine coverage ────────────────────────────────────────── */

if (existsSync(ENGINES_DIR)) {
  const engineFiles = new Set(
    readdirSync(ENGINES_DIR)
      .filter((f) => f.endsWith('Engine.ts'))
      .map((f) => f.replace(/Engine\.ts$/, ''))
  );
  const used = new Set<string>();
  for (const game of games.values()) {
    for (const level of game.levels) {
      used.add(level.engineType);
      if (engineFiles.size > 0 && !engineFiles.has(level.engineType)) {
        fail(`${game.id}.json`, `Level "${level.id}" needs engine "${level.engineType}Engine.ts", which does not exist`);
      }
    }
  }
  for (const engine of engineFiles) {
    if (engine !== 'scoring' && !used.has(engine)) {
      warnings.push(`  Engine "${engine}Engine.ts" is not referenced by any level`);
    }
  }
}

/* ── report ─────────────────────────────────────────────────── */

const levelCount = [...games.values()].reduce((n, g) => n + g.levels.length, 0);
const totalXp = [...games.values()].reduce((n, g) => n + g.xpReward, 0);

if (warnings.length > 0) {
  console.warn(`\n${warnings.length} warning(s):`);
  console.warn(warnings.join('\n'));
}

if (errors.length > 0) {
  console.error(`\n✗ Game validation failed — ${errors.length} error(s):\n`);
  console.error(errors.join('\n\n'));
  console.error('');
  process.exit(1);
}

console.log(
  `✓ ${games.size} chapters, ${levelCount} levels, ${totalXp} XP total — schema, manifest, unlock graph and engine coverage all valid.`
);
