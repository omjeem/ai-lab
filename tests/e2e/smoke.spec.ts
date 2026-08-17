import { test, expect, type Page, type Locator } from '@playwright/test';

/**
 * A3 (`plan-docs/REMAINING-WORK.md`) — the smoke test: onboarding → map →
 * open a chapter → clear a level → chapter unlocks.
 *
 * One continuous journey in one test (not several), deliberately — separate
 * `test()` blocks each get their own isolated browser context by default, so
 * the identity/progress created in one would never be visible to the next.
 *
 * Runs a real browser against `1-1-vectors` (the only chapter with an empty
 * `unlockRequires`, so it needs no prior progress) end to end, including its
 * real ~23MB embedding model download. The engine test suites already cover
 * scoring correctness — this only proves the wiring between UI, engine and
 * progress store actually holds together for a real player.
 *
 * Every "correct" placement/label/magnitude below is taken from this level's
 * own already-verified `hints` in `data/games/world-1-fundamentals/1-1-vectors.json`,
 * not guessed — see README, "Level hints".
 */

const MODEL_TIMEOUT = 90_000;

test('onboarding → map → clear the entry chapter → next chapter unlocks', async ({ page }) => {
  await test.step('onboarding creates an identity and lands on the map', async () => {
    await page.goto('/onboarding');
    await page.waitForLoadState('networkidle');

    const name = page.getByPlaceholder('display name');
    const startButton = page.getByRole('button', { name: /^start$/i });
    await name.fill(`Smoke Test ${Date.now()}`);
    // Next dev's first-visit compile can hot-refresh the route right after
    // load, remounting the form and silently dropping a fill that landed
    // just before it. Confirm the value stuck rather than racing that.
    if (!(await startButton.isEnabled())) {
      await name.fill(`Smoke Test ${Date.now()}`);
    }
    await expect(startButton).toBeEnabled();
    await startButton.click();
    await expect(page).toHaveURL(/\/map$/);
  });

  await test.step('the entry chapter is available; the next one is locked', async () => {
    await expect(
      page.getByRole('link', { name: /What is a Vector\?, available/i })
    ).toBeVisible();
    await expect(page.getByLabel(/Vector Arithmetic, locked/i)).toBeVisible();
  });

  await test.step('open the entry chapter', async () => {
    await page.getByRole('link', { name: /What is a Vector\?, available/i }).click();
    await expect(page).toHaveURL(/\/world\/1\/chapter\/1-1-vectors/);
  });

  await test.step('level 1 — place-and-cluster', async () => playLevel1PlaceAndCluster(page));
  await test.step('level 2 — guess-the-label', async () => playLevel2GuessTheLabel(page));
  await test.step('level 3 — magnitude', async () => playLevel3Magnitude(page));

  await test.step('the chapter completes and the next one unlocks on the map', async () => {
    await expect(page.getByRole('dialog', { name: 'Chapter complete' })).toBeVisible();
    await page.getByRole('link', { name: /^map$/i }).click();

    await expect(page).toHaveURL(/\/map$/);
    await expect(
      page.getByRole('link', { name: /What is a Vector\?, completed/i })
    ).toBeVisible();
    await expect(
      page.getByRole('link', { name: /Vector Arithmetic, available/i })
    ).toBeVisible();
  });
});

/** Level 1 — place-and-cluster: animals in one region, vehicles in another. */
async function playLevel1PlaceAndCluster(page: Page) {
  await page.getByRole('button', { name: /^begin$/i }).click();

  // First model download of the run — give the real HF Hub fetch room to finish.
  await expect(page.getByRole('button', { name: 'cat', exact: true })).toBeVisible({
    timeout: MODEL_TIMEOUT,
  });

  const animals = ['cat', 'dog', 'hamster'];
  const vehicles = ['truck', 'bus', 'bicycle'];

  for (const word of animals) await placeAndNudge(page, word, 'ArrowLeft');
  for (const word of vehicles) await placeAndNudge(page, word, 'ArrowRight');

  await page.getByRole('button', { name: /submit layout/i }).click();
  await expect(page.getByRole('button', { name: /next level/i })).toBeVisible();
  await page.getByRole('button', { name: /next level/i }).click();
}

/** Drops a tray chip at the plot's centre, then walks it toward one edge. */
async function placeAndNudge(page: Page, word: string, direction: 'ArrowLeft' | 'ArrowRight') {
  await page.getByRole('button', { name: word, exact: true }).click();
  const placed = page.getByRole('button', { name: new RegExp(`^${word},`) });
  for (let i = 0; i < 9; i++) await placed.press(direction);
}

/** Level 2 — guess-the-label: assign each word to its real cluster's label. */
async function playLevel2GuessTheLabel(page: Page) {
  await expect(page.getByRole('button', { name: /^begin$/i })).toBeVisible();
  await page.getByRole('button', { name: /^begin$/i }).click();

  const panel = page.locator('section', { has: page.getByText('assign labels', { exact: true }) });
  await expect(panel).toBeVisible({ timeout: MODEL_TIMEOUT });

  const truth: Record<string, string> = {
    violin: 'instruments',
    trumpet: 'instruments',
    cello: 'instruments',
    hydrogen: 'elements',
    oxygen: 'elements',
    nitrogen: 'elements',
    lasagna: 'food',
    risotto: 'food',
    paella: 'food',
  };

  for (const [word, label] of Object.entries(truth)) {
    const row = rowFor(panel, word);
    await row.getByRole('button', { name: label, exact: true }).click();
  }

  await page.getByRole('button', { name: /submit layout/i }).click();
  await expect(page.getByRole('button', { name: /next level/i })).toBeVisible();
  await page.getByRole('button', { name: /next level/i }).click();
}

/** The row containing a given word's label — the span, then its parent row. */
function rowFor(scope: Locator, word: string): Locator {
  return scope.getByText(word, { exact: true }).locator('..');
}

/** Level 3 — drag-vector-magnitude: every word here is ~unit length. */
async function playLevel3Magnitude(page: Page) {
  await expect(page.getByRole('button', { name: /^begin$/i })).toBeVisible();
  await page.getByRole('button', { name: /^begin$/i }).click();

  for (const word of ['ocean', 'the', 'quantum']) {
    const slider = page.getByRole('slider', { name: word });
    await expect(slider).toBeVisible({ timeout: MODEL_TIMEOUT });
    // Registers a real onChange at 1.0 — the slider's own displayed default,
    // but `magnitudeGuess` stays null (and scores 0) until it actually fires.
    await slider.press('ArrowRight');
    await slider.press('ArrowLeft');
  }

  await page.getByRole('button', { name: /submit layout/i }).click();
  await expect(page.getByRole('button', { name: /finish chapter/i })).toBeVisible();
  await page.getByRole('button', { name: /finish chapter/i }).click();
}
