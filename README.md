# AI Learning Lab

A browser-based course that teaches how language models work by making you operate one. Real
embeddings, real attention weights, real gradients — computed live on your machine and manipulated
directly, rather than described to you. There is no chatbot anywhere in the learning path.

30 chapters across 8 worlds so far, from "what is a vector" to red-teaming the course's own local
model's instruction-following. Worlds 1–7 are complete; World 8 (Scale, Efficiency & Safety) is in
progress — see `plan-docs/EXPANSION-PLAN.md` for what's built and what's next.

---

## Status

| Layer | State |
| --- | --- |
| Curriculum (30 chapters, 91 levels so far) | Complete for Worlds 1–7, World 8 in progress, schema-validated |
| Game logic engines (30) | Complete for every chapter built so far, 958 tests passing |
| Model wrappers (transformers.js, WebLLM, Ollama proxy) | Complete |
| Core UI shell, world map, onboarding, chapter frame | Complete |
| Per-chapter game canvases | **30 of 30 built so far** — every chapter in Worlds 1–7, plus World 8's first four |
| Backend, admin, offline sync, PWA | Complete |
| Sound design, offline path, Ollama Cloud and MongoDB round trips | Complete, verified for real |

Every chapter's logic, model wrapper and canvas built so far is finished and tested, and every
infrastructure item in `plan-docs/REMAINING-WORK.md` Part A that needed a live credential or a real
offline run has now been exercised for real — see that file for what each one found. The one item
still open needs a WebGPU-capable automated browser this environment cannot provide (canvas 20's live
playthrough). See [Adding a chapter's canvas](#adding-a-chapters-canvas) for the pattern, in case any
existing chapter needs revisiting, or `plan-docs/EXPANSION-PLAN.md` for World 8's remaining chapters.

---

## Tech stack

- **Next.js 15** (App Router), TypeScript strict mode
- **Tailwind CSS 4** with CSS custom properties for the per-world theming
- **Motion** (`motion/react`) for state transitions, unlock sequences and score reveals
- **Zustand** persisted to IndexedDB for progress and offline resume
- **@huggingface/transformers** for embeddings, tokenization, small causal LMs and attention
- **@mlc-ai/web-llm** for the World 6 local capstone model
- **Ollama Cloud** for the single optional cloud escalation, proxied server-side
- **MongoDB** via the official driver, reached only from API routes
- **idb** for the IndexedDB layer (progress, activity queue, model records)
- **Vitest** for the engine and library test suites
- **pnpm**

---

## Local setup

```bash
pnpm install
cp .env.local.example .env.local   # then fill in what you need
pnpm dev
```

The app runs with **no environment configuration at all** — every browser-tier chapter works, and
onboarding and activity fall back to local-only. Configure the environment when you want persistence,
the admin dashboard, or the cloud escalation.

### Environment variables

| Variable | Required for | Where to get it |
| --- | --- | --- |
| `MONGODB_URI` | Storing users and activity | MongoDB Atlas → Cluster → Connect → Drivers. Without it, `/api/users` and `/api/activity` accept and discard, so the client never retries forever. |
| `MONGODB_DB` | Database name | Any name; defaults to `ai_learning_lab`. |
| `ADMIN_EMAIL` | Admin login | Your choice. |
| `ADMIN_PASSWORD_HASH` | Admin login | Generate with `pnpm hash:password 'your-password'`. Only the hash is stored — never the plaintext. |
| `ADMIN_SESSION_SECRET` | Signing the admin session cookie | `openssl rand -hex 32` |
| `ADMIN_SESSION_HOURS` | Session lifetime | Optional, defaults to `12`. Refreshed on activity. |
| `OLLAMA_CLOUD_API_KEY` | World 6 cloud escalation | ollama.com → Settings → API keys. Never reaches the browser. |
| `OLLAMA_CLOUD_MODEL_ID` | World 6 cloud escalation | e.g. `gpt-oss:120b-cloud` |
| `OLLAMA_CLOUD_BASE_URL` | World 6 cloud escalation | Optional, defaults to `https://ollama.com`. |
| `GEO_LOOKUP_URL` | Optional IP geolocation | A provider of your own, with `{ip}` as the placeholder — e.g. `https://ipapi.co/{ip}/json/`. Unset means no geo lookup happens at all. |
| `GEO_LOOKUP_API_KEY` | Optional IP geolocation | Only if your provider needs one. |
| `RATE_LIMIT_ACTIVITY_PER_MIN` | Tuning | Optional, defaults to `60` per IP. |
| `RATE_LIMIT_CLOUD_INFERENCE_PER_MIN` | Tuning | Optional, defaults to `10` per IP. |
| `RATE_LIMIT_FEEDBACK_PER_MIN` | Tuning | Optional, defaults to `10` per IP. |

### Generating the admin password hash

```bash
pnpm hash:password 'a-long-password-you-choose'
# → ADMIN_PASSWORD_HASH="scrypt$<salt>$<hash>"
```

Paste the line into `.env.local`. The script refuses passwords under 12 characters.

---

## Commands

```bash
pnpm dev              # validates the curriculum, then starts the dev server
pnpm build            # validates the curriculum, then builds
pnpm test             # 958 unit tests, fully offline, no model downloads
pnpm test:watch
pnpm test:coverage
pnpm test:e2e         # one Playwright smoke test — real browser, real model download, opt-in
pnpm typecheck        # tsc --noEmit
pnpm lint
pnpm validate:games   # schema + cross-file curriculum validation
pnpm hash:password    # admin password hash
```

`pnpm validate:games` runs automatically before `dev` and `build`, so a malformed level config can
never reach the browser.

---

## Project structure

```
/data/games/            30 chapter definitions + curriculum-manifest.json
/public/corpora/        Bundled public-domain text the n-gram and RNN chapters count from
/scripts/               validate-games, calibrate-levels, hash-password
/src/engines/           Pure game logic, one module per game type. No React, no DOM.
/src/models/            Model lifecycle: transformers.js wrappers, WebLLM, the hand-rolled
                        TinyNet and TinyRNN, caching and progress
/src/components/        UI: design-system primitives, world map, chapter shell, game canvases
/src/lib/               MongoDB, admin auth, offline queue, sync manager, rate limiting
/src/store/             Zustand stores (durable progress, per-run session state)
/src/types/             Zod schemas and shared types
/tests/                 Mirrors src/engines, src/models and src/lib
```

### The two rules that shape everything

**Game logic lives in JSON, never in components.** Every level's parameters, pass criteria and star
bands come from `/data/games/**`. Components render engine state; they never own rules.

**Engines are pure and take their models by injection.** An engine never imports a model wrapper.
It receives one through a `prepare(config, deps)` parameter, which is why the whole test suite runs
offline in about a second with no downloads, while the app injects the real transformers.js wrapper
into the identical code path.

```ts
// Every engine exposes the same shape.
prepare(config, deps)          // optional; runs the real model, returns derived data
initState(config, rules, prepared)
applyAction(state, action)     // pure reducer, never mutates
evaluate(state) → ScoreResult
```

---

## Navigation, unlocking and sharing

A chapter's `unlockRequires` graph (`src/lib/curriculum.ts`) still drives everything about
progression — the map's lock icon, its "complete X first" tooltip, and what counts as the
legitimate next step. What changed is the *consequence* of a chapter being locked: it no longer
blocks navigation outright.

- **Every chapter route is reachable, including locked ones.** The map's `ChapterNode`
  (`src/components/map/WorldMap.tsx`) renders a real `<Link>` for unlocked/completed chapters and
  a `<button>` for locked ones. Locked chapters keep their dimmed styling and lock icon.
- **Locked chapters on the map open a warning modal** in place, rather than navigating immediately.
  The modal appears with a blurred/dimmed background over the map, names the specific unfinished
  prerequisite(s) by title, and offers "Play anyway" / "Stay on the map". Confirming navigates
  to the chapter URL; dismissing (or clicking the backdrop) closes the modal with no navigation.
- **Direct URL navigation never shows the warning** — typing or pasting a chapter URL, bookmarks,
  back-button navigation, and share links all render the chapter immediately, regardless of lock
  state. The gate lives in `WorldMap.tsx`, not on the chapter page.
- **Every chapter has a share icon** (map node and in-chapter header, both using
  `src/components/ui/ShareButton.tsx` → `src/lib/shareLink.ts`), producing a link with `?via=share`.
  Opening that link renders the chapter directly for anyone, regardless of their own progress —
  the point of a share link is to hand someone a working door into a specific chapter.
  - The marker is deliberately **not** named `source` — `src/middleware.ts` already intercepts any
    `?source=` on every route for an unrelated external-tracking beacon and strips it via redirect,
    which would destroy a same-named marker before it's ever read.
  - `public/sw.js`'s offline `networkFirst()` matches the request with `{ ignoreSearch: true }` so
    a `?via=share` URL still resolves to its precached chapter when opened offline as an installed
    PWA, instead of silently falling back to the `/map` shell.
- **One activity event type** tracks share opens (`src/types/activity.ts`):
  `chapter_shared_link_opened` (fired whenever `?via=share` is present, with a `detail.wasLocked`
  flag distinguishing "shared a chapter you'd already unlocked" from "the link let someone skip a
  gate"). The `chapter_jumped_ahead` event type is retained in the schema for historical data
  compatibility but is no longer fired by the application.

---

## SEO

Every chapter page (`src/app/(game)/world/[worldId]/chapter/[chapterId]/page.tsx`) is its own
independently indexable page, not a shared template:

- `generateMetadata` sets a real per-chapter `title`, `description` (the chapter's own
  `concept.shortExplanation`), `keywords` (`deriveChapterKeywords()` in `src/lib/seo.ts`), a
  canonical URL, and a **full** `openGraph`/`twitter` object — not a partial one. Next.js replaces
  rather than merges a parent layout's metadata per key, so a chapter route that returned only
  `{ title, description }` for `openGraph` was silently dropping the root layout's `type`/`url`/
  `siteName`/`summary_large_image` card the moment it touched that key at all.
- Each page also renders a `LearningResource` JSON-LD block (`teaches`, `isPartOf` the course) —
  same technique as the root page's `Course` schema in `src/app/page.tsx`, but naming the specific
  thing that one page teaches, which is the structured-data signal a topical search ("vectors
  explanation") actually keys off.
- `deriveChapterKeywords()` is **derived**, not hand-authored per chapter: Google's `keywords` meta
  tag hasn't affected ranking since ~2009, and the root layout already applies a site-wide keyword
  list to every page, so this exists to make the tag chapter-specific, not to fill a "zero
  keywords" gap. A curated field in each chapter's JSON would need a schema change and 26+ files of
  upkeep for a tag with no measurable ranking benefit.
- `src/app/sitemap.ts` and `robots.ts` already list/allow every chapter URL — no changes needed
  there. The real remaining lever after this is off-page (backlinks, domain authority) and time —
  nothing left here is a code problem.

---

## Adding a new chapter

1. **Write the JSON** in `/data/games/world-N-.../<id>.json`, matching the Zod schema in
   `src/types/game.ts`. Run `pnpm validate:games` — it checks the schema plus the cross-file
   invariants: manifest agreement, unlock-graph cycles, XP sums, and that an engine exists.
2. **Write the engine test** in `/tests/engines/<name>Engine.test.ts` first. Cover the initial state,
   valid transitions, invalid and edge-case input, scoring against each level's real config, and — for
   model-backed engines — behaviour with a fake model injected.
3. **Write the engine** in `/src/engines/<name>Engine.ts` until the tests pass. Keep it free of React
   and DOM imports.
4. **Wire the model**, if it needs one, by implementing or reusing an interface from
   `src/engines/deps.ts`.
5. **Build the canvas** in `/src/components/games/<chapter-id>/` and register it in
   `src/components/games/registry.tsx`.
6. **Add it to `curriculum-manifest.json`** with its unlock requirements.

### Adding a chapter's canvas

Two things every canvas that hides its answers until submit has to get right:

- **Keep the reveal in local component state, never `state.status === 'complete'`.** Engines set the
  status back to `active` on any subsequent action, so a post-reveal control — spinning the wheel,
  logging another attempt — silently un-reveals what was just shown.
- **Call `useRetrySignal`** (`src/components/games/useRetrySignal.ts`). The HUD's "Try again" only
  puts the shell back into `playing`; without the hook the player retries onto a board that still
  shows the answers and, for something like a fully merged BPE puzzle, cannot be replayed at all.

`src/components/games/registry.tsx` maps a chapter id to a lazily-loaded component. Use
`1-1-vectors/VectorCanvas.tsx` as the reference: it wraps its content in `<ModelGate>` (which owns
download progress, the failure state and retry), drives the engine through `applyAction`, reports the
live score to the HUD via `onScore`, and submits with `onSubmit`.

### Level hints

Every level should carry a `hints` array in its JSON — this is infrastructure, not per-canvas work.
`ChapterShell`'s HUD (`src/components/chapter/ChapterShell.tsx`) renders a "hints" panel automatically
for any level whose config includes one; a canvas needs no code of its own for this to work.

```json
"hints": [
  "First hint: names the approach or direction to try, without giving numbers away.",
  "Middle hint(s): goes deeper into *why* — the mechanism, not just a restatement of the first hint.",
  "Last hint: the concrete answer — actual numbers, an actual sequence, a verified worked example."
]
```

- **2–3 hints is typical; it can vary.** A quiz-style level (pick the right label) may only need one
  short hint. An open-ended level (tune a learning rate, build an analogy) may want three.
- **The last hint is a real, checked answer, never a vague nudge.** For engine-scored levels this
  means actually running the level's config — through the engine directly, or with a throwaway script
  — and reporting a verified value, not a plausible-sounding guess. Several existing hints (e.g. in
  `2-1-perceptron`, `2-3-gradient-descent`, `3-2-layers-forward-pass`) exist specifically because the
  "obvious" answer turned out to be wrong or suboptimal once actually run — see those files' hints for
  the pattern.
- **Open-ended levels with no single right answer** (free-text prompts, "bring your own words") still
  get a concrete last hint: a specific worked example that is verified to pass, framed as "a
  combination that works," not "the answer."
- **Never fabricate a number.** If a hint states a threshold, a computed value, or "the model's real
  top token," it must come from actually running the config — the same standard the rest of this
  project holds itself to (see "What 'real' means here" below).
- Schema: `hints` is `z.array(z.string().min(1)).min(1).max(6).optional()` in
  `src/types/game.ts` — optional so chapters without a canvas yet don't need it, but every level in a
  built chapter should have one before that chapter is considered done.

---

## Keeping levels honest

`pnpm tsx scripts/calibrate-levels.ts` plays every pure-computation level optimally and reports
whether its pass and 3-star thresholds are actually reachable.

This is not decoration. It caught four levels whose thresholds could never be met, and two whose
scoring could be gamed:

- A level scored "steps to converge" on data that label noise had made non-separable, so convergence
  was impossible by construction.
- Two levels had 3-star bands set beyond the achievable optimum.
- `2-4-l3` minimised the generalisation gap, which is trivially won by flattening the fit into a
  useless constant. Gap-scored levels now carry a `maxValidationLoss` ceiling.
- `3-4-l2` asked about batch size while the architecture was fixed at one that cannot learn the
  dataset at all, so it was scoring noise.
- `1-3-l3` asks where cosine and Euclidean disagree, but the embedding wrapper L2-normalises. On unit
  vectors Euclidean distance is `sqrt(2 - 2cos)`, strictly decreasing in cosine, so the two metrics
  cannot disagree about anything — 0 of 336 triples, against 74 of 336 unnormalised. Every answer was
  "they agree" and three identical clicks scored three stars. The level now takes its vectors from
  `rawEmbeddingModel`, selected by its `metric: "both"` config.

That last one is the case the calibration script cannot reach: it is model-backed, and the engine
suite injects planted unnormalised vectors, so both were satisfied while the real chapter was
unwinnable-by-understanding. Model-backed levels have to be played against the real model.

Add a case to the script whenever you add a pure-computation level. Model-backed levels are
calibrated against the real model in the browser instead.

---

## What "real" means here

The differentiator is that nothing shown to the player is fabricated:

- Word clusters come from k-means over live embeddings, and label truth from embedding the candidate
  labels — there is no answer key in the JSON.
- The BPE merge puzzle is scored against the tokenizer's own merge-rank table.
- `TinyNet` and `TinyRNN` are hand-written networks with real forward and backward passes; their
  analytic gradients are verified against numerical ones in the test suite.
- World 4.2's memory decay is measured by running two sequences that differ in one early token
  through the trained RNN and comparing its actual hidden states as the difference is overwritten.
- n-gram tables are counted from a bundled corpus at runtime, so changing the corpus changes the model.
- Attention weights and hidden states are read from a real transformer's forward pass. If the ONNX
  export does not expose them, the chapter says so and offers a retry rather than showing something
  plausible.
- The cloud endpoint returns no per-token detail, so the inspector shows an empty trace rather than
  invented probabilities.

---

## Offline behaviour

After the first visit, and once a chapter's model has been fetched once, Worlds 1–5 work with no
network:

- Model weights are cached by transformers.js and WebLLM in the browser's Cache Storage.
- The service worker caches the app shell, every chapter route and the bundled corpora. It
  deliberately does not re-cache model weights, which would double the storage cost for no benefit.
  Every chapter route is precached explicitly (`public/sw.js`'s `CHAPTER_URLS`) rather than left to
  cache-on-visit: a real player always reaches a chapter through a client-side `Link` transition from
  `/map`, which never arrives at the service worker as a `navigate`-mode request, so it was never
  being cached as a side effect — found and fixed while verifying this section for real (A4,
  `plan-docs/REMAINING-WORK.md`).
- Progress persists to IndexedDB and hydrates without a spinner.
- Activity is queued locally and synced when connectivity returns. The sync manager treats
  `navigator.onLine` and a real request to `/api/activity` as two separate signals, and clears only
  the event ids the server confirms.
- The sync manager's recurring timer (`src/lib/syncManager.ts`, every 30s by default) only probes
  `/api/activity` when the local queue actually has something in it (`queueSize()`, already
  exported from `src/lib/offlineQueue.ts`) — an idle device with nothing new to report doesn't keep
  making network requests forever. The initial check on mount, and the ones triggered by the
  browser's `online`/`visibilitychange` events, still probe unconditionally, since those are
  real signals worth refreshing the connectivity indicator on, not blind polling.
- World 6's cloud toggle is disabled while offline; the local model beside it keeps working.

Verified for real: `pnpm build && pnpm start`, open a World 1 chapter through the map so its model
caches, DevTools → Network → Offline, reload — the chapter itself reloads (not a fallback to
`/map`), progress and identity survive, further play queues, and the queue drains once back online.
Full account in `plan-docs/REMAINING-WORK.md`, A4.

---

## Admin dashboard

`/admin/dashboard` (cookie-session auth, see the environment variables above) has two tabs:

- **Users** — the original paginated user list and per-user activity drilldown.
- **Chapter analytics** — one row per chapter (`src/lib/adminAnalytics.ts`'s `shapeChapterAnalytics`,
  served by `src/app/api/admin/analytics/route.ts`): distinct users who started/completed it, the
  resulting completion rate, level pass/fail counts, and how often the new navigation events fired
  (`chapter_jumped_ahead`, `chapter_shared_link_opened`) — everything traced back to a specific
  chapter and world.
- Backed by a single `activity` aggregation grouped by `{chapterId, type}` (both a raw count and a
  distinct-`userId` count per group), joined onto `orderedChapters()` so every chapter shows up even
  with zero activity — not just the ones with rows. A `{chapterId: 1, type: 1}` index
  (`src/lib/mongodb.ts`) backs this.

---

## Privacy

Onboarding shows a one-line, non-blocking disclosure before anything is collected. What is stored:
display name, a client-generated id, IP address, approximate location (only when the operator
configures their own lookup), user agent, referrer, language, timezone and screen size. There is no
canvas or font fingerprinting. `/api/activity` is write-only and never returns anyone's data.
