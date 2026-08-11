# PROJECT BUILD SPEC — "AI Learning Lab"

### (Hand this entire document to Claude Code in an empty project folder)

You are building **AI Learning Lab** — a gamified, browser-based AI education platform for engineers (not children). Users learn AI/ML concepts — vectors, embeddings, neural networks, attention, transformers — by playing short, focused games that run **real AI models directly in the browser**, unlocking chapters sequentially as they progress. This is explicitly NOT a chatbot. There is no "ask the AI to explain it" flow anywhere in the core learning path. Understanding is built by direct manipulation and visualization of real model internals — not by reading model output.

Read this entire document before writing any code. Follow the **Build Order** section exactly — do not jump ahead to UI work before earlier phases are complete and tested.

---

## 1. NON-NEGOTIABLE PROJECT RULES

1. **100% frontend gameplay.** All game logic, all model inference (browser-tier), all game state, all scoring computation runs client-side. Next.js API routes exist ONLY for: (a) writing user activity/progress to MongoDB, (b) admin authentication + admin data reads, (c) proxying the one Ollama Cloud model call for the one chapter that needs it. No game logic ever lives server-side.
2. **JSON-driven game logic.** Every single game (all sub-chapters) has its levels, scoring rules, win conditions, prompts, and content defined in a standalone JSON file under `/data/games/`. No game's rules should be hardcoded inside a React component. Components are renderers of JSON, not owners of logic.
3. **Test-first.** For every game JSON file, write a corresponding test file that validates the game's logic engine (scoring, level progression, win/lose conditions, edge cases) BEFORE any UI component for that game is built. Do not write a single UI component until the game logic tests for that chapter are green.
4. **Mobile-first, distinctive UI.** Full design constraints are in Section 6. Read them carefully — this must NOT look like a generic Claude-Code-generated app, and must NOT look like a kids' app.
5. **Offline-capable.** The entire learning experience (browser-tier chapters) must work with zero internet connection after first load/model-cache. Activity is queued locally and synced silently when connectivity returns.
6. **No invented data.** Every score, every "real model output" shown to the user must come from an actual computation/inference — never fake/hardcoded to look real.

---

## 2. TECH STACK

- **Framework:** Next.js 15 (App Router), TypeScript strict mode
- **Styling:** Tailwind CSS + CSS variables for theming (see Section 6)
- **Animation:** Framer Motion (motion/react) for all transitions, game feedback, level-unlock sequences
- **State management:** Zustand (client game state, per-session), persisted to IndexedDB for offline resume
- **Browser ML inference:** `@huggingface/transformers` (transformers.js) for embeddings, tokenization, small causal LMs, attention extraction
- **Larger local model (capstone only):** WebLLM (`@mlc-ai/web-llm`)
- **Cloud model (selective heavy chapters only):** Ollama Cloud, called via a Next.js API route proxy (never expose the API key to the client)
- **Local offline queue:** IndexedDB via `idb` library
- **Database:** MongoDB (Atlas), accessed via Next.js API routes only, using the official `mongodb` driver
- **Admin auth:** simple credentials-based auth (email/password from env vars), signed session cookie — no third-party auth provider needed
- **Testing:** Vitest for game logic unit tests, Playwright (optional, later phase) for UI smoke tests
- **Package manager:** pnpm

---

## 3. PROJECT STRUCTURE

```
/ai-learning-lab
├── /data
│   └── /games
│       ├── /world-1-fundamentals
│       │   ├── 1-1-vectors.json
│       │   ├── 1-2-vector-arithmetic.json
│       │   ├── 1-3-similarity-distance.json
│       │   ├── 1-4-tokenization.json
│       │   └── 1-5-probability.json
│       ├── /world-2-classical-ml
│       │   ├── 2-1-perceptron.json
│       │   ├── 2-2-loss-functions.json
│       │   ├── 2-3-gradient-descent.json
│       │   └── 2-4-overfitting.json
│       ├── /world-3-neural-networks
│       │   ├── 3-1-neurons-activations.json
│       │   ├── 3-2-layers-forward-pass.json
│       │   ├── 3-3-backpropagation.json
│       │   └── 3-4-training-dynamics.json
│       ├── /world-4-sequence-models
│       │   ├── 4-1-ngrams.json
│       │   ├── 4-2-recurrence-memory.json
│       │   └── 4-3-sampling-strategies.json
│       ├── /world-5-transformers
│       │   ├── 5-1-positional-encoding.json
│       │   ├── 5-2-self-attention.json
│       │   ├── 5-3-multi-head-attention.json
│       │   ├── 5-4-residuals-layernorm.json
│       │   └── 5-5-full-transformer.json
│       ├── /world-6-capstone
│       │   └── 6-1-inspector-chat.json
│       └── curriculum-manifest.json      // ordered list of all worlds/chapters, unlock rules, XP values
│
├── /src
│   ├── /app
│   │   ├── /(game)
│   │   │   ├── /world/[worldId]/chapter/[chapterId]/page.tsx
│   │   │   ├── /onboarding/page.tsx
│   │   │   └── /map/page.tsx                 // world/chapter selection map
│   │   ├── /admin
│   │   │   ├── /login/page.tsx
│   │   │   └── /dashboard/page.tsx
│   │   └── /api
│   │       ├── /activity/route.ts            // POST — write-only activity ingestion
│   │       ├── /users/route.ts                // POST — create user on onboarding
│   │       ├── /admin/auth/route.ts
│   │       ├── /admin/users/route.ts          // GET — admin-only, protected
│   │       ├── /admin/activity/route.ts       // GET — admin-only, protected
│   │       └── /model/cloud-inference/route.ts // POST — Ollama Cloud proxy, selective chapters only
│   │
│   ├── /engines                              // pure game-logic engines, one per game TYPE (not per chapter)
│   │   ├── clusterPlacementEngine.ts
│   │   ├── vectorArithmeticEngine.ts
│   │   ├── similarityRankEngine.ts
│   │   ├── tokenMergeEngine.ts
│   │   ├── probabilityWheelEngine.ts
│   │   ├── linearClassifierEngine.ts
│   │   ├── lossMinimizationEngine.ts
│   │   ├── gradientDescentEngine.ts
│   │   ├── overfitFitEngine.ts
│   │   ├── neuronTuningEngine.ts
│   │   ├── networkBoundaryEngine.ts
│   │   ├── backpropVisualEngine.ts
│   │   ├── trainingDashboardEngine.ts
│   │   ├── ngramPredictionEngine.ts
│   │   ├── memoryDecayEngine.ts
│   │   ├── samplingEngine.ts
│   │   ├── positionalEncodingEngine.ts
│   │   ├── attentionGuessEngine.ts
│   │   ├── multiHeadDetectiveEngine.ts
│   │   ├── residualToggleEngine.ts
│   │   ├── transformerAssemblyEngine.ts
│   │   └── scoringEngine.ts                  // shared XP/star-rating logic used by all games
│   │
│   ├── /models                               // browser-model wrappers — ONE place that owns model lifecycle
│   │   ├── modelRegistry.ts                  // maps chapter → model id, size, tier (browser-light/browser-heavy/cloud)
│   │   ├── embeddingModel.ts                 // transformers.js wrapper: all-MiniLM-L6-v2
│   │   ├── tokenizerModel.ts                 // transformers.js AutoTokenizer wrapper (GPT-2 BPE)
│   │   ├── tinyCausalLM.ts                   // transformers.js wrapper: SmolLM-135M or similar
│   │   ├── attentionModel.ts                 // transformers.js wrapper exposing output_attentions + hidden_states
│   │   ├── tinyRNNTrainer.ts                 // small char-RNN trained live in-browser (World 4.2)
│   │   ├── tinyNetTrainer.ts                 // small feedforward net trained live in-browser (World 3)
│   │   ├── webllmCapstone.ts                 // WebLLM wrapper for World 6
│   │   ├── ollamaCloudClient.ts              // client-side caller that hits /api/model/cloud-inference
│   │   └── modelCache.ts                     // IndexedDB-backed model weight caching + load-progress events
│   │
│   ├── /components
│   │   ├── /ui                               // design-system primitives (Section 6) — buttons, cards, meters, etc.
│   │   ├── /games                            // one folder per chapter, renders its engine's state
│   │   │   ├── /1-1-vectors/VectorCanvas.tsx
│   │   │   └── ... (mirrors /data/games structure)
│   │   ├── /map                              // world map / chapter node UI
│   │   ├── /onboarding                       // name entry, id generation UI
│   │   └── /admin                            // admin dashboard components
│   │
│   ├── /lib
│   │   ├── mongodb.ts                        // connection singleton
│   │   ├── adminAuth.ts                      // session cookie verify/sign
│   │   ├── userIdentity.ts                   // client id generation (uuid), device fingerprint helpers
│   │   ├── offlineQueue.ts                   // IndexedDB queue: add/read/clear activity events
│   │   ├── syncManager.ts                    // background sync loop, online/offline event listeners
│   │   └── geoClient.ts                      // client-side best-effort geo/IP enrichment call
│   │
│   ├── /store
│   │   ├── useGameProgressStore.ts           // Zustand: unlocked chapters, XP, scores, persisted to IndexedDB
│   │   └── useSessionStore.ts                // current run state per active game
│   │
│   └── /types
│       ├── game.ts                           // shared GameDefinition, Level, ScoreResult types
│       ├── user.ts
│       └── activity.ts
│
├── /tests
│   └── /engines                              // one test file per engine, mirrors /src/engines structure
│       ├── clusterPlacementEngine.test.ts
│       ├── ... (one per engine, ALL must exist and pass before UI phase starts)
│
├── /public
├── .env.local.example
├── README.md
├── package.json
└── vitest.config.ts
```

---

## 4. GAME JSON SCHEMA

Every file in `/data/games/**/*.json` MUST conform to this shape. Define this as a Zod schema in `/src/types/game.ts` and validate every JSON file against it at build time (a `scripts/validate-games.ts` script, run in CI / pre-dev).

```ts
interface GameDefinition {
  id: string; // "1-1-vectors"
  world: number; // 1
  worldTitle: string; // "Fundamentals of AI"
  chapterTitle: string; // "What is a Vector?"
  order: number; // sequence within world
  concept: {
    shortExplanation: string; // 3-5 sentences, shown before play
    ahaMoment: string; // shown on chapter completion
  };
  modelRequirement: {
    tier: "none" | "browser-light" | "browser-heavy" | "cloud";
    modelId: string | null; // e.g. "Xenova/all-MiniLM-L6-v2", or ollama model id, or null
    estimatedSizeMB: number | null;
    fallbackMessage: string | null; // shown if tier === "cloud" and offline
  };
  xpReward: number;
  unlockRequires: string[]; // chapter ids that must be completed first
  levels: GameLevel[];
}

interface GameLevel {
  id: string;
  title: string;
  difficulty: "easy" | "medium" | "hard";
  instructions: string;
  engineType: string; // maps to one engine in /src/engines
  engineConfig: Record<string, unknown>; // level-specific parameters the engine consumes
  passCriteria: {
    metric: string; // e.g. "clusterSeparationScore", "cosineSimilarityError"
    threshold: number;
    comparator: "gte" | "lte" | "eq";
  };
  starsRules: { threshold: number; stars: number }[]; // e.g. score bands → 1-3 stars
}
```

Each **engine** (`/src/engines/*.ts`) exports a pure function set: `initState(config)`, `applyAction(state, action)`, `evaluate(state) → ScoreResult`. Engines must have ZERO React/DOM dependencies — this is what makes them independently testable. Model-backed engines call into `/src/models/*` wrappers, which must be mockable in tests (inject via parameter, not import side-effect, so tests can supply a fake model).

---

## 5. FULL CURRICULUM WITH MODEL ASSIGNMENT (build this exactly)

Legend: 🟢 none (pure math/JS) · 🔵 browser-light (transformers.js small model, <100MB) · 🟣 browser-heavy (transformers.js larger model or live-trained net, WebLLM) · ☁️ cloud (Ollama Cloud, selective only)

### World 1 — Fundamentals of AI

- **1.1 What is a Vector?** 🔵 — user types ANY word, live-embed with `Xenova/all-MiniLM-L6-v2`, PCA-project to 2D client-side (use a lightweight JS PCA, e.g. `ml-pca`), plot in real time. Levels: place-and-cluster, guess-the-label, drag-vector-magnitude.
- **1.2 Vector Arithmetic** 🔵 — same embedding model, user picks own 3 words, real vector math, reveal nearest real neighbor from a precomputed candidate pool (pool can be static, the math must be live).
- **1.3 Similarity & Distance** 🔵 — same embedding model, live cosine similarity heatmap, "odd one out" with user-chosen word sets allowed at hardest level.
- **1.4 Tokenization** 🔵 — real BPE tokenizer (`Xenova/gpt2` tokenizer via transformers.js `AutoTokenizer`), merge-puzzle against real merge order, free-text "break the tokenizer" mode.
- **1.5 Probability Basics** 🔵 — pull REAL next-token probability distribution from a tiny causal LM (`Xenova/SmolLM2-135M-Instruct` or similar) given a user seed word; wheel segments = real top-k probabilities, not synthetic ones.

### World 2 — Classical Machine Learning Foundations

- **2.1 Perceptron** 🟣 — a real perceptron, trained live in-browser (weights update on-screen per step; this is a genuine trained model, just tiny and untethered from any pretrained weights).
- **2.2 Loss Functions** 🟢 — pure math visualization; explicitly fine to stay model-free (see Section on model-use judgment below).
- **2.3 Gradient Descent** 🟢 (base levels) / 🟣 (bonus level: show the REAL loss surface of the live-trained net from 3.2, computed on demand — optional stretch, mark as `optional: true` in JSON if time-constrained).
- **2.4 Overfitting & Regularization** 🟢 — pure math polynomial fit; model-free is correct here.

### World 3 — Neural Networks

- **3.1 Neurons & Activations** 🟣 — a real single neuron, user-tunable weights, live math — count as browser-heavy since it's a real (if tiny) trained unit.
- **3.2 Layers & Forward Pass** 🟣 — real small feedforward net, trained live in-browser on 2D toy datasets (circles/XOR/spiral), TensorFlow-Playground-style, using plain JS/typed arrays (no need for a heavy ML framework — hand-roll it for full control over the visualization hooks).
- **3.3 Backpropagation** 🟣 — same live net, exposing real per-edge gradients during backward pass for the pulse animation.
- **3.4 Training Dynamics** 🟣 — same live net, exposing real loss-per-epoch and batch-size effects.

### World 4 — Sequence Models & Language

- **4.1 N-grams** 🔵 — build REAL n-gram frequency tables client-side at runtime from a small bundled public-domain corpus (don't precompute and ship static tables — compute in-browser so it's a genuine live model the user could rebuild with a different corpus later).
- **4.2 Why Recurrence Isn't Enough** 🟣 — **this must be a REAL tiny char-level RNN trained live in-browser**, not a scripted/faked decay animation (this was flagged as the weak point earlier — fix it here). Show its actual hidden-state vector's real values degrading over a real long input.
- **4.3 Sampling Strategies** 🔵 — same tiny causal LM as 1.5, real live temperature/top-k/top-p reshaping of the real output distribution.

### World 5 — The Transformer

- **5.1 Positional Encoding** 🟢/🔵 hybrid — sinusoidal formulas are pure math (🟢) but combine with REAL token embeddings from the 1.1 embedding model so it's presented as one ingredient in a real pipeline, not an isolated toy.
- **5.2 Self-Attention** 🟣 — real attention weights extracted from a small transformer (`Xenova/distilbert-base-uncased` or similar) via `output_attentions: true`. This is the centerpiece chapter — invest the most build time here.
- **5.3 Multi-Head Attention** 🟣 — same model, all heads extracted and rendered side-by-side.
- **5.4 Residuals & Layer Norm** 🟣 — same model, real intermediate hidden-state activations extracted per layer to show stabilized vs destabilized value traces.
- **5.5 Full Transformer Assembly** 🟣 — real end-to-end forward pass through a small causal LM (`Xenova/gpt2` or `SmolLM2-135M`), stitching outputs from all of 5.1–5.4's visualizations into one animated sequence.

### World 6 — Capstone

- **6.1 Inspector Chat** 🟣 (default) with optional ☁️ **cloud escalation** — default experience uses WebLLM with a small instruct model (`Llama-3.2-1B-Instruct-q4f16` or similar) fully offline-capable. Add a toggle: "Ask a bigger model" which routes through Ollama Cloud (model id `gpt-oss-120b-cloud` or whatever is configured in env) via the `/api/model/cloud-inference` route. **If offline, disable this toggle and show the fallback message**: _"This is a much larger model that runs in the cloud — connect to the internet to use it. Your local model above works fully offline."_

**Model-use judgment principle** (apply this consistently, including to any chapter you add later): use a live model whenever the concept IS something a real model produces (embeddings, attention, next-token probabilities, hidden states, gradients). Keep something math-only ONLY when the concept is a mathematical construct that exists independent of any trained model (positional encoding formulas, generic loss-surface visualization, generic overfitting curves). When in doubt, prefer the live model — that's this project's entire differentiator.

---

## 6. UI / UX DESIGN PRINCIPLES — STRICT RULES

**The #1 failure mode to avoid:** this must NOT look like a typical Claude-Code-generated dashboard (generic rounded cards, purple-to-blue gradients, inter font, centered hero + 3-column feature grid). It also must NOT look like a children's edutainment app (no cartoon mascots, no candy-color palettes, no comic-sans-adjacent fonts, no confetti-everywhere). The target audience is engineers who want to feel like they're inside a technical instrument — think "flight simulator cockpit" or "debugger/profiler tool" aesthetics crossed with a game HUD, not "Duolingo for AI."

**Required design direction:**

- **Visual identity:** dark-mode-first, near-black backgrounds (`#0a0a0f` range) with a single confident accent color per World (not per element — each of the 6 Worlds gets its own accent hue, used consistently across its chapters, so users feel "which world am I in" spatially). Use monospace or technical-grade fonts for numbers/data/code-like content (e.g. JetBrains Mono, Berkeley Mono) paired with a clean geometric sans for UI chrome (e.g. Inter is fine for chrome text ONLY, never for headers/branding — pick something more distinctive for headers, e.g. Space Grotesk or a custom-feel display face).
- **Data-forward, not decoration-forward.** Every screen's dominant visual element should be the actual visualization (the vector plot, the attention heatmap, the loss curve) — not a card containing a small chart. Charts/visualizations should feel like the primary UI, with controls (sliders, buttons) as a secondary HUD layer, not the other way around.
- **Motion must be purposeful, not decorative.** Use Framer Motion for: state transitions in the visualizations themselves (points moving to new positions, attention weights animating in, gradient pulses traveling along edges), level-unlock sequences (satisfying but quick — under 800ms, no long celebratory animations that can't be skipped), and score reveals. Avoid gratuitous micro-animations on every hover — motion should always be tied to a real state change in the underlying data/model.
- **Typography hierarchy:** numbers and live metrics (scores, loss values, probabilities) should be visually the loudest thing on screen — large, monospaced, high-contrast — because this is a tool for reading data, not marketing copy.
- **Mobile-first, but not mobile-only-thinking:** design the control layer (sliders, drag targets) for touch first, but ensure canvases/visualizations remain legible and are not cramped — use collapsible side panels on mobile that expand into persistent side rails on desktop/tablet, not just a smaller version of desktop.
- **World map screen:** should look like a technical systems diagram / circuit board / dependency graph connecting the 6 worlds and their sub-chapters — locked chapters rendered dimmed/desaturated with a lock glyph, completed chapters showing their earned star rating inline — NOT a cutesy platformer-style winding path with cartoon islands.
- **No stock illustration, no emoji as primary iconography.** Use a consistent custom icon set (Lucide is acceptable as a base, but customize weight/style) — emoji are acceptable only as small inline accents in copy, never as functional UI icons.
- **Sound design (optional but recommended):** subtle, technical-feeling audio feedback (short synth blips, not game-show fanfares) on correct answers/level completion — must be toggleable and off by default on first load.

Read and apply `/mnt/skills/public/frontend-design/SKILL.md` conventions during implementation for concrete spacing/token/component discipline, but the _creative direction_ above overrides any generic defaults that skill would otherwise produce — do not let it push you back toward generic SaaS-dashboard styling.

---

## 7. PROGRESSION & SCORING SYSTEM

- **XP system:** every level completion grants XP (defined per-level in JSON `xpReward`, scaled down from the parent chapter's total). Total XP determines a visible "Rank" (engineer-flavored titles, not childish ones — e.g. "Gradient Novice → Backprop Adept → Attention Architect → Transformer Engineer," you can refine naming).
- **Star rating per chapter:** 1-3 stars based on `starsRules` thresholds in the level JSON (e.g. based on accuracy, speed, or fewest attempts — vary the metric per game type so it doesn't feel repetitive).
- **Unlock graph:** driven by `unlockRequires` in each chapter's JSON + the `curriculum-manifest.json` ordering — build a generic unlock-resolver function, not per-chapter conditionals.
- **Persistent local state:** Zustand store persisted to IndexedDB (not localStorage — you're already using IndexedDB for the offline activity queue, keep one local data layer). On reload, hydrate from IndexedDB immediately (no loading spinner for progress — only for model downloads).

---

## 8. USER IDENTITY & ONBOARDING

- On first launch, show a minimal onboarding screen: user enters a display name only.
- On submit: generate a client-side UUID as the persistent user id (stored in IndexedDB), then POST to `/api/users` with `{ userId, name }`.
- Server-side, on that POST, capture and store everything reasonably available from the request: IP address (from request headers, respecting proxy headers like `x-forwarded-for`), approximate geolocation derived server-side from IP (use a free/self-hosted IP-geolocation lookup — do NOT call a paid third-party API without the user configuring their own key; document this as an env-configurable optional enrichment step), user-agent string, timezone offset, screen size/viewport, referrer if present, and timestamp.
- **Important addition (flagged, see Section 11):** show a one-line, non-blocking disclosure on the onboarding screen — something like _"We track anonymous play activity to improve this tool."_ This isn't optional legal boilerplate you can skip — collecting IP/location without ANY disclosure is the one part of this spec most likely to cause real problems later (App Store rejection, GDPR issues if any EU users, basic user trust). Keep it to one line, non-blocking, no cookie-banner theatrics — just don't skip it entirely.

---

## 9. ACTIVITY TRACKING, OFFLINE QUEUE & SYNC

- **Every meaningful in-game event** (chapter started, level completed, level failed, XP earned, model download started/completed, chapter unlocked) gets written to a local IndexedDB `activityQueue` store immediately, client-side, with a client-generated event id + timestamp + userId.
- **Sync manager** (`/src/lib/syncManager.ts`): runs on an interval (e.g. every 30s) AND on `online` browser events. When online, batches all queued events, POSTs to `/api/activity` in a single batch call, and on a successful response, clears exactly those synced events from IndexedDB (match by event id — never blind-clear the whole queue, in case new events were added mid-sync).
- **Offline detection:** use `navigator.onLine` plus an actual network check (a lightweight `HEAD` request to your own API) since `navigator.onLine` alone is unreliable — treat both as inputs.
- **`/api/activity` route:** write-only. Validates payload shape, inserts into MongoDB `activity` collection (batch insert), returns only a success/failure + which event ids were persisted (so the client can safely clear only those). Never returns other users' data — this route is intentionally one-directional as you specified.
- **MongoDB collections:** `users` (profile + enrichment data from onboarding), `activity` (append-only event log, indexed on `userId` + `timestamp`), keep these separate rather than embedding activity arrays inside user docs (embedding will blow past MongoDB's document size limits once users play for a while).

---

## 10. ADMIN DASHBOARD

- `/admin/login` — simple email/password form, credentials checked server-side against `ADMIN_EMAIL` / `ADMIN_PASSWORD_HASH` env vars (hash the password at setup time — store only the hash in env, never plaintext; document a small one-off script to generate the hash in the README).
- On success, set a signed, httpOnly session cookie (short-lived, e.g. 12h, with refresh-on-activity).
- `/admin/dashboard` — protected by middleware checking the session cookie on every request; redirect to login if missing/invalid/expired.
- Dashboard shows: total users, users list (name, id, first-seen, last-seen, derived location/IP, device info, current rank/XP), and a per-user drill-down showing their full activity timeline (chapter/level attempts, scores, timestamps) — paginated, not one giant unbounded query.
- All `/api/admin/*` routes must independently re-verify the session cookie server-side (never trust that reaching the route means the UI already checked — defense in depth).

---

## 11. THINGS I'M ADDING THAT WEREN'T IN YOUR REQUEST (your call whether to keep them)

I'm flagging these explicitly rather than silently adding them, since you asked me to include anything important I noticed missing:

1. **Onboarding disclosure line** (Section 8) — the single biggest gap. Collecting IP/location/device fingerprinting with zero disclosure is the part most likely to bite you later. One non-blocking sentence is a very low cost to add now.
2. **Model-load error handling & retry UI.** transformers.js/WebLLM downloads can fail (network drop mid-download, WebGPU unavailable, out of storage quota). Every model-backed chapter needs a defined failure state in its JSON (`fallbackMessage` already covers the cloud-tier case — extend a similar pattern for browser-tier load failures) with a retry button, not a silent blank screen.
3. **Device capability detection.** Before attempting WebGPU-accelerated inference, detect support and fall back to WASM automatically; show the user which mode they're running in (small, unobtrusive indicator) since performance will visibly differ.
4. **Rate limiting on `/api/model/cloud-inference` and `/api/activity`.** Since these are public-facing routes with no per-user auth, add basic IP-based rate limiting (even a simple in-memory or MongoDB-backed sliding window) to prevent abuse of your Ollama Cloud spend and database writes.
5. **Accessibility baseline.** Keyboard navigability for all drag/slider interactions (provide arrow-key alternatives to dragging), and respect `prefers-reduced-motion` by shortening/removing non-essential animations — don't skip this just because the target audience is engineers; some of them will still be on keyboard-only workflows or have motion sensitivity.
6. **Version the game JSON schema.** Add a `schemaVersion` field to every game JSON now, even at `1`, so future curriculum edits don't silently break old saved progress once real users have played.

---

## 12. BUILD ORDER — FOLLOW EXACTLY, DO NOT REORDER

**Phase 0 — Project init**

- `pnpm create next-app` with TypeScript, Tailwind, App Router, in this folder.
- Install all dependencies from Section 2.
- Set up `vitest.config.ts`, `.env.local.example` listing every env var this spec references (MongoDB URI, ADMIN_EMAIL, ADMIN_PASSWORD_HASH, OLLAMA_CLOUD_API_KEY, OLLAMA_CLOUD_MODEL_ID, session secret, optional IP-geolocation key).
- Scaffold the full folder structure from Section 3 (empty files/folders are fine at this stage).

**Phase 1 — All game JSON files**

- Write every JSON file listed in Section 5, conforming exactly to the schema in Section 4.
- Write the Zod schema and the `scripts/validate-games.ts` validator; run it and confirm every JSON file passes before moving on.
- Write `curriculum-manifest.json`.

**Phase 2 — Engines + tests (test-first, per the project rules)**

- For each engine in `/src/engines`, first write its test file in `/tests/engines` covering: initial state correctness, valid action transitions, invalid/edge-case inputs, score/pass-criteria evaluation against each level's JSON config, and (for model-backed engines) behavior with a mocked model injected.
- Then implement the engine to make tests pass.
- Do not proceed to Phase 3 until ALL engine tests are green.

**Phase 3 — Model wrapper layer**

- Implement `/src/models/*`, starting with `modelRegistry.ts` and `modelCache.ts`, then the individual model wrappers.
- Write integration-style tests where feasible (these can be lighter-touch than engine unit tests, since real model downloads are involved — mock at the transformers.js pipeline boundary).

**Phase 4 — Core UI shell**

- Design system primitives (`/src/components/ui`) per Section 6.
- World map screen, onboarding screen, chapter shell/layout (concept text panel, level HUD, progress indicator).

**Phase 5 — Per-chapter game UI**

- Build chapter UIs world-by-world, in curriculum order (World 1 → World 6), wiring each to its already-tested engine + model wrapper.
- Do not build a chapter's UI before its engine tests are passing — this includes Phase 5 itself; if you discover a gap in an engine while building UI, go back and fix/extend the engine + its tests first.

**Phase 6 — Backend: users, activity, offline sync**

- `/api/users`, `/api/activity`, MongoDB connection layer, `offlineQueue.ts`, `syncManager.ts`, wire the sync manager into the app root so it runs globally regardless of which screen the user is on.

**Phase 7 — Admin**

- Auth routes + middleware, dashboard UI, admin API routes with independent auth verification.

**Phase 8 — Ollama Cloud integration (World 6 only)**

- `/api/model/cloud-inference` proxy route, `ollamaCloudClient.ts`, the World 6 UI toggle + offline fallback messaging.

**Phase 9 — Offline polish + PWA**

- Service worker / PWA manifest so the app is installable and the shell loads offline after first visit; confirm the full browser-tier learning path (Worlds 1-5) works with devtools network set to offline.

**Phase 10 — README**

- Write the README per Section 13. Do this last, once you know what actually got built, not as a copy of this spec.

---

## 13. README REQUIREMENTS

The README must include: project overview (2-3 sentences), tech stack list, full env var table with descriptions of where to obtain each value (MongoDB URI, Ollama Cloud key/model id, how to generate the admin password hash), local dev setup steps, how to run `scripts/validate-games.ts`, how to run tests, folder structure explanation (can reference Section 3 of this doc), and a short "How to add a new chapter" guide (create JSON → write engine test → write engine → write UI component → register in curriculum-manifest.json) since this project structure is specifically designed to make that repeatable.

---

## 14. FINAL REMINDERS

- Every "real model output" shown to the user must be computed live, not hardcoded — this is the entire value proposition of the product. If you catch yourself about to fake a visualization for convenience (as flagged in World 4.2 above), stop and implement the real thing instead, even if it takes longer.
- Keep engines pure and model-injection-based specifically so the test suite never needs a real model download to pass — CI/local test runs should be fast and fully offline.
- If at any point a chapter's spec here is ambiguous, prefer the interpretation that keeps more computation live/real and client-side, and ask before defaulting to a server-side or faked shortcut.
