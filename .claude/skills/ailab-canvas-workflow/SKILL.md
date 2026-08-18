---
name: ailab-canvas-workflow
description: Tooling and debugging workflow for building/verifying AI Learning Lab game canvases in this repo — how to get a real browser running here to test a canvas end-to-end, how to introspect a real HF model from Node without the app's browser-only runtime getting in the way, how to find an alternate ONNX export when a model doesn't expose what a chapter needs (including ones missing config.json, splitting weights into external data, or failing session creation only in onnxruntime-web's WASM backend), a general engine-correctness check for any canvas with a live-adjustable parameter, a ModelGate trap specific to canvases whose levels don't all need the same model (including a second one for two models loaded at once), why Node calibration must use dtype q8 not fp32 and still isn't a perfect proxy for a real browser near a model's decision boundary (recall/instruction-following tasks diverge far more than mechanical ones like attention or timing), a long-single-prompt heavy-workload risk distinct from many-short-generations, why a chapter's durable progress survives a localStorage clear, why a `locator.click()` can hang even before any heavy decode starts and what to click with instead, why a raw coordinate click can silently miss inside a scrollable panel, and what to do when a chapter needs WebGPU and this environment can't reliably provide it. Load this before touching any World 5/6/7/8 canvas, before touching src/models/*.ts, or whenever a model load hangs/fails only in the browser. Complements plan-docs/REMAINING-WORK.md and plan-docs/EXPANSION-PLAN.md (the what/why per chapter) — this is the how, so the same false starts aren't repeated.
---

# AI Learning Lab canvas workflow

Process notes from building canvases 14 through 20 (all of Worlds 1–6, 22 chapters), World 7's four
chapters (7-1 through 7-4), and World 8's first four (8-1 through 8-4) — specifically the parts that
cost real time and aren't in `plan-docs/REMAINING-WORK.md` (which covers per-chapter findings for
Worlds 1–6) or `plan-docs/EXPANSION-PLAN.md` (the same, for World 7/8). Still worth reading before
*revisiting* any canvas: the same model-loading and verification traps apply to fixes as much as to
first builds.

## 1. Verifying a canvas in a real browser (Playwright)

No browser is preinstalled in this environment.

- `npx playwright install chromium` on its own **prints a warning and does not download
  anything** — it's a no-op here. Use `pnpm dlx playwright install chromium --force` instead
  (the `--force` matters; without it the first run may also silently skip the download).
- Playwright itself isn't a project dependency. Add it temporarily: `pnpm add -D playwright`,
  do the verification, then `pnpm remove playwright` and confirm
  `git status --short -- package.json pnpm-lock.yaml` is empty before committing anything.
- Run verification scripts with `npx tsx <script>.ts` **from the project root**, not the
  scratchpad directory — bare-specifier imports (`from '@huggingface/transformers'`, relative
  imports of `src/engines/...`) resolve against the executing file's own location, and the
  scratchpad has no `node_modules`. Write throwaway scripts as dotfiles at the repo root
  (`.scratch-foo.ts`) and delete them before committing; `git status` before staging anything
  to make sure none are left untracked-but-forgotten.
- Navigation pattern for this app, useful for any chapter:
  - URL is `/world/{worldNumber}/chapter/{chapterId}` (e.g. `/world/5/chapter/5-2-self-attention`).
  - **Every level opens on its own concept/instructions screen, not just the chapter's first
    one.** `ChapterShell` renders this screen whenever its session `phase` is `'concept'`, and
    advancing past a passed level (`goToLevel`, via the HUD's "Next level" button) resets `phase`
    back to `'concept'` — so a 3-level chapter needs **three separate `Begin` clicks**, one per
    level, not one for the whole chapter. While `phase === 'concept'` the game component is a
    ternary branch that isn't rendered at all (not a hidden overlay), so nothing in the actual
    canvas — its inputs, its `ModelGate` — exists in the DOM yet. Clicking "Next level" and
    immediately looking for the next level's board without an intervening `Begin` click hangs on
    whatever selector comes next, for a reason that looks like a slow model load but isn't one.
  - **Don't wait on a text string that might also appear in the concept prose — including a
    level's own title, across a level transition.** `waitForSelector('text=temperature')` matched
    the concept paragraph (which mentions "temperature" in prose) and resolved immediately, before
    the canvas or its model had even started loading. The same trap fires when advancing levels:
    the concept screen shown before level 2 renders that level's own title too (as `"level
    7-1-l2 · Break the Retriever"`), so `waitForSelector('text=Break the Retriever')` — intended to
    confirm level 2's *board* had mounted — instead resolves on the *concept* screen for level 2,
    one `Begin` click too early, and every following step then looks for elements that don't exist
    yet. In both cases the run times out on the *next* step for a confusing reason. Wait on a
    structural selector that only exists once the canvas is mounted instead, e.g.
    `input[type="range"]` or a label/`aria-label` unique to the rendered board.
  - Model downloads can take anywhere from a few seconds to ~2–3 minutes depending on size;
    give `waitForSelector` a generous timeout (120–180s) rather than polling.
  - Useful stable selectors once mounted: the HUD's metric name in caps (`text=ENTROPYERROR`,
    `text=REFERENCEFLIPSCORE`, etc. — matches `passCriteria.metric` uppercased), `getByRole('button',
    { name: /next level/i })` to advance after a pass, and `[role="img"][aria-label*="of 3 stars"]`
    for the star rating.
  - If a button can only be found by icon/position and not by accessible role+name in a
    Playwright query, that's usually a real accessibility gap in the canvas (missing text or
    `aria-label`), not a test problem — the same thing a screen reader would hit. Fix the
    component, not the selector.
  - **`page.goto(url, { waitUntil: 'domcontentloaded' })` followed immediately by `.click('Begin')`
    can silently swallow the click** on this app's first load (found building World 7's canvases) —
    the click fires before the client bundle has hydrated, the button is present in the DOM but not
    yet wired to React, and the run then sits on the concept screen forever with no error, timing out
    on whatever selector comes next for a confusing reason (looks like a slow model load, is actually
    a click that never registered). Fix: `waitUntil: 'networkidle'` on the `goto`, and
    `await page.getByRole('button', { name: 'Begin' }).waitFor({ state: 'visible' })` before calling
    `.click()` — don't just click as soon as `goto` resolves.
  - **A batch of many real generations behind one click can block the browser's main thread long
    enough that `locator.click()` itself times out** (found calibrating World 7.3's schema-reliability
    level, a 6-generation batch at `maxTokens: 40`) — the WASM CPU backend's forward passes run
    synchronously enough to starve the CDP round-trip `locator.click()` waits on for 80+ seconds,
    reproduced identically in a Node script (same `dtype: 'q8'`, see §10) timed at ~80s for an
    equivalent batch. This is a real UX problem too, not just a test artifact — a real player would
    see the same frozen-looking UI with no visible progress. Two fixes, both worth applying together:
    - In the **app itself**: `await` two `requestAnimationFrame`s right after setting a `busy` state,
      before starting the heavy decode loop, so "running the real model…" actually paints before the
      synchronous work begins:
      ```ts
      function yieldToPaint(): Promise<void> {
        return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
      }
      // setBusy(true); await yieldToPaint(); /* then the heavy loop */
      ```
    - In **verification scripts**: dispatch the click via `page.evaluate(() => el.click())` instead of
      `locator.click()` — a direct DOM `.click()` call only needs the JS execution context to accept
      one `Runtime.evaluate` call, not a full mousedown/hit-test/mouseup CDP round-trip, so it doesn't
      block on the same starved acknowledgement. Then wait on a real structural signal that the async
      work actually finished (e.g. the busy button's text reverting), not a fixed delay.
  - **A single long prompt is a *different* heavy-workload risk than many short generations, and a
    Node q8 timing probe badly underestimates it.** Found building World 8.2's needle-in-haystack
    level: a Node q8 script decoding against 6 haystack lengths up to 1,300 real filler words
    estimated a manageable ~40–70s total; the real browser instead took **364 seconds** just to reach
    a playable board — not a hang (`ps` showed the Chromium renderer process genuinely pegged at
    100%+ CPU the whole time), but a real cost this environment's WASM execution provider pays for
    long-context forward passes that Node's native onnxruntime does not reproduce at anywhere near the
    same ratio. §5's q8-not-fp32 lesson generalises: for anything that grows the *prompt itself* (not
    just the generation count), treat a Node timing estimate as a rough floor, not a real browser
    estimate, and prefer cutting the workload (fewer/shorter real lengths) over just raising a
    verification timeout — a real player faces the same slow load, not just the test script.
  - **A persistent Playwright context avoids re-downloading the model on every verification run.**
    `chromium.launch()` + `browser.newPage()` creates a fresh, ephemeral profile every time, so the
    Cache API entry from `useBrowserCache = true` never survives between script runs — every run
    redownloads the full model. Use `chromium.launchPersistentContext(profileDir, {})` with a fixed
    directory under the scratchpad instead; the model then downloads once and every subsequent run
    reuses it, both far faster to iterate on and closer to what a real returning player experiences.
    Chapter progress does persist too, but **not in `localStorage`** — correcting this in place, found
    repeatedly while verifying World 8.2: a chapter's durable per-level completion record lives in
    IndexedDB (Zustand's persisted store, per the README's own "Zustand persisted to IndexedDB for
    progress" line), so `page.evaluate(() => localStorage.clear())` does **not** give a clean start.
    A fresh `page.goto` after playing partway through a chapter resumes directly at the next
    **incomplete** level's concept screen — skipping level 1 entirely if it was already passed in an
    earlier run against the same profile — or, if every level in the chapter was already passed, at
    whatever level's board was last open, with no concept screen at all. Never assume "reload → level
    1"; check at runtime instead (`await page.getByRole('button', {name: 'Begin'}).isVisible()`) and
    only click `Begin` when the concept screen is actually the thing on screen. If a genuinely clean
    per-chapter restart is needed, clearing IndexedDB is the real fix, confirmed working while
    verifying 8.3/8.4: the whole app shares one database, `ai-learning-lab` (see `DB_NAME` in
    `src/models/modelCache.ts`), so a plain `indexedDB.deleteDatabase('ai-learning-lab')` inside
    `page.evaluate` (wrapped in a `Promise` resolving on `onsuccess`/`onerror`/`onblocked`) resets
    progress cleanly without needing to enumerate every database. This only clears progress/activity
    bookkeeping, not the cached model weights (those live in the separate Cache API via
    `useBrowserCache`), so the next run still loads instantly from cache — safe to call before every
    verification run, not just when debugging a stuck resume.
  - **A `locator.click()` can hang for the full default timeout on the very first click of a run —
    the concept screen's own "Begin" button — with every actionability check (visible, enabled,
    stable) already reported as passing**, a new variant of the "batch of generations starves the CDP
    round-trip" case above but firing *before* any heavy decode has even started (found verifying World
    8.3/8.4). Reproduced consistently across fresh headless launches in this environment. The same fix
    applies even though the trigger is different: dispatch every button click via
    `page.evaluate(() => el.click())` (a raw DOM click needs only one `Runtime.evaluate` call) rather
    than `locator.click()`, for every click in a verification script, not just the ones known to follow
    heavy decode work — cheap insurance once it's clearly not just a live-model-load artifact.
  - **A raw `page.mouse.click(x, y)` at a `boundingBox()`-derived coordinate can silently miss when the
    target has scrolled out of the visible container**, landing on whatever *is* at that pixel (or
    nothing) with no error at all — just a quietly-wrong result (a guess that never registers, an
    unrelated element that does). `locator.click()` auto-scrolls into view first; a manual coordinate
    click does not. Call `.scrollIntoViewIfNeeded()` on the locator immediately before reading its
    `boundingBox()` whenever clicking by coordinate (e.g. as a `page.evaluate`-based workaround to the
    bullet above) inside any scrollable results panel.
  - **Don't run a heavy Node model-calibration script and a live Playwright verification against the
    dev server at the same time.** Found while verifying World 7.4: running several
    `AutoModelForCausalLM.from_pretrained` Node scripts concurrently with `next dev` starved the dev
    server badly enough that its own page compiles hung indefinitely (`○ Compiling / ...` with no
    further output, `curl` timing out against a port that was still genuinely `LISTEN`ing). Not an app
    bug — a host-machine resource-contention issue. If the dev server seems wedged mid-verification,
    check for exactly this (`ps` for other heavy Node processes) before assuming the app broke; killing
    and restarting the dev server clean resolves it.
  - **A level's own HUD readout can collide with a canvas's identically-labelled text**, the same
    ambiguous-text trap as the concept-screen one above but recurring in a new place: `ChapterShell`
    renders a persistent `Readout` labelled with the level's own `passCriteria.metric` (e.g.
    `JSONVALIDRATE`) regardless of whether a real result exists yet. Waiting on that label's text
    resolves instantly, before any real generation finishes — it isn't proof anything ran. Likewise, a
    persistent "submit blocked" hint like `"get a real generation to validate"` contains the substring
    `"valid"` and can spuriously match a `text=/valid|invalid/` selector meant for a real result Tag.
    Scope waits to a structural signal instead (a specific Tag inside the round's own `<section>`, or
    the busy button leaving its busy state) — never a bare label/metric-name string that might already
    be on the page for an unrelated reason.

## 2. Introspecting a real HF model from Node (not the browser)

The app's own model wrappers (`src/models/*.ts`) all route through `transformersRuntime.ts`,
which sets `mod.env.useBrowserCache = true`. That throws under Node
(`Browser cache is not available in this environment`) — **don't try to import and run
`src/models/*.ts` directly in a Node script.** It will fail for an environment reason that has
nothing to do with whether the model itself works.

Instead, write a throwaway script that imports `@huggingface/transformers` directly (bypassing
the app's runtime wrapper entirely) and run it with `npx tsx` from the project root, per §1.
This is the fast way to check what output keys/dims a candidate model actually returns before
wiring it into the app — seconds per candidate instead of a full dev-server + browser round trip.

```ts
import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
const tokenizer = await AutoTokenizer.from_pretrained('org/repo');
const model = await AutoModel.from_pretrained('org/repo', { subfolder: '' } as any); // if weights aren't under onnx/
const outputs = await model(await tokenizer('the cat sat on the mat'));
console.log(Object.keys(outputs)); // and check outputs[key].dims per key
```

This needs the candidate repo to have a `config.json` (`AutoModel.from_pretrained` fetches it to
pick the model class) — if it doesn't, see §6 before spending time here. When a repo has **no**
config.json, an even faster first check that needs neither config nor tokenizer: download the
`.onnx` file directly (`curl`) and inspect it with `onnxruntime-node` (already a transitive dep
of `@huggingface/transformers`, no install needed):

```ts
import * as ort from 'onnxruntime-node';
const session = await ort.InferenceSession.create('/path/to/model.onnx');
console.log(session.inputNames, session.outputNames); // straight from the graph, nothing else required
```

This is how a plausible-sounding but wrong candidate (`xboluna/all-MiniLM-L6-v2_with_hidden_layer`
— only `last_hidden_state`/`pooler_output` despite the name) got ruled out in seconds instead of
after a full config-borrowing attempt (§6). If the graph's weights are split into an external-data
file, this fails with a `filesystem error` unless the two files sit in the same directory under
their **original** names (`model.onnx` + `model.onnx.data`) — rename downloaded files back to that
if you fetched them under a different local filename.

**Token ids can come back as `bigint` in a Node script, silently breaking array indexing.** Found
building World 8.1's teacher-forced surprisal probe: `tokenizer(text).input_ids`'s `.tolist()` can
return `bigint` entries (not plain `number`) for an int64 tensor under Node's onnxruntime backend.
Using such a value directly as an array index (`fullProbs[tokenId]`) doesn't throw — it just silently
returns `undefined`, which then silently becomes `0` or `NaN` a few steps later, producing plausible
but wrong numbers with no error to flag the cause. It also breaks `JSON.stringify` outright
(`TypeError: Do not know how to serialize a BigInt`), which is a more helpful failure since at least
it's loud. Coerce with `Number(id)` (or `typeof id === 'bigint' ? Number(id) : id`) on every token id
read out of a tensor before using it as an index, comparing it, or logging it — cheap insurance, and
the app's own browser-side wrappers (`toNestedArray` et al.) have not shown this specific issue, but
it costs nothing to guard for it in new Node-only calibration code either.

## 3. Finding an alternate ONNX export when a model doesn't expose what's needed

This is the exact situation canvas 17 was blocked on (`hidden_states` unavailable — see
REMAINING-WORK.md A1); the same technique, extended from `-with-attentions` to `hidden` more
generally, is what found its fix too. The technique that found canvas 15's fix:

- `https://huggingface.co/api/models/{owner}/{repo}` → `.siblings` lists every file in the repo
  without downloading anything. Look for multiple `.onnx` variants and where they live
  (`onnx/model.onnx` vs a root-level `model.onnx` — transformers.js defaults to the `onnx/`
  subfolder, so a root-level file needs an explicit `subfolder: ''` at load time).
- `.../resolve/main/config.json` → the `architectures` field is the real tell, often more
  reliable than the repo name. A `*ForMaskedLM`/`*ForSequenceClassification`/task-head
  architecture almost never carries `attentions`/`hidden_states` as extra ONNX outputs, because
  the conversion job only exports what that task's default pipeline needs. A plain base class
  (`BertModel`, no head) *can* carry them, but only if that specific conversion explicitly asked
  for `output_attentions`/`output_hidden_states` — check by loading it and reading real output
  keys (§2), never assume from the class name alone.
- `.../api/models/{owner}/{repo}/refs` → some repos ship the extra outputs on a **separate
  branch** rather than `main` (e.g. an `output_attentions` branch), reachable via
  `AutoModel.from_pretrained(id, { revision: 'branch-name' })`.
- The naming convention `-with-attentions` (and presumably `-with-hidden-states`) under orgs
  like `onnx-community` and individual contributors is used specifically for ONNX exports
  converted with those flags baked in at conversion time. Search
  `https://huggingface.co/api/models?search=with-attentions`.
- Fastest way to find which model id a *known, working, official demo* actually uses: HF Spaces
  bundle their JS. Search `https://huggingface.co/api/spaces?search=<topic>` for a relevant demo,
  fetch its `assets/*.js` (path is in the space's file listing, same `.siblings` API as above),
  and regex the bundle for `org/repo`-shaped strings
  (`/[\w.-]+\/[\w.-]+/` filtered to known orgs like `Xenova`/`onnx-community`). This is how
  `Qdrant/all_miniLM_L6_v2_with_attentions` and `damoncrockett/gpt2-with-attentions-onnx` were
  found, starting from `webml-community/attention-visualization`.

## 4. General engine check: parameters that change after data is derived

Any canvas with a live-adjustable parameter that a round/view was originally derived from
(a layer selector, a resolution, anything dispatched *after* `prepare()`/`initState()` has
already built per-round data) needs this check: **dispatch the change, then read the derived
field directly from the resulting state — don't just watch the canvas re-render.** A canvas can
recompute its own display independently of the engine (exactly what happened in canvas 15: the
matrix visual was always live and correct because the canvas recomputed it itself, which masked
that the engine's own `state.rounds`/`state.baseline` — the thing actually graded by
`evaluate()` — was frozen at whatever value was current during `initState`). The bug is invisible
from the UI alone; it only shows up by checking the state object after the action, or by playing
a full pass/reveal cycle after changing the parameter and seeing the score not match what's on
screen. See `src/engines/attentionGuessEngine.ts`'s `SET_LAYER` case and
`plan-docs/REMAINING-WORK.md` canvas 15 notes for the concrete fix pattern (`rebuildRounds`,
carrying over player-entered answers positionally).

Same family of bug: a `RESET` action that reconstructs state from already-derived fields instead
of the originally-injected raw data (`attentionGuessEngine`'s old `RESET` rebuilt rounds with
`attention: []`, discarding the real tensors — the game was unplayable after any reset). If an
engine's `RESET` doesn't just call `initState` with the *original* prepared data, check why not.

## 5. Calibrating level thresholds against a real model

Don't hand-derive what a real model will do. Once a canvas is minimally working end-to-end
(§1), sweep candidate settings with Playwright directly — `.fill()` a slider, read the HUD's
breakdown text (`page.locator('text=SOME_METRIC').locator('..').innerText()`), repeat. A dozen
sweeps this way is faster and more reliable than reasoning about the math by hand, and real
models are consistently more surprising than expected (e.g. canvas 15's model tracks which noun
is more *salient*/first-mentioned far more than it tracks adjective semantics — an assumption
that would have shipped a wrong hint if asserted from first principles instead of measured).

This applies to level *thresholds*, not just hints — a level authored before its model was
available (blocked on A1, or just built ahead of verification) can ship thresholds nobody ever
checked. Canvas 17's shipped defaults for two of its three levels turned out to already score a
perfect 3 stars with zero player interaction, because the "good" configuration reproduced the raw
captured data almost exactly — not obvious from reading the engine, only from actually running it
against real numbers and checking whether the *default* state already passes. If it does, there's
nothing for the level to teach.

**Calibrate a causal-LM chapter's Node script with `dtype: 'q8'`, never the library default
(`fp32`).** This is the single costliest mistake made building World 7.3/7.4, worth its own
callout: `tinyCausalLM.ts` loads with `dtype: backend === 'webgpu' ? 'q4' : 'q8'`, and this
environment cannot reliably provide WebGPU (§9) — so every real player on the WASM path gets a
**q8-quantized** model, never the fp32 a bare `AutoModelForCausalLM.from_pretrained(id)` call
gives you in a throwaway Node script. The gap is not cosmetic: a first pass calibrating 7.3
entirely against fp32 produced numbers that were simply wrong at the precision the app actually
ships — a "0 examples fails, 1 example is 100% reliable" story at fp32 dropped to 0% reliable at
q8 for the identical prompt; a "reorder this list, 100% reliable tool pick" mechanic at fp32
collapsed to an "always picks the same tool regardless of order" bias at q8. Both had to be
rebuilt from real q8 data. Always load calibration scripts with the real precision:
```ts
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype: 'q8' });
```
`AutoModel.from_pretrained` (§2, for output-shape introspection rather than generation behaviour)
is unaffected by this — dtype doesn't change what output keys/dims a model exposes, only what its
actual generated content and reliability look like. This matters specifically for **generation
behaviour**: JSON-validity rates, which of several options a model picks, anything sampled or
greedy-decoded and graded. If a Node q8 script isn't practical (e.g. checking something that only
manifests in the real WASM execution provider), a real browser run is the fallback of last resort,
but q8-in-Node is almost always fast enough to be the first move, and is what actually caught both
regressions above well before a slow browser round-trip would have.

**A naive per-token confidence/surprisal probe is dominated by the tokenization's own leading
formatting token, not by whether the model knows the answer.** Found building World 8.1's
teacher-forced perplexity-proxy level: reading the model's real probability for the *first* token of
a tokenized answer (`" 2,300"` → its first BPE token) measured almost pure noise across 8 real facts,
because that first token is a generic leading-space/formatting token nearly every answer shares
regardless of content — the metric was really answering "does this precision start a response the
same way," not "does it know the digits." Once the first token of the real tokenized continuation was
excluded from the average (teacher-forced, reading each subsequent step's probability off the real
full-vocabulary softmax), the intended signal appeared cleanly and consistently. Any future level that
scores "confidence in a specific real answer" via teacher-forced surprisal should isolate the *content*
tokens the same way — check what the tokenizer's first token of the target continuation actually
decodes to before trusting a metric built on it.

**A fragile, near-the-decision-boundary case can flip *qualitatively* between Node q8 and a real
browser, not just drift by a small margin.** Every other Node-vs-browser gap already documented in
this project (7-1's margin readout, 7-4's "62 times 14") was a small numeric delta that stayed inside
the same star band. Building World 8.2's level 3, a Node q8 probe found one specific
target/distractor/order combination genuinely deriving the wrong answer; testing that *exact* same
combination directly in a real Chromium browser found it answering correctly — not a small drift, a
full flip of pass/fail for the one case the whole level's "hard" mechanic depended on. Node q8 is
still the right first move (fast iteration, catches the big, structural findings reliably), but for
anything this close to the model's own decision boundary — a distractor that only sometimes derails an
answer, a threshold sitting right at a real measured value — **re-confirm the exact combinations a
level will score directly in a live browser before shipping**, not just a representative sample.

## 6. Loading a model whose repo is missing config.json/tokenizer, or splits weights into external data

Found unblocking canvas 17 (`yaww85/all-MiniLM-L6-v2-hidden-states-exposed-v1`) — two gotchas that
don't show up on any model already wired into this repo, so nothing existing hints at them:

- **No `config.json` in the repo at all**, just the ONNX graph and maybe a `vocab.txt`.
  `AutoModel`/`AutoTokenizer.from_pretrained(id, ...)` both fail immediately trying to fetch it —
  `Could not locate file: ".../config.json"`. Confirm the raw graph actually has what's needed
  first (§2's `onnxruntime-node` check, no config required for that). If it does, and the repo is
  a re-export of a model already used elsewhere in this app (check `base_model` in the HF repo's
  card, or match its `vocab.txt`/tokenizer against a known repo), borrow config and tokenizer from
  that base id instead of the weights-only repo:
  ```ts
  const config = await AutoConfig.from_pretrained('Xenova/all-MiniLM-L6-v2'); // the real base model
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
  const model = await AutoModel.from_pretrained('yaww85/...', { config, subfolder: '' } as any); // weights-only repo
  ```
- **Weights split into a separate external-data file** (`model.onnx` + `model.onnx.data` — the
  standard ONNX format for anything over the old 2GB single-file limit; some conversion scripts use
  it unconditionally even for a 90MB model). transformers.js has its own automatic external-data
  fetch, but it only recognises **its own naming convention** (`<name>_data`, `<name>_data_1`, ...)
  — a plain `<name>.data` file (what `onnx.save_model(..., save_as_external_data=True)` actually
  produces) is invisible to it, and the load fails deep inside onnxruntime with a
  filesystem/network error that gives no hint the fix is one option away. Fix: pass the second file
  explicitly through `session_options.externalData` — this still routes through transformers.js's
  own `getModelFile`, so it gets real browser-cache and progress-callback behaviour, not a
  hand-rolled fetch outside the app's normal loading path:
  ```ts
  const model = await AutoModel.from_pretrained(id, {
    config, subfolder: '', dtype: 'fp32',
    session_options: { externalData: [{ path: 'model.onnx.data', data: 'model.onnx.data' }] },
  } as any);
  ```
  Verify this pattern all the way through — a real forward pass in a Node script *and* in an
  actual Playwright-driven browser — before trusting it. The Node/browser split that caused canvas
  15's `dtype` default to only fail in the browser (see A1 in REMAINING-WORK.md) is reason enough
  not to assume Node success carries over.

## 7. `ModelGate`'s `load` prop never fires without a `modelId`

Only matters once a single canvas has to serve levels with genuinely different model needs — the
first time that came up was canvas 18, where levels 1–2 are pure maths and level 3 needs
`embeddingModel`. `ModelGate`'s own effect that calls `load` is gated on `if (!modelId) return`,
so passing `modelId={null}` (the documented way every World 2/3 canvas skips the gate, because
their `initState` is synchronous and never goes through `load` in the first place) means **`load`
is never invoked at all** for whichever levels get `null`. Those levels silently never call
`prepare()`/`initState()` — `state` stays `null` forever, and nothing renders. No error, no stuck
spinner, just a blank board, which makes it look like a data problem rather than a wiring one.

Fix: call `load()` directly from a `useEffect` in the top-level canvas component whenever the
current level doesn't need the gate, and let `ModelGate` handle the levels that do:
```ts
const needsModel = config.mode === 'combine-with-embeddings';
useEffect(() => { if (!needsModel) void load(); }, [needsModel, load]);
return <ModelGate modelId={needsModel ? EMBEDDING_MODEL_ID : null} load={load}>...</ModelGate>;
```
Worth checking first in any future chapter whose levels don't all need the same model.

**A related gap, hit building World 8.1: `ModelGate` was built assuming one model per chapter, and
has no way to show progress for two loaded at once.** 8.1 needs two full real instances of the same
model (an fp32 reference and a q8 quantized variant) loaded together for its speed-comparison level.
`ModelGate`'s `modelId` prop can only subscribe to one progress stream at a time
(`subscribeToModelLoads` filters by a single id). No custom dual-progress UI was built to fix this —
the pragmatic real fix used here: pass the **larger, slower** download's id as `modelId` (so the
visible progress bar tracks the one that actually dominates the wait), and have `load()` fetch that
one first, then the second inside the same callback, sequentially. The second model's real download
still happens and still succeeds — it just has no visible progress bar of its own once the first
finishes, and the "Loading the model" screen's breathing-square animation (which isn't tied to
progress at all) keeps the screen from looking frozen in the meantime. Fine for two models where one
clearly dominates; would need real UI work if a future chapter needed two comparably-sized models
loaded with equally-informative progress.

## 8. A quantized ONNX file can load fine under Node and still fail in the browser's WASM backend

Found unblocking canvas 19 (`Xenova/gpt2`, `gpt2CausalLM`). Every quantized variant of this repo —
`model_quantized.onnx` (404, doesn't even exist), `model_int8.onnx`, `model_uint8.onnx`, and
`decoder_model_merged_quantized.onnx` (reachable via an explicit `model_file_name` override) — fails
`InferenceSession.create` in a real browser with the exact same error:

```
Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: qdq_actions.cc:137
TransposeDQWeightsForMatMulNBits Missing required scale: transformer.wte.weight_merged_0_scale
```

**This only reproduces in an actual browser (Playwright, not headless-shell-only — regular
`chromium.launch()` reproduces it fine).** A plain Node script loading the identical file via
`onnxruntime-node` runs it without complaint — Node and onnxruntime-web are different runtimes with
different execution-provider code, so a "verified working in Node" check (§2) does **not** clear a
model for browser use once quantization is involved. This repo's conversion applies blockwise N-bit
quantization to the token-embedding weight specifically, in a way this onnxruntime-web version's WASM
EP can't resolve, regardless of which quantized file is picked. If a model's quantized variant throws
a `TransposeDQWeightsForMatMulNBits`/similar QDQ error only in the browser: don't chase a different
quantized file, they likely all carry the same weight-quantization scheme. Fall back to a dtype with
no quantize/dequantize nodes at all — `fp16` (a pure precision cast) is usually much smaller than
`fp32` and sidesteps the whole class of bug:
```ts
await transformers.AutoModelForCausalLM.from_pretrained(modelId, { dtype: 'fp16', device: backend });
```
Also worth knowing regardless of whether this bug is in play: `dtype` and the file it actually
requests are two different questions. `dtype: 'q8'` maps to a `_quantized` filename suffix; if you
need a *specific* file the repo ships under a different base name (e.g. `decoder_model_merged*.onnx`,
the merged KV-cache graph, versus the default `model*.onnx`), pass `model_file_name` explicitly:
```ts
await transformers.AutoModelForCausalLM.from_pretrained(modelId, {
  dtype: 'q8', model_file_name: 'decoder_model_merged',
} as any); // not in the public option type, same category as §3's output_attentions flag
```
Check `https://huggingface.co/api/models/{owner}/{repo}` → `.siblings` for the real filenames and
their real sizes (`curl -sIL .../resolve/main/onnx/<file>` → `content-length`) before picking a
dtype — the declared `estimatedSizeMB` in a chapter's JSON is sometimes a guess from before the model
was actually loaded successfully, and can be off by 2–4×.

## 9. This environment cannot reliably provide WebGPU — don't build a canvas's verification plan around it

Found while building canvas 20 (`webllmCapstone`, WebLLM, needs `navigator.gpu`). Checked thoroughly
before concluding this, not assumed: `navigator.gpu` is `undefined` in Playwright's bundled Chromium,
in both headless and headed mode, and also in this machine's own real, current, WebGPU-capable
system Chrome (channel `'chrome'`) launched through Playwright with every relevant flag
(`--ignore-gpu-blocklist --enable-unsafe-webgpu --use-angle=metal`) — it appeared exactly once,
transiently, on an internal `chrome://gpu` page load, and was consistently absent on every real page
otherwise. This reads as a GPU-process/automation-sandboxing limitation of launching Chrome through
Playwright specifically, not something fixable from application code.

**Consequence for any WebGPU-dependent chapter (currently just World 6): the actual generation
gameplay cannot be played end-to-end here the way every WASM/CPU-backed canvas in this course was.**
Don't spend more time chasing browser flags trying to force it — instead:
- Build the canvas so `isWebLLMSupported()` (or equivalent) is checked *before* `ModelGate` attempts
  the download, and verify that specific unsupported-browser path for real — it's the one thing about
  a WebGPU-gated chapter this environment can genuinely exercise, since `isWebLLMSupported()` will
  honestly return `false` here.
- Verify everything else that doesn't need the model to actually run: hints, concept panel, layout at
  375px, dark theme, zero console errors on the unsupported-state screen.
- Lean harder on the pre-existing engine test suite and a deliberate, careful self-review pass of the
  component against the engine's contract — this is how canvas 20's two real UI bugs (a dead
  `useReducedMotion` import driving nothing, and a trace label rendering `local — "local"` instead of
  the real prompt) were caught without ever running the generation flow live.
- Say so plainly in `plan-docs/REMAINING-WORK.md`, the same way A4/A5/A6 already document
  infrastructure that needs a real credential/cluster/network condition this environment doesn't
  have — don't claim "verified end-to-end in a real browser" for a flow that was never actually run.
- The cloud side of a chapter (an API route proxying an external key, as opposed to a model needing
  WebGPU) is a different situation and usually *is* testable — check `.env` for a real key before
  assuming it isn't; canvas 20's Ollama Cloud round trip was fully exercisable this way even though
  the local WebLLM half was not.

## 10. A stateful fake dependency can leak real state across unit tests

Found writing World 8.1's engine test: a `PrecisionModelDep` fake whose `ensureLoaded()` deliberately
mimics the real one's idempotence (returns a real duration on first call, `0` once "loaded") was
declared once at `describe`-block scope and reused across two `it()` blocks. The first test's call
left the fake's internal `loaded` flag `true`, so the second test's "real" load-time assertions
silently read back `0` instead of the fixture's configured value — not a crash, a quietly wrong
number that only surfaced because the test's own expected value didn't match it. Any fake dependency
that carries state across calls to mirror real caching/loading behaviour (a `loaded` flag, a call
counter, anything not reset between invocations) needs a **fresh instance per test**, not a shared
one at `describe` scope — build it in a small factory function and call that inside each `it()`.

## 11. Node q8 calibration is far less trustworthy for "what does this model know/do" than for mechanical properties — verify the real target set directly in a real browser before finalizing

§5 already establishes that Node calibration must use `dtype: 'q8'`, never `fp32`, and that it "is a
good first pass, never the final word" near a decision boundary. World 8.3 and 8.4 sharpened that into
something stronger: for **mechanical** properties — attention weights, load/inference timing, a model's
token-probability distribution shape — Node q8 and browser WASM q8 have matched almost exactly (8.2's
attention-dilution level matched to three decimal places). For **recall/instruction-following**
properties — does the model know this specific fact, does it obey this specific rule under this
specific follow-up text — the two environments diverged far more often and far more dramatically than
anywhere else in this project, to the point of flipping real, verified-correct facts and real,
verified-working defenses:

- 8.3 (Calibration & Hallucination): Node found `capital-japan` and `largest-planet` answered
  incorrectly; a real Chromium run answered both correctly. Node's most dramatic "confidently wrong"
  example (`capital-italy`, a fabricated city at real 71% confidence) barely resembled its own
  real-browser behaviour (56%, different wrong answer), and a *different* fact in the same round became
  the real highest-confidence-wrong case instead.
- 8.4 (Red-Teaming): Node calibration found only one of four classic attack framings (roleplay) broke
  a "answer in exactly one word" instruction. A real browser run found **all four** broke it — the
  premise that "most attacks fail, a clever one succeeds" was itself wrong for this tiny model, not
  just the specific numbers.

**Consequence for planning any future chapter that scores which specific facts a model does or doesn't
know, or which specific attacks/defenses do or don't work**: treat Node q8 as a hypothesis generator
only. Before finalizing which items ship in a level, widen that level's config to the full real
candidate pool (every fact, every attack, every defense under consideration), run it once for real in
a real browser, and read the actual per-item results straight off the rendered board — the same
"temporarily widen the config, sweep for real, narrow back down" move both 8.3 and 8.4 used. Don't
budget calibration time assuming Node gets you most of the way there for this category of chapter; for
World 7.3/7.4/8.1/8.2's category (structured-output reliability, quantization/timing/attention), Node
q8 was reliable enough to design against directly.

## 12. Verify a scoring premise's basic shape against the real model before designing rounds around it — and a defense that's real-ly guaranteed to fail is a legitimate way to keep a level honest

Two related lessons from 8.4, worth separating from §11's calibration-fidelity point because they'd
still apply even with perfect Node/browser agreement:

- **A level's entire framing can rest on an assumption that's simply false for the real model, not
  just imprecise.** 8.4's plan assumed "a naive instruction mostly holds, a well-chosen attack breaks
  it" — checking this directly (testing the *unattacked* baseline, and several genuinely mild
  non-attacks alongside the real attack candidates) revealed this tiny model's baseline
  instruction-following is far more fragile than that framing assumed: nearly any additional text after
  the instruction broke it, attack or not. The fix wasn't tuning numbers, it was checking the premise
  itself against real behaviour before building scoring logic on top of it — the same category of
  "verify before you design" as §5's calibration-threshold check, just one level up (the mechanic's
  shape, not a specific level's pass bar).
- **When a real sweep shows every candidate in a set succeeding (or failing), that's the "canvas 17
  already at 3 stars by default" anti-pattern from §5, and the fix doesn't require inventing a fake
  negative case.** 8.4's first real defense sweep found all 4 candidate defenses resisted the target
  attack — any first click would trivially solve the level. Rather than search for a weaker real defense
  to include (more calibration cycles, no guarantee of success), the fix was adding a defense whose text
  is **byte-identical to an already-established real failure** (the exact instruction level 1's attack
  already broke, unreinforced) — guaranteed to fail without any new probing, and a genuinely plausible
  thing a player might try first ("just repeat the same rule"), not a strawman. Worth remembering for
  any future "test several options, find the one that works" level where a real sweep comes back
  suspiciously one-sided: a guaranteed-real distractor built from an already-verified result is often
  cheaper and more honest than hunting for a new real negative case.
