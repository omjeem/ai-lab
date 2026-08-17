---
name: ailab-canvas-workflow
description: Tooling and debugging workflow for building/verifying AI Learning Lab game canvases in this repo — how to get a real browser running here to test a canvas end-to-end, how to introspect a real HF model from Node without the app's browser-only runtime getting in the way, how to find an alternate ONNX export when a model doesn't expose what a chapter needs (including ones missing config.json, splitting weights into external data, or failing session creation only in onnxruntime-web's WASM backend), a general engine-correctness check for any canvas with a live-adjustable parameter, a ModelGate trap specific to canvases whose levels don't all need the same model, and what to do when a chapter needs WebGPU and this environment can't reliably provide it. Load this before touching any World 5/6 canvas, before touching src/models/*.ts, or whenever a model load hangs/fails only in the browser. Complements plan-docs/REMAINING-WORK.md (the what/why per chapter) — this is the how, so the same false starts aren't repeated.
---

# AI Learning Lab canvas workflow

Process notes from building canvases 14 through 20 — all 22 chapters are now built — specifically
the parts that cost real time and aren't in `plan-docs/REMAINING-WORK.md` (which covers
per-chapter findings, not tooling). Still worth reading before *revisiting* any canvas: the same
model-loading and verification traps apply to fixes as much as to first builds.

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
