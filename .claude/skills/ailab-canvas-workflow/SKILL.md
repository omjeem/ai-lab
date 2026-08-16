---
name: ailab-canvas-workflow
description: Tooling and debugging workflow for building/verifying AI Learning Lab game canvases in this repo — how to get a real browser running here to test a canvas end-to-end, how to introspect a real HF model from Node without the app's browser-only runtime getting in the way, how to find an alternate ONNX export when a model doesn't expose what a chapter needs, and a general engine-correctness check for any canvas with a live-adjustable parameter. Load this before building canvas 16 or later, before touching src/models/*.ts, or whenever a model load hangs/fails only in the browser. Complements plan-docs/REMAINING-WORK.md (the what/why per chapter) — this is the how, so the same false starts aren't repeated.
---

# AI Learning Lab canvas workflow

Process notes from building canvases 14 and 15, specifically the parts that cost real time
and aren't in `plan-docs/REMAINING-WORK.md` (which covers per-chapter findings, not tooling).

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
  - Every chapter opens on a concept/instructions screen first. Click the exact text `Begin` to
    reach the actual canvas — the `ModelGate`/engine doesn't mount before that.
  - **Don't wait on a text string that might also appear in the concept prose.** `waitForSelector('text=temperature')`
    matched the concept paragraph (which mentions "temperature" in prose) and resolved
    immediately, before the canvas or its model had even started loading — the run then timed
    out on the next step for a confusing reason. Wait on a structural selector that only exists
    once the canvas is mounted instead, e.g. `input[type="range"]` or a label unique to the
    rendered board.
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

## 3. Finding an alternate ONNX export when a model doesn't expose what's needed

This is the exact situation for canvas 17 (`hidden_states` still unavailable — see
REMAINING-WORK.md A1). The technique that found canvas 15's fix:

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
