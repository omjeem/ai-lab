---
name: ailab-navigation-sharing
description: How chapter navigation, the locked-chapter warning dialog, the persistent prerequisite-gap banner, and shareable chapter links (`?via=share`) work in this repo — where the single gate lives, why it's client-side only, the two activity event types it fires, the service-worker offline gotcha shared links hit, and a Playwright-testing gotcha (React-controlled-input hydration race) hit while verifying this feature. Load this before changing anything about chapter unlocking, the map's `ChapterNode`, `ChapterPageClient.tsx`, or the share button.
---

# Chapter navigation, warning gate, and share links

Built when locked chapters went from a hard block (no link rendered at all) to "reachable, but
with a conscious warning." Read this before touching any of it — the design has a few
non-obvious constraints that are easy to accidentally undo.

## Where the gate actually lives, and why

`src/lib/curriculum.ts`'s `unlockRequires` graph and `isChapterUnlocked()` are unchanged — they
still decide what's "locked." What changed is only the *consequence*.

**The single enforcement point is `ChapterPageClient.tsx`**, not the map. This is deliberate:
unlock state depends on `completionRecord()` read from IndexedDB (`useGameProgressStore`), and
there is no server-side equivalent — no accounts, no session tied to a user identity server-side.
So the check can only run client-side, and it has to run at the one place every entry path
converges (map click, typed URL, bookmark, browser back button) rather than being duplicated in
`WorldMap.tsx`. If you ever find yourself adding an unlock check to the map component again,
that's a regression of this design, not an enhancement — the map's `ChapterNode` only *displays*
lock status now (dimmed styling, lock icon, tooltip); it does not gate anything.

**Must check `hydrated` before checking lock status.** `useGameProgressStore`'s `chapters` starts
empty until IndexedDB hydration finishes. Skipping the `if (!hydrated) return null` guard (copied
from `src/app/(game)/map/page.tsx`'s existing pattern) means a returning player who deep-links or
refreshes a chapter they already completed sees a **false "locked" warning flash** before real
state loads. This is the single easiest way to break this feature invisibly — it won't show up in
`tsc`/`eslint`/unit tests, only in an actual browser with actual persisted progress.

**Gate condition is `status === 'locked'` (i.e. `!isChapterUnlocked`), not "is this the next
chapter in `orderedChapters()`."** The curriculum's unlock graph is not strictly linear — e.g.
`7-1-retrieval` and `6-1-inspector-chat` both unlock directly off `5-5-full-transformer`, not off
each other — so a position-based "is this literally next" check would show a bogus warning on a
chapter that's actually fully earned. Always gate off the real per-chapter `unlockRequires`
graph.

## The share bypass

- Query param is **`via=share`**, read server-side in `page.tsx` (`searchParams` on the async
  Server Component), not via a client `useSearchParams()` call — one line away already, no extra
  Suspense boundary needed.
- **Never name this param `source`.** `src/middleware.ts` intercepts any `?source=` on *every*
  route for an unrelated external-tracking beacon and redirects to strip it — it would silently
  destroy a same-named share marker before `page.tsx` ever saw it.
- A shared link **always** skips the warning dialog, regardless of the visitor's own unlock state
  — that's the entire point of handing someone a working door into one chapter.
- **Offline PWA gotcha**: `public/sw.js`'s `networkFirst()` (used for all `navigate`-mode requests)
  originally did an exact-match `caches.match(request)`. A `?via=share` URL doesn't exact-match its
  precached chapter entry (which has no query string), so it silently fell back to the cached
  `/map` shell — the same bug class this file already documents fixing once before for the
  plain-URL case. Fixed with `caches.match(request, { ignoreSearch: true })`. If you add another
  query-string-bearing entry point to a chapter route in the future, check this function again.

## Activity events

Two new literals in `src/types/activity.ts`'s `activityEventTypeSchema`:
- `chapter_jumped_ahead` — fires only when the dialog was shown **and** the visitor clicked "Play
  anyway." Not fired for a shared-link bypass — that's a different action (following a link,
  not consciously overriding a visible warning).
- `chapter_shared_link_opened` — fires whenever a chapter loads with `via=share`, **regardless**
  of lock state, with `detail: { wasLocked: 1|0 }` distinguishing "shared a chapter you'd already
  unlocked" from "the link let someone skip a gate." One event type with a flag here, not two —
  unlike the jumped-ahead/dialog case, both cases are the *same* action (opening a shared link),
  just with different outcomes.

Both go through the existing `enqueueActivity()` (`src/lib/offlineQueue.ts`) — no new queue or
sync logic needed for this feature itself (see the separate sync-manager polling fix for that
system).

## The persistent gap banner

`ChapterShell`'s `prerequisiteGap` prop (`{ missing: { id, title, world }[] }`) renders a
dismissible banner below `ChapterBar` for as long as the chapter is unlocked-but-played (via
"Play anyway" or a share-link bypass into a locked chapter). Dismissal is local `useState` — it
resets on remount/reload by design. The gap hasn't actually closed just because the banner was
closed once, so it should resurface next time the player comes back to that chapter, not be
permanently silenced.

## Testing gotcha found while verifying this in a real browser

**A fresh onboarding submit button can appear to "never enable" in Playwright even though `.fill()`
succeeded — this is a React-controlled-input hydration race, not a bug in the feature.**
`Onboarding.tsx`'s Start button is `disabled={name.trim().length === 0 || busy}`. If you
`page.goto(..., { waitUntil: 'domcontentloaded' })` and immediately `.fill()` the name input,
Playwright dispatches a native `input` event before React has necessarily attached its listeners
(hydration hasn't finished), so the native DOM value changes but React's `name` state never
updates — the button then sits disabled forever and a `.click()` on it times out with
"element is not enabled," which reads like a real assertion failure. Fix: wait a short beat after
the input becomes visible before filling (`page.waitForTimeout(300–500)` is enough in practice),
or poll for the button to become enabled (`page.waitForFunction(() => !btn.disabled)`) before
clicking it. This is a general trap for any fresh-onboarding Playwright script in this app, not
specific to this feature — worth remembering for verifying anything that requires a fresh
identity first.

Also: `Locator.isVisible()` does **not** poll/wait the way `.click()` or `waitForSelector` do —
calling it immediately after `page.goto()` on a route that hasn't finished rendering yet returns a
false negative, not a "not found yet, keep trying" result. Use `waitForSelector`/`.waitFor()` when
you need to wait for something to appear, and reserve `isVisible()` for a snapshot check after
you already know the DOM has settled.
