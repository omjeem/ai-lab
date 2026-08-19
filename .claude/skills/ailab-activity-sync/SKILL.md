---
name: ailab-activity-sync
description: How the offline activity queue and its background sync manager work in this repo — the event schema, the queue's IndexedDB cap/dedupe behavior, why enqueueActivity() fires an immediate single-event POST instead of waiting for the interval, why the recurring sync timer skips its network probe when the queue is empty, and why this area deliberately has no unit test for either fix. Load this before touching src/lib/offlineQueue.ts, src/lib/syncManager.ts, src/types/activity.ts, or src/app/api/activity/route.ts.
---

# Activity queue and sync manager

## The pieces

- `src/types/activity.ts` — `activityEventTypeSchema`, a **closed** Zod enum (not an open string).
  Adding a new event type anywhere in the app means adding its literal here first — see the
  `ailab-navigation-sharing` and `ailab-admin-analytics` skills for the two most recently added
  (`chapter_jumped_ahead`, `chapter_shared_link_opened`) as a model.
- `src/lib/offlineQueue.ts` — `enqueueActivity()` writes to IndexedDB, capped at 5000 events
  (oldest dropped first, never newest), deduped only by `eventId` (never by content — queuing the
  "same" event twice with two different ids just means two rows). After the write succeeds it also
  fires `sendImmediately()` (fire-and-forget, same file): a single-event POST to `/api/activity`
  that clears itself out of the queue on success. `queueSize()` is a cheap `db.count()` — already
  exported, already used by the sync manager's idle check (see below); don't reintroduce a
  duplicate way to get this number.
- `src/lib/syncManager.ts` — `startSyncManager()`, mounted once at app root in
  `AppShell.tsx`. Runs on a recurring interval (30s default) plus the browser's
  `online`/`visibilitychange` events. `syncOnce()` posts a batch to `/api/activity` and clears
  exactly the event ids the server confirms via `clearSynced()` — never a blanket clear, since
  events queued mid-flight must survive a sync that started before they existed. This is now the
  *fallback* path (offline at enqueue time, or `sendImmediately()` failed) rather than the only
  delivery path — see below.

## Instant delivery (`sendImmediately`)

`enqueueActivity()` used to only write to IndexedDB and let the interval/online/visibility
triggers in `syncManager.ts` do the actual sending — meaning a freshly-recorded event could sit
for up to 30s before reaching the server. It now also calls `sendImmediately(event)` right after
the IndexedDB write succeeds: a direct `fetch('/api/activity', { method: 'POST', body: { events:
[event] } })` for just that one event. On a 200 with the event id present in
`persistedEventIds`, it calls `clearSynced([event.eventId])` immediately. On any failure (offline,
non-200, thrown error) it does nothing and swallows the error — the event is already sitting in
the queue from the write above, so `syncManager.ts`'s existing interval/online/visibility triggers
pick it up later with no special-casing needed.

This lives in `offlineQueue.ts`, not `syncManager.ts` — it does its own `fetch` rather than calling
`syncOnce()`, specifically to avoid a circular import (`syncManager.ts` already imports *from*
`offlineQueue.ts`). It also intentionally does not add any retry/backoff logic beyond the one
attempt: "no caching" here means don't build new delivery infrastructure for this path, since the
queue + existing sync manager already *is* the retry mechanism.

**Race with the interval sync is safe by design, not by luck**: if `syncOnce()`'s batch sync picks
up the same still-queued event before `sendImmediately()`'s single-event POST clears it (or vice
versa), both requests hit the server. `activity.eventId` has a unique index
(`src/lib/mongodb.ts`), so the second insert throws a Mongo duplicate-key error (code 11000), which
`route.ts`'s `extractDuplicateIds()` already treats as "persisted" and returns in
`persistedEventIds` — so the client still clears it correctly. Don't add de-duplication logic to
prevent this race; the existing unique-index + duplicate-key handling is what makes it safe to not
bother.

## The idle-polling fix

**Before**: every single interval tick did a `HEAD /api/activity` connectivity probe, unconditionally,
forever — even on a device that hadn't queued anything in hours. `syncOnce()` itself short-circuited
on an empty queue, but only *after* the network probe had already fired.

**After**: the interval-tick path (and only that path) checks `queueSize() === 0` first and returns
immediately if so — no network request at all. The initial `attempt()` call on mount, and the ones
triggered by `online`/`visibilitychange`, still probe unconditionally. That's deliberate, not an
inconsistency: those are real, infrequent, meaningful signals (the app just mounted; the browser
just came back online; the tab just became visible again) worth refreshing the connectivity
indicator on. The thing being fixed is *blind, unconditional polling on a fixed timer regardless of
whether there's anything to do* — not "probe less often" in general.

This idle-skip only applies to the interval-tick fallback path. It has nothing to do with instant
delivery, which is handled separately by `sendImmediately()` in `offlineQueue.ts` (see above) — that
path fires unconditionally on every `enqueueActivity()` call, queue-empty or not, since it's sending
the one event that was just enqueued.

If you're asked to make the *fallback* sync faster: don't wire `enqueueActivity()` to call
`syncOnce()` from `syncManager.ts` directly — `syncManager.ts` already imports *from*
`offlineQueue.ts`, so the reverse direction risks a circular import. `sendImmediately()`'s
standalone `fetch` (above) is how instant delivery was actually built, precisely to sidestep this.
The interval + connectivity-event triggers remain the right shape for the fallback path itself;
only the *unconditional probing on an empty queue* was ever the bug there.

## Why there's no unit test for either fix

This repo has no `fake-indexeddb` dependency and no existing timer-mocked (`vi.useFakeTimers()`)
test anywhere — introducing that machinery for a one-line early-return branch (the idle-skip) or a
fire-and-forget `fetch` call with two outcomes (`sendImmediately`) was judged not worth it,
especially since `startSyncManager()` itself no-ops entirely outside a browser (`typeof window ===
'undefined'`) and this project's vitest config runs in a plain Node environment, not jsdom. Both
changes were verified by direct code inspection rather than an automated test: the idle-skip guard
runs strictly before `running` is set (can't deadlock a later real sync); `sendImmediately`'s only
observable side effect on failure is "do nothing," and its success path reuses the already-tested
`clearSynced()`. If you're adding more logic to either file in the future and it's getting complex
enough that "read it carefully" stops being sufficient verification, that's the point to introduce
`fake-indexeddb` + fake timers properly — not before.

## Verifying this live

For the interval idle-skip: there's no way to observe "no network request fired" faster than the
interval itself without temporarily passing a shorter `intervalMs` through `SyncManagerOptions`
(which `AppShell.tsx` doesn't currently expose) — a real browser check means either waiting out a
full 30s window while watching DevTools → Network for `/api/activity` HEAD requests with an empty
queue, or temporarily patching `AppShell.tsx`'s `startSyncManager({...})` call to pass a short
`intervalMs` for the duration of the check and reverting it before committing.

For `sendImmediately`: much easier — trigger any event (e.g. complete a level) with DevTools →
Network open and confirm a `POST /api/activity` fires immediately, with a body containing exactly
that one event, well before the 30s interval would have ticked.
