---
name: ailab-activity-sync
description: How the offline activity queue and its background sync manager work in this repo — the event schema, the queue's IndexedDB cap/dedupe behavior, why the recurring sync timer now skips its network probe when the queue is empty, and why this area deliberately has no unit test for that specific fix. Load this before touching src/lib/offlineQueue.ts, src/lib/syncManager.ts, src/types/activity.ts, or src/app/api/activity/route.ts.
---

# Activity queue and sync manager

## The pieces

- `src/types/activity.ts` — `activityEventTypeSchema`, a **closed** Zod enum (not an open string).
  Adding a new event type anywhere in the app means adding its literal here first — see the
  `ailab-navigation-sharing` and `ailab-admin-analytics` skills for the two most recently added
  (`chapter_jumped_ahead`, `chapter_shared_link_opened`) as a model.
- `src/lib/offlineQueue.ts` — `enqueueActivity()` writes to IndexedDB, capped at 5000 events
  (oldest dropped first, never newest), deduped only by `eventId` (never by content — queuing the
  "same" event twice with two different ids just means two rows). `queueSize()` is a cheap
  `db.count()` — already exported, already used by the sync manager's idle check (see below); don't
  reintroduce a duplicate way to get this number.
- `src/lib/syncManager.ts` — `startSyncManager()`, mounted once at app root in
  `AppShell.tsx`. Runs on a recurring interval (30s default) plus the browser's
  `online`/`visibilitychange` events. `syncOnce()` posts a batch to `/api/activity` and clears
  exactly the event ids the server confirms via `clearSynced()` — never a blanket clear, since
  events queued mid-flight must survive a sync that started before they existed.

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

If you're asked to make this "sync faster" or "sync immediately when something is queued": don't
wire `enqueueActivity()` to trigger `syncManager.ts` directly. This was considered and rejected —
`syncManager.ts` already imports *from* `offlineQueue.ts`, so the reverse direction risks a
circular import, and nothing in this app reads queued activity live; background telemetry doesn't
need sub-30s latency. The existing interval + connectivity-event triggers are the right shape;
only the *unconditional probing* was the bug.

## Why there's no unit test for this specific change

This repo has no `fake-indexeddb` dependency and no existing timer-mocked (`vi.useFakeTimers()`)
test anywhere — introducing that machinery for one one-line early-return branch was judged not
worth it, especially since `startSyncManager()` itself no-ops entirely outside a browser
(`typeof window === 'undefined'`) and this project's vitest config runs in a plain Node
environment, not jsdom. The change was verified by direct code inspection (the guard runs strictly
before `running` is set, so it can't deadlock a later real sync) rather than an automated test. If
you're adding more logic to `syncManager.ts` in the future and it's getting complex enough that
"read it carefully" stops being sufficient verification, that's the point to introduce
`fake-indexeddb` + fake timers properly — not before.

## Verifying this live

There's no way to observe "no network request fired" faster than the interval itself without
temporarily passing a shorter `intervalMs` through `SyncManagerOptions` (which `AppShell.tsx`
doesn't currently expose) — a real browser check means either waiting out a full 30s window while
watching DevTools → Network for `/api/activity` HEAD requests with an empty queue, or temporarily
patching `AppShell.tsx`'s `startSyncManager({...})` call to pass a short `intervalMs` for the
duration of the check and reverting it before committing.
