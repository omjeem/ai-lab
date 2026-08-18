---
name: ailab-admin-analytics
description: How the admin dashboard's per-chapter/per-world analytics work in this repo — the Mongo aggregation shape, why shaping logic is a separate pure/tested function from the route, how to add a new metric or a new activity event type to the table, and the verification limits of this area (no live Mongo or admin credentials in most dev environments). Load this before touching src/lib/adminAnalytics.ts, src/app/api/admin/analytics/route.ts, src/app/admin/dashboard/page.tsx, or src/lib/mongodb.ts's indexes.
---

# Admin chapter analytics

## The shape of it

One Mongo aggregation, over the whole `activity` collection, grouped by `{chapterId, type}`:

```js
activity.aggregate([
  { $match: { chapterId: { $exists: true, $ne: null } } },
  { $group: { _id: { chapterId: '$chapterId', type: '$type' }, count: { $sum: 1 }, users: { $addToSet: '$userId' } } },
]);
```

This one pass gets both a raw event count and a distinct-user count per group (`users.length`) —
no second aggregation stage needed. The route (`src/app/api/admin/analytics/route.ts`) does
nothing but run this, reshape the driver's `_id.chapterId`/`_id.type` back into flat fields, and
hand the array to `shapeChapterAnalytics()` (`src/lib/adminAnalytics.ts`).

**`shapeChapterAnalytics()` is a pure function, deliberately separate from the route**, so it's
unit-testable with fabricated rows (`tests/lib/adminAnalytics.test.ts`) with no live Mongo needed —
matching this repo's existing convention that only pure/shaping logic gets unit tests, not routes
that need a real connection (`api/admin/users/route.ts`, `api/admin/activity/route.ts` are
untested for the same reason). If you're tempted to inline the shaping into the route "since it's
only used once," don't — that's the only reason this piece is testable at all.

It joins onto `orderedChapters()`/`worldOfChapter()` (`src/lib/curriculum.ts`), not onto whatever
chapter ids happen to appear in the aggregation output — **every chapter in the curriculum shows a
row, including ones with zero activity**, rather than the table silently shrinking to "chapters
someone has touched." This matters for a course that's still growing: a brand-new chapter should
be visible in the table at zero, not absent.

## Metric conventions — pick the right one when adding a new column

- **Distinct-user counts** (`startedUsers`, `completedUsers`) for "how many people did this at
  least once" — milestone events that only make sense counted once per person.
- **Raw event counts** (`levelCompleted`, `levelFailed`, `jumpedAhead`, `sharedLinkOpened`) for
  things a single person can legitimately do more than once (retry a level, open a shared link
  again, get warned and proceed on a later visit). Don't switch these to distinct-user counts
  without thinking about whether "did it once" or "did it N times" is actually the more useful
  number for that specific metric — they're not interchangeable defaults.
- `completionRate = completedUsers / startedUsers`, guarded to `0` when `startedUsers` is `0` —
  never divide without that guard, a chapter with zero activity is the first row you'll hit in
  practice (see the "even with zero activity" point above).

## Adding a new activity event to the table

1. Add the literal to `activityEventTypeSchema` in `src/types/activity.ts` (see
   `ailab-navigation-sharing` skill for the two that exist today as a model).
2. Fire it via `enqueueActivity()` wherever the real trigger is, with `chapterId` set — the
   aggregation's `$match` filters on `chapterId` existing, so an event without one never reaches
   this table at all (by design; this table is chapter-scoped, not a general event log — that's
   what the Users tab's per-user drilldown is for).
3. Add a field to `ChapterAnalyticsRow` and a `get('your_new_type')?.count` (or `.distinctUsers`)
   line in `shapeChapterAnalytics()`, following the existing pattern.
4. Add a column to the table in `ChapterAnalyticsPanel` (`src/app/admin/dashboard/page.tsx`).
No schema/index change needed beyond step 1 — the `{chapterId: 1, type: 1}` index already covers
any event type, and the aggregation is already type-agnostic.

## Verification limits — know this before promising a live check

This whole area needs `MONGODB_URI` and admin credentials (`ADMIN_EMAIL`/`ADMIN_PASSWORD_HASH`)
configured to interact with for real. In an environment without those set (no `.env.local`, no
seeded database — the common case for a fresh clone or this project's own dev sandbox), the most
you can verify is:
- `curl http://localhost:PORT/api/admin/analytics` returns `401 {"ok":false,"error":"Unauthorized"}`
  with no session — confirms the route is wired into `requireAdmin()` correctly.
- `curl -o /dev/null -w '%{http_code} -> %{redirect_url}'` on `/admin/dashboard` shows a `307` to
  `/admin/login?next=...` — confirms `middleware.ts` still gates the page.
- `tsc`/`eslint`/`vitest`/`pnpm validate:games` all passing.

Don't claim to have "verified the dashboard shows real numbers" without an actual seeded Mongo
instance and a logged-in session — this is the same limit the existing Users tab and its tests
already live with, not a new gap introduced by this feature.
