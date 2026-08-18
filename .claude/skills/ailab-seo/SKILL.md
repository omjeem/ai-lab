---
name: ailab-seo
description: How per-chapter SEO works in this repo — the generateMetadata/JSON-LD pattern in the chapter route, the Next.js metadata merge-semantics trap that silently drops inherited openGraph/twitter fields, why chapter keywords are derived instead of hand-authored per chapter, and what to actually check (and not bother checking) when asked to "improve SEO" here. Load this before touching src/app/(game)/world/[worldId]/chapter/[chapterId]/page.tsx, src/lib/seo.ts, sitemap.ts, robots.ts, or the root layout's metadata.
---

# Per-chapter SEO

## The pattern

Every chapter route (`src/app/(game)/world/[worldId]/chapter/[chapterId]/page.tsx`) is a real,
independently indexable page with its own:
- `title`/`description` pulled from the chapter's own written content (`chapterTitle`,
  `concept.shortExplanation`) — never the site-wide default.
- `keywords` via `deriveChapterKeywords()` in `src/lib/seo.ts`.
- A full `openGraph` and `twitter` object (see the merge trap below).
- A `LearningResource` JSON-LD `<script>` block in the page body, naming what that specific page
  teaches (`teaches: game.chapterTitle`) and its parent `Course` (`isPartOf`) — modelled on the
  root page's own `Course` schema in `src/app/page.tsx`.

`src/app/sitemap.ts` (iterates `allGames()`) and `robots.ts` (allows everything except `/admin` and
`/api`) already handle discovery/crawlability for every chapter — don't add to either unless a
genuinely new route type is introduced.

## The metadata merge trap — the most important thing here

**Next.js does not deep-merge a route's `generateMetadata` return value with its parent layout's
`metadata` export.** If a route returns an `openGraph` (or `twitter`) key *at all*, that whole
object **replaces** the parent's — it does not layer specific fields on top of inherited ones.

The root layout (`src/app/layout.tsx`) sets a complete `openGraph` (`type`, `url`, `siteName`,
`title`, `description`, `locale`) and `twitter` (`card: 'summary_large_image'`, `title`,
`description`). Before this was understood, the chapter route's `generateMetadata` returned only
`openGraph: { title, description }` — which meant **every chapter page was silently missing
`type`/`url`/`siteName`/`locale`, and had no `twitter` override at all** (so it inherited the
generic site-wide Twitter card by accident, not by design). This is easy to reintroduce: if you
touch `generateMetadata` and add a new field to `openGraph` or `twitter`, you must redeclare the
**whole** object (all of `type`, `url`, `siteName`, `locale`, `title`, `description` for
`openGraph`; `card`, `title`, `description` for `twitter`), not just the field you're adding.
Verify with `curl <url> | grep 'og:\|twitter:'` — a missing `og:type`/`og:site_name` or a
`twitter:card` that isn't `summary_large_image` means this broke again.

`keywords` is not subject to this trap — it's a flat string array, not a nested object, so it
merges fine on its own. Only `openGraph`/`twitter` (and any other nested metadata object) need the
full-redeclare treatment.

## Why `deriveChapterKeywords()` is derived, not a per-chapter JSON field

Google's `keywords` meta tag has had no ranking effect since ~2009, and the root layout already
applies `SITE_KEYWORDS` to every page — so there was never a "zero keywords" gap to fill here, only
a "not chapter-specific" one. A hand-curated `seoKeywords: string[]` field on each chapter's JSON
would need: a schema change (`src/types/game.ts`), a `scripts/validate-games.ts` rule to keep it
from silently rotting as chapters are added, and manual upkeep across 26+ files — all for a tag
with no measurable ranking benefit. `deriveChapterKeywords(game)` in `src/lib/seo.ts` gets the same
practical outcome (chapter-specific terms) from data that's already real and already current,
with zero maintenance surface. Don't "improve" this by moving to hand-authored keywords unless
something concrete changes that assumption (e.g. evidence a *different* search engine or AI
crawler this project cares about actually weighs the tag).

## What "improve SEO" should actually mean here, if asked again

In order of actual leverage, highest first:
1. **On-page content matching search intent** — already true: `game.concept.shortExplanation`
   renders unconditionally in `ChapterShell`'s concept panel (not gated behind a client-only mount
   effect), so it's present in the initial server-rendered HTML a crawler sees. Confirm this stays
   true if `ConceptPanel` is ever refactored — a `mounted`-gated render would silently remove the
   one thing giving a chapter page real indexable text about its own topic.
2. **Structured data** (`LearningResource` JSON-LD) — done, see above.
3. **Correct, unique per-page metadata** (`title`/`description`/canonical/`openGraph`/`twitter`) —
   done, see the merge trap above.
4. **Crawlability/discovery** (`sitemap.ts`, `robots.ts`) — already correct, nothing to add.
5. **`keywords` meta tag** — lowest leverage of all of these; already present, chapter-specific,
   effectively decorative for Google specifically.

What this can never fix: actual search ranking depends on domain authority, backlinks, and query
competition — all off-page factors. Don't imply to a user that shipping more on-page SEO code will
make a chapter rank for a competitive query; it removes technical barriers, it doesn't buy
position. If the user has Google Search Console access connected, checking real indexing coverage
after a deploy is the only way to confirm any of this actually landed — don't guess from the code
alone when that's available.
