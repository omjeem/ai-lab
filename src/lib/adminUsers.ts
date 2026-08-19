/**
 * Shapes one Mongo `users` document (plus its aggregated activity edges) into
 * the row the admin dashboard renders — kept separate from the route so it's
 * unit-testable with a fabricated document, no live Mongo needed (same
 * convention as `shapeChapterAnalytics`).
 */
import type { AdminUserRow, UserDocument } from '@/types/user';

/** Earliest/latest `page_viewed` event for one user, from a Mongo aggregation. */
export interface PageViewEdges {
  landingPage: string | null;
  landingTitle: string | null;
  lastPage: string | null;
}

export function shapeAdminUserRow(
  doc: UserDocument,
  eventCount: number,
  edges: PageViewEdges | undefined
): AdminUserRow {
  const client = doc.client;
  const geo = doc.enrichment?.geo;
  const device = doc.enrichment?.device;

  return {
    userId: doc.userId,
    name: doc.name,
    firstSeen: new Date(doc.firstSeen).toISOString(),
    lastSeen: new Date(doc.lastSeen).toISOString(),
    country: geo?.country ?? null,
    city: geo?.city ?? null,
    ip: doc.enrichment?.ip ?? null,
    userAgent: doc.enrichment?.userAgent ?? null,
    browser: device?.browser ?? null,
    browserVersion: device?.browserVersion ?? null,
    os: device?.os ?? null,
    osVersion: device?.osVersion ?? null,
    deviceType: device?.deviceType ?? null,
    eventCount,
    latitude: geo?.latitude ?? null,
    longitude: geo?.longitude ?? null,
    geoTimezone: geo?.timezone ?? null,
    host: doc.enrichment?.host ?? null,
    source: doc.firstSource ?? null,
    landingPage: edges?.landingPage ?? null,
    landingTitle: edges?.landingTitle ?? null,
    lastPage: edges?.lastPage ?? null,
    platform: client?.platform ?? null,
    touchSupport: client?.touchSupport ?? null,
    deviceMemoryGb: client?.deviceMemoryGb ?? null,
    cpuCores: client?.cpuCores ?? null,
    networkType: client?.networkType ?? null,
    colorScheme: client?.colorScheme ?? null,
    languages: client?.languages && client.languages.length > 0 ? client.languages.join(', ') : null,
    acceptLanguage: doc.enrichment?.acceptLanguage ?? null,
    timezone: client?.timezone ?? null,
    screen: dimensions(client?.screenWidth, client?.screenHeight),
    viewport: dimensions(client?.viewportWidth, client?.viewportHeight),
    pixelRatio: client?.devicePixelRatio ?? null,
  };
}

function dimensions(width: number | undefined, height: number | undefined): string | null {
  if (!width || !height) return null;
  return `${width}×${height}`;
}
