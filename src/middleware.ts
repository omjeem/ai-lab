import { NextResponse, type NextRequest, type NextFetchEvent } from 'next/server';
import { ADMIN_COOKIE, verifySessionTokenEdge } from '@/lib/adminSessionEdge';

const EXTERNAL_TRACK_URL = 'https://ommishra.tech/api/track/external';

/**
 * Gate on the admin pages, and record + strip the `source` attribution param.
 *
 * Uses the Edge-safe verifier, since middleware cannot load node:crypto. The
 * API routes verify independently with the Node implementation; this exists so
 * an unauthenticated visitor is redirected rather than shown a dashboard shell
 * that then fails to load its data.
 */
export async function middleware(request: NextRequest, event: NextFetchEvent) {
  const { pathname } = request.nextUrl;

  const source = request.nextUrl.searchParams.get('source');
  if (source) {
    const trackUrl = new URL(EXTERNAL_TRACK_URL);
    trackUrl.searchParams.set('source', source);
    trackUrl.searchParams.set('domain', request.nextUrl.hostname);
    // Fire-and-forget: waitUntil lets this complete after the redirect is sent,
    // without the response waiting on it.
    event.waitUntil(fetch(trackUrl).catch(() => {}));

    const cleanUrl = request.nextUrl.clone();
    cleanUrl.searchParams.delete('source');
    return NextResponse.redirect(cleanUrl);
  }

  if (!pathname.startsWith('/admin') || pathname.startsWith('/admin/login')) {
    return NextResponse.next();
  }

  const session = await verifySessionTokenEdge(request.cookies.get(ADMIN_COOKIE)?.value);
  if (session) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.pathname = '/admin/login';
  url.searchParams.set('next', pathname);
  return NextResponse.redirect(url);
}

export const config = {
  // All pages except API routes, Next internals, and static files (anything with an extension).
  matcher: ['/((?!api|_next/static|_next/image|.*\\..*).*)'],
};
