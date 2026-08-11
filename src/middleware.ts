import { NextResponse, type NextRequest } from 'next/server';
import { ADMIN_COOKIE, verifySessionTokenEdge } from '@/lib/adminSessionEdge';

/**
 * Gate on the admin pages.
 *
 * Uses the Edge-safe verifier, since middleware cannot load node:crypto. The
 * API routes verify independently with the Node implementation; this exists so
 * an unauthenticated visitor is redirected rather than shown a dashboard shell
 * that then fails to load its data.
 */
export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
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
  matcher: ['/admin/:path*'],
};
