import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import {
  ADMIN_COOKIE,
  sessionCookieOptions,
  createSessionToken,
  verifySessionToken,
  type AdminSession,
} from './adminAuth';

/**
 * Re-verifies the session inside the route itself.
 *
 * Section 10 asks for defence in depth: middleware already redirects the pages,
 * but every `/api/admin/*` handler checks again rather than trusting that
 * reaching the route means something upstream approved it.
 */
export async function requireAdmin(): Promise<
  { ok: true; session: AdminSession } | { ok: false; response: Response }
> {
  const store = await cookies();
  const session = verifySessionToken(store.get(ADMIN_COOKIE)?.value);

  if (!session) {
    return {
      ok: false,
      response: NextResponse.json({ ok: false, error: 'Unauthorized' }, { status: 401 }),
    };
  }
  return { ok: true, session };
}

/** Slides the expiry forward, so an active admin is not logged out mid-task. */
export function refreshSession(response: NextResponse, session: AdminSession): NextResponse {
  const token = createSessionToken(session.email);
  if (token) response.cookies.set(ADMIN_COOKIE, token, sessionCookieOptions());
  return response;
}
