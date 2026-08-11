/**
 * Admin login and logout.
 *
 * On success sets a signed, httpOnly session cookie. Rate limited by IP, since
 * this is a public endpoint guarding every admin read.
 */
import { NextResponse } from 'next/server';
import { z } from 'zod';
import {
  ADMIN_COOKIE,
  checkCredentials,
  createSessionToken,
  isAdminConfigured,
  sessionCookieOptions,
} from '@/lib/adminAuth';
import { clientIpFrom, rateLimit } from '@/lib/rateLimit';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const loginSchema = z.object({
  email: z.string().min(1).max(200),
  password: z.string().min(1).max(400),
});

export async function POST(request: Request): Promise<Response> {
  const ip = clientIpFrom(request.headers) ?? 'unknown';
  // Deliberately tight — this is a credential check, not gameplay traffic.
  const limit = rateLimit(`admin-login:${ip}`, 10);
  if (!limit.allowed) {
    return NextResponse.json(
      { ok: false, error: 'Too many attempts' },
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } }
    );
  }

  if (!isAdminConfigured()) {
    return NextResponse.json(
      { ok: false, error: 'Admin access is not configured on this deployment' },
      { status: 503 }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid request' }, { status: 400 });
  }

  const parsed = loginSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ ok: false, error: 'Invalid request' }, { status: 400 });
  }

  if (!checkCredentials(parsed.data.email, parsed.data.password)) {
    // One message for both failure modes, so it cannot be used to enumerate.
    return NextResponse.json({ ok: false, error: 'Invalid credentials' }, { status: 401 });
  }

  const token = createSessionToken(parsed.data.email);
  if (!token) {
    return NextResponse.json({ ok: false, error: 'Session secret missing' }, { status: 503 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(ADMIN_COOKIE, token, sessionCookieOptions());
  return response;
}

export async function DELETE(): Promise<Response> {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(ADMIN_COOKIE, '', { ...sessionCookieOptions(0), maxAge: 0 });
  return response;
}
