/**
 * Admin session handling.
 *
 * Credentials come from env vars, the password is only ever stored as a scrypt
 * hash, and the session is a signed cookie this app verifies itself. No
 * third-party auth provider, per Section 2.
 */
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from 'node:crypto';

export const ADMIN_COOKIE = 'ail_admin';

const DEFAULT_SESSION_HOURS = 12;
const SCRYPT_KEYLEN = 64;

/* ── password hashing ───────────────────────────────────────── */

/** `scrypt$<saltHex>$<hashHex>` — the format `pnpm hash:password` emits. */
export function hashPassword(password: string, salt = randomBytes(16).toString('hex')): string {
  const derived = scryptSync(password, salt, SCRYPT_KEYLEN).toString('hex');
  return `scrypt$${salt}$${derived}`;
}

export function verifyPassword(password: string, stored: string): boolean {
  const [scheme, salt, expected] = stored.split('$');
  if (scheme !== 'scrypt' || !salt || !expected) return false;

  const derived = scryptSync(password, salt, SCRYPT_KEYLEN).toString('hex');
  const a = Buffer.from(derived, 'hex');
  const b = Buffer.from(expected, 'hex');
  if (a.length !== b.length) return false;
  // Constant-time, so a wrong password cannot be narrowed down by timing.
  return timingSafeEqual(a, b);
}

/* ── session cookie ─────────────────────────────────────────── */

export interface AdminSession {
  email: string;
  issuedAt: number;
  expiresAt: number;
}

function sessionSecret(): string | null {
  return process.env.ADMIN_SESSION_SECRET || null;
}

function sign(payload: string, secret: string): string {
  return createHmac('sha256', secret).update(payload).digest('base64url');
}

export function sessionLifetimeMs(): number {
  const hours = Number(process.env.ADMIN_SESSION_HOURS ?? DEFAULT_SESSION_HOURS);
  return (Number.isFinite(hours) && hours > 0 ? hours : DEFAULT_SESSION_HOURS) * 3600_000;
}

export function createSessionToken(email: string): string | null {
  const secret = sessionSecret();
  if (!secret) return null;

  const now = Date.now();
  const session: AdminSession = { email, issuedAt: now, expiresAt: now + sessionLifetimeMs() };
  const payload = Buffer.from(JSON.stringify(session)).toString('base64url');
  return `${payload}.${sign(payload, secret)}`;
}

/**
 * Verifies a token's signature and expiry.
 *
 * Returns null for anything even slightly wrong — this is the only gate in front
 * of every admin read.
 */
export function verifySessionToken(token: string | undefined | null): AdminSession | null {
  const secret = sessionSecret();
  if (!secret || !token) return null;

  const [payload, signature] = token.split('.');
  if (!payload || !signature) return null;

  const expected = sign(payload, secret);
  const a = Buffer.from(signature);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !timingSafeEqual(a, b)) return null;

  try {
    const session = JSON.parse(Buffer.from(payload, 'base64url').toString()) as AdminSession;
    if (typeof session.expiresAt !== 'number' || session.expiresAt < Date.now()) return null;
    return session;
  } catch {
    return null;
  }
}

/** True when the credentials in the environment are usable at all. */
export function isAdminConfigured(): boolean {
  return Boolean(
    process.env.ADMIN_EMAIL && process.env.ADMIN_PASSWORD_HASH && process.env.ADMIN_SESSION_SECRET
  );
}

export function checkCredentials(email: string, password: string): boolean {
  const expectedEmail = process.env.ADMIN_EMAIL;
  const expectedHash = process.env.ADMIN_PASSWORD_HASH;
  if (!expectedEmail || !expectedHash) return false;

  // Compare both, always, so a wrong email and a wrong password cost the same.
  const emailMatches = email.trim().toLowerCase() === expectedEmail.trim().toLowerCase();
  const passwordMatches = verifyPassword(password, expectedHash);
  return emailMatches && passwordMatches;
}

export function sessionCookieOptions(maxAgeMs = sessionLifetimeMs()) {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax' as const,
    path: '/',
    maxAge: Math.floor(maxAgeMs / 1000),
  };
}
