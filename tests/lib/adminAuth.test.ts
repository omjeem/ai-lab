import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  hashPassword,
  verifyPassword,
  createSessionToken,
  verifySessionToken,
  checkCredentials,
  isAdminConfigured,
  sessionCookieOptions,
} from '@/lib/adminAuth';

const ORIGINAL = { ...process.env };

beforeEach(() => {
  process.env.ADMIN_SESSION_SECRET = 'a'.repeat(64);
  process.env.ADMIN_EMAIL = 'admin@example.com';
  process.env.ADMIN_PASSWORD_HASH = hashPassword('correct-horse-battery');
});

afterEach(() => {
  process.env = { ...ORIGINAL };
});

describe('password hashing', () => {
  it('produces the documented scrypt format', () => {
    const hash = hashPassword('some-password');
    expect(hash.split('$')).toHaveLength(3);
    expect(hash.startsWith('scrypt$')).toBe(true);
  });

  it('salts, so the same password hashes differently each time', () => {
    expect(hashPassword('same')).not.toBe(hashPassword('same'));
  });

  it('verifies the right password and rejects the wrong one', () => {
    const hash = hashPassword('correct-horse-battery');
    expect(verifyPassword('correct-horse-battery', hash)).toBe(true);
    expect(verifyPassword('wrong', hash)).toBe(false);
  });

  it('rejects a malformed stored hash rather than throwing', () => {
    expect(verifyPassword('x', 'not-a-hash')).toBe(false);
    expect(verifyPassword('x', 'md5$salt$hash')).toBe(false);
    expect(verifyPassword('x', '')).toBe(false);
  });

  it('never stores the plaintext anywhere in the hash', () => {
    expect(hashPassword('hunter2-hunter2')).not.toContain('hunter2');
  });
});

describe('session tokens', () => {
  it('round-trips a valid session', () => {
    const token = createSessionToken('admin@example.com')!;
    const session = verifySessionToken(token);
    expect(session?.email).toBe('admin@example.com');
    expect(session?.expiresAt).toBeGreaterThan(Date.now());
  });

  it('rejects a tampered payload', () => {
    const token = createSessionToken('admin@example.com')!;
    const [, signature] = token.split('.');
    const forged = Buffer.from(
      JSON.stringify({ email: 'attacker@evil.com', issuedAt: Date.now(), expiresAt: Date.now() + 1e6 })
    ).toString('base64url');

    expect(verifySessionToken(`${forged}.${signature}`)).toBeNull();
  });

  it('rejects a tampered signature', () => {
    const token = createSessionToken('admin@example.com')!;
    const [payload] = token.split('.');
    expect(verifySessionToken(`${payload}.badsignature`)).toBeNull();
  });

  it('rejects a token signed with a different secret', () => {
    const token = createSessionToken('admin@example.com')!;
    process.env.ADMIN_SESSION_SECRET = 'b'.repeat(64);
    expect(verifySessionToken(token)).toBeNull();
  });

  it('rejects an expired session', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    process.env.ADMIN_SESSION_HOURS = '1';
    const token = createSessionToken('admin@example.com')!;
    expect(verifySessionToken(token)).not.toBeNull();

    // Past the configured lifetime, the same token must stop working.
    vi.setSystemTime(new Date('2026-01-01T02:00:00Z'));
    expect(verifySessionToken(token)).toBeNull();
    vi.useRealTimers();
  });

  it('rejects missing, empty and malformed tokens', () => {
    expect(verifySessionToken(undefined)).toBeNull();
    expect(verifySessionToken(null)).toBeNull();
    expect(verifySessionToken('')).toBeNull();
    expect(verifySessionToken('nodot')).toBeNull();
    expect(verifySessionToken('not-base64.signature')).toBeNull();
  });

  it('issues nothing when no secret is configured', () => {
    delete process.env.ADMIN_SESSION_SECRET;
    expect(createSessionToken('admin@example.com')).toBeNull();
    expect(verifySessionToken('anything.atall')).toBeNull();
  });
});

describe('credential checking', () => {
  it('accepts the configured credentials', () => {
    expect(checkCredentials('admin@example.com', 'correct-horse-battery')).toBe(true);
  });

  it('is case-insensitive on the email and tolerates surrounding space', () => {
    expect(checkCredentials('  ADMIN@Example.com ', 'correct-horse-battery')).toBe(true);
  });

  it('rejects a wrong password and a wrong email', () => {
    expect(checkCredentials('admin@example.com', 'nope')).toBe(false);
    expect(checkCredentials('someone@else.com', 'correct-horse-battery')).toBe(false);
  });

  it('rejects everything when nothing is configured', () => {
    delete process.env.ADMIN_EMAIL;
    delete process.env.ADMIN_PASSWORD_HASH;
    expect(checkCredentials('admin@example.com', 'correct-horse-battery')).toBe(false);
    expect(isAdminConfigured()).toBe(false);
  });

  it('reports configured only when all three variables are present', () => {
    expect(isAdminConfigured()).toBe(true);
    delete process.env.ADMIN_SESSION_SECRET;
    expect(isAdminConfigured()).toBe(false);
  });
});

describe('cookie options', () => {
  it('is httpOnly and same-site, so it cannot be read from script', () => {
    const options = sessionCookieOptions();
    expect(options.httpOnly).toBe(true);
    expect(options.sameSite).toBe('lax');
    expect(options.path).toBe('/');
    expect(options.maxAge).toBeGreaterThan(0);
  });

  it('respects the configured lifetime', () => {
    process.env.ADMIN_SESSION_HOURS = '2';
    expect(sessionCookieOptions().maxAge).toBe(7200);
  });

  it('falls back to the default for a nonsense lifetime', () => {
    process.env.ADMIN_SESSION_HOURS = 'not-a-number';
    expect(sessionCookieOptions().maxAge).toBe(12 * 3600);
  });
});
