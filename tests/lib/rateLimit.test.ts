import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { rateLimit, clientIpFrom, readLimit, resetRateLimits } from '@/lib/rateLimit';

beforeEach(() => resetRateLimits());
afterEach(() => vi.useRealTimers());

describe('rateLimit', () => {
  it('allows requests up to the limit', () => {
    for (let i = 0; i < 5; i++) {
      expect(rateLimit('key', 5).allowed).toBe(true);
    }
  });

  it('blocks the request past the limit', () => {
    for (let i = 0; i < 5; i++) rateLimit('key', 5);
    const result = rateLimit('key', 5);
    expect(result.allowed).toBe(false);
    expect(result.remaining).toBe(0);
    expect(result.retryAfterSeconds).toBeGreaterThan(0);
  });

  it('counts each key separately', () => {
    for (let i = 0; i < 5; i++) rateLimit('a', 5);
    expect(rateLimit('a', 5).allowed).toBe(false);
    expect(rateLimit('b', 5).allowed).toBe(true);
  });

  it('reports the remaining allowance', () => {
    expect(rateLimit('key', 3).remaining).toBe(2);
    expect(rateLimit('key', 3).remaining).toBe(1);
    expect(rateLimit('key', 3).remaining).toBe(0);
  });

  it('lets requests through again once the window slides past', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));

    for (let i = 0; i < 3; i++) rateLimit('key', 3);
    expect(rateLimit('key', 3).allowed).toBe(false);

    vi.setSystemTime(new Date('2026-01-01T00:01:01Z'));
    expect(rateLimit('key', 3).allowed).toBe(true);
  });

  it('reports a retry delay inside the window length', () => {
    for (let i = 0; i < 2; i++) rateLimit('key', 2);
    const { retryAfterSeconds } = rateLimit('key', 2);
    expect(retryAfterSeconds).toBeGreaterThan(0);
    expect(retryAfterSeconds).toBeLessThanOrEqual(60);
  });
});

describe('clientIpFrom', () => {
  it('prefers the leftmost x-forwarded-for entry, the original client', () => {
    const headers = new Headers({ 'x-forwarded-for': '203.0.113.7, 70.41.3.18, 150.172.238.178' });
    expect(clientIpFrom(headers)).toBe('203.0.113.7');
  });

  it('trims surrounding whitespace', () => {
    expect(clientIpFrom(new Headers({ 'x-forwarded-for': '  203.0.113.7 ' }))).toBe('203.0.113.7');
  });

  it('falls back through the other proxy headers', () => {
    expect(clientIpFrom(new Headers({ 'cf-connecting-ip': '198.51.100.4' }))).toBe('198.51.100.4');
    expect(clientIpFrom(new Headers({ 'x-real-ip': '198.51.100.5' }))).toBe('198.51.100.5');
  });

  it('returns null when no header identifies the client', () => {
    expect(clientIpFrom(new Headers())).toBeNull();
  });

  it('ignores an empty forwarded-for value', () => {
    expect(clientIpFrom(new Headers({ 'x-forwarded-for': '' }))).toBeNull();
  });
});

describe('readLimit', () => {
  const original = { ...process.env };
  afterEach(() => {
    process.env = { ...original };
  });

  it('reads a configured limit', () => {
    process.env.TEST_LIMIT = '25';
    expect(readLimit('TEST_LIMIT', 10)).toBe(25);
  });

  it('falls back when unset, non-numeric or non-positive', () => {
    delete process.env.TEST_LIMIT;
    expect(readLimit('TEST_LIMIT', 10)).toBe(10);
    process.env.TEST_LIMIT = 'lots';
    expect(readLimit('TEST_LIMIT', 10)).toBe(10);
    process.env.TEST_LIMIT = '0';
    expect(readLimit('TEST_LIMIT', 10)).toBe(10);
    process.env.TEST_LIMIT = '-5';
    expect(readLimit('TEST_LIMIT', 10)).toBe(10);
  });
});
