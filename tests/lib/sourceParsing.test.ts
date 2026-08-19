import { describe, it, expect } from 'vitest';
import { deriveSource } from '@/lib/sourceParsing';

describe('deriveSource', () => {
  it('calls it direct when there is no referrer at all', () => {
    expect(deriveSource(null, 'ailab.example.com')).toBe('Direct · direct');
  });

  it('calls it direct when the referrer is not a parseable URL', () => {
    expect(deriveSource('not a url', 'ailab.example.com')).toBe('Direct · direct');
  });

  it('calls it internal when the referrer host matches the request host', () => {
    expect(deriveSource('https://ailab.example.com/map', 'ailab.example.com')).toBe(
      'Internal · internal'
    );
  });

  it('ignores the port when comparing referrer host to request host', () => {
    expect(deriveSource('http://localhost:3100/onboarding', 'localhost:3100')).toBe(
      'Internal · internal'
    );
  });

  it('reports the external domain when the referrer is a different site', () => {
    expect(deriveSource('https://www.google.com/search?q=ai+learning', 'ailab.example.com')).toBe(
      'www.google.com · external'
    );
  });
});
