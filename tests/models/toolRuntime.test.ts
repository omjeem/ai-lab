import { describe, it, expect } from 'vitest';
import { findMatchingFact, toolRuntime, type RetrievalFact } from '@/models/toolRuntime';

// evaluateArithmetic itself is pure, shared math and lives in
// tests/engines/shared.test.ts — this file only covers what's specific to
// toolRuntime: the ToolDep wrapper and the corpus-matching logic.

describe('toolRuntime — calculator', () => {
  it('returns a real evaluated result', async () => {
    const result = await toolRuntime.run('calculator', { expression: '47 + 89' });
    expect(result).toEqual({ ok: true, output: '136' });
  });

  it('fails honestly on a bad expression rather than throwing', async () => {
    const result = await toolRuntime.run('calculator', { expression: 'not math' });
    expect(result.ok).toBe(false);
  });

  it('fails honestly when args.expression is missing', async () => {
    const result = await toolRuntime.run('calculator', {});
    expect(result.ok).toBe(false);
  });
});

describe('toolRuntime — lookup', () => {
  it('fails honestly when args.topic is missing', async () => {
    const result = await toolRuntime.run('lookup', {});
    expect(result.ok).toBe(false);
  });
});

describe('findMatchingFact', () => {
  const facts: RetrievalFact[] = [
    { id: 'mariana', topic: 'Mariana Trench', sentences: ['It reaches 10,935 meters deep.'], query: 'How deep is the Mariana Trench?', answer: '10,935' },
    { id: 'burj', topic: 'Burj Khalifa', sentences: ['It stands 828 meters tall.'], query: 'How tall is the Burj Khalifa?', answer: '828' },
  ];

  it('finds the fact whose topic/query best overlaps the given words', () => {
    const match = findMatchingFact(facts, 'Mariana Trench depth');
    expect(match?.id).toBe('mariana');
  });

  it('is case-insensitive', () => {
    expect(findMatchingFact(facts, 'MARIANA trench')?.id).toBe('mariana');
  });

  it('returns null when nothing overlaps', () => {
    expect(findMatchingFact(facts, 'zzz nonsense topic zzz')).toBeNull();
  });
});

describe('toolRuntime — unknown tool', () => {
  it('fails honestly rather than throwing', async () => {
    const result = await toolRuntime.run('teleporter', {});
    expect(result.ok).toBe(false);
  });
});
