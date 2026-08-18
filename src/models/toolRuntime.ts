/**
 * Real tools World 7.4's agent loop can call — deterministic, offline-safe,
 * no model or network involved. `calculator` evaluates real arithmetic with
 * `evaluateArithmetic` (never `eval`/`new Function` on model-generated text);
 * `lookup` does real keyword matching against the same bundled fact corpus
 * 7.1/7.2 already use, returning "no match" honestly rather than inventing an
 * answer when nothing fits.
 */
import type { ToolDep, ToolResult } from '@/engines/deps';
import { evaluateArithmetic } from '@/engines/shared';
import { corpusLoader } from './corpusLoader';

export { evaluateArithmetic };

export interface RetrievalFact {
  id: string;
  topic: string;
  sentences: string[];
  query: string;
  answer: string;
}

async function runCalculator(args: Record<string, unknown>): Promise<ToolResult> {
  const expression = args.expression;
  if (typeof expression !== 'string') {
    return { ok: false, output: 'calculator needs a string "expression" argument' };
  }
  const result = evaluateArithmetic(expression);
  return result === null
    ? { ok: false, output: `could not evaluate "${expression}"` }
    : { ok: true, output: String(result) };
}

/** Pure keyword-overlap match — no I/O, so it's directly testable against a
 * fixture corpus rather than only through a real `fetch`. */
export function findMatchingFact(facts: readonly RetrievalFact[], topic: string): RetrievalFact | null {
  const words = topic.toLowerCase().split(/\W+/).filter(Boolean);
  let best: RetrievalFact | null = null;
  let bestScore = 0;
  for (const fact of facts) {
    const haystack = `${fact.topic} ${fact.query}`.toLowerCase();
    const score = words.filter((w) => haystack.includes(w)).length;
    if (score > bestScore) {
      bestScore = score;
      best = fact;
    }
  }
  return best;
}

/** Real keyword overlap against the bundled fact corpus — no fabricated match. */
async function runLookup(args: Record<string, unknown>): Promise<ToolResult> {
  const topic = args.topic;
  if (typeof topic !== 'string') {
    return { ok: false, output: 'lookup needs a string "topic" argument' };
  }
  const raw = await corpusLoader.load('retrieval-facts');
  const { facts } = JSON.parse(raw) as { facts: RetrievalFact[] };

  const match = findMatchingFact(facts, topic);
  return match
    ? { ok: true, output: match.sentences.join(' ') }
    : { ok: false, output: 'No matching fact found in the reference corpus.' };
}

export const toolRuntime: ToolDep = {
  async run(toolName, args) {
    switch (toolName) {
      case 'calculator':
        return runCalculator(args);
      case 'lookup':
        return runLookup(args);
      default:
        return { ok: false, output: `unknown tool "${toolName}"` };
    }
  },
};
