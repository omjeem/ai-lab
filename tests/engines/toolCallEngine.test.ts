import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  sampledDecode,
  buildSchemaPrompt,
  parseToolCall,
  buildToolSelectPrompt,
  extractToolName,
  type ToolCallConfig,
} from '@/engines/toolCallEngine';
import type { CausalLMDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

describe('buildSchemaPrompt', () => {
  it('includes zero worked examples by default', () => {
    const prompt = buildSchemaPrompt('What is 2 plus 2?', 0);
    expect(prompt).not.toContain('Question: What is 3 plus 4?');
    expect(prompt).toContain('What is 2 plus 2?');
    expect(prompt).toContain('"tool": "calculator"');
  });

  it('includes exactly N worked examples for exampleCount N', () => {
    const zero = buildSchemaPrompt('Q?', 0);
    const one = buildSchemaPrompt('Q?', 1);
    const two = buildSchemaPrompt('Q?', 2);
    const countShots = (s: string) => (s.match(/\nJSON: \{"tool": "calculator", "args"/g) ?? []).length;
    expect(countShots(zero)).toBe(0);
    expect(countShots(one)).toBe(1);
    expect(countShots(two)).toBe(2);
  });
});

describe('parseToolCall', () => {
  it('accepts a well-formed calculator call', () => {
    const result = parseToolCall('{"tool": "calculator", "args": {"expression": "3 + 4"}}');
    expect(result.valid).toBe(true);
    expect(result.tool).toBe('calculator');
    expect(result.expression).toBe('3 + 4');
  });

  it('accepts a call with extra keys, as long as the required shape is present', () => {
    const result = parseToolCall('{"tool": "calculator", "args": {"expression": "3", "operator": "+"}, "result": 7}');
    expect(result.valid).toBe(true);
  });

  it('rejects text with no JSON object at all', () => {
    expect(parseToolCall('sure, the answer is 7').valid).toBe(false);
  });

  it('rejects an unbalanced / truncated object', () => {
    expect(parseToolCall('{"tool": "calculator", "args": {"expression": "3"').valid).toBe(false);
  });

  it('rejects a schema placeholder echoed back literally (invalid JSON)', () => {
    expect(parseToolCall('{"tool": "calculator", "args": {"expression": string}}').valid).toBe(false);
  });

  it('rejects an object missing args.expression', () => {
    expect(parseToolCall('{"tool": "calculator", "args": {}}').valid).toBe(false);
  });

  it('extracts only the first balanced object, ignoring trailing text', () => {
    const result = parseToolCall('{"tool": "calculator", "args": {"expression": "3 + 4"}}\nQuestion: next one\nJSON:');
    expect(result.valid).toBe(true);
    expect(result.expression).toBe('3 + 4');
  });
});

describe('buildToolSelectPrompt', () => {
  it('lists the requested tool last', () => {
    const calcLast = buildToolSelectPrompt('What is 2 plus 2?', 'calculator');
    const lookupIdx = calcLast.indexOf('- lookup');
    const calcIdx = calcLast.indexOf('- calculator');
    expect(lookupIdx).toBeGreaterThanOrEqual(0);
    expect(calcIdx).toBeGreaterThan(lookupIdx);

    const lookupLast = buildToolSelectPrompt('What is 2 plus 2?', 'lookup');
    const lookupIdx2 = lookupLast.indexOf('- lookup');
    const calcIdx2 = lookupLast.indexOf('- calculator');
    expect(calcIdx2).toBeLessThan(lookupIdx2);
  });

  it('includes the instruction text', () => {
    expect(buildToolSelectPrompt('How deep is the trench?', 'lookup')).toContain('How deep is the trench?');
  });
});

describe('extractToolName', () => {
  it('reads the tool field out of a JSON-shaped response', () => {
    expect(extractToolName('{"tool": "calculator"}')).toBe('calculator');
    expect(extractToolName('  {\n "tool": "lookup" \n}')).toBe('lookup');
  });

  it('returns null when no tool field is present', () => {
    expect(extractToolName('no json here')).toBeNull();
  });
});

describe('greedyDecode', () => {
  it('accumulates real tokens until the model stops emitting', async () => {
    const tokens = ['{', '"', 'a', '}', '\n\n'];
    let i = 0;
    const fake: CausalLMDep = {
      async nextTokenDistribution() {
        const t = tokens[i++];
        return t ? { tokens: [t], probs: [1] } : { tokens: [], probs: [] };
      },
    };
    expect(await greedyDecode(fake, 'JSON:', 10)).toBe('{"a}\n\n');
  });

  it('respects the token budget', async () => {
    const fake: CausalLMDep = { async nextTokenDistribution() { return { tokens: ['x'], probs: [1] }; } };
    expect(await greedyDecode(fake, 'JSON:', 4)).toBe('xxxx');
  });
});

describe('sampledDecode', () => {
  it('is a real, seeded, repeatable draw from the model’s own top-k distribution', async () => {
    const fake: CausalLMDep = {
      async nextTokenDistribution(_prompt, topK) {
        return { tokens: ['a', 'b'].slice(0, topK), probs: [0.9, 0.1].slice(0, topK) };
      },
    };
    const a = await sampledDecode(fake, 'p', 3, { temperature: 1, topK: 2, seed: 42 });
    const b = await sampledDecode(fake, 'p', 3, { temperature: 1, topK: 2, seed: 42 });
    expect(a).toBe(b);
  });

  it('stops on an end-of-turn marker', async () => {
    const fake: CausalLMDep = { async nextTokenDistribution() { return { tokens: ['<|im_end|>'], probs: [1] }; } };
    expect(await sampledDecode(fake, 'p', 5, { temperature: 1, topK: 1, seed: 1 })).toBe('');
  });
});

const schemaRules: EngineRules = {
  passCriteria: { metric: 'jsonValidRate', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.8, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const selectRules: EngineRules = {
  passCriteria: { metric: 'toolPickAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.8, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 45,
};

const retryRules: EngineRules = {
  passCriteria: { metric: 'attemptsToSolve', threshold: 4, comparator: 'lte' },
  starsRules: [
    { threshold: 4, stars: 1 },
    { threshold: 2, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 50,
};

describe('toolCallEngine — schema-reliability mode', () => {
  const config: ToolCallConfig = {
    mode: 'schema-reliability',
    maxTokens: 40,
    temperature: 0.7,
    sampleTopK: 20,
    samplesPerQuestion: 2,
    questions: ['Q1?', 'Q2?'],
  };

  it('initialises with example count 0 and no recorded attempts', async () => {
    const prepared = await prepare(config);
    const state = initState(config, schemaRules, prepared);
    expect(state.exampleCount).toBe(0);
    expect(state.schemaResults).toHaveLength(0);
    expect(evaluate(state).value).toBe(0);
  });

  it('SET_EXAMPLE_COUNT changes the lever', async () => {
    const prepared = await prepare(config);
    let state = initState(config, schemaRules, prepared);
    state = applyAction(state, { type: 'SET_EXAMPLE_COUNT', count: 2 });
    expect(state.exampleCount).toBe(2);
  });

  it('RUN_SCHEMA_TEST records real results and scores the valid rate', async () => {
    const prepared = await prepare(config);
    let state = initState(config, schemaRules, prepared);
    state = applyAction(state, {
      type: 'RUN_SCHEMA_TEST',
      results: [
        { question: 'Q1?', text: '{"tool":"calculator","args":{"expression":"1+1"}}', valid: true },
        { question: 'Q1?', text: 'garbage', valid: false },
        { question: 'Q2?', text: '{"tool":"calculator","args":{"expression":"2+2"}}', valid: true },
        { question: 'Q2?', text: '{"tool":"calculator","args":{"expression":"2+2"}}', valid: true },
      ],
    });
    const result = evaluate(state);
    expect(result.metric).toBe('jsonValidRate');
    expect(result.value).toBeCloseTo(0.75);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(1);
  });

  it('a later RUN_SCHEMA_TEST replaces the previous results, not append', async () => {
    const prepared = await prepare(config);
    let state = initState(config, schemaRules, prepared);
    state = applyAction(state, { type: 'RUN_SCHEMA_TEST', results: [{ question: 'Q1?', text: 'x', valid: false }] });
    state = applyAction(state, {
      type: 'RUN_SCHEMA_TEST',
      results: [
        { question: 'Q1?', text: 'ok', valid: true },
        { question: 'Q2?', text: 'ok', valid: true },
      ],
    });
    expect(state.schemaResults).toHaveLength(2);
    expect(evaluate(state).value).toBe(1);
  });
});

describe('toolCallEngine — tool-select mode', () => {
  const config: ToolCallConfig = {
    mode: 'tool-select',
    maxTokens: 20,
    instructions: [
      { text: 'What is 2 plus 2?', expectedTool: 'calculator' },
      { text: 'How deep is the ocean?', expectedTool: 'lookup' },
    ],
  };

  it('builds one round per configured instruction, untested', async () => {
    const prepared = await prepare(config);
    const state = initState(config, selectRules, prepared);
    expect(state.selectRounds).toHaveLength(2);
    expect(state.selectRounds.every((r) => !r.tested)).toBe(true);
    expect(state.selectRounds.every((r) => r.lastOrder === null)).toBe(true);
  });

  it('SET_TOOL_ORDER records which tool the player placed last, per round', async () => {
    const prepared = await prepare(config);
    let state = initState(config, selectRules, prepared);
    state = applyAction(state, { type: 'SET_TOOL_ORDER', roundIndex: 0, lastTool: 'calculator' });
    expect(state.selectRounds[0]!.lastOrder).toBe('calculator');
    expect(state.selectRounds[1]!.lastOrder).toBeNull();
  });

  it('TEST_TOOL_SELECT records the real picked tool and scores accuracy', async () => {
    const prepared = await prepare(config);
    let state = initState(config, selectRules, prepared);
    state = applyAction(state, { type: 'TEST_TOOL_SELECT', roundIndex: 0, pickedTool: 'calculator' });
    state = applyAction(state, { type: 'TEST_TOOL_SELECT', roundIndex: 1, pickedTool: 'calculator' });

    const result = evaluate(state);
    expect(result.metric).toBe('toolPickAccuracy');
    expect(result.value).toBeCloseTo(0.5);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(1);
  });

  it('scores full accuracy when every real pick matches the expected tool', async () => {
    const prepared = await prepare(config);
    let state = initState(config, selectRules, prepared);
    state = applyAction(state, { type: 'TEST_TOOL_SELECT', roundIndex: 0, pickedTool: 'calculator' });
    state = applyAction(state, { type: 'TEST_TOOL_SELECT', roundIndex: 1, pickedTool: 'lookup' });

    const result = evaluate(state);
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
  });

  it('an untested round never counts as correct', async () => {
    const prepared = await prepare(config);
    let state = initState(config, selectRules, prepared);
    state = applyAction(state, { type: 'TEST_TOOL_SELECT', roundIndex: 0, pickedTool: 'calculator' });
    expect(evaluate(state).value).toBeCloseTo(0.5);
  });
});

describe('toolCallEngine — retry-fix mode', () => {
  const config: ToolCallConfig = { mode: 'retry-fix', maxTokens: 40, question: 'What is 9 times 6?' };

  it('starts unsolved with zero attempts', async () => {
    const prepared = await prepare(config);
    const state = initState(config, retryRules, prepared);
    expect(state.retryAttempts).toBe(0);
    expect(state.retrySolved).toBe(false);
    expect(evaluate(state).passed).toBe(false);
  });

  it('TEST_RETRY records an attempt and solves once a real generation validates', async () => {
    const prepared = await prepare(config);
    let state = initState(config, retryRules, prepared);
    state = applyAction(state, { type: 'TEST_RETRY', text: '{"tool": "calculator", "args": {"expression": string}}' });
    expect(state.retryAttempts).toBe(1);
    expect(state.retrySolved).toBe(false);

    state = applyAction(state, { type: 'TEST_RETRY', text: '{"tool": "calculator", "args": {"expression": "9 * 6"}}' });
    expect(state.retryAttempts).toBe(2);
    expect(state.retrySolved).toBe(true);
    expect(state.retrySolvedAtAttempt).toBe(2);

    const result = evaluate(state);
    expect(result.metric).toBe('attemptsToSolve');
    expect(result.value).toBe(2);
    expect(result.stars).toBe(2);
  });

  it('does not overwrite an earlier solve on a later attempt', async () => {
    const prepared = await prepare(config);
    let state = initState(config, retryRules, prepared);
    state = applyAction(state, { type: 'TEST_RETRY', text: '{"tool": "calculator", "args": {"expression": "9 * 6"}}' });
    state = applyAction(state, { type: 'TEST_RETRY', text: '{"tool": "calculator", "args": {"expression": "9 * 6"}}' });
    expect(state.retrySolvedAtAttempt).toBe(1);
  });
});

describe('toolCallEngine — SET_EXAMPLE_COUNT applies to schema-reliability and retry-fix only', () => {
  it('is a no-op outside those modes but never throws', async () => {
    const config: ToolCallConfig = {
      mode: 'tool-select',
      maxTokens: 20,
      instructions: [{ text: 'Q', expectedTool: 'calculator' }],
    };
    const prepared = await prepare(config);
    const state = initState(config, selectRules, prepared);
    expect(() => applyAction(state, { type: 'SET_EXAMPLE_COUNT', count: 1 })).not.toThrow();
  });
});

describe('toolCallEngine — RESET', () => {
  it('restores a fresh state for the same config without needing new model calls', async () => {
    const config: ToolCallConfig = { mode: 'retry-fix', maxTokens: 40, question: 'Q?' };
    const prepared = await prepare(config);
    let state = initState(config, retryRules, prepared);
    state = applyAction(state, { type: 'TEST_RETRY', text: '{"tool": "calculator", "args": {"expression": "1"}}' });
    state = applyAction(state, { type: 'RESET' });
    expect(state.retryAttempts).toBe(0);
    expect(state.retrySolved).toBe(false);
  });
});

describe('toolCallEngine — SUBMIT', () => {
  it('marks the run complete', async () => {
    const config: ToolCallConfig = { mode: 'retry-fix', maxTokens: 40, question: 'Q?' };
    const prepared = await prepare(config);
    let state = initState(config, retryRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
