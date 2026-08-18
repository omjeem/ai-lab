import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  sampledDecode,
  buildSingleHopPrompt,
  buildToolSelectHopPrompt,
  buildFinalAnswerPrompt,
  parseAgentCall,
  containsAnswer,
  type AgentLoopConfig,
} from '@/engines/agentLoopEngine';
import type { CausalLMDep, CorpusDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id !== 'retrieval-facts') throw new Error(`unexpected corpus ${id}`);
    return JSON.stringify({
      facts: [
        { id: 'mariana', topic: 'Mariana Trench', sentences: ['It reaches 10,935 meters deep.'], query: 'How deep is the Mariana Trench?', answer: '10,935' },
        { id: 'burj', topic: 'Burj Khalifa', sentences: ['It stands 828 meters tall.'], query: 'How tall is the Burj Khalifa?', answer: '828' },
      ],
    });
  },
};

describe('buildSingleHopPrompt', () => {
  it('includes exactly N worked examples for exampleCount N', () => {
    const zero = buildSingleHopPrompt('Q?', 0);
    const one = buildSingleHopPrompt('Q?', 1);
    const two = buildSingleHopPrompt('Q?', 2);
    const countShots = (s: string) => (s.match(/\nTool result: /g) ?? []).length;
    expect(countShots(zero)).toBe(0);
    expect(countShots(one)).toBe(1);
    expect(countShots(two)).toBe(2);
  });

  it('includes the question', () => {
    expect(buildSingleHopPrompt('What is 2 plus 2?', 0)).toContain('What is 2 plus 2?');
  });
});

describe('buildToolSelectHopPrompt', () => {
  it('lists and demonstrates the requested tool last', () => {
    const calcLast = buildToolSelectHopPrompt('What is 2 plus 2?', 'calculator');
    expect(calcLast.indexOf('- calculator')).toBeGreaterThan(calcLast.indexOf('- lookup'));
    expect(calcLast).toContain('"tool": "calculator"');

    const lookupLast = buildToolSelectHopPrompt('What is 2 plus 2?', 'lookup');
    expect(lookupLast.indexOf('- lookup')).toBeGreaterThan(lookupLast.indexOf('- calculator'));
    expect(lookupLast).toContain('"tool": "lookup"');
  });
});

describe('parseAgentCall', () => {
  it('extracts a well-formed calculator call', () => {
    const result = parseAgentCall('{"tool": "calculator", "args": {"expression": "3 + 4"}}');
    expect(result?.tool).toBe('calculator');
    expect(result?.args.expression).toBe('3 + 4');
    expect(result?.cleanJson).toBe('{"tool": "calculator", "args": {"expression": "3 + 4"}}');
  });

  it('extracts a well-formed lookup call', () => {
    const result = parseAgentCall('{"tool": "lookup", "args": {"topic": "Everest height"}}');
    expect(result?.tool).toBe('lookup');
    expect(result?.args.topic).toBe('Everest height');
  });

  it('ignores trailing text after the first balanced object', () => {
    const result = parseAgentCall('{"tool": "calculator", "args": {"expression": "1+1"}}\nQuestion: next\nJSON:');
    expect(result?.cleanJson).toBe('{"tool": "calculator", "args": {"expression": "1+1"}}');
  });

  it('returns null for text with no JSON object', () => {
    expect(parseAgentCall('sure, the answer is 7')).toBeNull();
  });

  it('returns null for an unbalanced object', () => {
    expect(parseAgentCall('{"tool": "calculator"')).toBeNull();
  });

  it('returns null when tool is missing', () => {
    expect(parseAgentCall('{"args": {"expression": "1+1"}}')).toBeNull();
  });
});

describe('buildFinalAnswerPrompt', () => {
  it('appends the clean tool call and real tool result', () => {
    const base = 'Question: What is 3 plus 4?\nJSON:';
    const prompt = buildFinalAnswerPrompt(base, '{"tool": "calculator", "args": {"expression": "3 + 4"}}', '7');
    expect(prompt).toBe(
      'Question: What is 3 plus 4?\nJSON:{"tool": "calculator", "args": {"expression": "3 + 4"}}\nTool result: 7\nFinal answer:'
    );
  });
});

describe('containsAnswer', () => {
  it('matches regardless of comma formatting or case', () => {
    expect(containsAnswer(' The answer is 8,849 meters', '8,849')).toBe(true);
    expect(containsAnswer(' the answer is 8849 meters', '8,849')).toBe(true);
    expect(containsAnswer(' nothing useful', '8,849')).toBe(false);
  });
});

describe('greedyDecode', () => {
  it('accumulates real tokens until the model stops emitting', async () => {
    const tokens = ['8', '7', '9', '8', '\n\n'];
    let i = 0;
    const fake: CausalLMDep = {
      async nextTokenDistribution() {
        const t = tokens[i++];
        return t ? { tokens: [t], probs: [1] } : { tokens: [], probs: [] };
      },
    };
    expect(await greedyDecode(fake, 'Final answer:', 10)).toBe('8798\n\n');
  });
});

describe('sampledDecode', () => {
  it('is a real, seeded, repeatable draw from the model’s own top-k distribution', async () => {
    const fake: CausalLMDep = {
      async nextTokenDistribution(_prompt, topK) {
        return { tokens: ['a', 'b'].slice(0, topK), probs: [0.9, 0.1].slice(0, topK) };
      },
    };
    const a = await sampledDecode(fake, 'p', 3, { temperature: 1, topK: 2, seed: 7 });
    const b = await sampledDecode(fake, 'p', 3, { temperature: 1, topK: 2, seed: 7 });
    expect(a).toBe(b);
  });
});

const hopRules: EngineRules = {
  passCriteria: { metric: 'toolHopAccuracy', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 40,
};

const selectHopRules: EngineRules = {
  passCriteria: { metric: 'agentTaskAccuracy', threshold: 0.4, comparator: 'gte' },
  starsRules: [
    { threshold: 0.4, stars: 1 },
    { threshold: 0.6, stars: 2 },
    { threshold: 0.8, stars: 3 },
  ],
  xpReward: 45,
};

const budgetRules: EngineRules = {
  passCriteria: { metric: 'tasksSolvedRate', threshold: 0.4, comparator: 'gte' },
  starsRules: [
    { threshold: 0.4, stars: 1 },
    { threshold: 0.6, stars: 2 },
    { threshold: 0.9, stars: 3 },
  ],
  xpReward: 50,
};

describe('agentLoopEngine — single-hop mode', () => {
  const config: AgentLoopConfig = {
    mode: 'single-hop',
    maxTokens: 40,
    finalAnswerMaxTokens: 16,
    tasks: [
      { question: 'What is 3 plus 4?', canonicalExpression: '3 + 4' },
      { question: 'What is 20 minus 6?', canonicalExpression: '20 - 6' },
    ],
  };

  it('builds one round per task, with the real expected answer computed at runtime', async () => {
    const prepared = await prepare(config, {});
    expect(prepared.hopRounds).toHaveLength(2);
    expect(prepared.hopRounds[0]!.expectedAnswer).toBe('7');
    expect(prepared.hopRounds[1]!.expectedAnswer).toBe('14');
  });

  it('starts with example count 0 and no rounds tested', async () => {
    const prepared = await prepare(config, {});
    const state = initState(config, hopRules, prepared);
    expect(state.exampleCount).toBe(0);
    expect(state.hopRounds.every((r) => !r.tested)).toBe(true);
    expect(evaluate(state).value).toBe(0);
  });

  it('TEST_HOP scores correct once the real final answer contains the real expected value', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, hopRules, prepared);
    state = applyAction(state, {
      type: 'TEST_HOP',
      roundIndex: 0,
      toolCallText: '{"tool": "calculator", "args": {"expression": "3 + 4"}}',
      toolResult: '7',
      finalAnswerText: ' 7\n\n',
    });
    state = applyAction(state, {
      type: 'TEST_HOP',
      roundIndex: 1,
      toolCallText: '{"tool": "calculator", "args": {"expression": "20 - 6"}}',
      toolResult: '14',
      finalAnswerText: ' 14\n\n',
    });

    const result = evaluate(state);
    expect(result.metric).toBe('toolHopAccuracy');
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
  });

  it('scores wrong when the real final answer does not contain the real expected value', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, hopRules, prepared);
    state = applyAction(state, {
      type: 'TEST_HOP',
      roundIndex: 0,
      toolCallText: '{"tool": "calculator", "args": {"expression": "3 + 4"}}',
      toolResult: '7',
      finalAnswerText: ' 42\n\n',
    });
    expect(evaluate(state).value).toBe(0);
  });

  it('SET_EXAMPLE_COUNT changes the lever', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, hopRules, prepared);
    state = applyAction(state, { type: 'SET_EXAMPLE_COUNT', count: 2 });
    expect(state.exampleCount).toBe(2);
  });
});

describe('agentLoopEngine — tool-select-hop mode', () => {
  const config: AgentLoopConfig = {
    mode: 'tool-select-hop',
    maxTokens: 30,
    finalAnswerMaxTokens: 20,
    corpus: 'retrieval-facts',
    instructions: [
      { text: 'What is 2 plus 2?', expectedTool: 'calculator', canonicalExpression: '2 + 2' },
      { text: 'How deep is the Mariana Trench?', expectedTool: 'lookup', factId: 'mariana' },
    ],
  };

  it('resolves each round’s real expected answer — computed for calculator, from the real corpus for lookup', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    expect(prepared.selectRounds[0]!.expectedAnswer).toBe('4');
    expect(prepared.selectRounds[1]!.expectedAnswer).toBe('10,935');
  });

  it('throws if a configured factId does not exist in the corpus', async () => {
    const badConfig: AgentLoopConfig = {
      ...config,
      instructions: [{ text: 'x', expectedTool: 'lookup', factId: 'nope' }],
    };
    await expect(prepare(badConfig, { corpus: fakeCorpus })).rejects.toThrow();
  });

  it('SET_TOOL_ORDER records which tool the player placed last, per round', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, selectHopRules, prepared);
    state = applyAction(state, { type: 'SET_TOOL_ORDER', roundIndex: 0, lastTool: 'calculator' });
    expect(state.selectRounds[0]!.lastOrder).toBe('calculator');
    expect(state.selectRounds[1]!.lastOrder).toBeNull();
  });

  it('scores correct only when the real picked tool matches AND the real final answer is right', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, selectHopRules, prepared);
    state = applyAction(state, {
      type: 'TEST_SELECT_HOP',
      roundIndex: 0,
      toolCallText: '{"tool": "calculator", "args": {"expression": "2 + 2"}}',
      pickedTool: 'calculator',
      toolResult: '4',
      finalAnswerText: ' 4\n\n',
    });
    state = applyAction(state, {
      type: 'TEST_SELECT_HOP',
      roundIndex: 1,
      toolCallText: '{"tool": "lookup", "args": {"topic": "Mariana Trench"}}',
      pickedTool: 'lookup',
      toolResult: 'It reaches 10,935 meters deep.',
      finalAnswerText: ' 10,935 meters\n\n',
    });

    const result = evaluate(state);
    expect(result.metric).toBe('agentTaskAccuracy');
    expect(result.value).toBe(1);
  });

  it('scores wrong when the real picked tool does not match the expected one, even if the final text happens to contain the right number', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, selectHopRules, prepared);
    state = applyAction(state, {
      type: 'TEST_SELECT_HOP',
      roundIndex: 0,
      toolCallText: '{"tool": "lookup", "args": {"topic": "2 plus 2"}}',
      pickedTool: 'lookup',
      toolResult: 'No matching fact found in the reference corpus.',
      finalAnswerText: ' 4\n\n',
    });
    expect(evaluate(state).value).toBe(0);
  });
});

describe('agentLoopEngine — budget-tuning mode', () => {
  const config: AgentLoopConfig = {
    mode: 'budget-tuning',
    maxTokens: 40,
    finalAnswerMaxTokens: 16,
    temperature: 0.7,
    sampleTopK: 20,
    retryBudgetMax: 3,
    tasks: [
      { question: 'What is 3 plus 4?', canonicalExpression: '3 + 4' },
      { question: 'What is 20 minus 6?', canonicalExpression: '20 - 6' },
    ],
  };

  it('starts with example count 0, retry budget 1, and no recorded run', async () => {
    const prepared = await prepare(config, {});
    const state = initState(config, budgetRules, prepared);
    expect(state.exampleCount).toBe(0);
    expect(state.retryBudget).toBe(1);
    expect(state.budgetResults).toHaveLength(0);
    expect(evaluate(state).value).toBe(0);
  });

  it('SET_RETRY_BUDGET clamps to the configured maximum', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, budgetRules, prepared);
    state = applyAction(state, { type: 'SET_RETRY_BUDGET', budget: 99 });
    expect(state.retryBudget).toBe(3);
    state = applyAction(state, { type: 'SET_RETRY_BUDGET', budget: 0 });
    expect(state.retryBudget).toBe(1);
  });

  it('RUN_BUDGET_TEST records the real per-task results and scores the solved rate', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, budgetRules, prepared);
    state = applyAction(state, {
      type: 'RUN_BUDGET_TEST',
      results: [
        { question: 'What is 3 plus 4?', solved: true, attemptsUsed: 1 },
        { question: 'What is 20 minus 6?', solved: false, attemptsUsed: 3 },
      ],
    });
    const result = evaluate(state);
    expect(result.metric).toBe('tasksSolvedRate');
    expect(result.value).toBe(0.5);
  });

  it('a later RUN_BUDGET_TEST replaces the previous results, not append', async () => {
    const prepared = await prepare(config, {});
    let state = initState(config, budgetRules, prepared);
    state = applyAction(state, { type: 'RUN_BUDGET_TEST', results: [{ question: 'a', solved: false, attemptsUsed: 3 }] });
    state = applyAction(state, {
      type: 'RUN_BUDGET_TEST',
      results: [
        { question: 'a', solved: true, attemptsUsed: 1 },
        { question: 'b', solved: true, attemptsUsed: 2 },
      ],
    });
    expect(state.budgetResults).toHaveLength(2);
    expect(evaluate(state).value).toBe(1);
  });
});

describe('agentLoopEngine — RESET never re-runs the model', () => {
  it('restores a fresh state for the same config without needing new model calls', async () => {
    const config: AgentLoopConfig = {
      mode: 'single-hop',
      maxTokens: 40,
      finalAnswerMaxTokens: 16,
      tasks: [{ question: 'What is 3 plus 4?', canonicalExpression: '3 + 4' }],
    };
    const prepared = await prepare(config, {});
    let state = initState(config, hopRules, prepared);
    state = applyAction(state, {
      type: 'TEST_HOP',
      roundIndex: 0,
      toolCallText: 'x',
      toolResult: '7',
      finalAnswerText: '7',
    });
    state = applyAction(state, { type: 'RESET' });
    expect(state.hopRounds[0]!.tested).toBe(false);
  });
});

describe('agentLoopEngine — SUBMIT', () => {
  it('marks the run complete', async () => {
    const config: AgentLoopConfig = {
      mode: 'single-hop',
      maxTokens: 40,
      finalAnswerMaxTokens: 16,
      tasks: [{ question: 'Q?', canonicalExpression: '1 + 1' }],
    };
    const prepared = await prepare(config, {});
    let state = initState(config, hopRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});
