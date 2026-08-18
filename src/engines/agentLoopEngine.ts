/**
 * Chapter 7.4 — Agent Loop.
 *
 * Every tool call, tool result and final answer here is real: the model
 * really can't add four-digit numbers or recall a bundled fact on its own —
 * `toolRuntime`'s real calculator (`evaluateArithmetic`, shared with this
 * engine via `./shared`) and real corpus lookup genuinely extend it. As in
 * 7.3, there is no local `ChatModelDep`, only repeated single-token
 * `nextTokenDistribution` calls via `greedyDecode`/`sampledDecode`. The
 * canvas orchestrates the real async calls (decode the tool call, run the
 * real tool, decode the final answer) and hands the real results back into
 * the engine; the engine only ever grades what actually happened.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep, CorpusDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { createRng, evaluateArithmetic } from './shared';

export type AgentLoopMode = 'single-hop' | 'tool-select-hop' | 'budget-tuning';

export interface AgentLoopTask {
  question: string;
  /** Real arithmetic, evaluated at runtime — never a hand-authored answer. */
  canonicalExpression: string;
}

export interface AgentLoopInstruction {
  text: string;
  expectedTool: 'calculator' | 'lookup';
  canonicalExpression?: string;
  /** Resolved against the real bundled corpus during prepare(). */
  factId?: string;
}

export interface AgentLoopConfig {
  mode: AgentLoopMode;
  /** Real token budget for the tool-call decode. */
  maxTokens: number;
  /** Real token budget for the final-answer decode. */
  finalAnswerMaxTokens: number;
  corpus?: string;
  tasks?: AgentLoopTask[];
  instructions?: AgentLoopInstruction[];
  retryBudgetMax?: number;
  temperature?: number;
  sampleTopK?: number;
}

export interface HopRound {
  question: string;
  canonicalExpression: string;
  expectedAnswer: string;
  toolCallText: string | null;
  toolResult: string | null;
  finalAnswerText: string | null;
  tested: boolean;
}

export interface SelectHopRound {
  text: string;
  expectedTool: 'calculator' | 'lookup';
  expectedAnswer: string;
  lastOrder: 'calculator' | 'lookup' | null;
  toolCallText: string | null;
  pickedTool: string | null;
  toolResult: string | null;
  finalAnswerText: string | null;
  tested: boolean;
}

export interface BudgetTaskResult {
  question: string;
  solved: boolean;
  attemptsUsed: number;
}

export interface BudgetTask {
  question: string;
  canonicalExpression: string;
  expectedAnswer: string;
}

export interface PreparedAgentLoopData {
  hopRounds: HopRound[];
  selectRounds: SelectHopRound[];
  budgetTasks: BudgetTask[];
}

export interface AgentLoopState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: AgentLoopMode;
  config: AgentLoopConfig;

  exampleCount: 0 | 1 | 2;
  hopRounds: HopRound[];

  selectRounds: SelectHopRound[];

  retryBudget: number;
  budgetTasks: BudgetTask[];
  budgetResults: BudgetTaskResult[];
}

export type AgentLoopAction =
  | { type: 'SET_EXAMPLE_COUNT'; count: 0 | 1 | 2 }
  | { type: 'TEST_HOP'; roundIndex: number; toolCallText: string; toolResult: string | null; finalAnswerText: string }
  | { type: 'SET_TOOL_ORDER'; roundIndex: number; lastTool: 'calculator' | 'lookup' }
  | {
      type: 'TEST_SELECT_HOP';
      roundIndex: number;
      toolCallText: string;
      pickedTool: string | null;
      toolResult: string | null;
      finalAnswerText: string;
    }
  | { type: 'SET_RETRY_BUDGET'; budget: number }
  | { type: 'RUN_BUDGET_TEST'; results: BudgetTaskResult[] }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/* ── real, verified worked examples ──────────────────────────────
   Zero examples reliably fails to complete the loop (verified against the
   real model, same finding as 7.3); one worked example per tool is already
   enough for the recency-driven tool pick to hold. */
const CALC_EXAMPLES = [
  { q: 'What is 3 plus 4?', json: '{"tool": "calculator", "args": {"expression": "3 + 4"}}', result: '7' },
  { q: 'What is 20 minus 6?', json: '{"tool": "calculator", "args": {"expression": "20 - 6"}}', result: '14' },
];

export function buildSingleHopPrompt(question: string, exampleCount: 0 | 1 | 2): string {
  const header =
    'You are an assistant with access to a calculator tool. When you need to compute something, respond with ONLY JSON: {"tool": "calculator", "args": {"expression": "..."}}. After you receive the tool\'s result, state the final answer as: Final answer: <number>.\n';
  const shots = CALC_EXAMPLES.slice(0, exampleCount)
    .map((e) => `Question: ${e.q}\nJSON: ${e.json}\nTool result: ${e.result}\nFinal answer: ${e.result}\n`)
    .join('');
  return `${header}${shots}Question: ${question}\nJSON:`;
}

/**
 * Same recency-driven tool pick 7.3 discovered: whichever tool is described
 * and demonstrated last is what the real model picks. `lastTool` is the tool
 * this prompt places last.
 */
export function buildToolSelectHopPrompt(instruction: string, lastTool: 'calculator' | 'lookup'): string {
  const descriptions = {
    calculator: '- calculator: for arithmetic questions',
    lookup: '- lookup: for questions about facts in a reference document',
  };
  const order: ('calculator' | 'lookup')[] = lastTool === 'calculator' ? ['lookup', 'calculator'] : ['calculator', 'lookup'];
  const toolList = order.map((t) => descriptions[t]).join('\n');
  const example =
    lastTool === 'calculator'
      ? 'Instruction: What is 9 plus 9?\nJSON: {"tool": "calculator", "args": {"expression": "9 + 9"}}\nTool result: 18\nFinal answer: 18\n'
      : 'Instruction: How tall is Mount Everest?\nJSON: {"tool": "lookup", "args": {"topic": "Mount Everest height"}}\nTool result: 8849 meters\nFinal answer: 8849 meters\n';
  return `Tools available:\n${toolList}\nRespond with ONLY JSON: {"tool": "...", "args": {...}}. After a tool result, state: Final answer: <answer>.\n${example}Instruction: ${instruction}\nJSON:`;
}

export interface AgentToolCall {
  tool: string;
  args: Record<string, unknown>;
  /** The exact balanced {...} substring, for building the continuation prompt. */
  cleanJson: string;
}

export function parseAgentCall(text: string): AgentToolCall | null {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let end = -1;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '{') depth++;
    if (text[i] === '}') {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  if (end === -1) return null;
  const cleanJson = text.slice(start, end + 1);
  try {
    const parsed: unknown = JSON.parse(cleanJson);
    if (typeof parsed !== 'object' || parsed === null || typeof (parsed as Record<string, unknown>).tool !== 'string') {
      return null;
    }
    const args = (parsed as Record<string, unknown>).args;
    return { tool: (parsed as { tool: string }).tool, args: (args as Record<string, unknown>) ?? {}, cleanJson };
  } catch {
    return null;
  }
}

export function buildFinalAnswerPrompt(toolCallPrompt: string, toolCallText: string, toolResult: string): string {
  return `${toolCallPrompt}${toolCallText}\nTool result: ${toolResult}\nFinal answer:`;
}

export function containsAnswer(text: string, answer: string): boolean {
  const normalise = (s: string) => s.replace(/,/g, '').toLowerCase();
  return normalise(text).includes(normalise(answer));
}

/** Real greedy decode: repeated single-token forward passes, argmax each time. */
export async function greedyDecode(causalLM: CausalLMDep, prompt: string, maxTokens: number): Promise<string> {
  let text = prompt;
  let out = '';
  for (let i = 0; i < maxTokens; i++) {
    const dist = await causalLM.nextTokenDistribution(text, 1);
    const token = dist.tokens[0];
    if (!token) break;
    if (token.includes('<|im_end|>')) break;
    out += token;
    text += token;
    if (out.includes('\n\n')) break;
  }
  return out;
}

/** Real stochastic decode from the model's own top-k distribution, seeded. */
export async function sampledDecode(
  causalLM: CausalLMDep,
  prompt: string,
  maxTokens: number,
  options: { temperature: number; topK: number; seed: number }
): Promise<string> {
  const rng = createRng(options.seed);
  let text = prompt;
  let out = '';
  for (let i = 0; i < maxTokens; i++) {
    const dist = await causalLM.nextTokenDistribution(text, options.topK);
    if (dist.tokens.length === 0) break;
    const temperature = Math.max(options.temperature, 1e-6);
    const scaled = dist.probs.map((p) => Math.log(Math.max(p, 1e-12)) / temperature);
    const max = Math.max(...scaled);
    const exps = scaled.map((l) => Math.exp(l - max));
    const total = exps.reduce((a, b) => a + b, 0);
    const reshaped = exps.map((e) => e / total);

    let r = rng();
    let token = dist.tokens[dist.tokens.length - 1]!;
    for (let j = 0; j < dist.tokens.length; j++) {
      r -= reshaped[j]!;
      if (r <= 0) {
        token = dist.tokens[j]!;
        break;
      }
    }
    if (token.includes('<|im_end|>')) break;
    out += token;
    text += token;
    if (out.includes('\n\n')) break;
  }
  return out;
}

interface RawCorpus {
  facts: { id: string; answer: string }[];
}

export async function prepare(
  config: AgentLoopConfig,
  deps: { corpus?: CorpusDep }
): Promise<PreparedAgentLoopData> {
  const hopRounds: HopRound[] = (config.tasks ?? []).map((task) => {
    const value = evaluateArithmetic(task.canonicalExpression);
    if (value === null) {
      throw new Error(`Task "${task.question}" has an unevaluable canonicalExpression "${task.canonicalExpression}"`);
    }
    return {
      question: task.question,
      canonicalExpression: task.canonicalExpression,
      expectedAnswer: String(value),
      toolCallText: null,
      toolResult: null,
      finalAnswerText: null,
      tested: false,
    };
  });

  const budgetTasks: BudgetTask[] = (config.tasks ?? []).map((task) => {
    const value = evaluateArithmetic(task.canonicalExpression);
    if (value === null) {
      throw new Error(`Task "${task.question}" has an unevaluable canonicalExpression "${task.canonicalExpression}"`);
    }
    return { question: task.question, canonicalExpression: task.canonicalExpression, expectedAnswer: String(value) };
  });

  let factById: Map<string, string> | null = null;
  const selectRounds: SelectHopRound[] = [];
  for (const instr of config.instructions ?? []) {
    let expectedAnswer: string;
    if (instr.expectedTool === 'calculator') {
      const value = evaluateArithmetic(instr.canonicalExpression ?? '');
      if (value === null) {
        throw new Error(`Instruction "${instr.text}" has an unevaluable canonicalExpression "${instr.canonicalExpression}"`);
      }
      expectedAnswer = String(value);
    } else {
      if (!factById) {
        if (!deps.corpus || !config.corpus) throw new Error('tool-select-hop needs a CorpusDep and config.corpus');
        const raw = await deps.corpus.load(config.corpus);
        const parsed = JSON.parse(raw) as RawCorpus;
        factById = new Map(parsed.facts.map((f) => [f.id, f.answer]));
      }
      const answer = instr.factId ? factById.get(instr.factId) : undefined;
      if (!answer) throw new Error(`Instruction "${instr.text}" references unknown factId "${instr.factId}"`);
      expectedAnswer = answer;
    }
    selectRounds.push({
      text: instr.text,
      expectedTool: instr.expectedTool,
      expectedAnswer,
      lastOrder: null,
      toolCallText: null,
      pickedTool: null,
      toolResult: null,
      finalAnswerText: null,
      tested: false,
    });
  }

  return { hopRounds, selectRounds, budgetTasks };
}

export function initState(
  config: AgentLoopConfig,
  rules: EngineRules,
  prepared: PreparedAgentLoopData
): AgentLoopState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,

    exampleCount: 0,
    hopRounds: prepared.hopRounds.map((r) => ({ ...r, toolCallText: null, toolResult: null, finalAnswerText: null, tested: false })),

    selectRounds: prepared.selectRounds.map((r) => ({
      ...r,
      lastOrder: null,
      toolCallText: null,
      pickedTool: null,
      toolResult: null,
      finalAnswerText: null,
      tested: false,
    })),

    retryBudget: 1,
    budgetTasks: prepared.budgetTasks,
    budgetResults: [],
  };
}

export function applyAction(state: AgentLoopState, action: AgentLoopAction): AgentLoopState {
  const bump = (next: Partial<AgentLoopState>): AgentLoopState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_EXAMPLE_COUNT':
      return bump({ exampleCount: action.count });

    case 'TEST_HOP': {
      const round = state.hopRounds[action.roundIndex];
      if (!round) return state;
      const hopRounds = [...state.hopRounds];
      hopRounds[action.roundIndex] = {
        ...round,
        toolCallText: action.toolCallText,
        toolResult: action.toolResult,
        finalAnswerText: action.finalAnswerText,
        tested: true,
      };
      return bump({ hopRounds });
    }

    case 'SET_TOOL_ORDER': {
      const round = state.selectRounds[action.roundIndex];
      if (!round) return state;
      const selectRounds = [...state.selectRounds];
      selectRounds[action.roundIndex] = { ...round, lastOrder: action.lastTool };
      return bump({ selectRounds });
    }

    case 'TEST_SELECT_HOP': {
      const round = state.selectRounds[action.roundIndex];
      if (!round) return state;
      const selectRounds = [...state.selectRounds];
      selectRounds[action.roundIndex] = {
        ...round,
        toolCallText: action.toolCallText,
        pickedTool: action.pickedTool,
        toolResult: action.toolResult,
        finalAnswerText: action.finalAnswerText,
        tested: true,
      };
      return bump({ selectRounds });
    }

    case 'SET_RETRY_BUDGET': {
      const max = state.config.retryBudgetMax ?? 3;
      return bump({ retryBudget: Math.max(1, Math.min(action.budget, max)) });
    }

    case 'RUN_BUDGET_TEST':
      return bump({ budgetResults: action.results });

    case 'RESET':
      return initState(state.config, state.rules, {
        hopRounds: state.hopRounds,
        selectRounds: state.selectRounds,
        budgetTasks: state.budgetTasks,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: AgentLoopState): ScoreResult {
  switch (state.mode) {
    case 'single-hop': {
      const total = state.hopRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'toolHopAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.hopRounds.filter(
        (r) => r.tested && r.finalAnswerText !== null && containsAnswer(r.finalAnswerText, r.expectedAnswer)
      ).length;
      return scoreLevel({
        metric: 'toolHopAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'tool-select-hop': {
      const total = state.selectRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'agentTaskAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.selectRounds.filter(
        (r) =>
          r.tested &&
          r.pickedTool === r.expectedTool &&
          r.finalAnswerText !== null &&
          containsAnswer(r.finalAnswerText, r.expectedAnswer)
      ).length;
      return scoreLevel({
        metric: 'agentTaskAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'budget-tuning': {
      const total = state.budgetResults.length;
      if (total === 0) {
        return scoreLevel({ metric: 'tasksSolvedRate', value: 0, rules: state.rules });
      }
      const solved = state.budgetResults.filter((r) => r.solved).length;
      return scoreLevel({
        metric: 'tasksSolvedRate',
        value: solved / total,
        rules: state.rules,
        breakdown: { solved, total },
      });
    }
  }
}
