/**
 * Chapter 7.3 — Structured Output & Tool Calling.
 *
 * Every JSON string, tool pick and pass/fail here is a real generation from
 * the tiny causal LM (`tinyCausalLM`, `CausalLMDep`) — read one real token at
 * a time via `greedyDecode`/`sampledDecode`, the same pattern World 7.2's
 * `groundedGenerationEngine` uses (there is still no local `ChatModelDep`
 * wrapper, only the single-token distribution). Nothing is pre-scored: the
 * canvas runs a real decode and hands the text back into the engine, which
 * only ever parses and grades what actually came out of the model.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep } from './deps';
import { scoreLevel } from './scoringEngine';
import { createRng } from './shared';

export type ToolCallMode = 'schema-reliability' | 'tool-select' | 'retry-fix';

export interface ToolSelectInstruction {
  text: string;
  expectedTool: 'calculator' | 'lookup';
}

export interface ToolCallConfig {
  mode: ToolCallMode;
  maxTokens: number;
  /** schema-reliability */
  questions?: string[];
  samplesPerQuestion?: number;
  temperature?: number;
  sampleTopK?: number;
  /** tool-select */
  instructions?: ToolSelectInstruction[];
  /** retry-fix */
  question?: string;
}

export interface SchemaTestResult {
  question: string;
  text: string;
  valid: boolean;
}

export interface ToolSelectRound {
  text: string;
  expectedTool: 'calculator' | 'lookup';
  lastOrder: 'calculator' | 'lookup' | null;
  pickedTool: string | null;
  tested: boolean;
}

export interface PreparedToolCallData {
  selectRounds: ToolSelectRound[];
}

export interface ToolCallState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: ToolCallMode;
  config: ToolCallConfig;

  // schema-reliability
  exampleCount: 0 | 1 | 2;
  schemaResults: SchemaTestResult[];

  // tool-select
  selectRounds: ToolSelectRound[];

  // retry-fix
  retryAttempts: number;
  retrySolved: boolean;
  retrySolvedAtAttempt: number | null;
  retryLastText: string | null;
}

export type ToolCallAction =
  | { type: 'SET_EXAMPLE_COUNT'; count: 0 | 1 | 2 }
  | { type: 'RUN_SCHEMA_TEST'; results: SchemaTestResult[] }
  | { type: 'SET_TOOL_ORDER'; roundIndex: number; lastTool: 'calculator' | 'lookup' }
  | { type: 'TEST_TOOL_SELECT'; roundIndex: number; pickedTool: string | null }
  | { type: 'TEST_RETRY'; text: string }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

/* ── real few-shot examples, shared by schema-reliability and retry-fix ──
   Verified directly against the real model: zero examples reliably fails to
   close the JSON object (extra keys sneak in, or the schema's own placeholder
   text gets echoed back literally); one worked example is already enough. */
const SCHEMA_EXAMPLES = [
  { q: 'What is 3 plus 4?', json: '{"tool": "calculator", "args": {"expression": "3 + 4"}}' },
  { q: 'What is 10 minus 2?', json: '{"tool": "calculator", "args": {"expression": "10 - 2"}}' },
];

export function buildSchemaPrompt(question: string, exampleCount: 0 | 1 | 2): string {
  const header =
    'You are a tool-calling assistant. You must respond with ONLY a single JSON object matching this exact schema, and nothing else: {"tool": "calculator", "args": {"expression": string}}\nDo not explain. Do not add extra text.\n';
  const shots = SCHEMA_EXAMPLES.slice(0, exampleCount)
    .map((e) => `Question: ${e.q}\nJSON: ${e.json}\n`)
    .join('');
  return `${header}${shots}Question: ${question}\nJSON:`;
}

/**
 * Extracts the first balanced `{...}` object from real model output and
 * checks it against the calculator tool-call schema. Extra keys are allowed —
 * a permissive schema check, the same way a real tool router would validate
 * only the fields it actually needs.
 */
export function parseToolCall(text: string): { valid: boolean; tool?: string; expression?: string } {
  const start = text.indexOf('{');
  if (start === -1) return { valid: false };
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
  if (end === -1) return { valid: false };
  try {
    const parsed: unknown = JSON.parse(text.slice(start, end + 1));
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      typeof (parsed as Record<string, unknown>).tool === 'string' &&
      typeof (parsed as Record<string, unknown>).args === 'object' &&
      (parsed as Record<string, unknown>).args !== null &&
      typeof ((parsed as Record<string, { expression?: unknown }>).args as { expression?: unknown }).expression ===
        'string'
    ) {
      const p = parsed as { tool: string; args: { expression: string } };
      return { valid: true, tool: p.tool, expression: p.args.expression };
    }
    return { valid: false };
  } catch {
    return { valid: false };
  }
}

/**
 * Real, measured behaviour: a small model's tool selection is dominated by
 * which tool it was most recently shown being used — not by the
 * instruction's actual content. Verified directly, at the real quantized
 * precision the browser actually runs (q8, not the higher-precision fp32
 * that first suggested tool *order* alone was enough): reordering the bare
 * tool list flips the pick only inconsistently under q8's added noise, but
 * adding one worked example of whichever tool goes last restores total
 * reliability — confirmed by deliberately ordering the *wrong* tool last and
 * watching the model follow it anyway, 6 real generations out of 6.
 * `lastTool` is the tool this prompt places last and demonstrates.
 */
export function buildToolSelectPrompt(instruction: string, lastTool: 'calculator' | 'lookup'): string {
  const descriptions = {
    calculator: '- calculator: for arithmetic questions',
    lookup: '- lookup: for questions about facts in a reference document',
  };
  const order: ('calculator' | 'lookup')[] = lastTool === 'calculator' ? ['lookup', 'calculator'] : ['calculator', 'lookup'];
  const toolList = order.map((t) => descriptions[t]).join('\n');
  const exampleInstruction = lastTool === 'calculator' ? 'What is 9 plus 9?' : 'How tall is Mount Everest?';
  const shot = `Instruction: ${exampleInstruction}\nJSON: {"tool": "${lastTool}"}\n`;
  return `Tools available:\n${toolList}\nRespond with ONLY JSON: {"tool": "calculator" or "lookup"}\n${shot}Instruction: ${instruction}\nJSON:`;
}

export function extractToolName(text: string): string | null {
  const match = /"tool"\s*:\s*"(\w+)"/.exec(text);
  return match?.[1] ?? null;
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

/**
 * Real stochastic decode: at each step, draws from the model's own top-k
 * distribution (renormalised, exactly what `nextTokenDistribution` already
 * returns) using a seeded RNG — deterministic given the same seed, so a test
 * or a "repeat this exact run" replay is reproducible, but genuinely
 * different generations across seeds, unlike greedy decoding.
 */
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

export async function prepare(config: ToolCallConfig): Promise<PreparedToolCallData> {
  const selectRounds: ToolSelectRound[] = (config.instructions ?? []).map((instr) => ({
    text: instr.text,
    expectedTool: instr.expectedTool,
    lastOrder: null,
    pickedTool: null,
    tested: false,
  }));
  return { selectRounds };
}

export function initState(config: ToolCallConfig, rules: EngineRules, prepared: PreparedToolCallData): ToolCallState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,

    exampleCount: 0,
    schemaResults: [],

    selectRounds: prepared.selectRounds.map((r) => ({ ...r, lastOrder: null, pickedTool: null, tested: false })),

    retryAttempts: 0,
    retrySolved: false,
    retrySolvedAtAttempt: null,
    retryLastText: null,
  };
}

export function applyAction(state: ToolCallState, action: ToolCallAction): ToolCallState {
  const bump = (next: Partial<ToolCallState>): ToolCallState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_EXAMPLE_COUNT':
      return bump({ exampleCount: action.count });

    case 'RUN_SCHEMA_TEST':
      return bump({ schemaResults: action.results });

    case 'SET_TOOL_ORDER': {
      const round = state.selectRounds[action.roundIndex];
      if (!round) return state;
      const selectRounds = [...state.selectRounds];
      selectRounds[action.roundIndex] = { ...round, lastOrder: action.lastTool };
      return bump({ selectRounds });
    }

    case 'TEST_TOOL_SELECT': {
      const round = state.selectRounds[action.roundIndex];
      if (!round) return state;
      const selectRounds = [...state.selectRounds];
      selectRounds[action.roundIndex] = { ...round, pickedTool: action.pickedTool, tested: true };
      return bump({ selectRounds });
    }

    case 'TEST_RETRY': {
      const attempts = state.retryAttempts + 1;
      const hit = parseToolCall(action.text).valid;
      const alreadySolved = state.retrySolved;
      return bump({
        retryAttempts: attempts,
        retryLastText: action.text,
        retrySolved: alreadySolved || hit,
        retrySolvedAtAttempt: alreadySolved ? state.retrySolvedAtAttempt : hit ? attempts : null,
      });
    }

    case 'RESET':
      return initState(state.config, state.rules, { selectRounds: state.selectRounds });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: ToolCallState): ScoreResult {
  switch (state.mode) {
    case 'schema-reliability': {
      const total = state.schemaResults.length;
      if (total === 0) {
        return scoreLevel({ metric: 'jsonValidRate', value: 0, rules: state.rules });
      }
      const valid = state.schemaResults.filter((r) => r.valid).length;
      return scoreLevel({
        metric: 'jsonValidRate',
        value: valid / total,
        rules: state.rules,
        breakdown: { valid, total },
      });
    }

    case 'tool-select': {
      const total = state.selectRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'toolPickAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.selectRounds.filter((r) => r.tested && r.pickedTool === r.expectedTool).length;
      return scoreLevel({
        metric: 'toolPickAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }

    case 'retry-fix': {
      const value = state.retrySolved ? state.retrySolvedAtAttempt! : Infinity;
      return scoreLevel({
        metric: 'attemptsToSolve',
        value,
        rules: state.rules,
        breakdown: { attempts: state.retryAttempts, solved: state.retrySolved ? 1 : 0 },
      });
    }
  }
}
