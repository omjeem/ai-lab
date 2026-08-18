/**
 * Model dependencies, as interfaces.
 *
 * Engines never import a model wrapper directly. They receive one of these
 * through a `prepare()` parameter, so the test suite can inject a fake and run
 * offline in milliseconds while the app injects the real transformers.js
 * wrapper from /src/models.
 */

export interface Embedder {
  /** One vector per input string, in the same order. */
  embed(texts: string[]): Promise<number[][]>;
}

export interface TokenizerDep {
  /** Token strings as the real tokenizer segments them. */
  tokenize(text: string): Promise<string[]>;
  encode(text: string): Promise<number[]>;
  decode(ids: number[]): Promise<string>;
  /**
   * BPE merge ranks keyed by "left right". Lower rank means the pair is merged
   * earlier, which is the order a merge puzzle is scored against.
   */
  mergeRanks(): Promise<ReadonlyMap<string, number>>;
}

export interface TokenDistribution {
  tokens: string[];
  probs: number[];
}

export interface CausalLMDep {
  /** Real next-token distribution, truncated to the top k entries. */
  nextTokenDistribution(prompt: string, topK: number): Promise<TokenDistribution>;
}

export interface GenerationStep {
  token: string;
  probability: number;
  entropyBits: number;
  /** Top alternatives the sampler was choosing between at this position. */
  alternatives: TokenDistribution;
}

export interface GenerationTrace {
  text: string;
  steps: GenerationStep[];
}

export interface ChatModelDep {
  generateWithTrace(
    prompt: string,
    options: { maxTokens: number; temperature: number; topP: number }
  ): Promise<GenerationTrace>;
}

export interface AttentionResult {
  tokens: string[];
  /** [layer][head][query][key], every value a real softmaxed attention weight. */
  attention: number[][][][];
}

export interface AttentionDep {
  attention(sentence: string): Promise<AttentionResult>;
}

export interface HiddenStateResult {
  tokens: string[];
  /** [layer][token][dimension] — includes the embedding output as layer 0. */
  hiddenStates: number[][][];
}

export interface HiddenStateDep {
  hiddenStates(sentence: string): Promise<HiddenStateResult>;
}

export interface CorpusDep {
  /** Raw text of a bundled public-domain corpus, fetched at runtime. */
  load(corpusId: string): Promise<string>;
}

export interface ToolResult {
  ok: boolean;
  output: string;
}

/**
 * A real, deterministic, offline-safe tool an agent-loop engine can execute —
 * no network call, no model involved. Implemented first as a calculator
 * (real arithmetic, not `eval`) and a corpus lookup, both in
 * `src/models/toolRuntime.ts`.
 */
export interface ToolDep {
  run(toolName: string, args: Record<string, unknown>): Promise<ToolResult>;
}

/** A net the engine can train and inspect, implemented by /src/models. */
export interface TrainableNetDep {
  forward(inputs: number[][]): number[];
  trainEpoch(options: { learningRate: number; batchSize: number }): number;
  /** Per-parameter gradients from the most recent backward pass. */
  gradients(): { layer: number; from: number; to: number; value: number }[];
  weights(): number[][][];
  reset(seed: number): void;
}
