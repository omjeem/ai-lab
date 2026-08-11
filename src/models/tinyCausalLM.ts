/**
 * A small causal language model, run in the browser for its real next-token
 * distribution.
 *
 * Worlds 1.5, 4.3 and 5.5 all read from here. The probabilities the wheel and
 * the sampling controls operate on are this model's actual softmaxed logits at
 * the final position — never a synthetic distribution shaped to look plausible.
 */
import type { CausalLMDep, TokenDistribution } from '@/engines/deps';
import { loadOnce, softmaxRow, toNestedArray, type TransformersModule } from './transformersRuntime';
import { prettifyToken } from './tokenizerModel';

export const CAUSAL_LM_MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct';
export const GPT2_MODEL_ID = 'Xenova/gpt2';

interface LoadedCausalLM {
  tokenizer: {
    (text: string): Promise<Record<string, unknown>>;
    decode: (ids: number[], options?: Record<string, unknown>) => string;
  };
  model: (inputs: Record<string, unknown>) => Promise<{ logits: unknown }>;
}

async function loadCausalLM(modelId: string, sizeMB: number): Promise<LoadedCausalLM> {
  return loadOnce<LoadedCausalLM>(
    `causal-lm:${modelId}`,
    { modelId, estimatedSizeMB: sizeMB },
    async ({ transformers, backend, onProgress }) => {
      const [tokenizer, model] = await Promise.all([
        transformers.AutoTokenizer.from_pretrained(modelId, { progress_callback: onProgress }),
        transformers.AutoModelForCausalLM.from_pretrained(modelId, {
          device: backend,
          dtype: backend === 'webgpu' ? 'q4' : 'q8',
          progress_callback: onProgress,
        }),
      ]);
      return { tokenizer, model } as unknown as LoadedCausalLM;
    }
  );
}

/**
 * Runs one forward pass and reads the distribution at the final position.
 *
 * `logits` comes back as [batch][sequence][vocab]; the last sequence entry is
 * the model's prediction for what comes next.
 */
async function nextTokenDistribution(
  modelId: string,
  sizeMB: number,
  prompt: string,
  k: number
): Promise<TokenDistribution> {
  const { tokenizer, model } = await loadCausalLM(modelId, sizeMB);

  const inputs = await tokenizer(prompt);
  const output = await model(inputs);
  const nested = toNestedArray(output.logits) as unknown as number[][][];

  const sequence = Array.isArray(nested[0]) ? nested[0] : null;
  const finalLogits = sequence?.at(-1);
  if (!Array.isArray(finalLogits) || finalLogits.length === 0) {
    throw new Error(`Model ${modelId} returned no usable logits for the prompt`);
  }

  const probabilities = softmaxRow(finalLogits as number[]);

  const ranked = probabilities
    .map((probability, id) => ({ id, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, Math.max(1, k));

  return {
    tokens: ranked.map((entry) => prettifyToken(tokenizer.decode([entry.id]))),
    // Renormalised over the top k, so the wheel's segments genuinely sum to one.
    probs: normalise(ranked.map((entry) => entry.probability)),
  };
}

function normalise(values: readonly number[]): number[] {
  const total = values.reduce((a, b) => a + b, 0);
  if (total === 0) return values.map(() => 1 / Math.max(values.length, 1));
  return values.map((v) => v / total);
}

/** SmolLM2-135M — Worlds 1.5 and 4.3. */
export const tinyCausalLM: CausalLMDep = {
  nextTokenDistribution: (prompt, topK) =>
    nextTokenDistribution(CAUSAL_LM_MODEL_ID, 92, prompt, topK),
};

/** GPT-2 — World 5.5's end-to-end forward pass. */
export const gpt2CausalLM: CausalLMDep = {
  nextTokenDistribution: (prompt, topK) => nextTokenDistribution(GPT2_MODEL_ID, 124, prompt, topK),
};

export async function preloadCausalLM(modelId = CAUSAL_LM_MODEL_ID): Promise<void> {
  await loadCausalLM(modelId, modelId === GPT2_MODEL_ID ? 124 : 92);
}

/** Exposed for the World 5.5 stage inspector, which shows intermediate output. */
export async function tokenizePrompt(
  prompt: string,
  modelId = GPT2_MODEL_ID
): Promise<{ ids: number[]; tokens: string[] }> {
  const { tokenizer } = await loadCausalLM(modelId, 124);
  const encoded = (await tokenizer(prompt)) as { input_ids?: unknown };
  const ids = (toNestedArray(encoded.input_ids) as unknown as number[][])[0] ?? [];
  return { ids, tokens: ids.map((id) => prettifyToken(tokenizer.decode([id]))) };
}

export type { TransformersModule };
