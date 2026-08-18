/**
 * Two precision variants of the same model, for chapter 8.1.
 *
 * Resolves plan-docs A4 via the cheaper path it flagged: rather than finding a
 * second Hub repo, `transformers.js`'s own `dtype` option on `from_pretrained`
 * loads `HuggingFaceTB/SmolLM2-135M-Instruct` (the exact model `tinyCausalLM`
 * already ships) at two genuinely different real precisions — `fp32` and `q8`
 * — so this chapter needs zero new Hub downloads beyond a second dtype of a
 * model already in the app. Confirmed directly, not assumed: both dtypes load
 * and run real forward passes (see the throwaway calibration in plan-docs).
 *
 * Real file sizes below are measured directly against the Hub, not guessed:
 * `curl -sIL https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/onnx/model.onnx`
 * → 540,345,794 bytes; the same for `model_quantized.onnx` (what `dtype: 'q8'`
 * actually requests — `transformers.js`'s `DEFAULT_DTYPE_SUFFIX_MAPPING` maps
 * `q8` to the `_quantized` filename suffix) → 137,147,981 bytes.
 */
import type { PrecisionModelDep, PrecisionRunResult } from '@/engines/deps';
import { loadOnce, toNestedArray, softmaxRow } from './transformersRuntime';
import { CAUSAL_LM_MODEL_ID } from './tinyCausalLM';
import { prettifyToken } from './tokenizerModel';

export type PrecisionDtype = 'fp32' | 'q8';

/** Real, measured (not estimated) download sizes in MB — see file header. */
const REAL_SIZES_MB: Record<PrecisionDtype, number> = {
  fp32: 540,
  q8: 137,
};

function modelIdFor(dtype: PrecisionDtype): string {
  return `${CAUSAL_LM_MODEL_ID}#${dtype}`;
}

/** The exact modelId `loadOnce` reports progress under for each variant — subscribe a `ModelGate` to one of these. */
export const REFERENCE_MODEL_ID = modelIdFor('fp32');
export const QUANTIZED_MODEL_ID = modelIdFor('q8');

interface LoadedVariant {
  tokenizer: {
    (text: string): Promise<Record<string, unknown>>;
    decode: (ids: number[], options?: Record<string, unknown>) => string;
  };
  model: (inputs: Record<string, unknown>) => Promise<{ logits: unknown }>;
}

function loadVariant(dtype: PrecisionDtype): Promise<LoadedVariant> {
  return loadOnce<LoadedVariant>(
    `quantization-compare:${dtype}`,
    { modelId: modelIdFor(dtype), estimatedSizeMB: REAL_SIZES_MB[dtype] },
    async ({ transformers, backend, onProgress }) => {
      const [tokenizer, model] = await Promise.all([
        transformers.AutoTokenizer.from_pretrained(CAUSAL_LM_MODEL_ID, { progress_callback: onProgress }),
        transformers.AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID, {
          device: backend,
          // Forced explicitly rather than left to the usual
          // backend-dependent default (`tinyCausalLM.ts` picks q4 on webgpu,
          // q8 on wasm) — this chapter's whole point is comparing two named
          // precisions directly, not whatever a given device would pick.
          dtype,
          progress_callback: onProgress,
        } as Parameters<typeof transformers.AutoModelForCausalLM.from_pretrained>[1]),
      ]);
      return { tokenizer, model } as unknown as LoadedVariant;
    }
  );
}

/** Tokenizer ids can come back as bigint from some ONNX Runtime backends. */
function tokenId(id: number | bigint): number {
  return typeof id === 'bigint' ? Number(id) : id;
}

/** Tokenizes only — no forward pass — to read real token ids for length/diff bookkeeping. */
async function tokenIds(loaded: LoadedVariant, text: string): Promise<number[]> {
  const inputs = await loaded.tokenizer(text);
  const ids = toNestedArray((inputs as { input_ids?: unknown }).input_ids) as unknown as number[][];
  return (ids[0] ?? []).map(tokenId);
}

/**
 * Runs the real tokenizer + model pipeline on `text` and returns the full
 * (not top-k-truncated) softmax at the final position. Always re-tokenizes
 * real text rather than constructing a raw input-ids tensor by hand — the
 * tokenizer call is what produces genuine `Tensor` inputs (attention mask
 * included) the model actually expects, the same path every other wrapper in
 * this app goes through.
 */
async function fullVocabProbsForText(loaded: LoadedVariant, text: string): Promise<number[]> {
  const inputs = await loaded.tokenizer(text);
  const output = await loaded.model(inputs);
  const nested = toNestedArray(output.logits) as unknown as number[][][];
  const finalLogits = nested[0]?.at(-1);
  if (!Array.isArray(finalLogits) || finalLogits.length === 0) {
    throw new Error('Model returned no usable logits');
  }
  return softmaxRow(finalLogits);
}

async function runForward(loaded: LoadedVariant, prompt: string, topK: number): Promise<PrecisionRunResult> {
  const t0 = performance.now();
  const probs = await fullVocabProbsForText(loaded, prompt);
  const inferenceTimeMs = performance.now() - t0;

  const ranked = probs
    .map((p, id) => ({ id, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, Math.max(1, topK));
  const total = ranked.reduce((sum, r) => sum + r.p, 0) || 1;

  return {
    topK: {
      tokens: ranked.map((r) => prettifyToken(loaded.tokenizer.decode([r.id]))),
      probs: ranked.map((r) => r.p / total),
    },
    inferenceTimeMs,
  };
}

/**
 * Real per-token surprisal (bits) for `continuation`, teacher-forced: each
 * step decodes the true prefix (prompt + the correct continuation tokens seen
 * so far — never a sampled or decoded one) back to text, re-tokenizes it for
 * a real forward pass, and reads that step's real probability for the next
 * true token straight off the full-vocabulary softmax, never the
 * renormalised top-k.
 */
async function continuationSurprisal(
  loaded: LoadedVariant,
  prompt: string,
  continuation: string
): Promise<number[]> {
  const promptIds = await tokenIds(loaded, prompt);
  const fullIds = await tokenIds(loaded, prompt + continuation);
  const targetIds = fullIds.slice(promptIds.length);

  const bits: number[] = [];
  for (let i = 0; i < targetIds.length; i++) {
    const contextIds = fullIds.slice(0, promptIds.length + i);
    const contextText = loaded.tokenizer.decode(contextIds);
    const probs = await fullVocabProbsForText(loaded, contextText);
    const p = probs[targetIds[i]!] ?? 1e-12;
    bits.push(-Math.log2(p || 1e-12));
  }
  return bits;
}

function makeVariant(dtype: PrecisionDtype): PrecisionModelDep {
  return {
    label: dtype,
    sizeMB: REAL_SIZES_MB[dtype],
    async ensureLoaded(): Promise<number> {
      const t0 = performance.now();
      await loadVariant(dtype);
      return performance.now() - t0;
    },
    async run(prompt: string, topK: number): Promise<PrecisionRunResult> {
      const loaded = await loadVariant(dtype);
      return runForward(loaded, prompt, topK);
    },
    async continuationSurprisal(prompt: string, continuation: string): Promise<number[]> {
      const loaded = await loadVariant(dtype);
      return continuationSurprisal(loaded, prompt, continuation);
    },
  };
}

/** fp32 — the real, unquantized reference. */
export const referencePrecisionModel: PrecisionModelDep = makeVariant('fp32');

/** q8 — the real quantized variant, what every WASM-backed player already runs elsewhere in this app. */
export const quantizedPrecisionModel: PrecisionModelDep = makeVariant('q8');

export async function preloadPrecisionModels(): Promise<void> {
  await Promise.all([loadVariant('fp32'), loadVariant('q8')]);
}
