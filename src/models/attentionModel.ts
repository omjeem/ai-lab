/**
 * Real attention weights from a small transformer.
 *
 * This is what Worlds 5.2 and 5.3 are built on. The model is loaded with
 * `output_attentions` so the tensors the chapters visualise are the ones the
 * forward pass actually produced.
 *
 * `Xenova/distilbert-base-uncased` (the model this was originally built
 * against — see plan-docs A1) turned out not to work: its ONNX export was
 * converted for the `DistilBertForMaskedLM` task, so its graph only has a
 * `logits` output, no `attentions`/`hidden_states`, regardless of what flags
 * are passed at load time. `Qdrant/all_miniLM_L6_v2_with_attentions` is a
 * base `BertModel` export (sentence-transformers/all-MiniLM-L6-v2, 6 layers,
 * 12 heads — the same layer/head count the original config assumed) whose
 * graph was built with `output_attentions: true`, verified by actually
 * running it: it returns real `attention_1..attention_6` tensors. It has no
 * `hidden_states` output — only `last_hidden_state` (the final layer) — so
 * World 5.4 (which needs the full per-layer stack) is served by a different
 * model entirely; see `./hiddenStateModel.ts`.
 */
import type { AttentionDep, AttentionResult } from '@/engines/deps';
import { loadOnce, toNestedArray } from './transformersRuntime';

export const ATTENTION_MODEL_ID = 'Qdrant/all_miniLM_L6_v2_with_attentions';
const ATTENTION_SIZE_MB = 87;

interface LoadedAttentionModel {
  tokenizer: {
    (text: string): Promise<Record<string, unknown>>;
    convert_ids_to_tokens?: (ids: number[]) => string[];
    decode: (ids: number[], options?: Record<string, unknown>) => string;
  };
  model: (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>;
}

async function getModel(): Promise<LoadedAttentionModel> {
  return loadOnce<LoadedAttentionModel>(
    'attention',
    { modelId: ATTENTION_MODEL_ID, estimatedSizeMB: ATTENTION_SIZE_MB },
    async ({ transformers, backend, onProgress }) => {
      const [tokenizer, model] = await Promise.all([
        transformers.AutoTokenizer.from_pretrained(ATTENTION_MODEL_ID, {
          progress_callback: onProgress,
        }),
        transformers.AutoModel.from_pretrained(ATTENTION_MODEL_ID, {
          device: backend,
          // This repo ships exactly one weights file — a full-precision
          // model.onnx, no quantized variant. Every other model wrapper in
          // this codebase leaves dtype unset and lets transformers.js pick,
          // which defaults to requesting a quantized file on the WASM
          // backend (model_quantized.onnx) — a 404 against this repo. Forcing
          // fp32 is what makes this the one wrapper that has to say so.
          dtype: 'fp32',
          progress_callback: onProgress,
          // transformers.js surfaces attention tensors only when the ONNX
          // graph exports them; this flag is read from the config rather than
          // the load options, so it is not in the public option type. Passing
          // it through is the supported way to ask.
          ...({
            output_attentions: true,
            // This repo's model.onnx sits at the repo root, not the usual
            // onnx/ subfolder transformers.js defaults to.
            subfolder: '',
          } as Record<string, unknown>),
        } as Parameters<typeof transformers.AutoModel.from_pretrained>[1]),
      ]);
      return { tokenizer, model } as unknown as LoadedAttentionModel;
    }
  );
}

async function runForward(sentence: string): Promise<{
  tokens: string[];
  outputs: Record<string, unknown>;
}> {
  const { tokenizer, model } = await getModel();
  const inputs = await tokenizer(sentence);
  const outputs = await model(inputs);

  const ids = (toNestedArray((inputs as { input_ids?: unknown }).input_ids) as unknown as number[][])[0] ?? [];
  const tokens =
    typeof tokenizer.convert_ids_to_tokens === 'function'
      ? tokenizer.convert_ids_to_tokens(ids)
      : ids.map((id) => tokenizer.decode([id]));

  return { tokens, outputs };
}

/**
 * Collects the per-layer tensors for whichever output family this export
 * used — every ONNX conversion that includes them names them differently:
 * a single `attentions` array, numbered `attentions.0`/`attentions_0` keys
 * (0-indexed), or numbered `attention_1` keys (1-indexed, singular — what
 * `Qdrant/all_miniLM_L6_v2_with_attentions` actually returns). Every shape
 * is checked rather than assumed, so a future model swap only needs a new
 * prefix added here, not a rewrite.
 */
function collectLayers(outputs: Record<string, unknown>, prefixes: string[]): unknown[] {
  for (const prefix of prefixes) {
    const direct = outputs[prefix];
    if (Array.isArray(direct)) return direct;

    for (const start of [0, 1]) {
      const numbered: unknown[] = [];
      for (let layer = start; ; layer++) {
        const value = outputs[`${prefix}.${layer}`] ?? outputs[`${prefix}_${layer}`];
        if (value === undefined) break;
        numbered.push(value);
      }
      if (numbered.length > 0) return numbered;
    }
  }
  return [];
}

export const attentionModel: AttentionDep = {
  async attention(sentence: string): Promise<AttentionResult> {
    const { tokens, outputs } = await runForward(sentence);
    const layers = collectLayers(outputs, ['attentions', 'attention']);

    if (layers.length === 0) {
      throw new Error(
        'The model returned no attention tensors — it was not loaded with output_attentions'
      );
    }

    // Each layer arrives as [batch][head][query][key]; drop the batch dimension.
    const attention = layers.map((layer) => {
      const nested = toNestedArray(layer) as unknown as number[][][][];
      return (nested[0] ?? []) as number[][][];
    });

    return { tokens, attention };
  },
};

export async function preloadAttentionModel(): Promise<void> {
  await getModel();
}
