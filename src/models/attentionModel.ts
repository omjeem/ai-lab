/**
 * Real attention weights and hidden states from a small transformer.
 *
 * This is what Worlds 5.2 to 5.4 are built on. The model is loaded with
 * `output_attentions` and `output_hidden_states` so the tensors the chapters
 * visualise are the ones the forward pass actually produced.
 */
import type { AttentionDep, AttentionResult, HiddenStateDep, HiddenStateResult } from '@/engines/deps';
import { loadOnce, toNestedArray } from './transformersRuntime';

export const ATTENTION_MODEL_ID = 'Xenova/distilbert-base-uncased';
const ATTENTION_SIZE_MB = 67;

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
          progress_callback: onProgress,
          // transformers.js surfaces attention and hidden-state tensors only
          // when the ONNX graph exports them; these flags are read from the
          // config rather than the load options, so they are not in the public
          // option type. Passing them through is the supported way to ask.
          ...({ output_attentions: true, output_hidden_states: true } as Record<string, unknown>),
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
 * Collects the per-layer attention tensors.
 *
 * transformers.js returns them either as an `attentions` array or as numbered
 * `attentions.0` keys depending on the export, so both are handled.
 */
function collectLayers(outputs: Record<string, unknown>, prefix: string): unknown[] {
  const direct = outputs[prefix];
  if (Array.isArray(direct)) return direct;

  const numbered: unknown[] = [];
  for (let layer = 0; ; layer++) {
    const value = outputs[`${prefix}.${layer}`] ?? outputs[`${prefix}_${layer}`];
    if (value === undefined) break;
    numbered.push(value);
  }
  return numbered;
}

export const attentionModel: AttentionDep = {
  async attention(sentence: string): Promise<AttentionResult> {
    const { tokens, outputs } = await runForward(sentence);
    const layers = collectLayers(outputs, 'attentions');

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

export const hiddenStateModel: HiddenStateDep = {
  async hiddenStates(sentence: string): Promise<HiddenStateResult> {
    const { tokens, outputs } = await runForward(sentence);
    const layers = collectLayers(outputs, 'hidden_states');

    if (layers.length === 0) {
      throw new Error(
        'The model returned no hidden states — it was not loaded with output_hidden_states'
      );
    }

    // [batch][token][dimension] per layer; batch dropped.
    const hiddenStates = layers.map((layer) => {
      const nested = toNestedArray(layer) as unknown as number[][][];
      return (nested[0] ?? []) as number[][];
    });

    return { tokens, hiddenStates };
  },
};

export async function preloadAttentionModel(): Promise<void> {
  await getModel();
}
