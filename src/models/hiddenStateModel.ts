/**
 * Real per-layer hidden states from a small transformer.
 *
 * `Qdrant/all_miniLM_L6_v2_with_attentions` (what `attentionModel.ts` uses for
 * 5.2 and 5.3) only exports `last_hidden_state` — the final layer, not the
 * per-layer stack this chapter needs — see plan-docs A1. Found via the same
 * technique that turned up that model: searching the Hub's `-with-attentions`
 * naming convention, this time for `hidden`. `yaww85/all-MiniLM-L6-v2-hidden-
 * states-exposed-v1` is a plain ONNX export of the same base model
 * (`sentence-transformers/all-MiniLM-L6-v2`) whose graph was built with
 * `output_hidden_states=True` — verified by loading it directly and reading
 * real tensors: `hidden_states.0` through `hidden_states.6` (the embedding
 * output plus all 6 encoder layers), and `last_hidden_state` is bit-identical
 * to `hidden_states.6`.
 *
 * Two gotchas fixed here, neither of which show up on any other model in this
 * repo:
 * - The repo ships no `config.json` or tokenizer files at all, only the ONNX
 *   graph and a `vocab.txt` — `AutoModel`/`AutoTokenizer.from_pretrained`
 *   against this id directly fail looking for `config.json`. Both are instead
 *   borrowed from `Xenova/all-MiniLM-L6-v2`, the exact same base model, so the
 *   vocabulary and architecture line up exactly.
 * - The weights are saved using the standard ONNX external-data mechanism as
 *   a second file, `model.onnx.data`, alongside `model.onnx`. transformers.js
 *   does not fetch that automatically: its own automatic external-data
 *   convention only recognises a `<name>_data`-suffixed chunk naming, not the
 *   plain `<name>.data` this repo actually uses. Passing an explicit
 *   `session_options.externalData` entry routes the fetch through the same
 *   `getModelFile` path as the main weights file, which is what gives it
 *   proper browser-cache and progress-callback behaviour rather than a
 *   one-off manual fetch.
 */
import type { HiddenStateDep, HiddenStateResult } from '@/engines/deps';
import { loadOnce, toNestedArray } from './transformersRuntime';

export const HIDDEN_STATE_MODEL_ID = 'yaww85/all-MiniLM-L6-v2-hidden-states-exposed-v1';
const HIDDEN_STATE_SIZE_MB = 90;
const CONFIG_SOURCE_ID = 'Xenova/all-MiniLM-L6-v2';

interface LoadedHiddenStateModel {
  tokenizer: {
    (text: string): Promise<Record<string, unknown>>;
    convert_ids_to_tokens?: (ids: number[]) => string[];
    decode: (ids: number[], options?: Record<string, unknown>) => string;
  };
  model: (inputs: Record<string, unknown>) => Promise<Record<string, unknown>>;
}

async function getModel(): Promise<LoadedHiddenStateModel> {
  return loadOnce<LoadedHiddenStateModel>(
    'hidden-state',
    { modelId: HIDDEN_STATE_MODEL_ID, estimatedSizeMB: HIDDEN_STATE_SIZE_MB },
    async ({ transformers, backend, onProgress }) => {
      const [tokenizer, config] = await Promise.all([
        transformers.AutoTokenizer.from_pretrained(CONFIG_SOURCE_ID, { progress_callback: onProgress }),
        transformers.AutoConfig.from_pretrained(CONFIG_SOURCE_ID),
      ]);

      const model = await transformers.AutoModel.from_pretrained(HIDDEN_STATE_MODEL_ID, {
        config,
        device: backend,
        // The only full-precision (non-quantized) weights this repo ships.
        dtype: 'fp32',
        progress_callback: onProgress,
        ...({
          subfolder: '',
          session_options: {
            externalData: [{ path: 'model.onnx.data', data: 'model.onnx.data' }],
          },
        } as Record<string, unknown>),
      } as Parameters<typeof transformers.AutoModel.from_pretrained>[1]);

      return { tokenizer, model } as unknown as LoadedHiddenStateModel;
    }
  );
}

export const hiddenStateModel: HiddenStateDep = {
  async hiddenStates(sentence: string): Promise<HiddenStateResult> {
    const { tokenizer, model } = await getModel();
    const inputs = await tokenizer(sentence);
    const outputs = await model(inputs);

    const ids = (toNestedArray((inputs as { input_ids?: unknown }).input_ids) as unknown as number[][])[0] ?? [];
    const tokens =
      typeof tokenizer.convert_ids_to_tokens === 'function'
        ? tokenizer.convert_ids_to_tokens(ids)
        : ids.map((id) => tokenizer.decode([id]));

    const layers: unknown[] = [];
    for (let layer = 0; ; layer++) {
      const value = outputs[`hidden_states.${layer}`];
      if (value === undefined) break;
      layers.push(value);
    }

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

export async function preloadHiddenStateModel(): Promise<void> {
  await getModel();
}
