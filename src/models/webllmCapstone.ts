/**
 * The capstone's local model: a 1B instruct model running entirely in the
 * browser through WebLLM.
 *
 * Generation is streamed with per-token logprobs so the inspector panel can show
 * the real distribution the sampler was choosing between at each step — which is
 * what makes World 6 an instrument rather than a chat box.
 */
import type { ChatModelDep, GenerationStep, GenerationTrace } from '@/engines/deps';
import { markDownloading, markError, markReady, recordModelCached, reportProgress } from './modelCache';

export const WEBLLM_MODEL_ID = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const WEBLLM_SIZE_MB = 880;

type WebLLMModule = typeof import('@mlc-ai/web-llm');
type Engine = Awaited<ReturnType<WebLLMModule['CreateMLCEngine']>>;

let enginePromise: Promise<Engine> | null = null;

export async function getEngine(): Promise<Engine> {
  if (enginePromise) return enginePromise;

  enginePromise = (async () => {
    markDownloading(WEBLLM_MODEL_ID);
    try {
      const webllm = await import('@mlc-ai/web-llm');
      const engine = await webllm.CreateMLCEngine(WEBLLM_MODEL_ID, {
        initProgressCallback: (report) => {
          reportProgress(WEBLLM_MODEL_ID, { progress: report.progress, file: report.text });
        },
      });
      markReady(WEBLLM_MODEL_ID);
      void recordModelCached(WEBLLM_MODEL_ID, WEBLLM_SIZE_MB);
      return engine;
    } catch (error) {
      markError(WEBLLM_MODEL_ID, error);
      // Drop the rejection so the retry button starts a genuine second attempt.
      enginePromise = null;
      throw error;
    }
  })();

  return enginePromise;
}

/** Shannon entropy in bits over a set of logprobs. */
function entropyBitsOf(logprobs: readonly { logprob: number }[]): number {
  if (logprobs.length === 0) return 0;
  const probs = logprobs.map((entry) => Math.exp(entry.logprob));
  const total = probs.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;

  let entropy = 0;
  for (const p of probs) {
    const normalised = p / total;
    if (normalised > 0) entropy -= normalised * Math.log2(normalised);
  }
  return entropy;
}

interface TopLogprob {
  token: string;
  logprob: number;
}

interface ChoiceLogprobs {
  content?: { token: string; logprob: number; top_logprobs?: TopLogprob[] }[];
}

export const webllmCapstone: ChatModelDep = {
  async generateWithTrace(prompt, options): Promise<GenerationTrace> {
    const engine = await getEngine();

    const completion = await engine.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      top_p: options.topP,
      // The whole point of the chapter: keep the per-step distributions.
      logprobs: true,
      top_logprobs: 5,
    });

    const choice = completion.choices?.[0];
    const text = choice?.message?.content ?? '';
    const content = (choice?.logprobs as ChoiceLogprobs | undefined)?.content ?? [];

    const steps: GenerationStep[] = content.map((entry) => {
      const alternatives = entry.top_logprobs ?? [];
      return {
        token: entry.token,
        probability: Math.exp(entry.logprob),
        entropyBits: entropyBitsOf(alternatives),
        alternatives: {
          tokens: alternatives.map((a) => a.token),
          probs: alternatives.map((a) => Math.exp(a.logprob)),
        },
      };
    });

    return { text, steps };
  },
};

/** True when the browser can run WebLLM at all — it requires WebGPU. */
export async function isWebLLMSupported(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

export function resetEngine(): void {
  enginePromise = null;
}
