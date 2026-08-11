/**
 * Client-side caller for the one cloud model in the course.
 *
 * Never talks to Ollama directly — it posts to this app's own API route, which
 * holds the key server-side. The only cloud call in the product is the World 6
 * escalation toggle, and it refuses to fire when the device is offline.
 */
import type { ChatModelDep, GenerationStep, GenerationTrace } from '@/engines/deps';

export interface CloudInferenceRequest {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
}

export interface CloudInferenceResponse {
  text: string;
  model: string;
  steps?: {
    token: string;
    probability: number;
    entropyBits: number;
    alternatives?: { tokens: string[]; probs: number[] };
  }[];
}

export class CloudUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CloudUnavailableError';
  }
}

export class CloudRateLimitedError extends Error {
  readonly retryAfterSeconds: number;
  constructor(message: string, retryAfterSeconds: number) {
    super(message);
    this.name = 'CloudRateLimitedError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * The larger cloud model, reached through the proxy route.
 *
 * The trace it returns carries whatever per-token detail the provider gave us.
 * When the provider returns none, `steps` is empty rather than filled with
 * plausible-looking numbers — the inspector shows what it has and says so.
 */
export const ollamaCloudClient: ChatModelDep = {
  async generateWithTrace(prompt, options): Promise<GenerationTrace> {
    if (typeof navigator !== 'undefined' && navigator.onLine === false) {
      throw new CloudUnavailableError('This device is offline, so the cloud model cannot be reached');
    }

    const response = await fetch('/api/model/cloud-inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
        topP: options.topP,
      } satisfies CloudInferenceRequest),
    });

    if (response.status === 429) {
      const retryAfter = Number(response.headers.get('Retry-After') ?? '60');
      throw new CloudRateLimitedError(
        'Too many cloud requests from this address. Try again shortly.',
        Number.isFinite(retryAfter) ? retryAfter : 60
      );
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      throw new CloudUnavailableError(
        `Cloud model request failed (${response.status})${detail ? `: ${detail}` : ''}`
      );
    }

    const payload = (await response.json()) as CloudInferenceResponse;

    const steps: GenerationStep[] = (payload.steps ?? []).map((step) => ({
      token: step.token,
      probability: step.probability,
      entropyBits: step.entropyBits,
      alternatives: step.alternatives ?? { tokens: [], probs: [] },
    }));

    return { text: payload.text, steps };
  },
};

/**
 * Whether the cloud path is usable right now.
 *
 * `navigator.onLine` alone is unreliable, so a real request to our own origin
 * settles it — the same two-signal approach the sync manager uses.
 */
export async function isCloudReachable(timeoutMs = 3000): Promise<boolean> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return false;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch('/api/model/cloud-inference', {
      method: 'HEAD',
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
