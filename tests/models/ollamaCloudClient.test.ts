import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ollamaCloudClient,
  isCloudReachable,
  CloudUnavailableError,
  CloudRateLimitedError,
} from '@/models/ollamaCloudClient';

const options = { maxTokens: 64, temperature: 0.7, topP: 0.95 };

function mockFetch(response: Partial<Response> & { json?: () => Promise<unknown> }) {
  const fn = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers(),
    text: async () => '',
    json: async () => ({}),
    ...response,
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

beforeEach(() => {
  vi.stubGlobal('navigator', { onLine: true });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ollamaCloudClient', () => {
  it('posts the prompt and sampling settings to the proxy route', async () => {
    const fetchMock = mockFetch({
      json: async () => ({ text: 'hello from the cloud', model: 'gpt-oss:120b-cloud' }),
    });

    await ollamaCloudClient.generateWithTrace('why is the sky blue?', options);

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('/api/model/cloud-inference');
    expect(init.method).toBe('POST');

    const body = JSON.parse(init.body as string);
    expect(body).toEqual({
      prompt: 'why is the sky blue?',
      maxTokens: 64,
      temperature: 0.7,
      topP: 0.95,
    });
  });

  it('never contacts an Ollama host directly', async () => {
    const fetchMock = mockFetch({ json: async () => ({ text: 'x', model: 'm' }) });
    await ollamaCloudClient.generateWithTrace('p', options);

    for (const [url] of fetchMock.mock.calls) {
      expect(String(url)).not.toMatch(/ollama\.com|https?:\/\//);
    }
  });

  it('refuses to call out while the device is offline', async () => {
    vi.stubGlobal('navigator', { onLine: false });
    const fetchMock = mockFetch({ json: async () => ({ text: 'x', model: 'm' }) });

    await expect(ollamaCloudClient.generateWithTrace('p', options)).rejects.toBeInstanceOf(
      CloudUnavailableError
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns the text and the per-step trace the route supplied', async () => {
    mockFetch({
      json: async () => ({
        text: 'answer',
        model: 'gpt-oss:120b-cloud',
        steps: [
          {
            token: 'an',
            probability: 0.6,
            entropyBits: 1.2,
            alternatives: { tokens: ['an', 'the'], probs: [0.6, 0.4] },
          },
        ],
      }),
    });

    const trace = await ollamaCloudClient.generateWithTrace('p', options);
    expect(trace.text).toBe('answer');
    expect(trace.steps).toHaveLength(1);
    expect(trace.steps[0]!.probability).toBeCloseTo(0.6);
    expect(trace.steps[0]!.alternatives.tokens).toEqual(['an', 'the']);
  });

  it('reports an empty trace rather than inventing per-token detail', async () => {
    mockFetch({ json: async () => ({ text: 'answer', model: 'm' }) });
    const trace = await ollamaCloudClient.generateWithTrace('p', options);
    expect(trace.steps).toEqual([]);
  });

  it('surfaces rate limiting with the retry delay', async () => {
    mockFetch({
      ok: false,
      status: 429,
      headers: new Headers({ 'Retry-After': '30' }),
    });

    const error = await ollamaCloudClient
      .generateWithTrace('p', options)
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(CloudRateLimitedError);
    expect((error as CloudRateLimitedError).retryAfterSeconds).toBe(30);
  });

  it('defaults the retry delay when the header is missing or unusable', async () => {
    mockFetch({ ok: false, status: 429, headers: new Headers() });
    const error = await ollamaCloudClient.generateWithTrace('p', options).catch((e) => e);
    expect((error as CloudRateLimitedError).retryAfterSeconds).toBe(60);
  });

  it('reports a failed request with its status', async () => {
    mockFetch({ ok: false, status: 502, text: async () => 'upstream is down' });
    await expect(ollamaCloudClient.generateWithTrace('p', options)).rejects.toThrow(/502/);
  });
});

describe('isCloudReachable', () => {
  it('is false straight away when the browser reports offline', async () => {
    vi.stubGlobal('navigator', { onLine: false });
    const fetchMock = mockFetch({});
    expect(await isCloudReachable()).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('confirms reachability with a real request rather than trusting the flag', async () => {
    const fetchMock = mockFetch({ ok: true });
    expect(await isCloudReachable()).toBe(true);
    expect(fetchMock.mock.calls[0]![1].method).toBe('HEAD');
  });

  it('is false when the probe request fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    expect(await isCloudReachable()).toBe(false);
  });

  it('is false when the route answers with an error', async () => {
    mockFetch({ ok: false, status: 503 });
    expect(await isCloudReachable()).toBe(false);
  });
});
