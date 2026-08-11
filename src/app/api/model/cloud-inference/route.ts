/**
 * Ollama Cloud proxy — the only server-side model call in the product.
 *
 * The API key never leaves this process. Rate limited by IP, since the route is
 * public and unauthenticated and would otherwise be an open tap on someone's
 * cloud spend (Section 11.4).
 */
import { NextResponse } from 'next/server';
import { z } from 'zod';
import { clientIpFrom, rateLimit, readLimit } from '@/lib/rateLimit';
import type { CloudInferenceResponse } from '@/models/ollamaCloudClient';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const UPSTREAM_TIMEOUT_MS = 60_000;

const requestSchema = z.object({
  prompt: z.string().min(1).max(4000),
  maxTokens: z.number().int().min(1).max(2048),
  temperature: z.number().min(0).max(2),
  topP: z.number().min(0).max(1),
});

function isConfigured(): boolean {
  return Boolean(process.env.OLLAMA_CLOUD_API_KEY && process.env.OLLAMA_CLOUD_MODEL_ID);
}

/** Reachability probe used by the World 6 toggle. */
export async function HEAD(): Promise<Response> {
  return new Response(null, { status: isConfigured() ? 200 : 503 });
}

export async function POST(request: Request): Promise<Response> {
  const ip = clientIpFrom(request.headers) ?? 'unknown';
  const limit = rateLimit(
    `cloud:${ip}`,
    readLimit('RATE_LIMIT_CLOUD_INFERENCE_PER_MIN', 10)
  );

  if (!limit.allowed) {
    return NextResponse.json(
      { error: 'Rate limit exceeded' },
      { status: 429, headers: { 'Retry-After': String(limit.retryAfterSeconds) } }
    );
  }

  if (!isConfigured()) {
    return NextResponse.json(
      { error: 'Cloud escalation is not configured on this deployment' },
      { status: 503 }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const parsed = requestSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid request shape' }, { status: 400 });
  }

  const baseUrl = process.env.OLLAMA_CLOUD_BASE_URL ?? 'https://ollama.com';
  const model = process.env.OLLAMA_CLOUD_MODEL_ID!;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, '')}/api/chat`, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.OLLAMA_CLOUD_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        stream: false,
        messages: [{ role: 'user', content: parsed.data.prompt }],
        options: {
          num_predict: parsed.data.maxTokens,
          temperature: parsed.data.temperature,
          top_p: parsed.data.topP,
        },
      }),
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '');
      // The upstream body may name the key or account; do not pass it through.
      console.error('Ollama Cloud request failed', upstream.status, detail.slice(0, 200));
      return NextResponse.json(
        { error: `Cloud model returned ${upstream.status}` },
        { status: 502 }
      );
    }

    const data = (await upstream.json()) as { message?: { content?: string } };

    // No per-token detail is returned by this endpoint, so `steps` is omitted
    // rather than filled with numbers that would look real and not be.
    return NextResponse.json({
      text: data.message?.content ?? '',
      model,
    } satisfies CloudInferenceResponse);
  } catch (error) {
    const aborted = error instanceof Error && error.name === 'AbortError';
    return NextResponse.json(
      { error: aborted ? 'Cloud model timed out' : 'Cloud model unreachable' },
      { status: 504 }
    );
  } finally {
    clearTimeout(timer);
  }
}
