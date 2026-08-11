/**
 * Edge-safe session verification.
 *
 * The middleware runs on the Edge runtime, which has no `node:crypto`, so the
 * signature check is repeated here against Web Crypto. This is only the page
 * redirect gate — every `/api/admin/*` route still re-verifies with the Node
 * implementation, which is the check that actually protects the data.
 */
import type { AdminSession } from './adminAuth';

export const ADMIN_COOKIE = 'ail_admin';

function base64UrlToBytes(value: string): ArrayBuffer {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4));
  // Backed by a plain ArrayBuffer so it satisfies BufferSource for subtle.verify.
  const buffer = new ArrayBuffer(binary.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return buffer;
}

export async function verifySessionTokenEdge(
  token: string | undefined | null
): Promise<AdminSession | null> {
  const secret = process.env.ADMIN_SESSION_SECRET;
  if (!secret || !token) return null;

  const [payload, signature] = token.split('.');
  if (!payload || !signature) return null;

  try {
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['verify']
    );

    const valid = await crypto.subtle.verify(
      'HMAC',
      key,
      base64UrlToBytes(signature),
      new TextEncoder().encode(payload)
    );
    if (!valid) return null;

    const session = JSON.parse(new TextDecoder().decode(base64UrlToBytes(payload))) as AdminSession;
    if (typeof session.expiresAt !== 'number' || session.expiresAt < Date.now()) return null;
    return session;
  } catch {
    return null;
  }
}
