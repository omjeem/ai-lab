'use client';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import { Button, Heading } from '@/components/ui';

/**
 * Only ever returns an in-app admin path.
 *
 * `next` comes from the query string, so without this a crafted login link could
 * bounce a freshly authenticated admin to an external site. Protocol-relative
 * URLs (`//evil.com`) are rejected too, since those are not local.
 */
function safeNext(value: string | null): string {
  const fallback = '/admin/dashboard';
  if (!value) return fallback;
  if (!value.startsWith('/admin') || value.startsWith('//')) return fallback;
  // Bouncing back to the login page would loop.
  if (value.startsWith('/admin/login')) return fallback;
  return value;
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch('/api/admin/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const payload = (await response.json()) as { ok: boolean; error?: string };
      if (payload.ok) {
        router.push(safeNext(params.get('next')));
        return;
      }
      setError(payload.error ?? 'Login failed');
    } catch {
      setError('Could not reach the server');
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex w-full max-w-sm flex-col gap-4">
      <label className="flex flex-col gap-1.5">
        <span className="label">email</span>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="username"
          required
          className="border border-line bg-inset px-3 py-2.5 font-mono text-sm text-primary focus:border-accent"
          style={{ borderRadius: 'var(--radius)' }}
        />
      </label>

      <label className="flex flex-col gap-1.5">
        <span className="label">password</span>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          required
          className="border border-line bg-inset px-3 py-2.5 font-mono text-sm text-primary focus:border-accent"
          style={{ borderRadius: 'var(--radius)' }}
        />
      </label>

      {error && <p className="text-xs text-bad">{error}</p>}

      <Button type="submit" variant="primary" disabled={busy}>
        {busy ? 'Checking…' : 'Sign in'}
      </Button>
    </form>
  );
}

export default function AdminLoginPage() {
  return (
    <div className="grid-field flex min-h-dvh flex-col items-center justify-center gap-6 px-6">
      <div className="w-full max-w-sm">
        <p className="label mb-2">ai learning lab</p>
        <Heading level={2}>Admin</Heading>
      </div>
      <Suspense fallback={null}>
        <LoginForm />
      </Suspense>
    </div>
  );
}
