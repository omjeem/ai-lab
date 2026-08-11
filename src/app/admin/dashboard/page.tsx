'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ChevronLeft, ChevronRight, LogOut, X } from 'lucide-react';
import { Button, Heading, Panel, Readout, Tag, cx } from '@/components/ui';
import type { AdminUserRow } from '@/types/user';
import type { ActivityEvent } from '@/types/activity';

interface UsersResponse {
  ok: boolean;
  total: number;
  page: number;
  size: number;
  users: AdminUserRow[];
  error?: string;
}

interface ActivityResponse {
  ok: boolean;
  total: number;
  events: (ActivityEvent & { receivedAt: string })[];
  error?: string;
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const [data, setData] = useState<UsersResponse | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<AdminUserRow | null>(null);

  const load = useCallback(async (nextPage: number) => {
    try {
      const response = await fetch(`/api/admin/users?page=${nextPage}&size=25`);
      if (response.status === 401) {
        router.push('/admin/login');
        return;
      }
      const payload = (await response.json()) as UsersResponse;
      if (payload.ok) setData(payload);
      else setError(payload.error ?? 'Could not load users');
    } catch {
      setError('Could not reach the server');
    }
  }, [router]);

  useEffect(() => {
    void load(page);
  }, [page, load]);

  const signOut = async () => {
    await fetch('/api/admin/auth', { method: 'DELETE' });
    router.push('/admin/login');
  };

  const pages = data ? Math.max(1, Math.ceil(data.total / data.size)) : 1;

  return (
    <div className="flex min-h-dvh flex-col bg-base">
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-baseline gap-3">
          <span className="font-display text-sm font-semibold text-primary">AI Learning Lab</span>
          <span className="label">admin</span>
        </div>
        <Button onClick={() => void signOut()}>
          <LogOut size={13} strokeWidth={2} />
          Sign out
        </Button>
      </header>

      <main className="flex-1 space-y-4 p-4">
        {error && (
          <Panel label="error">
            <p className="text-sm text-bad">{error}</p>
          </Panel>
        )}

        <div className="flex flex-wrap gap-8 border-b border-line pb-4">
          <Readout label="total users" value={data?.total ?? '—'} size="lg" tone="accent" />
          <Readout label="page" value={`${(data?.page ?? 0) + 1}/${pages}`} size="md" />
        </div>

        <Panel label="users" flush>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[840px] text-left text-xs">
              <thead>
                <tr className="border-b border-line">
                  {['name', 'id', 'first seen', 'last seen', 'location', 'ip', 'events'].map((h) => (
                    <th key={h} className="label px-3 py-2 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data?.users.map((user) => (
                  <tr
                    key={user.userId}
                    onClick={() => setSelected(user)}
                    className="cursor-pointer border-b border-line-faint transition-colors hover:bg-raised"
                  >
                    <td className="px-3 py-2 text-primary">{user.name}</td>
                    <td className="readout px-3 py-2 text-[10px] text-muted">
                      {user.userId.slice(0, 8)}…
                    </td>
                    <td className="readout px-3 py-2 text-muted">{formatDate(user.firstSeen)}</td>
                    <td className="readout px-3 py-2 text-muted">{formatDate(user.lastSeen)}</td>
                    <td className="px-3 py-2 text-secondary">
                      {[user.city, user.country].filter(Boolean).join(', ') || '—'}
                    </td>
                    <td className="readout px-3 py-2 text-muted">{user.ip ?? '—'}</td>
                    <td className="readout px-3 py-2 text-accent">{user.eventCount}</td>
                  </tr>
                ))}
                {data?.users.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-muted">
                      No users yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="flex items-center gap-2">
          <Button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
            <ChevronLeft size={13} strokeWidth={2} />
            Previous
          </Button>
          <Button onClick={() => setPage((p) => p + 1)} disabled={page + 1 >= pages}>
            Next
            <ChevronRight size={13} strokeWidth={2} />
          </Button>
        </div>
      </main>

      {selected && <UserDrilldown user={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function UserDrilldown({ user, onClose }: { user: AdminUserRow; onClose: () => void }) {
  const [data, setData] = useState<ActivityResponse | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    void (async () => {
      const response = await fetch(
        `/api/admin/activity?userId=${encodeURIComponent(user.userId)}&page=${page}&size=50`
      );
      setData((await response.json()) as ActivityResponse);
    })();
  }, [user.userId, page]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-void/70 backdrop-blur-sm">
      <div className="flex h-full w-full max-w-2xl flex-col border-l border-line bg-panel">
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div>
            <Heading level={3}>{user.name}</Heading>
            <p className="readout mt-1 text-[10px] text-muted">{user.userId}</p>
          </div>
          <button onClick={onClose} aria-label="Close" className="p-1 text-muted hover:text-primary">
            <X size={16} strokeWidth={2} />
          </button>
        </header>

        <div className="flex flex-wrap gap-2 border-b border-line px-4 py-3">
          {user.country && <Tag>{[user.city, user.country].filter(Boolean).join(', ')}</Tag>}
          {user.ip && <Tag>{user.ip}</Tag>}
          <Tag tone="accent">{data?.total ?? user.eventCount} events</Tag>
        </div>

        {user.userAgent && (
          <p className="border-b border-line px-4 py-2 text-[11px] leading-relaxed text-muted">
            {user.userAgent}
          </p>
        )}

        <ol className="flex-1 divide-y divide-line-faint overflow-y-auto">
          {data?.events.map((event) => (
            <li key={event.eventId} className="flex items-start gap-3 px-4 py-2.5">
              <span className="readout w-32 shrink-0 text-[10px] text-muted">
                {formatDate(new Date(event.timestamp).toISOString())}
              </span>
              <div className="min-w-0 flex-1">
                <span className={cx('font-mono text-[11px]', toneFor(event.type))}>
                  {event.type}
                </span>
                {(event.chapterId || event.levelId) && (
                  <span className="ml-2 text-[11px] text-secondary">
                    {[event.chapterId, event.levelId].filter(Boolean).join(' · ')}
                  </span>
                )}
                {event.detail && (
                  <p className="readout mt-0.5 truncate text-[10px] text-muted">
                    {Object.entries(event.detail)
                      .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
                      .join('  ')}
                  </p>
                )}
              </div>
            </li>
          ))}
          {data?.events.length === 0 && (
            <li className="px-4 py-8 text-center text-xs text-muted">No activity recorded</li>
          )}
        </ol>

        <footer className="flex items-center gap-2 border-t border-line px-4 py-3">
          <Button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
            <ChevronLeft size={13} strokeWidth={2} />
          </Button>
          <span className="label">page {page + 1}</span>
          <Button
            onClick={() => setPage((p) => p + 1)}
            disabled={!data || (page + 1) * 50 >= data.total}
          >
            <ChevronRight size={13} strokeWidth={2} />
          </Button>
        </footer>
      </div>
    </div>
  );
}

function toneFor(type: string): string {
  if (type.includes('failed')) return 'text-bad';
  if (type.includes('completed') || type.includes('unlocked')) return 'text-good';
  return 'text-secondary';
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? '—'
    : date.toISOString().replace('T', ' ').slice(0, 16);
}
