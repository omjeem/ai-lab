'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ChevronLeft, ChevronRight, LogOut, Trash2, X } from 'lucide-react';
import { Button, Heading, Meter, Panel, Readout, Tag, cx } from '@/components/ui';
import type { AdminUserRow } from '@/types/user';
import type { ActivityEvent } from '@/types/activity';
import type { ChapterAnalyticsRow } from '@/lib/adminAnalytics';
import type { AdminFeedbackRow } from '@/types/feedback';

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

interface ChapterAnalyticsResponse {
  ok: boolean;
  chapters: ChapterAnalyticsRow[];
  error?: string;
}

interface FeedbackResponse {
  ok: boolean;
  total: number;
  page: number;
  size: number;
  feedback: AdminFeedbackRow[];
  error?: string;
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const [tab, setTab] = useState<'users' | 'chapters' | 'feedback'>('users');
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
        <div className="flex items-center gap-2">
          <Button variant={tab === 'users' ? 'primary' : 'ghost'} onClick={() => setTab('users')}>
            Users
          </Button>
          <Button
            variant={tab === 'chapters' ? 'primary' : 'ghost'}
            onClick={() => setTab('chapters')}
          >
            Chapter analytics
          </Button>
          <Button
            variant={tab === 'feedback' ? 'primary' : 'ghost'}
            onClick={() => setTab('feedback')}
          >
            Feedback
          </Button>
        </div>

        {error && (
          <Panel label="error">
            <p className="text-sm text-bad">{error}</p>
          </Panel>
        )}

        {tab === 'users' ? (
          <>
            <div className="flex flex-wrap gap-8 border-b border-line pb-4">
              <Readout label="total users" value={data?.total ?? '—'} size="lg" tone="accent" />
              <Readout label="page" value={`${(data?.page ?? 0) + 1}/${pages}`} size="md" />
            </div>

            <Panel label="users" flush>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[840px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-line">
                      {['name', 'id', 'first seen', 'last seen', 'location', 'ip', 'events'].map(
                        (h) => (
                          <th key={h} className="label px-3 py-2 font-normal">
                            {h}
                          </th>
                        )
                      )}
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
          </>
        ) : tab === 'chapters' ? (
          <ChapterAnalyticsPanel />
        ) : (
          <FeedbackPanel />
        )}
      </main>

      {selected && (
        <UserDrilldown
          user={selected}
          onClose={() => setSelected(null)}
          onDeleted={() => {
            setSelected(null);
            void load(page);
          }}
        />
      )}
    </div>
  );
}

function UserDrilldown({
  user,
  onClose,
  onDeleted,
}: {
  user: AdminUserRow;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [data, setData] = useState<ActivityResponse | null>(null);
  const [page, setPage] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    void (async () => {
      const response = await fetch(
        `/api/admin/activity?userId=${encodeURIComponent(user.userId)}&page=${page}&size=50`
      );
      setData((await response.json()) as ActivityResponse);
    })();
  }, [user.userId, page]);

  const deleteUser = async () => {
    const confirmed = window.confirm(
      `Permanently delete ${user.name} and all ${user.eventCount} recorded activity events? ` +
        'This cannot be undone.'
    );
    if (!confirmed || deleting) return;

    setDeleting(true);
    try {
      const response = await fetch(`/api/admin/users?userId=${encodeURIComponent(user.userId)}`, {
        method: 'DELETE',
      });
      const payload = (await response.json()) as { ok: boolean; error?: string };
      if (!payload.ok) {
        window.alert(payload.error ?? 'Could not delete this user');
        return;
      }
      onDeleted();
    } catch {
      window.alert('Could not reach the server');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-void/70 backdrop-blur-sm">
      <div className="flex h-full w-full max-w-2xl flex-col border-l border-line bg-panel">
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div>
            <Heading level={3}>{user.name}</Heading>
            <p className="readout mt-1 text-[10px] text-muted">{user.userId}</p>
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={() => void deleteUser()} variant="danger" disabled={deleting}>
              <Trash2 size={13} strokeWidth={2} />
              {deleting ? 'Deleting…' : 'Delete user'}
            </Button>
            <button onClick={onClose} aria-label="Close" className="p-1 text-muted hover:text-primary">
              <X size={16} strokeWidth={2} />
            </button>
          </div>
        </header>

        <div className="flex flex-wrap gap-2 border-b border-line px-4 py-3">
          {user.country && <Tag>{[user.city, user.country].filter(Boolean).join(', ')}</Tag>}
          {user.ip && <Tag>{user.ip}</Tag>}
          {(user.browser || user.os) && (
            <Tag>{[user.browser, user.os].filter(Boolean).join(' · ')}</Tag>
          )}
          {user.deviceType && user.deviceType !== 'desktop' && <Tag>{user.deviceType}</Tag>}
          <Tag tone="accent">{data?.total ?? user.eventCount} events</Tag>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-1 gap-3 border-b border-line p-4 sm:grid-cols-2">
            <DetailSection
              title="Location & network"
              rows={[
                { label: 'Country', value: user.country },
                { label: 'City', value: user.city },
                { label: 'IP address', value: user.ip },
                {
                  label: 'Coordinates',
                  value: user.latitude && user.longitude ? `${user.latitude}, ${user.longitude}` : null,
                },
                { label: 'Geo timezone', value: user.geoTimezone },
                { label: 'Host', value: user.host },
              ]}
            />
            <DetailSection
              title="How they arrived"
              rows={[
                { label: 'First seen', value: formatDate(user.firstSeen) },
                { label: 'Last seen', value: formatDate(user.lastSeen) },
                { label: 'First source', value: user.source },
                { label: 'Landing page', value: user.landingPage },
                { label: 'Landing title', value: user.landingTitle },
                { label: 'Last page', value: user.lastPage },
              ]}
            />
            <DetailSection
              title="Device"
              rows={[
                { label: 'Device', value: user.deviceType },
                { label: 'OS', value: [user.os, user.osVersion].filter(Boolean).join(' ') || null },
                {
                  label: 'Browser',
                  value: [user.browser, user.browserVersion].filter(Boolean).join(' ') || null,
                },
                { label: 'Platform', value: user.platform },
                { label: 'Screen', value: user.screen },
                { label: 'Viewport', value: user.viewport },
                { label: 'Pixel ratio', value: user.pixelRatio },
                {
                  label: 'Touch',
                  value: user.touchSupport === null ? null : user.touchSupport ? 'yes' : 'no',
                },
                { label: 'Memory', value: user.deviceMemoryGb ? `${user.deviceMemoryGb} GB` : null },
                { label: 'CPU cores', value: user.cpuCores },
                { label: 'Network', value: user.networkType },
                { label: 'Color scheme', value: user.colorScheme },
              ]}
            />
            <DetailSection
              title="Locale & agent"
              rows={[
                { label: 'Timezone', value: user.timezone },
                { label: 'Languages', value: user.languages },
                { label: 'Accept-Language', value: user.acceptLanguage },
                { label: 'User-Agent', value: user.userAgent, wrap: true },
              ]}
            />
          </div>

          <ol className="divide-y divide-line-faint">
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
        </div>

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

interface DetailRow {
  label: string;
  value: string | number | boolean | null | undefined;
  /** Long values (the raw user-agent string) wrap instead of truncating. */
  wrap?: boolean;
}

/** One labeled group of key/value rows — empty/null rows and empty sections are hidden. */
function DetailSection({ title, rows }: { title: string; rows: DetailRow[] }) {
  const visible = rows.filter((r) => r.value !== null && r.value !== undefined && r.value !== '');
  if (visible.length === 0) return null;

  return (
    <div className="border border-line p-3">
      <p className="label mb-2 text-accent">{title}</p>
      <dl className="divide-y divide-line-faint">
        {visible.map((row) => (
          <div key={row.label} className="flex items-start justify-between gap-3 py-1.5 text-[11px]">
            <dt className="shrink-0 text-muted">{row.label}</dt>
            <dd className={cx('text-right text-secondary', row.wrap ? 'wrap-break-word' : 'truncate')}>
              {String(row.value)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Per-world/per-chapter breakdown: who started/completed each chapter, and what else they did there. */
function ChapterAnalyticsPanel() {
  const router = useRouter();
  const [data, setData] = useState<ChapterAnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch('/api/admin/analytics');
        if (response.status === 401) {
          router.push('/admin/login');
          return;
        }
        const payload = (await response.json()) as ChapterAnalyticsResponse;
        if (payload.ok) setData(payload);
        else setError(payload.error ?? 'Could not load chapter analytics');
      } catch {
        setError('Could not reach the server');
      }
    })();
  }, [router]);

  if (error) {
    return (
      <Panel label="error">
        <p className="text-sm text-bad">{error}</p>
      </Panel>
    );
  }

  return (
    <Panel label="chapter analytics" flush>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[960px] text-left text-xs">
          <thead>
            <tr className="border-b border-line">
              {[
                'world',
                'chapter',
                'started',
                'completed',
                'completion',
                'jumped ahead',
                'shared opens',
                'levels pass/fail',
              ].map((h) => (
                <th key={h} className="label px-3 py-2 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data?.chapters.map((row) => (
              <tr key={row.chapterId} className="border-b border-line-faint">
                <td className="readout px-3 py-2 text-muted">W{row.world}</td>
                <td className="px-3 py-2 text-primary">
                  {row.chapterTitle}
                  <span className="readout ml-2 text-[10px] text-muted">{row.chapterId}</span>
                </td>
                <td className="readout px-3 py-2 text-secondary">{row.startedUsers}</td>
                <td className="readout px-3 py-2 text-secondary">{row.completedUsers}</td>
                <td className="px-3 py-2">
                  <div className="w-24">
                    <Meter
                      value={row.completionRate}
                      tone={row.completionRate >= 0.5 ? 'good' : 'accent'}
                      label={`${Math.round(row.completionRate * 100)}%`}
                    />
                  </div>
                </td>
                <td className="readout px-3 py-2 text-warn">{row.jumpedAhead || '—'}</td>
                <td className="readout px-3 py-2 text-accent">{row.sharedLinkOpened || '—'}</td>
                <td className="readout px-3 py-2 text-muted">
                  {row.levelCompleted}/{row.levelFailed}
                </td>
              </tr>
            ))}
            {data?.chapters.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-8 text-center text-muted">
                  No activity recorded yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/** Free-form feedback submitted from the chapter-completion screen, newest first. */
function FeedbackPanel() {
  const router = useRouter();
  const [data, setData] = useState<FeedbackResponse | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch(`/api/admin/feedback?page=${page}&size=25`);
        if (response.status === 401) {
          router.push('/admin/login');
          return;
        }
        const payload = (await response.json()) as FeedbackResponse;
        if (payload.ok) setData(payload);
        else setError(payload.error ?? 'Could not load feedback');
      } catch {
        setError('Could not reach the server');
      }
    })();
  }, [page, router]);

  if (error) {
    return (
      <Panel label="error">
        <p className="text-sm text-bad">{error}</p>
      </Panel>
    );
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / data.size)) : 1;

  return (
    <>
      <div className="flex flex-wrap gap-8 border-b border-line pb-4">
        <Readout label="total feedback" value={data?.total ?? '—'} size="lg" tone="accent" />
        <Readout label="page" value={`${(data?.page ?? 0) + 1}/${pages}`} size="md" />
      </div>

      <Panel label="feedback" flush>
        <ol className="divide-y divide-line-faint">
          {data?.feedback.map((row) => (
            <li key={row.feedbackId} className="flex flex-col gap-1.5 px-4 py-3">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="readout text-[10px] text-muted">{formatDate(row.receivedAt)}</span>
                <span className="text-xs text-primary">{row.displayName ?? 'anonymous'}</span>
                <span className="readout text-[10px] text-muted">{row.userId.slice(0, 8)}…</span>
                {row.email && <Tag tone="accent">{row.email}</Tag>}
                {row.chapterTitle && <Tag>{row.chapterTitle}</Tag>}
              </div>
              <p className="whitespace-pre-wrap text-xs leading-relaxed text-secondary">
                {row.message}
              </p>
            </li>
          ))}
          {data?.feedback.length === 0 && (
            <li className="px-4 py-8 text-center text-xs text-muted">No feedback submitted yet</li>
          )}
        </ol>
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
    </>
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
