'use client';

/**
 * The frame every screen sits inside.
 *
 * Owns the three things that must run regardless of which screen is open: the
 * background sync loop (Section 9), connectivity state, and the small status
 * rail that tells the player which inference backend they are on (Section 11.3).
 */
import { useEffect, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Cpu, Wifi, WifiOff, Volume2, VolumeX } from 'lucide-react';
import { detectCapabilities, type DeviceCapabilities } from '@/lib/deviceCapabilities';
import { startSyncManager } from '@/lib/syncManager';
import { useGameProgressStore } from '@/store/useGameProgressStore';
import { cx, Tag } from '@/components/ui';
import { ServiceWorkerRegistrar } from '@/components/ServiceWorkerRegistrar';

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [capabilities, setCapabilities] = useState<DeviceCapabilities | null>(null);
  const [online, setOnline] = useState(true);
  const soundEnabled = useGameProgressStore((s) => s.soundEnabled);
  const setSoundEnabled = useGameProgressStore((s) => s.setSoundEnabled);

  useEffect(() => {
    void detectCapabilities().then(setCapabilities);
  }, []);

  // The sync manager runs for the whole app lifetime, not per screen.
  useEffect(() => startSyncManager({ onConnectivityChange: setOnline }), []);

  const isAdmin = pathname?.startsWith('/admin') ?? false;
  if (isAdmin) return <>{children}</>;

  return (
    <div className="flex min-h-dvh flex-col bg-base">
      <ServiceWorkerRegistrar />
      <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-line bg-base/90 px-3 py-2 backdrop-blur-sm sm:px-4">
        <Link href="/map" className="flex items-baseline gap-2">
          <span className="font-display text-sm font-semibold tracking-tight text-primary">
            AI Learning Lab
          </span>
          <span className="label hidden sm:inline">instrument</span>
        </Link>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSoundEnabled(!soundEnabled)}
            aria-label={soundEnabled ? 'Mute feedback sounds' : 'Enable feedback sounds'}
            title={soundEnabled ? 'Sound on' : 'Sound off'}
            className="p-1.5 text-muted transition-colors hover:text-primary"
          >
            {soundEnabled ? <Volume2 size={14} strokeWidth={1.75} /> : <VolumeX size={14} strokeWidth={1.75} />}
          </button>

          <BackendIndicator capabilities={capabilities} />

          <span
            title={online ? 'Online — activity is syncing' : 'Offline — activity is queued locally'}
            className={cx('flex items-center gap-1', online ? 'text-muted' : 'text-warn')}
          >
            {online ? <Wifi size={14} strokeWidth={1.75} /> : <WifiOff size={14} strokeWidth={1.75} />}
            <span className="label hidden sm:inline">{online ? 'online' : 'offline'}</span>
          </span>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col">{children}</main>
    </div>
  );
}

/**
 * Which backend inference is running on.
 *
 * Section 11.3: performance differs visibly between WebGPU and WASM, so the
 * mode is stated rather than left as a mystery.
 */
function BackendIndicator({ capabilities }: { capabilities: DeviceCapabilities | null }) {
  if (!capabilities) return null;

  return (
    <span
      title={
        capabilities.webgpu
          ? 'WebGPU acceleration is active'
          : `Running on WASM — ${capabilities.reason ?? 'WebGPU unavailable'}. Models will be noticeably slower.`
      }
    >
      <Tag tone={capabilities.webgpu ? 'good' : 'warn'}>
        <Cpu size={10} strokeWidth={2} />
        {capabilities.backend}
      </Tag>
    </span>
  );
}
