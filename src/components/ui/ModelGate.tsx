'use client';

/**
 * The screen between "chapter opened" and "chapter playable".
 *
 * Section 11.2: model downloads fail — a dropped connection, no WebGPU, a full
 * storage quota — and the failure state has to be a real one with a retry, not a
 * blank canvas. This component owns that whole path, including the chapter's own
 * `loadFailureMessage` from its JSON.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { RotateCw, HardDriveDownload, TriangleAlert } from 'lucide-react';
import { subscribeToModelLoads, clearError, type ModelLoadState } from '@/models/modelCache';
import { detectCapabilities, hasRoomFor } from '@/lib/deviceCapabilities';
import { Button, Heading, Meter, Tag, cx } from '@/components/ui';

export interface ModelGateProps {
  modelId: string | null;
  estimatedSizeMB: number | null;
  /** Shown when the download fails — comes from the chapter definition. */
  loadFailureMessage: string | null;
  /** Kicks off the load. Called again by the retry button. */
  load: () => Promise<void>;
  children: ReactNode;
}

type GateStatus = 'idle' | 'loading' | 'ready' | 'error';

export function ModelGate({
  modelId,
  estimatedSizeMB,
  loadFailureMessage,
  load,
  children,
}: ModelGateProps) {
  const [status, setStatus] = useState<GateStatus>(modelId ? 'idle' : 'ready');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ModelLoadState | null>(null);
  const [storageWarning, setStorageWarning] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!modelId) return;
    return subscribeToModelLoads((state) => {
      if (state.modelId === modelId) setProgress(state);
    });
  }, [modelId]);

  useEffect(() => {
    if (!modelId) return;
    let cancelled = false;

    void (async () => {
      setStatus('loading');
      setError(null);

      // Warn before spending the download rather than failing part-way through.
      if (estimatedSizeMB && estimatedSizeMB > 50) {
        const capabilities = await detectCapabilities();
        if (!hasRoomFor(capabilities, estimatedSizeMB)) {
          setStorageWarning(
            `This model needs about ${estimatedSizeMB}MB and your browser reports limited storage. The download may fail.`
          );
        }
      }

      try {
        await load();
        if (!cancelled) setStatus('ready');
      } catch (caught) {
        if (cancelled) return;
        setError(caught instanceof Error ? caught.message : String(caught));
        setStatus('error');
      }
    })();

    return () => {
      cancelled = true;
    };
    // `attempt` is the retry trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, attempt]);

  if (status === 'ready') return <>{children}</>;

  if (status === 'error') {
    return (
      <GateFrame>
        <span className="text-bad">
          <TriangleAlert size={22} strokeWidth={1.75} />
        </span>
        <Heading level={2}>This chapter could not start</Heading>

        <p className="max-w-prose text-sm leading-relaxed text-secondary">
          {loadFailureMessage ??
            'The model this chapter needs could not be loaded. Check your connection and try again.'}
        </p>

        {error && (
          <p className="readout max-w-prose text-xs text-muted">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            onClick={() => {
              if (modelId) clearError(modelId);
              setAttempt((n) => n + 1);
            }}
          >
            <RotateCw size={13} strokeWidth={2} />
            Retry download
          </Button>
        </div>
      </GateFrame>
    );
  }

  return (
    <GateFrame>
      <LoadingPulse />
      <Heading level={2}>Loading the model</Heading>

      <p className="max-w-prose text-sm leading-relaxed text-secondary">
        This chapter runs a real model in your browser. It downloads once, then works offline.
      </p>

      <div className="w-full max-w-sm space-y-3">
        <Meter
          value={progress?.progress ?? 0}
          max={1}
          label={progress?.file ? `fetching ${trimFile(progress.file)}` : 'preparing'}
        />
        <div className="flex items-baseline justify-between gap-3">
          <span className="readout text-sm text-accent">
            {progress?.progress !== null && progress?.progress !== undefined
              ? `${Math.round(progress.progress * 100)}%`
              : '—'}
          </span>
          <span className="label">
            {progress?.totalMB
              ? `${progress.loadedMB ?? 0} / ${progress.totalMB} MB`
              : estimatedSizeMB
                ? `~${estimatedSizeMB} MB`
                : ''}
          </span>
        </div>
      </div>

      {storageWarning && (
        <Tag tone="warn">
          <HardDriveDownload size={10} strokeWidth={2} />
          {storageWarning}
        </Tag>
      )}
    </GateFrame>
  );
}

function GateFrame({ children }: { children: ReactNode }) {
  return (
    <div className="grid-field flex flex-1 flex-col items-center justify-center gap-4 px-6 py-16 text-center">
      {children}
    </div>
  );
}

/** A slow breathing square — deliberately not a spinner. */
function LoadingPulse() {
  const reduce = useReducedMotion();
  return (
    <motion.span
      className={cx('block h-5 w-5 border border-accent')}
      animate={reduce ? undefined : { opacity: [0.25, 1, 0.25], scale: [0.9, 1, 0.9] }}
      transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
      aria-hidden
    />
  );
}

function trimFile(file: string): string {
  const parts = file.split('/');
  return parts[parts.length - 1] ?? file;
}
