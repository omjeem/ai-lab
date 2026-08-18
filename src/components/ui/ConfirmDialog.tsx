'use client';

/**
 * A full-screen "are you sure" dialog with real body copy — for the one case
 * in the app where a native `window.confirm()` (used for the two simpler
 * "delete my data" flows) can't hold enough explanation. Modelled on
 * `ChapterShell`'s `AhaReveal` overlay markup, kept as its own primitive
 * rather than folding that already-shipped completion screen through a
 * shared abstraction.
 */
import type { ReactNode } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Button, Heading } from '@/components/ui';

export function ConfirmDialog({
  title,
  body,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onCancel,
}: {
  title: string;
  body: ReactNode;
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const reduce = useReducedMotion();

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-void/95 px-6 backdrop-blur-sm"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: reduce ? 0 : 0.22 }}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <motion.div
        className="w-full max-w-lg border border-line bg-panel p-6"
        style={{ borderRadius: 'var(--radius-lg)' }}
        initial={reduce ? false : { y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: reduce ? 0 : 0.32, ease: [0.16, 1, 0.3, 1] }}
      >
        <p className="label mb-3">heads up</p>
        <Heading level={2} className="mb-4">
          {title}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">{body}</p>

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" onClick={onConfirm}>
            {confirmLabel}
          </Button>
          <Button onClick={onCancel}>{cancelLabel}</Button>
        </div>
      </motion.div>
    </motion.div>
  );
}
