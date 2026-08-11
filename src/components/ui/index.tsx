'use client';

/**
 * Design-system primitives.
 *
 * The rule these follow (Section 6): the visualisation is the interface and
 * these are the HUD around it. Nothing here is a rounded card with a shadow —
 * structure comes from hairlines, monospace labels and the world accent.
 */
import { motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}

/* ── panel ──────────────────────────────────────────────────── */

export function Panel({
  children,
  className,
  label,
  actions,
  flush,
}: {
  children: ReactNode;
  className?: string;
  label?: string;
  actions?: ReactNode;
  /** Removes inner padding, for panels that hold a canvas edge to edge. */
  flush?: boolean;
}) {
  return (
    <section
      className={cx(
        'border border-line bg-panel/60 backdrop-blur-[1px]',
        'flex min-h-0 flex-col',
        className
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {(label || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line-faint px-3 py-2">
          {label && <span className="label">{label}</span>}
          {actions && <div className="flex items-center gap-1">{actions}</div>}
        </header>
      )}
      <div className={cx('min-h-0 flex-1', !flush && 'p-3')}>{children}</div>
    </section>
  );
}

/* ── buttons ────────────────────────────────────────────────── */

type ButtonVariant = 'primary' | 'ghost' | 'danger';

export function Button({
  children,
  onClick,
  variant = 'ghost',
  disabled,
  type = 'button',
  className,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: ButtonVariant;
  disabled?: boolean;
  type?: 'button' | 'submit';
  className?: string;
  title?: string;
}) {
  const styles: Record<ButtonVariant, string> = {
    primary:
      'bg-accent text-inverse border-accent hover:brightness-110 font-medium',
    ghost: 'bg-transparent text-primary border-line hover:border-line-strong hover:bg-raised',
    danger: 'bg-transparent text-bad border-bad/40 hover:bg-bad/10',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cx(
        'inline-flex items-center justify-center gap-2 border px-3 py-2',
        'font-mono text-xs uppercase tracking-[0.1em]',
        'transition-colors duration-[var(--dur-fast)]',
        // Touch-first: never smaller than a comfortable tap target.
        'min-h-[40px]',
        'disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent',
        styles[variant],
        className
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      {children}
    </button>
  );
}

/* ── readouts ───────────────────────────────────────────────── */

/**
 * A live metric. Section 6 asks for these to be the loudest thing on screen,
 * so they are large, monospaced and high contrast by default.
 */
export function Readout({
  label,
  value,
  unit,
  tone = 'neutral',
  size = 'md',
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: 'neutral' | 'accent' | 'good' | 'warn' | 'bad';
  size?: 'sm' | 'md' | 'lg';
}) {
  const tones = {
    neutral: 'text-primary',
    accent: 'text-accent',
    good: 'text-good',
    warn: 'text-warn',
    bad: 'text-bad',
  };
  const sizes = { sm: 'text-lg', md: 'text-2xl', lg: 'text-4xl' };

  return (
    <div className="flex flex-col gap-0.5">
      <span className="label">{label}</span>
      <span className={cx('readout leading-none', sizes[size], tones[tone])}>
        {typeof value === 'number' ? formatNumber(value) : value}
        {unit && <span className="ml-1 text-[0.5em] text-muted">{unit}</span>}
      </span>
    </div>
  );
}

/** Keeps long decimals from shoving the layout around. */
export function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '—';
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
  return value.toFixed(digits);
}

/* ── meter ──────────────────────────────────────────────────── */

export function Meter({
  value,
  max = 1,
  threshold,
  label,
  tone = 'accent',
}: {
  value: number;
  max?: number;
  /** Draws the pass line, so progress is read against the bar, not a caption. */
  threshold?: number;
  label?: string;
  tone?: 'accent' | 'good' | 'bad';
}) {
  const reduce = useReducedMotion();
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  const thresholdPct = threshold === undefined ? null : Math.max(0, Math.min(1, threshold / max)) * 100;
  const fill = { accent: 'bg-accent', good: 'bg-good', bad: 'bg-bad' }[tone];

  return (
    <div className="flex flex-col gap-1">
      {label && <span className="label">{label}</span>}
      <div className="relative h-1.5 w-full bg-inset">
        <motion.div
          className={cx('absolute inset-y-0 left-0', fill)}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={reduce ? { duration: 0 } : { duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        />
        {thresholdPct !== null && (
          <div
            className="absolute inset-y-[-3px] w-px bg-text-muted"
            style={{ left: `${thresholdPct}%`, background: 'var(--text-muted)' }}
            aria-hidden
          />
        )}
      </div>
    </div>
  );
}

/* ── stars ──────────────────────────────────────────────────── */

export function StarRating({ stars, size = 12 }: { stars: number; size?: number }) {
  return (
    <span
      className="inline-flex items-center gap-0.5"
      role="img"
      aria-label={`${stars} of 3 stars`}
    >
      {[1, 2, 3].map((n) => (
        <svg key={n} width={size} height={size} viewBox="0 0 12 12" aria-hidden>
          {/* Angular, not a cartoon star — this is an instrument readout. */}
          <path
            d="M6 0.5 L7.6 4.4 L11.5 6 L7.6 7.6 L6 11.5 L4.4 7.6 L0.5 6 L4.4 4.4 Z"
            fill={n <= stars ? 'var(--accent)' : 'none'}
            stroke={n <= stars ? 'var(--accent)' : 'var(--line-strong)'}
            strokeWidth="1"
            strokeLinejoin="round"
          />
        </svg>
      ))}
    </span>
  );
}

/* ── controls ───────────────────────────────────────────────── */

/**
 * A slider that is genuinely keyboard-operable.
 *
 * Section 11.5 asks for arrow-key alternatives to dragging; a native range input
 * already gives that, so this wraps one rather than reimplementing it on a div.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  disabled?: boolean;
}) {
  const display = format ? format(value) : formatNumber(value);
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between gap-3">
        <span className="label">{label}</span>
        <span className="readout text-sm text-accent">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step ?? (max - min) / 100}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={label}
        aria-valuetext={display}
      />
    </label>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  disabled,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <label
      className={cx(
        'flex min-h-[40px] cursor-pointer items-center justify-between gap-3',
        disabled && 'cursor-not-allowed opacity-40'
      )}
    >
      <span className="flex flex-col">
        <span className="font-mono text-xs uppercase tracking-[0.1em] text-primary">{label}</span>
        {hint && <span className="text-[11px] text-muted">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          'relative h-5 w-9 shrink-0 border transition-colors duration-[var(--dur-fast)]',
          checked ? 'border-accent bg-accent-dim' : 'border-line bg-inset'
        )}
        style={{ borderRadius: 'var(--radius)' }}
      >
        <span
          className={cx(
            'absolute top-[2px] h-3 w-3 transition-transform duration-[var(--dur-fast)]',
            checked ? 'bg-accent' : 'bg-line-strong'
          )}
          style={{ transform: `translateX(${checked ? 20 : 3}px)` }}
        />
      </button>
    </label>
  );
}

/* ── status ─────────────────────────────────────────────────── */

export function Tag({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'accent' | 'good' | 'warn' | 'bad';
}) {
  const tones = {
    neutral: 'border-line text-muted',
    accent: 'border-accent/40 text-accent',
    good: 'border-good/40 text-good',
    warn: 'border-warn/40 text-warn',
    bad: 'border-bad/40 text-bad',
  };
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 border px-1.5 py-0.5',
        'font-mono text-[10px] uppercase tracking-[0.12em]',
        tones[tone]
      )}
      style={{ borderRadius: '2px' }}
    >
      {children}
    </span>
  );
}

/** Section heading in the display face — never Inter, per Section 6. */
export function Heading({
  children,
  level = 2,
  className,
}: {
  children: ReactNode;
  level?: 1 | 2 | 3;
  className?: string;
}) {
  const Tag = (['h1', 'h2', 'h3'] as const)[level - 1];
  const sizes = { 1: 'text-3xl sm:text-4xl', 2: 'text-xl sm:text-2xl', 3: 'text-base' };
  return (
    <Tag className={cx('font-display font-semibold text-primary', sizes[level], className)}>
      {children}
    </Tag>
  );
}
