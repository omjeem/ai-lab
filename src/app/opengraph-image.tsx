import { ImageResponse } from 'next/og';
import { SITE_DESCRIPTION } from '@/lib/seo';

export const runtime = 'edge';
export const alt = 'AI Learning Lab — learn how language models actually work by operating one.';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

/** Same three-node mark as public/icon.svg, redrawn at social-card scale. */
function Mark() {
  return (
    <svg width="120" height="120" viewBox="0 0 64 64">
      <path d="M16 44 L32 20 L48 44" stroke="#4cc2ff" strokeWidth="2.4" fill="none" />
      <path d="M16 44 L48 44" stroke="#4cc2ff" strokeWidth="2.4" fill="none" />
      <rect x="12" y="40" width="8" height="8" fill="#4cc2ff" />
      <rect x="28" y="16" width="8" height="8" fill="#4cc2ff" />
      <rect x="44" y="40" width="8" height="8" fill="#4cc2ff" />
    </svg>
  );
}

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '72px',
          background: '#0a0a0f',
          backgroundImage:
            'radial-gradient(circle at 82% 18%, rgba(76,194,255,0.16), transparent 55%)',
          fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <Mark />
          <span
            style={{
              fontSize: 30,
              fontWeight: 600,
              letterSpacing: '0.02em',
              color: '#f4f6f9',
            }}
          >
            AI Learning Lab
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 980 }}>
          <div
            style={{
              fontSize: 60,
              fontWeight: 700,
              lineHeight: 1.12,
              color: '#f4f6f9',
              letterSpacing: '-0.01em',
            }}
          >
            Learn how language models actually work — by operating one.
          </div>
          <div style={{ fontSize: 25, lineHeight: 1.5, color: '#9aa1ad', maxWidth: 900 }}>
            {SITE_DESCRIPTION}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 14 }}>
          {['real embeddings', 'real attention', 'real gradients', 'runs in your browser'].map(
            (label) => (
              <div
                key={label}
                style={{
                  display: 'flex',
                  padding: '10px 18px',
                  border: '1px solid rgba(76,194,255,0.35)',
                  borderRadius: 6,
                  color: '#4cc2ff',
                  fontSize: 18,
                  fontFamily: 'monospace',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                }}
              >
                {label}
              </div>
            )
          )}
        </div>
      </div>
    ),
    { ...size }
  );
}
