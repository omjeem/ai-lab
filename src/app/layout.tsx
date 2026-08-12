import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono, Space_Grotesk } from 'next/font/google';
import './globals.css';
import { AppShell } from '@/components/AppShell';
import { THEME_INIT_SCRIPT } from '@/lib/theme';

/* Self-hosted at build time, so first paint works offline after install. */
const inter = Inter({ variable: '--font-inter', subsets: ['latin'], display: 'swap' });

const jetbrainsMono = JetBrains_Mono({
  variable: '--font-jetbrains-mono',
  subsets: ['latin'],
  display: 'swap',
});

const spaceGrotesk = Space_Grotesk({
  variable: '--font-space-grotesk',
  subsets: ['latin'],
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'AI Learning Lab',
  description:
    'Learn how models actually work by running them. Real embeddings, real attention, real gradients — in your browser.',
  applicationName: 'AI Learning Lab',
  manifest: '/manifest.webmanifest',
  appleWebApp: { capable: true, title: 'AI Lab', statusBarStyle: 'black-translucent' },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0f' },
    { media: '(prefers-color-scheme: light)', color: '#f4f6f9' },
  ],
  colorScheme: 'dark light',
  width: 'device-width',
  initialScale: 1,
  // The canvases are drag surfaces; a double-tap zoom would fight them.
  maximumScale: 1,
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} ${spaceGrotesk.variable} h-full`}
      suppressHydrationWarning
    >
      <head>
        {/* Blocking on purpose: it must land before the first paint, or the
            player sees a flash of the theme they did not pick. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
