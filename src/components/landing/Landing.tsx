/**
 * The landing page.
 *
 * Server-rendered on purpose — a technical visitor should see the pitch and
 * the architecture on first paint, with no client JS gating the content the
 * way the map and onboarding screens gate theirs. Only the CTA needs the
 * client: it has to read IndexedDB progress to decide whether this visitor
 * is new or returning (`HeroActions`).
 */
import {
  Activity,
  Braces,
  Cloud,
  Cpu,
  Database,
  GitBranch,
} from 'lucide-react';
import { manifest } from '@/lib/curriculum';
import { Heading, Panel, Tag } from '@/components/ui';
import { HeroActions } from './HeroActions';

const PROOF_POINTS = [
  '0 API keys',
  '0 servers in the loop to play',
  '780 engine tests',
  '22 chapters · 6 worlds',
  'works offline after first load',
];

const ARCHITECTURE_MODULES = [
  {
    icon: Cpu,
    title: 'Inference runs in this tab',
    body: "transformers.js (ONNX Runtime Web) does embeddings, tokenization, small causal LMs and attention extraction on WebGPU, falling back to WASM automatically — the backend in use is shown live, top right. Model weights cache in the browser's Cache Storage after the first fetch, so a chapter you've opened once needs no network again.",
  },
  {
    icon: Activity,
    title: 'Some models are trained, not downloaded',
    body: 'TinyNet and TinyRNN are hand-rolled nets with real forward and backward passes, trained live in your browser as you play — not pretrained weights. Their analytic gradients are checked against numerical ones in the test suite, so the backprop pulse you watch on screen is verified, not staged.',
  },
  {
    icon: GitBranch,
    title: 'Engines are pure functions',
    body: 'Every game is prepare → initState → applyAction → evaluate. An engine never imports a model — one is injected through a deps parameter. That is why the same code path runs 780 tests offline in about a second and drives real inference in the browser, with nothing mocked in between.',
  },
  {
    icon: Braces,
    title: 'Curriculum is data, not code',
    body: "Every level's parameters, pass criteria and star bands live in versioned JSON, validated by a Zod schema plus cross-file checks — unlock-graph cycles, XP sums, and a calibration script that plays every pure-computation level optimally to confirm its thresholds are actually reachable. Components render engine state; they never own rules.",
  },
  {
    icon: Database,
    title: 'Progress is local-first',
    body: 'A Zustand store persists to IndexedDB, not localStorage — the same local data layer backs the offline activity queue and cached model records. A service worker caches the app shell and bundled corpora; it deliberately does not cache model weights, which would double the storage cost for nothing.',
  },
  {
    icon: Cloud,
    title: 'The one thing that touches a server',
    body: "Next.js API routes exist only to optionally persist progress to MongoDB and to proxy a single optional cloud escalation in the capstone chapter — the API key never reaches the browser. Every exercise before that runs, and is scored, entirely on your machine.",
  },
];

const STACK = [
  'Next.js 15',
  'TypeScript, strict',
  'Tailwind CSS 4',
  'transformers.js',
  'WebLLM',
  'Zustand',
  'IndexedDB',
  'Vitest',
];

const REAL_NOT_FAKED = [
  'Word clusters come from k-means over live embeddings — there is no answer key sitting in the JSON.',
  'The tokenizer puzzle is scored against the real BPE merge-rank table, not a hand-picked sequence.',
  "A trained RNN's memory decay is measured from its actual hidden states as an early token gets overwritten.",
  'N-gram tables are counted from a bundled corpus at runtime — change the corpus, and the model changes.',
  "Attention weights are read from a real transformer's forward pass. If a model doesn't expose them, the chapter says so and offers a retry, rather than showing something plausible.",
];

export function Landing() {
  return (
    <div className="flex flex-1 flex-col">
      <Hero />
      <NotAChatbot />
      <Architecture />
      <Curriculum />
      <WhatsReal />
      <ClosingCta />
    </div>
  );
}

function Hero() {
  return (
    <section className="grid-field border-b border-line px-4 py-14 sm:px-8 sm:py-20">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <p className="label">ai learning lab · an instrument, not a course reader</p>

        <Heading level={1} className="text-balance">
          Learn how language models actually work — by operating one.
        </Heading>

        <p className="text-sm leading-relaxed text-secondary sm:text-[15px]">
          Vectors, gradients, attention — normally taught as equations on a slide, or as a chatbot
          describing its own internals back to you in prose. Here they&apos;re controls on a real
          instrument: nudge a weight and watch an actual perceptron&apos;s decision boundary move,
          train a tiny RNN in your browser and watch its memory genuinely fail, read attention
          weights straight off a real transformer&apos;s forward pass. Real numbers, computed on
          your machine, the instant you touch them.
        </p>

        <p className="text-sm leading-relaxed text-secondary sm:text-[15px]">
          No API key. No signup wall to try it. No server round-trip for a single exercise — every
          model in the browser-tier chapters runs in this tab and keeps working after you go offline.
        </p>

        <HeroActions />

        <div className="mt-2 flex flex-wrap gap-2">
          {PROOF_POINTS.map((point) => (
            <Tag key={point} tone="accent">
              {point}
            </Tag>
          ))}
        </div>
      </div>
    </section>
  );
}

function NotAChatbot() {
  const cards = [
    {
      label: 'reading about it',
      body: 'A textbook page or a blog post describes attention with a diagram. You never see a real weight — only an illustration of one.',
    },
    {
      label: 'asking a chatbot',
      body: "A model describes its own mechanism in prose. It's still zero real numbers — you're reading a guess about internals, generated the same way any other answer is.",
    },
    {
      label: 'operating it — this course',
      body: 'You embed your own word. The vector is real. You drag it, and the cosine similarity that updates is computed from what you just did, not looked up.',
      highlight: true,
    },
  ];

  return (
    <section className="border-b border-line px-4 py-12 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <Heading level={2} className="mb-2">
          This is not another &ldquo;ask AI to explain AI&rdquo;
        </Heading>
        <p className="mb-6 max-w-2xl text-sm text-secondary">
          There is no chatbot anywhere in the learning path. Understanding is built by direct
          manipulation of real model internals, not by reading model output about itself.
        </p>

        <div className="grid gap-3 sm:grid-cols-3">
          {cards.map((card) => (
            <Panel
              key={card.label}
              label={card.label}
              className={card.highlight ? 'border-accent/50' : undefined}
            >
              <p className="text-[13px] leading-relaxed text-secondary">{card.body}</p>
            </Panel>
          ))}
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section id="architecture" className="grid-field border-b border-line px-4 py-14 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <p className="label mb-2">for the technical audience</p>
        <Heading level={2} className="mb-2">
          How it actually runs
        </Heading>
        <p className="mb-8 max-w-2xl text-sm text-secondary">
          No server in the loop for anything you play. Here&apos;s the stack, end to end.
        </p>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {ARCHITECTURE_MODULES.map(({ icon: Icon, title, body }) => (
            <Panel key={title} className="gap-2">
              <div className="mb-2 flex items-center gap-2">
                <Icon size={15} strokeWidth={1.75} className="text-accent" />
                <span className="font-display text-[13px] font-semibold text-primary">{title}</span>
              </div>
              <p className="text-[13px] leading-relaxed text-secondary">{body}</p>
            </Panel>
          ))}
        </div>

        <div className="mt-6 flex flex-wrap gap-2">
          {STACK.map((item) => (
            <Tag key={item}>{item}</Tag>
          ))}
        </div>
      </div>
    </section>
  );
}

function Curriculum() {
  return (
    <section id="curriculum" className="border-b border-line px-4 py-14 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <p className="label mb-2">the course</p>
        <Heading level={2} className="mb-2">
          6 worlds. 22 chapters.
        </Heading>
        <p className="mb-3 max-w-2xl text-sm text-secondary">
          Each world owns one accent color, carried through into the map you&apos;ll actually play
          on. Chapters unlock in order — every arrow below is a real dependency, not a suggestion.
        </p>
        <p className="label mb-8">every chapter below is live — open any of them from the map</p>

        <div className="flex flex-col gap-8">
          {manifest.worlds.map((world) => {
            const chapters = [...world.chapters].sort((a, b) => a.order - b.order);
            return (
              <div key={world.world} data-world={world.world}>
                <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span
                    className="block h-2.5 w-2.5 shrink-0 border"
                    style={{ borderColor: 'var(--accent)', background: 'var(--accent)' }}
                    aria-hidden
                  />
                  <h3
                    className="font-display text-[1rem] font-semibold"
                    style={{ color: 'var(--accent)' }}
                  >
                    World {world.world} · {world.title}
                  </h3>
                  <span className="text-xs text-muted">{world.subtitle}</span>
                </div>

                <div className="ml-[22px] flex flex-wrap gap-1.5">
                  {chapters.map((chapter) => (
                    <Tag key={chapter.id} tone="accent">
                      {chapter.title}
                    </Tag>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function WhatsReal() {
  return (
    <section className="grid-field border-b border-line px-4 py-14 sm:px-8">
      <div className="mx-auto max-w-3xl">
        <p className="label mb-2">no invented data</p>
        <Heading level={2} className="mb-6">
          Nothing shown to you is fabricated
        </Heading>

        <ul className="flex flex-col gap-4">
          {REAL_NOT_FAKED.map((line) => (
            <li key={line} className="flex gap-3 text-sm leading-relaxed text-secondary">
              <span className="readout mt-0.5 shrink-0 text-accent">→</span>
              <span>{line}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function ClosingCta() {
  return (
    <section className="px-4 py-16 sm:px-8">
      <div className="mx-auto flex max-w-2xl flex-col items-start gap-4">
        <Heading level={2} className="text-balance">
          Your machine is already fast enough for this.
        </Heading>
        <p className="text-sm leading-relaxed text-secondary">
          First model download is a few dozen megabytes, cached after that. Everything else —
          gradients, attention, a real hidden state forgetting — happens on the CPU or GPU you
          already have.
        </p>
        <HeroActions />
      </div>
    </section>
  );
}
