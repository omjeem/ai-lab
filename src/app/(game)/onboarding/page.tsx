import type { Metadata } from 'next';
import { Onboarding } from '@/components/onboarding/Onboarding';

export const metadata: Metadata = {
  title: 'Get Started',
  description: 'Pick a name and start learning AI by running real models in your browser — no account, no install.',
  alternates: { canonical: '/onboarding' },
};

export default function OnboardingPage() {
  return <Onboarding />;
}
