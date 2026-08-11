import { redirect } from 'next/navigation';

export default function RootPage() {
  // The map decides whether onboarding is still needed; it is the only screen
  // that can read local progress.
  redirect('/map');
}
