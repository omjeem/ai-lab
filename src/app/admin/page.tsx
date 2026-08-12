import { redirect } from 'next/navigation';

/**
 * `/admin` is an entry point, not a screen.
 *
 * The middleware already gates everything under /admin, so anyone reaching here
 * holds a valid session and just needs sending to the dashboard.
 */
export default function AdminIndexPage() {
  redirect('/admin/dashboard');
}
