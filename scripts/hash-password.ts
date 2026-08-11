/**
 * Generates the scrypt hash for ADMIN_PASSWORD_HASH.
 *
 *   pnpm hash:password 'your-password-here'
 *
 * Only the hash goes in the environment — never the plaintext.
 */
import { hashPassword } from '../src/lib/adminAuth';

const password = process.argv[2];

if (!password) {
  console.error('\nUsage: pnpm hash:password \'your-password-here\'\n');
  process.exit(1);
}

if (password.length < 12) {
  console.error('\nRefusing to hash a password under 12 characters.\n');
  process.exit(1);
}

console.log('\nAdd this to .env.local:\n');
console.log(`ADMIN_PASSWORD_HASH="${hashPassword(password)}"`);
console.log('\nAnd generate a session secret with:  openssl rand -hex 32\n');
