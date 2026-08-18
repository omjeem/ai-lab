import { MongoClient, type Db, type Collection } from 'mongodb';
import type { UserDocument } from '@/types/user';
import type { ActivityEvent } from '@/types/activity';

/**
 * One MongoClient for the process.
 *
 * Next.js reloads modules in development, so the client is parked on globalThis
 * to avoid opening a new pool on every hot reload.
 */
const globalForMongo = globalThis as unknown as {
  __mongoClient?: MongoClient;
  __mongoPromise?: Promise<MongoClient>;
  __mongoIndexed?: boolean;
};

export class DatabaseNotConfiguredError extends Error {
  constructor() {
    super('MONGODB_URI is not set');
    this.name = 'DatabaseNotConfiguredError';
  }
}

export function isDatabaseConfigured(): boolean {
  return Boolean(process.env.MONGODB_URI);
}

function getClientPromise(): Promise<MongoClient> {
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new DatabaseNotConfiguredError();

  if (!globalForMongo.__mongoPromise) {
    const client = new MongoClient(uri, {
      // Keeps a stalled cluster from holding an API route open indefinitely.
      serverSelectionTimeoutMS: 8000,
      maxPoolSize: 10,
    });
    globalForMongo.__mongoClient = client;
    globalForMongo.__mongoPromise = client.connect();
  }
  return globalForMongo.__mongoPromise;
}

export async function getDatabase(): Promise<Db> {
  const client = await getClientPromise();
  const db = client.db(process.env.MONGODB_DB ?? 'ai_learning_lab');
  await ensureIndexes(db);
  return db;
}

export interface ActivityDocument extends ActivityEvent {
  /** Server clock, so a skewed client cannot reorder the log. */
  receivedAt: Date;
}

export async function usersCollection(): Promise<Collection<UserDocument>> {
  return (await getDatabase()).collection<UserDocument>('users');
}

export async function activityCollection(): Promise<Collection<ActivityDocument>> {
  return (await getDatabase()).collection<ActivityDocument>('activity');
}

/**
 * Indexes the queries this app actually runs.
 *
 * Section 9 keeps activity in its own collection rather than embedded in the
 * user document, so the log can grow past the 16MB document limit.
 */
async function ensureIndexes(db: Db): Promise<void> {
  if (globalForMongo.__mongoIndexed) return;
  globalForMongo.__mongoIndexed = true;

  try {
    await Promise.all([
      db.collection('users').createIndex({ userId: 1 }, { unique: true }),
      db.collection('users').createIndex({ lastSeen: -1 }),
      // The admin drill-down reads one user's timeline newest first.
      db.collection('activity').createIndex({ userId: 1, timestamp: -1 }),
      // Deduplicates a batch replayed after a dropped response.
      db.collection('activity').createIndex({ eventId: 1 }, { unique: true }),
      // Backs the admin per-chapter analytics aggregation (group by chapterId + type).
      db.collection('activity').createIndex({ chapterId: 1, type: 1 }),
    ]);
  } catch {
    // A permissions-restricted user may not be allowed to create indexes; the
    // app still works, just more slowly.
    globalForMongo.__mongoIndexed = true;
  }
}
