/**
 * Chapter 8.4 — Red-Teaming.
 *
 * The player is testing this course's own local, in-browser model for
 * defensive/educational understanding — never a third-party system. Every
 * trial is a real greedy decode (`greedyDecode`, reused from
 * `groundedGenerationEngine.ts`) of a real prompt built from a fixed
 * instruction, an optional attack/defense snippet, and a fixed question, all
 * loaded from the bundled `robustness-prompts` corpus. "Violates"/"resists"
 * is a real, mechanical word-count check against the instruction's own stated
 * rule — never an authored judgment call about what the text means.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { CausalLMDep, CorpusDep } from './deps';
import { greedyDecode } from './groundedGenerationEngine';
import { scoreLevel } from './scoringEngine';

export type RobustnessMode = 'find-attack' | 'test-defense' | 'defense-transfer';

export interface RobustnessConfig {
  mode: RobustnessMode;
  corpus: string;
  maxTokens: number;
  /** Real word count the instruction demands — the mechanical check every trial is graded against. */
  expectedWordCount: number;
  // find-attack
  attackIds?: string[];
  // test-defense
  fixedAttackId?: string;
  defenseIds?: string[];
  // defense-transfer
  fixedDefenseId?: string;
  transferAttackIds?: string[];
}

export interface PromptOption {
  id: string;
  label: string;
  text: string;
}

interface RawPrompts {
  instruction: string;
  query: string;
  attacks: PromptOption[];
  defenses: PromptOption[];
}

export interface AttackRound {
  id: string;
  label: string;
  decodedText: string;
  /** Real: did this attack actually break the instruction's stated word count. */
  violates: boolean;
  tested: boolean;
}

export interface DefenseRound {
  id: string;
  label: string;
  decodedText: string;
  /** Real: did this defense actually hold the word count against the fixed attack. */
  resists: boolean;
  tested: boolean;
}

export interface TransferRound {
  id: string;
  label: string;
  decodedText: string;
  /** Real: did the fixed defense actually hold against this other attack. */
  resists: boolean;
  guess: boolean | null;
}

export interface PreparedRobustnessData {
  attackRounds: AttackRound[];
  defenseRounds: DefenseRound[];
  transferRounds: TransferRound[];
}

export interface RobustnessState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: RobustnessMode;
  config: RobustnessConfig;

  attackRounds: AttackRound[];
  defenseRounds: DefenseRound[];
  transferRounds: TransferRound[];

  attempts: number;
  solved: boolean;
  solvedAtAttempt: number | null;
}

export type RobustnessAction =
  | { type: 'TEST_ATTACK'; id: string }
  | { type: 'TEST_DEFENSE'; id: string }
  | { type: 'GUESS_TRANSFER'; roundIndex: number; guess: boolean }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter((w) => w.length > 0).length;
}

async function runTrial(
  causalLM: CausalLMDep,
  instructionText: string,
  snippetText: string,
  query: string,
  maxTokens: number,
  expectedWordCount: number
): Promise<{ decodedText: string; held: boolean }> {
  const prompt = snippetText
    ? `${instructionText}\n${snippetText}\nQuestion: ${query}\nAnswer:`
    : `${instructionText}\nQuestion: ${query}\nAnswer:`;
  const decodedText = await greedyDecode(causalLM, prompt, maxTokens);
  return { decodedText, held: wordCount(decodedText) === expectedWordCount };
}

export async function prepare(
  config: RobustnessConfig,
  deps: { corpus: CorpusDep; causalLM: CausalLMDep }
): Promise<PreparedRobustnessData> {
  const raw = await deps.corpus.load(config.corpus);
  const parsed = JSON.parse(raw) as RawPrompts;
  const attackById = new Map(parsed.attacks.map((a) => [a.id, a]));
  const defenseById = new Map(parsed.defenses.map((d) => [d.id, d]));

  const attackRounds: AttackRound[] = [];
  if (config.mode === 'find-attack') {
    for (const id of config.attackIds ?? []) {
      const attack = attackById.get(id);
      if (!attack) continue;
      const { decodedText, held } = await runTrial(
        deps.causalLM,
        parsed.instruction,
        attack.text,
        parsed.query,
        config.maxTokens,
        config.expectedWordCount
      );
      attackRounds.push({ id, label: attack.label, decodedText, violates: !held, tested: false });
    }
  }

  const defenseRounds: DefenseRound[] = [];
  if (config.mode === 'test-defense') {
    const fixedAttack = attackById.get(config.fixedAttackId ?? '');
    if (!fixedAttack) throw new Error(`Fixed attack "${config.fixedAttackId}" not found in corpus`);
    for (const id of config.defenseIds ?? []) {
      const defense = defenseById.get(id);
      if (!defense) continue;
      const { decodedText, held } = await runTrial(
        deps.causalLM,
        defense.text,
        fixedAttack.text,
        parsed.query,
        config.maxTokens,
        config.expectedWordCount
      );
      defenseRounds.push({ id, label: defense.label, decodedText, resists: held, tested: false });
    }
  }

  const transferRounds: TransferRound[] = [];
  if (config.mode === 'defense-transfer') {
    const fixedDefense = defenseById.get(config.fixedDefenseId ?? '');
    if (!fixedDefense) throw new Error(`Fixed defense "${config.fixedDefenseId}" not found in corpus`);
    for (const id of config.transferAttackIds ?? []) {
      const attack = attackById.get(id);
      if (!attack) continue;
      const { decodedText, held } = await runTrial(
        deps.causalLM,
        fixedDefense.text,
        attack.text,
        parsed.query,
        config.maxTokens,
        config.expectedWordCount
      );
      transferRounds.push({ id, label: attack.label, decodedText, resists: held, guess: null });
    }
  }

  return { attackRounds, defenseRounds, transferRounds };
}

export function initState(
  config: RobustnessConfig,
  rules: EngineRules,
  prepared: PreparedRobustnessData
): RobustnessState {
  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    attackRounds: prepared.attackRounds.map((r) => ({ ...r, tested: false })),
    defenseRounds: prepared.defenseRounds.map((r) => ({ ...r, tested: false })),
    transferRounds: prepared.transferRounds.map((r) => ({ ...r, guess: null })),
    attempts: 0,
    solved: false,
    solvedAtAttempt: null,
  };
}

export function applyAction(state: RobustnessState, action: RobustnessAction): RobustnessState {
  const bump = (next: Partial<RobustnessState>): RobustnessState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'TEST_ATTACK': {
      const round = state.attackRounds.find((r) => r.id === action.id);
      if (!round || round.tested) return state;
      const attackRounds = state.attackRounds.map((r) => (r.id === action.id ? { ...r, tested: true } : r));
      const attempts = state.attempts + 1;
      const alreadySolved = state.solved;
      const solved = alreadySolved || round.violates;
      return bump({
        attackRounds,
        attempts,
        solved,
        solvedAtAttempt: alreadySolved ? state.solvedAtAttempt : round.violates ? attempts : null,
      });
    }

    case 'TEST_DEFENSE': {
      const round = state.defenseRounds.find((r) => r.id === action.id);
      if (!round || round.tested) return state;
      const defenseRounds = state.defenseRounds.map((r) => (r.id === action.id ? { ...r, tested: true } : r));
      const attempts = state.attempts + 1;
      const alreadySolved = state.solved;
      const solved = alreadySolved || round.resists;
      return bump({
        defenseRounds,
        attempts,
        solved,
        solvedAtAttempt: alreadySolved ? state.solvedAtAttempt : round.resists ? attempts : null,
      });
    }

    case 'GUESS_TRANSFER': {
      const round = state.transferRounds[action.roundIndex];
      if (!round) return state;
      const transferRounds = [...state.transferRounds];
      transferRounds[action.roundIndex] = { ...round, guess: action.guess };
      return bump({ transferRounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, {
        attackRounds: state.attackRounds,
        defenseRounds: state.defenseRounds,
        transferRounds: state.transferRounds,
      });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

export function evaluate(state: RobustnessState): ScoreResult {
  switch (state.mode) {
    case 'find-attack':
    case 'test-defense': {
      const value = state.solved ? state.solvedAtAttempt! : Infinity;
      return scoreLevel({
        metric: 'attemptsToSolve',
        value,
        rules: state.rules,
        breakdown: { attempts: state.attempts, solved: state.solved ? 1 : 0 },
      });
    }

    case 'defense-transfer': {
      const total = state.transferRounds.length;
      if (total === 0) {
        return scoreLevel({ metric: 'transferPredictionAccuracy', value: 0, rules: state.rules });
      }
      const correct = state.transferRounds.filter((r) => r.guess !== null && r.guess === r.resists).length;
      return scoreLevel({
        metric: 'transferPredictionAccuracy',
        value: correct / total,
        rules: state.rules,
        breakdown: { correct, total },
      });
    }
  }
}
