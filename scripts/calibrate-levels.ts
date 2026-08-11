/**
 * Reports what each pure-computation level's metric actually reaches when
 * played well, so the thresholds in the level JSON stay honest.
 *
 * Levels whose metric depends on live model output are skipped — those are
 * calibrated against the real model in the browser, not here.
 *
 *   pnpm tsx scripts/calibrate-levels.ts
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import type { EngineRules, GameDefinition, StarRule } from '../src/types/game';

import * as linearClassifier from '../src/engines/linearClassifierEngine';
import * as lossMinimization from '../src/engines/lossMinimizationEngine';
import * as gradientDescent from '../src/engines/gradientDescentEngine';
import * as overfitFit from '../src/engines/overfitFitEngine';

const ROOT = path.resolve(__dirname, '..');

function loadGame(rel: string): GameDefinition {
  return JSON.parse(readFileSync(path.join(ROOT, 'data', 'games', rel), 'utf8'));
}

function rulesOf(game: GameDefinition, i: number): EngineRules {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria,
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
}

function describe(
  levelId: string,
  metric: string,
  best: number,
  comparator: string,
  threshold: number,
  starsRules: StarRule[]
) {
  const three = starsRules.find((r) => r.stars === 3)?.threshold;
  const reachable =
    comparator === 'lte' ? best <= threshold : best >= threshold;
  const threeStar =
    three === undefined ? false : comparator === 'lte' ? best <= three : best >= three;

  const flag = !reachable ? '  ✗ PASS UNREACHABLE' : !threeStar ? '  ! 3-STAR UNREACHABLE' : '';
  console.log(
    `${levelId.padEnd(10)} ${metric.padEnd(28)} best=${best.toFixed(4).padStart(10)}  ` +
      `pass ${comparator} ${threshold}  3★ ${three}${flag}`
  );
}

/* ── 2.1 perceptron ─────────────────────────────────────────── */
{
  const game = loadGame('world-2-classical-ml/2-1-perceptron.json');
  console.log('\n2-1 perceptron');
  for (const [i, level] of game.levels.entries()) {
    const config = level.engineConfig as unknown as linearClassifier.LinearClassifierConfig;
    const rules = rulesOf(game, i);

    // Search the learning rate the level exposes, and brute-force the best
    // linear boundary, to find the real ceiling rather than assuming one.
    let best = level.passCriteria.comparator === 'lte' ? Infinity : -Infinity;
    const rates = config.learningRateRange
      ? [0.01, 0.05, 0.1, 0.3, 0.6, 1]
      : [config.learningRate];

    for (const rate of rates) {
      let state = linearClassifier.initState({ ...config, learningRate: rate }, rules);
      state = linearClassifier.applyAction(state, { type: 'RUN', steps: config.maxSteps });
      const value = linearClassifier.evaluate(state).value;
      best = level.passCriteria.comparator === 'lte' ? Math.min(best, value) : Math.max(best, value);
    }

    if (level.passCriteria.metric === 'classificationAccuracy') {
      const state = linearClassifier.initState(config, rules);
      for (let a = 0; a < 180; a++) {
        const th = (a / 180) * Math.PI * 2;
        const w = [Math.cos(th), Math.sin(th)];
        for (let b = -1.5; b <= 1.5; b += 0.01) {
          best = Math.max(best, linearClassifier.accuracyOf(w, b, state.samples));
        }
      }
    }

    describe(level.id, level.passCriteria.metric, best, level.passCriteria.comparator, level.passCriteria.threshold, level.starsRules);
  }
}

/* ── 2.2 loss functions ─────────────────────────────────────── */
{
  const game = loadGame('world-2-classical-ml/2-2-loss-functions.json');
  console.log('\n2-2 loss functions');
  for (const [i, level] of game.levels.entries()) {
    const config = level.engineConfig as unknown as lossMinimization.LossMinimizationConfig;
    const rules = rulesOf(game, i);
    let best = level.passCriteria.comparator === 'lte' ? Infinity : -Infinity;

    if (config.mode === 'minimize') {
      const [sMin, sMax] = config.slopeRange ?? [-5, 5];
      const [iMin, iMax] = config.interceptRange ?? [-5, 5];
      for (let s = sMin; s <= sMax; s += 0.01) {
        for (let b = iMin; b <= iMax; b += 0.01) {
          let state = lossMinimization.initState(config, rules);
          state = lossMinimization.applyAction(state, { type: 'SET_SLOPE', value: s });
          state = lossMinimization.applyAction(state, { type: 'SET_INTERCEPT', value: b });
          best = Math.min(best, lossMinimization.evaluate(state).value);
        }
      }
    } else {
      let state = lossMinimization.initState(config, rules);
      for (const scenario of config.scenarios ?? []) {
        state = lossMinimization.applyAction(state, {
          type: 'ANSWER_SCENARIO',
          scenarioId: scenario.id,
          answer: scenario.answer,
        });
      }
      best = lossMinimization.evaluate(state).value;
    }

    describe(level.id, level.passCriteria.metric, best, level.passCriteria.comparator, level.passCriteria.threshold, level.starsRules);
  }
}

/* ── 2.3 gradient descent ───────────────────────────────────── */
{
  const game = loadGame('world-2-classical-ml/2-3-gradient-descent.json');
  console.log('\n2-3 gradient descent');
  for (const [i, level] of game.levels.entries()) {
    const config = level.engineConfig as unknown as gradientDescent.GradientDescentConfig;
    if (config.surface === 'liveNet') {
      console.log(`${level.id.padEnd(10)} (live-net surface — calibrated in browser)`);
      continue;
    }
    const rules = rulesOf(game, i);
    let best = level.passCriteria.comparator === 'lte' ? Infinity : -Infinity;

    const [lrMin, lrMax] = config.learningRateRange ?? [config.learningRate, config.learningRate];
    const momenta = config.momentumRange ? [0, 0.5, 0.8, 0.9, 0.95, 0.99] : [0];
    const noises = config.noiseRange ? [0, 0.1, 0.2, 0.3, 0.4, 0.5] : [0];

    for (let lr = lrMin; lr <= lrMax; lr *= 1.25) {
      for (const momentum of momenta) {
        for (const noise of noises) {
          let state = gradientDescent.initState(config, rules);
          state = gradientDescent.applyAction(state, { type: 'SET_LEARNING_RATE', value: lr });
          state = gradientDescent.applyAction(state, { type: 'SET_MOMENTUM', value: momentum });
          state = gradientDescent.applyAction(state, { type: 'SET_NOISE', value: noise });
          state = gradientDescent.applyAction(state, { type: 'RUN', steps: config.maxSteps });
          const value = gradientDescent.evaluate(state).value;
          if (Number.isFinite(value)) {
            best =
              level.passCriteria.comparator === 'lte' ? Math.min(best, value) : Math.max(best, value);
          }
        }
      }
    }

    describe(level.id, level.passCriteria.metric, best, level.passCriteria.comparator, level.passCriteria.threshold, level.starsRules);
  }
}

/* ── 2.4 overfitting ────────────────────────────────────────── */
{
  const game = loadGame('world-2-classical-ml/2-4-overfitting.json');
  console.log('\n2-4 overfitting');
  for (const [i, level] of game.levels.entries()) {
    const config = level.engineConfig as unknown as overfitFit.OverfitFitConfig;
    const rules = rulesOf(game, i);
    let best = Infinity;

    const degrees = config.degreeLocked
      ? [config.degree]
      : Array.from(
          { length: (config.degreeRange?.[1] ?? config.degree) - (config.degreeRange?.[0] ?? 1) + 1 },
          (_, d) => (config.degreeRange?.[0] ?? 1) + d
        );
    const [lMin, lMax] = config.lambdaRange ?? [0, 0];

    for (const degree of degrees) {
      for (let lambda = lMin; lambda <= lMax; lambda += Math.max((lMax - lMin) / 200, 0.001)) {
        let state = overfitFit.initState(config, rules);
        state = overfitFit.applyAction(state, { type: 'SET_DEGREE', value: degree });
        state = overfitFit.applyAction(state, { type: 'SET_LAMBDA', value: lambda });
        const value = overfitFit.evaluate(state).value;
        if (Number.isFinite(value)) best = Math.min(best, value);
        if (lMax === lMin) break;
      }
    }

    describe(level.id, level.passCriteria.metric, best, level.passCriteria.comparator, level.passCriteria.threshold, level.starsRules);
  }
}

console.log('');
