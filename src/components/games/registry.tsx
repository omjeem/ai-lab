'use client';

/**
 * Chapter id → its game component.
 *
 * Each entry is loaded on demand so a chapter's model wrapper and canvas never
 * enter the bundle until that chapter is opened.
 */
import dynamic from 'next/dynamic';
import type { ComponentType } from 'react';
import type { GameRenderProps } from '@/components/chapter/ChapterShell';
import type { GameDefinition } from '@/types/game';

export interface GameComponentProps extends GameRenderProps {
  game: GameDefinition;
}

type GameComponent = ComponentType<GameComponentProps>;

const loading = () => (
  <div className="grid-field flex flex-1 items-center justify-center">
    <span className="label">preparing instrument</span>
  </div>
);

const REGISTRY: Record<string, GameComponent> = {
  '1-1-vectors': dynamic(
    () => import('./1-1-vectors/VectorCanvas').then((m) => m.VectorCanvas),
    { ssr: false, loading }
  ),
  '1-2-vector-arithmetic': dynamic(
    () => import('./1-2-vector-arithmetic/ArithmeticCanvas').then((m) => m.ArithmeticCanvas),
    { ssr: false, loading }
  ),
  '1-3-similarity-distance': dynamic(
    () => import('./1-3-similarity-distance/SimilarityCanvas').then((m) => m.SimilarityCanvas),
    { ssr: false, loading }
  ),
  '1-4-tokenization': dynamic(
    () => import('./1-4-tokenization/TokenizationCanvas').then((m) => m.TokenizationCanvas),
    { ssr: false, loading }
  ),
  '1-5-probability': dynamic(
    () => import('./1-5-probability/ProbabilityCanvas').then((m) => m.ProbabilityCanvas),
    { ssr: false, loading }
  ),
  '2-1-perceptron': dynamic(
    () => import('./2-1-perceptron/PerceptronCanvas').then((m) => m.PerceptronCanvas),
    { ssr: false, loading }
  ),
  '2-2-loss-functions': dynamic(
    () => import('./2-2-loss-functions/LossCanvas').then((m) => m.LossCanvas),
    { ssr: false, loading }
  ),
  '2-3-gradient-descent': dynamic(
    () => import('./2-3-gradient-descent/GradientDescentCanvas').then((m) => m.GradientDescentCanvas),
    { ssr: false, loading }
  ),
  '2-4-overfitting': dynamic(
    () => import('./2-4-overfitting/OverfitCanvas').then((m) => m.OverfitCanvas),
    { ssr: false, loading }
  ),
  '3-1-neurons-activations': dynamic(
    () => import('./3-1-neurons-activations/NeuronCanvas').then((m) => m.NeuronCanvas),
    { ssr: false, loading }
  ),
  '3-2-layers-forward-pass': dynamic(
    () => import('./3-2-layers-forward-pass/LayersCanvas').then((m) => m.LayersCanvas),
    { ssr: false, loading }
  ),
  '3-3-backpropagation': dynamic(
    () => import('./3-3-backpropagation/BackpropCanvas').then((m) => m.BackpropCanvas),
    { ssr: false, loading }
  ),
  '3-4-training-dynamics': dynamic(
    () => import('./3-4-training-dynamics/TrainingCanvas').then((m) => m.TrainingCanvas),
    { ssr: false, loading }
  ),
  '4-1-ngrams': dynamic(
    () => import('./4-1-ngrams/NgramCanvas').then((m) => m.NgramCanvas),
    { ssr: false, loading }
  ),
  '4-2-recurrence-memory': dynamic(
    () => import('./4-2-recurrence-memory/RecurrenceCanvas').then((m) => m.RecurrenceCanvas),
    { ssr: false, loading }
  ),
  '4-3-sampling-strategies': dynamic(
    () => import('./4-3-sampling-strategies/SamplingCanvas').then((m) => m.SamplingCanvas),
    { ssr: false, loading }
  ),
  '5-1-positional-encoding': dynamic(
    () => import('./5-1-positional-encoding/PositionalEncodingCanvas').then((m) => m.PositionalEncodingCanvas),
    { ssr: false, loading }
  ),
  '5-2-self-attention': dynamic(
    () => import('./5-2-self-attention/SelfAttentionCanvas').then((m) => m.SelfAttentionCanvas),
    { ssr: false, loading }
  ),
  '5-3-multi-head-attention': dynamic(
    () => import('./5-3-multi-head-attention/MultiHeadCanvas').then((m) => m.MultiHeadCanvas),
    { ssr: false, loading }
  ),
  '5-4-residuals-layernorm': dynamic(
    () => import('./5-4-residuals-layernorm/ResidualCanvas').then((m) => m.ResidualCanvas),
    { ssr: false, loading }
  ),
  '5-5-full-transformer': dynamic(
    () => import('./5-5-full-transformer/FullTransformerCanvas').then((m) => m.FullTransformerCanvas),
    { ssr: false, loading }
  ),
};

export function getGameComponent(chapterId: string): GameComponent | null {
  return REGISTRY[chapterId] ?? null;
}

export function implementedChapters(): string[] {
  return Object.keys(REGISTRY);
}
