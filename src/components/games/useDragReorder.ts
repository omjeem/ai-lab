'use client';

/**
 * Backs a vertical drag-to-reorder list — a grip handle plus up/down buttons
 * is the pattern in every ranking canvas (similarity ranking, gradient
 * ranking). Only the target-finding and drag bookkeeping live here; the
 * pointer capture itself is `usePointerDrag`, wired up per row, since a hook
 * can't be called a variable number of times inside `.map()` — each row has
 * to be its own component for that.
 *
 * The drop target is whichever item's vertical midpoint is nearest the
 * pointer, which is what lets a finger dragged anywhere over a short list
 * resolve to one unambiguous position, the same way most reorderable lists
 * behave.
 */
import { useCallback, useRef, useState } from 'react';

export interface DragReorderState {
  dragIndex: number | null;
  overIndex: number | null;
  registerItem: (index: number) => (element: HTMLElement | null) => void;
  startDrag: (index: number) => void;
  dragTo: (clientY: number) => void;
  dropAt: (clientY: number) => void;
}

export function useDragReorder(onMove: (from: number, to: number) => void): DragReorderState {
  const itemRefs = useRef(new Map<number, HTMLElement>());
  const dragFrom = useRef<number | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [overIndex, setOverIndex] = useState<number | null>(null);

  const registerItem = useCallback(
    (index: number) => (element: HTMLElement | null) => {
      if (element) itemRefs.current.set(index, element);
      else itemRefs.current.delete(index);
    },
    []
  );

  const findTarget = useCallback((clientY: number): number | null => {
    let bestIndex: number | null = null;
    let bestDistance = Infinity;
    for (const [index, element] of itemRefs.current) {
      const rect = element.getBoundingClientRect();
      const distance = Math.abs(clientY - (rect.top + rect.height / 2));
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    return bestIndex;
  }, []);

  const startDrag = useCallback((index: number) => {
    dragFrom.current = index;
    setDragIndex(index);
  }, []);

  const dragTo = useCallback((clientY: number) => setOverIndex(findTarget(clientY)), [findTarget]);

  const dropAt = useCallback(
    (clientY: number) => {
      const from = dragFrom.current;
      const to = findTarget(clientY);
      dragFrom.current = null;
      setDragIndex(null);
      setOverIndex(null);
      if (from !== null && to !== null && to !== from) onMove(from, to);
    },
    [findTarget, onMove]
  );

  return { dragIndex, overIndex, registerItem, startDrag, dragTo, dropAt };
}
