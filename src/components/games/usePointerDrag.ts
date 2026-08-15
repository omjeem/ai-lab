'use client';

/**
 * Touch-and-mouse dragging via the Pointer Events API.
 *
 * Several canvases used to drag with the native HTML5 drag-and-drop API
 * (`draggable` + `onDragStart`/`onDragOver`/`onDrop`), which only ever fires
 * from a mouse — no touch browser translates a finger drag into those events,
 * so dragging silently did nothing on a phone or tablet. Pointer events cover
 * mouse, touch and pen uniformly, and pointer capture keeps delivering move
 * and up events to the element a gesture started on even once the finger has
 * moved off it, which is what makes a small drag target usable at all.
 *
 * A small movement threshold keeps a tap-to-select `onClick` working — a
 * press that never moves past it is a tap, not a drag, and never fires
 * `onDragStart`/`onDragMove`/`onDragEnd` at all.
 */
import { useCallback, useRef, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react';

export interface PointerDragHandlers {
  onDragMove: (clientX: number, clientY: number) => void;
  onDragStart?: (clientX: number, clientY: number) => void;
  onDragEnd?: (clientX: number, clientY: number) => void;
  /** Pixels of movement before a press counts as a drag rather than a tap. */
  threshold?: number;
}

export interface PointerDragProps {
  onPointerDown: (event: ReactPointerEvent) => void;
  onPointerMove: (event: ReactPointerEvent) => void;
  onPointerUp: (event: ReactPointerEvent) => void;
  onPointerCancel: (event: ReactPointerEvent) => void;
  /** Stops the browser from treating the gesture as a page scroll/pan. */
  style: CSSProperties;
}

export function usePointerDrag({
  onDragMove,
  onDragStart,
  onDragEnd,
  threshold = 4,
}: PointerDragHandlers): PointerDragProps {
  const gesture = useRef<{ pointerId: number; startX: number; startY: number; dragging: boolean } | null>(
    null
  );

  const onPointerDown = useCallback((event: ReactPointerEvent) => {
    (event.currentTarget as Element).setPointerCapture(event.pointerId);
    gesture.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      dragging: false,
    };
  }, []);

  const onPointerMove = useCallback(
    (event: ReactPointerEvent) => {
      const g = gesture.current;
      if (!g || event.pointerId !== g.pointerId) return;

      if (!g.dragging) {
        const dx = event.clientX - g.startX;
        const dy = event.clientY - g.startY;
        if (Math.hypot(dx, dy) < threshold) return;
        g.dragging = true;
        onDragStart?.(event.clientX, event.clientY);
      }
      onDragMove(event.clientX, event.clientY);
    },
    [onDragMove, onDragStart, threshold]
  );

  const endGesture = useCallback(
    (event: ReactPointerEvent) => {
      const g = gesture.current;
      if (!g || event.pointerId !== g.pointerId) return;
      if (g.dragging) onDragEnd?.(event.clientX, event.clientY);
      gesture.current = null;
    },
    [onDragEnd]
  );

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp: endGesture,
    onPointerCancel: endGesture,
    style: { touchAction: 'none' },
  };
}
