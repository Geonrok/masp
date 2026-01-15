import { useCallback, useRef } from 'react';

/**
 * RequestAnimationFrame based throttle hook.
 * Allows at most one call per frame (~16ms).
 */
export function useRAFThrottle<T extends (...args: any[]) => void>(fn: T): T {
  const rafIdRef = useRef<number | null>(null);
  const lastArgsRef = useRef<any[] | null>(null);

  return useCallback(
    (...args: any[]) => {
      lastArgsRef.current = args;

      if (rafIdRef.current !== null) {
        return;
      }

      rafIdRef.current = requestAnimationFrame(() => {
        if (lastArgsRef.current !== null) {
          fn(...lastArgsRef.current);
        }
        rafIdRef.current = null;
      });
    },
    [fn]
  ) as T;
}
