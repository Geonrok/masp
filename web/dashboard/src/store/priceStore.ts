import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { useShallow } from 'zustand/react/shallow';
import type { ConnectionState } from '../api/types';

interface PriceData {
  price: number;
  timestamp: number;
}

interface PriceState {
  prices: Record<string, PriceData>;
  connectionState: ConnectionState;
}

interface PriceActions {
  updatePrice: (symbol: string, price: number) => void;
  setConnectionState: (state: ConnectionState) => void;
  reset: () => void;
}

const getInitialState = (): PriceState => ({
  prices: {},
  connectionState: 'CLOSED',
});

export const usePriceStore = create<PriceState & PriceActions>()(
  immer((set) => ({
    ...getInitialState(),

    updatePrice: (symbol, price) =>
      set((state) => {
        state.prices[symbol] = { price, timestamp: Date.now() };
      }),

    setConnectionState: (connectionState) =>
      set((state) => {
        state.connectionState = connectionState;
      }),

    reset: () => set(getInitialState()),
  }))
);

export const usePrices = () => usePriceStore(useShallow((state) => state.prices));
export const useConnectionState = () => usePriceStore((state) => state.connectionState);

export const usePriceValues = () =>
  usePriceStore(
    useShallow((state) =>
      Object.fromEntries(Object.entries(state.prices).map(([key, value]) => [key, value.price]))
    )
  );

export const usePriceActions = () =>
  usePriceStore(
    useShallow((state) => ({
      updatePrice: state.updatePrice,
      setConnectionState: state.setConnectionState,
      reset: state.reset,
    }))
  );
