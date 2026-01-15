import React, { useEffect, useRef, useCallback, useState } from 'react';
import { createChart, LineSeries } from 'lightweight-charts';
import type { UTCTimestamp } from 'lightweight-charts';
import { useShallow } from 'zustand/react/shallow';
import { useRAFThrottle } from '../../hooks/useRAFThrottle';
import { usePriceStore } from '../../store/priceStore';

interface Props {
  symbol: string;
  height?: number;
  maxPoints?: number;
}

type SeriesApi = ReturnType<ReturnType<typeof createChart>['addSeries']>;

type LinePoint = { time: UTCTimestamp; value: number };

function RealtimeLineChartInner({
  symbol,
  height = 300,
  maxPoints = 1200,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef = useRef<SeriesApi | null>(null);
  const bufferRef = useRef<LinePoint[]>([]);
  const renderCountRef = useRef(0);

  const [hasData, setHasData] = useState(false);
  const hasDataRef = useRef(false);

  const priceData = usePriceStore((state) => state.prices[symbol]);

  useEffect(() => {
    renderCountRef.current += 1;
    if (import.meta.env.DEV) {
      console.log(`[${symbol}]  RENDER #${renderCountRef.current}`);
    }
  });

  useEffect(() => {
    if (import.meta.env.DEV && priceData) {
      console.log(`[${symbol}]  PRICE_UPDATE:`, priceData.price);
    }
  }, [priceData, symbol]);

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log(`[${symbol}] ⬆️ MOUNT_START (render #${renderCountRef.current})`);
    }

    if (!containerRef.current) {
      if (import.meta.env.DEV) {
        console.error(`[${symbol}] ❌ ContainerRef is null`);
      }
      return;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: { background: { color: '#ffffff' }, textColor: '#333' },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: { borderColor: '#d1d4dc' },
      timeScale: {
        borderColor: '#d1d4dc',
        timeVisible: true,
        secondsVisible: true,
      },
    });

    const line = chart.addSeries(LineSeries, {
      color: '#2962FF',
      lineWidth: 2,
    });

    chartRef.current = chart;
    seriesRef.current = line;

    if (import.meta.env.DEV) {
      console.log(
        `[${symbol}] ✅ MOUNT_COMPLETE - size: ${containerRef.current.clientWidth}x${height}`
      );
    }

    if (bufferRef.current.length > 0) {
      line.setData(bufferRef.current);
      chart.timeScale().fitContent();
      if (import.meta.env.DEV) {
        console.log(`[${symbol}]  BUFFER_REPLAY: ${bufferRef.current.length} points`);
      }
    }

    const ro = new ResizeObserver(() => {
      if (!containerRef.current || !chartRef.current) return;
      if (import.meta.env.DEV) {
        console.log(`[${symbol}]  RESIZE_OBSERVER`);
      }
      chartRef.current.resize(containerRef.current.clientWidth, height);
    });
    ro.observe(containerRef.current);

    return () => {
      if (import.meta.env.DEV) {
        console.log(`[${symbol}] ⬇️ UNMOUNT`);
      }
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log(`[${symbol}]  HEIGHT_EFFECT: ${height}px`);
    }
    if (chartRef.current && containerRef.current) {
      chartRef.current.resize(containerRef.current.clientWidth, height);
    }
  }, [height, symbol]);

  const pushPoint = useCallback(
    (price: number) => {
      if (!seriesRef.current) return;

      const now = Math.floor(Date.now() / 1000) as UTCTimestamp;
      bufferRef.current.push({ time: now, value: price });

      if (bufferRef.current.length > maxPoints) {
        bufferRef.current = bufferRef.current.slice(-maxPoints);
        seriesRef.current.setData(bufferRef.current);
      } else {
        seriesRef.current.update({ time: now, value: price });
      }
    },
    [maxPoints]
  );

  const throttledPush = useRAFThrottle(pushPoint);

  useEffect(() => {
    if (!priceData || typeof priceData.price !== 'number' || priceData.price <= 0) {
      return;
    }

    if (!hasDataRef.current) {
      hasDataRef.current = true;
      setHasData(true);
    }

    throttledPush(priceData.price);
  }, [priceData, throttledPush]);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-sm font-medium text-gray-700 mb-2">{symbol} Realtime</div>
      <div className="relative" style={{ minHeight: height }}>
        <div ref={containerRef} className="w-full" style={{ height }} />
        {!hasData && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50/80 rounded z-10">
            <div className="text-center">
              <div className="animate-pulse text-gray-400 mb-2 text-2xl"></div>
              <span className="text-gray-500 text-sm">Waiting for {symbol} data...</span>
            </div>
          </div>
        )}
      </div>
      {import.meta.env.DEV && <StoreKeysDebug />}
    </div>
  );
}

function StoreKeysDebug() {
  if (!import.meta.env.DEV) return null;

  const keys = usePriceStore(useShallow((state) => Object.keys(state.prices)));

  if (keys.length === 0) {
    return (
      <div className="text-xs text-gray-400 mt-2 border-t pt-2">
        Store keys: <span className="text-yellow-500">(empty)</span>
      </div>
    );
  }

  return (
    <div className="text-xs text-gray-400 mt-2 border-t pt-2">
      Store keys ({keys.length}): {keys.slice(0, 5).join(', ')}
      {keys.length > 5 && ` (+${keys.length - 5})`}
    </div>
  );
}

const propsAreEqual = (prev: Props, next: Props): boolean => {
  const isSame =
    prev.symbol === next.symbol &&
    prev.height === next.height &&
    prev.maxPoints === next.maxPoints;

  if (import.meta.env.DEV) {
    if (isSame) {
      console.log(`[${prev.symbol}] ✅ Props Same → MEMO_SKIP`);
    } else {
      console.group(` [${prev.symbol}] Props Changed`);
      if (prev.symbol !== next.symbol) {
        console.log(`  symbol: "${prev.symbol}" → "${next.symbol}"`);
      }
      if (prev.height !== next.height) {
        console.log(`  height: ${prev.height} → ${next.height}`);
      }
      if (prev.maxPoints !== next.maxPoints) {
        console.log(`  maxPoints: ${prev.maxPoints} → ${next.maxPoints}`);
      }
      console.log('   Result: ❌ RENDER');
      console.groupEnd();
    }
  }

  return isSame;
};

export default React.memo(RealtimeLineChartInner, propsAreEqual);
