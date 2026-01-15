import { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries } from 'lightweight-charts';
import type { CandlestickData, UTCTimestamp } from 'lightweight-charts';

interface Props {
  symbol: string;
  data?: CandlestickData<UTCTimestamp>[];
  height?: number;
}

type SeriesApi = ReturnType<ReturnType<typeof createChart>['addSeries']>;

export default function PriceChart({ symbol, data = [], height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef = useRef<SeriesApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: { background: { color: '#ffffff' }, textColor: '#333' },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: { borderColor: '#d1d4dc' },
      timeScale: { borderColor: '#d1d4dc', timeVisible: true },
    });

    const candlestick = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    chartRef.current = chart;
    seriesRef.current = candlestick;

    if (data.length > 0) {
      candlestick.setData(data);
    }

    const ro = new ResizeObserver(() => {
      if (!containerRef.current || !chartRef.current) return;
      chartRef.current.resize(containerRef.current.clientWidth, height);
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height, data]);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-sm font-medium text-gray-700 mb-2">{symbol} Price Chart</div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
