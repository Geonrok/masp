import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { SystemStatus } from '../api/types';
import RealtimeLineChart from '../components/charts/RealtimeLineChart';

const StatusPanel = React.memo(function StatusPanel() {
  const { data: status, isLoading, isError, isFetching } = useQuery<SystemStatus>({
    queryKey: ['status'],
    queryFn: async () => {
      const res = await apiClient.get<SystemStatus>('/status');
      return res.data;
    },
    refetchInterval: 60_000,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div className="bg-white p-4 rounded shadow">
        <div className="text-sm text-gray-500">Status</div>
        <div className="text-xl font-bold text-green-600">
          {isLoading ? (
            'Loading...'
          ) : isError ? (
            'Error'
          ) : (
            status?.success ? 'Online' : 'Offline'
          )}
          {isFetching && !isLoading && <span className="text-xs"> (syncing)</span>}
        </div>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <div className="text-sm text-gray-500">Active Strategies</div>
        <div className="text-xl font-bold">
          {isLoading ? '...' : status?.active_strategies ?? 0}
        </div>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <div className="text-sm text-gray-500">Uptime</div>
        <div className="text-xl font-bold">
          {isLoading ? '...' : `${Math.floor((status?.uptime_seconds ?? 0) / 3600)}h`}
        </div>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <div className="text-sm text-gray-500">Version</div>
        <div className="text-xl font-bold">
          {isLoading ? '...' : status?.version ?? 'N/A'}
        </div>
      </div>
    </div>
  );
});

StatusPanel.displayName = 'StatusPanel';

const ChartSection = React.memo(function ChartSection() {
  const chartSymbols = useMemo(() => ['KRW-BTC', 'KRW-ETH'] as const, []);

  if (import.meta.env.DEV) {
    console.log('[ChartSection] RENDER');
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
      {chartSymbols.map((symbol) => (
        <RealtimeLineChart key={symbol} symbol={symbol} height={300} maxPoints={1200} />
      ))}
    </div>
  );
});

ChartSection.displayName = 'ChartSection';

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <StatusPanel />
      <ChartSection />
    </div>
  );
}
