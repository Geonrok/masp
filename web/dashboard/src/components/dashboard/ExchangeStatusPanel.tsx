import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { ExchangeStatusResponse, ExchangeStatus } from '../../api/types';

const exchangeDisplayNames: Record<string, string> = {
  upbit: 'Upbit',
  bithumb: 'Bithumb',
  binance_spot: 'Binance Spot',
  binance_futures: 'Binance Futures',
  paper: 'Paper Trading',
};

const exchangeIcons: Record<string, string> = {
  upbit: 'U',
  bithumb: 'B',
  binance_spot: 'BN',
  binance_futures: 'BF',
  paper: 'P',
};

function ExchangeCard({ exchange }: { exchange: ExchangeStatus }) {
  const displayName = exchangeDisplayNames[exchange.exchange] || exchange.exchange;
  const icon = exchangeIcons[exchange.exchange] || '?';
  const isBinance = exchange.exchange.startsWith('binance');

  return (
    <div
      className={`p-4 rounded-lg border ${
        exchange.enabled
          ? 'bg-white border-green-200'
          : 'bg-gray-50 border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
              isBinance
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-blue-100 text-blue-700'
            }`}
          >
            {icon}
          </div>
          <span className="font-medium">{displayName}</span>
        </div>
        <span
          className={`px-2 py-1 rounded text-xs font-medium ${
            exchange.enabled
              ? 'bg-green-100 text-green-700'
              : 'bg-gray-100 text-gray-500'
          }`}
        >
          {exchange.enabled ? 'Active' : 'Disabled'}
        </span>
      </div>

      <div className="space-y-1 text-sm text-gray-600">
        <div className="flex justify-between">
          <span>Currency:</span>
          <span className="font-medium">{exchange.quote_currency}</span>
        </div>
        {exchange.schedule && (
          <div className="flex justify-between">
            <span>Schedule:</span>
            <span className="font-medium">{exchange.schedule}</span>
          </div>
        )}
        <div className="flex justify-between">
          <span>Symbols:</span>
          <span className="font-medium">
            {exchange.symbols_count === -1 ? 'Dynamic' : exchange.symbols_count}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function ExchangeStatusPanel() {
  const {
    data: response,
    isLoading,
    isError,
  } = useQuery<ExchangeStatusResponse>({
    queryKey: ['exchanges'],
    queryFn: async () => {
      const res = await apiClient.get<ExchangeStatusResponse>('/exchanges');
      return res.data;
    },
    refetchInterval: 60_000,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });

  if (isLoading) {
    return (
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-lg font-semibold mb-4">Exchanges</h2>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-lg font-semibold mb-4">Exchanges</h2>
        <div className="text-red-500">Failed to load exchange status</div>
      </div>
    );
  }

  const exchanges = response?.exchanges || [];

  // Sort: enabled first, then alphabetically
  const sortedExchanges = [...exchanges].sort((a, b) => {
    if (a.enabled !== b.enabled) return b.enabled ? 1 : -1;
    return a.exchange.localeCompare(b.exchange);
  });

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="text-lg font-semibold mb-4">Exchanges</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {sortedExchanges.map((exchange) => (
          <ExchangeCard key={exchange.exchange} exchange={exchange} />
        ))}
      </div>
    </div>
  );
}
