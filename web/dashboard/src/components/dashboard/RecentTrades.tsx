import { useTrades } from '../../api/hooks/useTrades';
import LoadingSpinner from '../common/LoadingSpinner';

export default function RecentTrades() {
  const { data, isLoading } = useTrades();

  if (isLoading) {
    return <LoadingSpinner />;
  }

  const trades = data?.trades ?? [];

  return (
    <div className="bg-white p-4 rounded shadow">
      <div className="text-sm text-gray-500 mb-2">Recent Trades</div>
      {trades.length === 0 ? (
        <div className="text-sm text-gray-400">No trades yet.</div>
      ) : (
        <ul className="space-y-2">
          {trades.slice(0, 5).map((trade) => (
            <li key={trade.id} className="text-sm text-gray-700">
              {trade.symbol} {trade.side} @ {trade.price.toLocaleString()}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
