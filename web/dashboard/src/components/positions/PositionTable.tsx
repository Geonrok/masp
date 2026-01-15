import { usePositions } from '../../api/hooks/usePositions';
import LoadingSpinner from '../common/LoadingSpinner';

export default function PositionTable() {
  const { data, isLoading } = usePositions();

  if (isLoading) {
    return <LoadingSpinner />;
  }

  const positions = data?.positions ?? [];

  return (
    <div className="bg-white rounded shadow overflow-hidden">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            <th className="px-4 py-2 text-left">Symbol</th>
            <th className="px-4 py-2 text-left">Side</th>
            <th className="px-4 py-2 text-right">Qty</th>
            <th className="px-4 py-2 text-right">Entry</th>
            <th className="px-4 py-2 text-right">PnL</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 ? (
            <tr>
              <td colSpan={5} className="px-4 py-6 text-center text-gray-400">
                No positions yet.
              </td>
            </tr>
          ) : (
            positions.map((pos) => (
              <tr key={pos.symbol} className="border-t">
                <td className="px-4 py-2">{pos.symbol}</td>
                <td className="px-4 py-2">{pos.side}</td>
                <td className="px-4 py-2 text-right">{pos.quantity}</td>
                <td className="px-4 py-2 text-right">{pos.entry_price}</td>
                <td className="px-4 py-2 text-right">{pos.pnl}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
