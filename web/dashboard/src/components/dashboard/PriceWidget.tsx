import { usePriceValues } from '../../store/priceStore';

export default function PriceWidget() {
  const prices = usePriceValues();
  const rows = Object.entries(prices).slice(0, 6);

  return (
    <div className="bg-white p-4 rounded shadow">
      <div className="text-sm text-gray-500 mb-2">Live Prices</div>
      {rows.length === 0 ? (
        <div className="text-sm text-gray-400">No price updates yet.</div>
      ) : (
        <ul className="space-y-2">
          {rows.map(([symbol, price]) => (
            <li key={symbol} className="flex justify-between text-sm">
              <span className="text-gray-700">{symbol}</span>
              <span className="font-medium">{price.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
