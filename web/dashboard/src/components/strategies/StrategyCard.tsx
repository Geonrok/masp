import type { StrategyInfo } from '../../api/types';

interface StrategyCardProps {
  strategy: StrategyInfo;
}

export default function StrategyCard({ strategy }: StrategyCardProps) {
  return (
    <div className="bg-white p-4 rounded shadow">
      <div className="font-bold">{strategy.name}</div>
      <div className="text-sm text-gray-500">{strategy.description}</div>
      <div className="mt-2">
        <span
          className={`px-2 py-1 rounded text-xs ${
            strategy.active ? 'bg-green-100 text-green-800' : 'bg-gray-100'
          }`}
        >
          {strategy.active ? 'Active' : 'Inactive'}
        </span>
      </div>
    </div>
  );
}
