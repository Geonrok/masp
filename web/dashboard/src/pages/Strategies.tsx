import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { StrategyListResponse } from '../api/types';

export default function Strategies() {
  const { data, isLoading } = useQuery<StrategyListResponse>({
    queryKey: ['strategies'],
    queryFn: async () => {
      const res = await apiClient.get<StrategyListResponse>('/strategy/list');
      return res.data;
    },
  });

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Strategies</h1>
      <div className="grid gap-4">
        {data?.strategies?.map((strategy) => (
          <div key={strategy.strategy_id} className="bg-white p-4 rounded shadow">
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
        ))}
      </div>
    </div>
  );
}
