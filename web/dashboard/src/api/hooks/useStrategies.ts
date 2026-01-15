import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../client';
import type { StrategyListResponse } from '../types';

export function useStrategies() {
  return useQuery<StrategyListResponse>({
    queryKey: ['strategies'],
    queryFn: async () => {
      const res = await apiClient.get<StrategyListResponse>('/strategy/list');
      return res.data;
    },
  });
}
