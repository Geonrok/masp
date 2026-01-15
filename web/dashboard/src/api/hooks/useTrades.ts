import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../client';
import type { TradesResponse } from '../types';

export function useTrades() {
  return useQuery<TradesResponse>({
    queryKey: ['trades'],
    queryFn: async () => {
      const res = await apiClient.get<TradesResponse>('/trades');
      return res.data;
    },
    refetchInterval: 5000,
  });
}
