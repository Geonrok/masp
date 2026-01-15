import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../client';
import type { PositionsResponse } from '../types';

export function usePositions() {
  return useQuery<PositionsResponse>({
    queryKey: ['positions'],
    queryFn: async () => {
      const res = await apiClient.get<PositionsResponse>('/positions');
      return res.data;
    },
    refetchInterval: 5000,
  });
}
