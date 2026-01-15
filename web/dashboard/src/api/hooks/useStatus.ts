import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../client';
import type { SystemStatus } from '../types';

export function useStatus() {
  return useQuery<SystemStatus>({
    queryKey: ['status'],
    queryFn: async () => {
      const res = await apiClient.get<SystemStatus>('/status');
      return res.data;
    },
    refetchInterval: 5000,
  });
}
