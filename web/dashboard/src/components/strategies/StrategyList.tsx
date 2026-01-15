import { useStrategies } from '../../api/hooks/useStrategies';
import LoadingSpinner from '../common/LoadingSpinner';
import StrategyCard from './StrategyCard';

export default function StrategyList() {
  const { data, isLoading } = useStrategies();

  if (isLoading) {
    return <LoadingSpinner />;
  }

  const strategies = data?.strategies ?? [];

  return (
    <div className="grid gap-4">
      {strategies.length === 0 ? (
        <div className="bg-white p-4 rounded shadow text-gray-400">No strategies.</div>
      ) : (
        strategies.map((strategy) => (
          <StrategyCard key={strategy.strategy_id} strategy={strategy} />
        ))
      )}
    </div>
  );
}
