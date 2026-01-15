interface StatusCardProps {
  label: string;
  value: string | number;
  tone?: 'up' | 'down' | 'neutral';
}

const toneClass = {
  up: 'text-trade-up',
  down: 'text-trade-down',
  neutral: 'text-trade-neutral',
};

export default function StatusCard({ label, value, tone = 'neutral' }: StatusCardProps) {
  return (
    <div className="bg-white p-4 rounded shadow">
      <div className="text-sm text-gray-500">{label}</div>
      <div className={`text-xl font-bold ${toneClass[tone]}`}>{value}</div>
    </div>
  );
}
