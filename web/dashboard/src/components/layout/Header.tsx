import { useConnectionState } from '../../store/priceStore';

export default function Header() {
  const connectionState = useConnectionState();

  const statusColor = {
    OPEN: 'bg-green-500',
    CONNECTING: 'bg-yellow-500',
    RECONNECTING: 'bg-yellow-500',
    CLOSED: 'bg-red-500',
  }[connectionState];

  return (
    <header className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
      <h1 className="text-lg font-semibold text-gray-800">
        Multi-Asset Strategy Platform
      </h1>
      <div className="flex items-center gap-2">
        <span className={`w-3 h-3 rounded-full ${statusColor}`} />
        <span className="text-sm text-gray-600">
          {connectionState === 'OPEN' ? 'Connected' : connectionState}
        </span>
      </div>
    </header>
  );
}
