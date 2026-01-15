import { useCallback, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import { useWebSocket } from '../../hooks/useWebSocket';
import { usePriceActions } from '../../store/priceStore';
import type { WSMessage } from '../../api/types';

export default function Layout() {
  const { updatePrice, setConnectionState } = usePriceActions();

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      if (msg.type === 'price' && msg.data?.symbol && typeof msg.data.price === 'number') {
        updatePrice(msg.data.symbol, msg.data.price);
      }
    },
    [updatePrice]
  );

  const { connectionState } = useWebSocket(handleMessage);

  useEffect(() => {
    setConnectionState(connectionState);
  }, [connectionState, setConnectionState]);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
