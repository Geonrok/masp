import { useEffect, useRef, useCallback, useState } from 'react';
import type { WSMessage, ConnectionState } from '../api/types';

const BASE_DELAY = 1000;
const MAX_DELAY = 30000;

export function useWebSocket(onMessage: (msg: WSMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>('CLOSED');
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<number | null>(null);
  const shouldReconnect = useRef(true);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimer.current !== null) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
  }, []);

  const getReconnectDelay = useCallback(() => {
    const exponentialDelay = Math.min(
      BASE_DELAY * Math.pow(2, reconnectAttempt.current),
      MAX_DELAY
    );
    const jitter = Math.random() * BASE_DELAY;
    reconnectAttempt.current += 1;
    return exponentialDelay + jitter;
  }, []);

  const connect = useCallback(() => {
    clearReconnectTimer();

    if (!shouldReconnect.current) {
      return;
    }

    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;

      if (wsRef.current.readyState !== WebSocket.CLOSED) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}/ws/stream`;
    console.log('[WS] Connecting to:', wsUrl);

    setConnectionState('CONNECTING');
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WS] Connected');
      setIsConnected(true);
      setConnectionState('OPEN');
      reconnectAttempt.current = 0;

      if (import.meta.env.DEV) {
        ws.send(JSON.stringify({ type: 'ping', ts: new Date().toISOString() }));
      }
    };

    ws.onmessage = (event) => {
      if (import.meta.env.DEV) {
        const raw = typeof event.data === 'string' ? event.data : '[non-string]';
        const preview = raw.length > 500 ? `${raw.slice(0, 500)}...<truncated>` : raw;
        console.log('[WS] Raw:', preview);
      }

      try {
        const msg: WSMessage = JSON.parse(event.data);

        if (import.meta.env.DEV && msg.type === 'ping') {
          console.log('[DEV] Mock: Generating fake price data');

          const mockSymbols = ['KRW-BTC', 'KRW-ETH'];
          mockSymbols.forEach((symbol, idx) => {
            const basePrice = symbol === 'KRW-BTC' ? 50000000 : 3500000;
            const variance = basePrice * 0.001;
            const mockPrice = basePrice + (Math.random() - 0.5) * variance;

            const mockMsg: WSMessage = {
              type: 'price',
              data: {
                symbol,
                price: Math.round(mockPrice),
                timestamp: new Date().toISOString(),
              },
              ts: new Date().toISOString(),
            };

            setTimeout(() => {
              console.log(`[DEV] Mock: ${symbol} = ${mockMsg.data?.price}`);
              onMessage(mockMsg);
            }, idx * 50);
          });
        }

        if (import.meta.env.DEV && msg.type !== 'ping' && msg.type !== 'pong') {
          console.log('[WS] Parsed:', msg.type, msg);
        }

        if (msg.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong', ts: new Date().toISOString() }));
          return;
        }
        onMessage(msg);
      } catch (e) {
        console.warn('[WS] Parse error:', e, 'Raw:', event.data);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected');
      setIsConnected(false);
      if (!shouldReconnect.current) {
        setConnectionState('CLOSED');
        return;
      }
      setConnectionState('RECONNECTING');

      const delay = getReconnectDelay();
      console.log(
        `[WS] Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttempt.current})`
      );
      reconnectTimer.current = window.setTimeout(connect, delay);
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
    };

    wsRef.current = ws;
  }, [onMessage, getReconnectDelay, clearReconnectTimer]);

  useEffect(() => {
    shouldReconnect.current = true;
    connect();
    return () => {
      shouldReconnect.current = false;
      clearReconnectTimer();
      wsRef.current?.close();
    };
  }, [connect, clearReconnectTimer]);

  return { isConnected, connectionState };
}
