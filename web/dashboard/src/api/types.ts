// Backend schema-aligned types
export interface BaseResponse {
  success: boolean;
  message?: string;
  request_id?: string;
  timestamp?: string;
}

export interface SystemStatus extends BaseResponse {
  version: string;
  uptime_seconds: number;
  exchanges: string[];
  active_strategies: number;
}

export interface ExchangeStatus {
  exchange: string;
  enabled: boolean;
  connected: boolean;
  quote_currency: string;
  schedule?: string;
  next_run?: string;
  last_run?: string;
  symbols_count: number;
}

export interface ExchangeStatusResponse extends BaseResponse {
  exchanges: ExchangeStatus[];
}

export interface StrategyInfo {
  strategy_id: string;
  name: string;
  version: string;
  description: string;
  active: boolean;
}

export interface StrategyListResponse extends BaseResponse {
  strategies: StrategyInfo[];
}

export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

export interface PositionsResponse extends BaseResponse {
  positions: Position[];
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  strategy_id: string;
}

export interface TradesResponse extends BaseResponse {
  trades: Trade[];
}

export interface KillSwitchResponse extends BaseResponse {
  positions_closed: number;
  strategies_stopped: number;
}

export type ConnectionState = 'CONNECTING' | 'OPEN' | 'CLOSED' | 'RECONNECTING';

export interface WSMessage {
  type: 'price' | 'signal' | 'ping' | 'pong';
  data?: {
    symbol?: string;
    price?: number;
    change_24h?: number;
    signal?: string;
    reason?: string;
    timestamp?: string;
  };
  ts: string;
}
