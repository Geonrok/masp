"""
Config Utilities for Phase 2C
환경변수 기반 설정 유틸리티
"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def get_trading_limits() -> dict:
    """
    환경변수에서 Trading Limits 로드
    
    Returns:
        dict: Trading limits configuration
    """
    return {
        "max_order_value_krw": int(os.getenv("MAX_ORDER_VALUE_KRW", "1000000")),
        "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.10")),
        "max_daily_loss_krw": int(os.getenv("MAX_DAILY_LOSS_KRW", "100000")),
        "adapter_mode": os.getenv("ADAPTER_MODE", "paper")
    }


def is_live_mode() -> bool:
    """
    실거래 모드 여부 확인
    
    Returns:
        bool: True if live mode, False if paper mode
    """
    return os.getenv("ADAPTER_MODE", "paper").lower() == "live"


def validate_api_keys(exchange: str) -> bool:
    """
    환경변수 API 키 유효성 검증
    
    Args:
        exchange: "upbit" | "bithumb" | "binance"
        
    Returns:
        bool: True if all required keys are set
    """
    if exchange.lower() == "upbit":
        access = os.getenv("UPBIT_ACCESS_KEY", "")
        secret = os.getenv("UPBIT_SECRET_KEY", "")
        return bool(access and secret and access != "your_access_key_here")
    
    elif exchange.lower() == "bithumb":
        api_key = os.getenv("BITHUMB_API_KEY", "")
        secret = os.getenv("BITHUMB_SECRET_KEY", "")
        return bool(api_key and secret and api_key != "your_api_key_here")
    
    elif exchange.lower() == "binance":
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")
        return bool(api_key and secret and api_key != "your_api_key_here")
    
    return False


def get_adapter_mode_description() -> str:
    """
    현재 Adapter Mode 설명
    
    Returns:
        str: Mode description with warnings
    """
    if is_live_mode():
        return "⚠️ LIVE MODE - 실거래 활성화! 실제 자금이 사용됩니다."
    else:
        return "✅ PAPER MODE - 모의 거래 (안전)"


def print_config_summary():
    """환경변수 기반 설정 요약 출력"""
    print("=" * 60)
    print("Configuration Summary (from .env)")
    print("=" * 60)
    
    limits = get_trading_limits()
    print(f"\nTrading Limits:")
    print(f"  Max Order Value: {limits['max_order_value_krw']:,} KRW")
    print(f"  Max Position: {limits['max_position_pct']*100:.0f}%")
    print(f"  Max Daily Loss: {limits['max_daily_loss_krw']:,} KRW")
    
    print(f"\nAdapter Mode:")
    print(f"  {get_adapter_mode_description()}")
    
    print(f"\nAPI Keys:")
    for exchange in ["upbit", "bithumb", "binance"]:
        status = "✅ SET" if validate_api_keys(exchange) else "❌ NOT SET"
        print(f"  {exchange.capitalize()}: {status}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
