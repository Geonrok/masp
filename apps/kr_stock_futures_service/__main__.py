"""
KR Stock Futures Service - Entry point.

Usage:
    python -m apps.kr_stock_futures_service --once
    python -m apps.kr_stock_futures_service --daemon
"""

from libs.core.config import AssetClass
from apps.service_base import create_service_main

main = create_service_main(AssetClass.KR_STOCK_FUTURES)

if __name__ == "__main__":
    main()


