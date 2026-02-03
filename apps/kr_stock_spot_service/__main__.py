"""
KR Stock Spot Service - Entry point.

Usage:
    python -m apps.kr_stock_spot_service --once
    python -m apps.kr_stock_spot_service --daemon
"""

from apps.service_base import create_service_main
from libs.core.config import AssetClass

main = create_service_main(AssetClass.KR_STOCK_SPOT)

if __name__ == "__main__":
    main()
