"""
Crypto Futures Service - Entry point.

Usage:
    python -m apps.crypto_futures_service --once
    python -m apps.crypto_futures_service --daemon
"""

from apps.service_base import create_service_main
from libs.core.config import AssetClass

main = create_service_main(AssetClass.CRYPTO_FUTURES)

if __name__ == "__main__":
    main()
