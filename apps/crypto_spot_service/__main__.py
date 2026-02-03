"""
Crypto Spot Service - Entry point.

Usage:
    python -m apps.crypto_spot_service --once
    python -m apps.crypto_spot_service --daemon
"""

from apps.service_base import create_service_main
from libs.core.config import AssetClass

main = create_service_main(AssetClass.CRYPTO_SPOT)

if __name__ == "__main__":
    main()
