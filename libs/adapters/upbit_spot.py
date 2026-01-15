"""
Alias module for Upbit spot adapters.
Use libs.adapters.real_upbit_spot for actual implementations.
"""

from libs.adapters.real_upbit_spot import UpbitSpotExecution, UpbitSpotMarketData

__all__ = ["UpbitSpotExecution", "UpbitSpotMarketData"]
