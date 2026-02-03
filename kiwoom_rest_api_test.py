# -*- coding: utf-8 -*-
"""
키움 REST API 테스트
- 32-bit Python 불필요
- REST API로 투자자 데이터 조회
"""

import json
import logging

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API 키
APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"

# 기본 URL
BASE_URL = "https://api.kiwoom.com"  # 실전투자
# BASE_URL = "https://mockapi.kiwoom.com"  # 모의투자


def get_access_token():
    """접근 토큰 발급"""
    url = f"{BASE_URL}/oauth2/token"

    headers = {"Content-Type": "application/json;charset=UTF-8"}

    data = {
        "grant_type": "client_credentials",
        "appkey": APPKEY,
        "secretkey": SECRETKEY,
    }

    logger.info("토큰 발급 요청 중...")

    try:
        response = requests.post(url, headers=headers, json=data)
        logger.info(f"응답 상태: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.info("토큰 발급 성공!")
            logger.info(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result.get("access_token") or result.get("token")
        else:
            logger.error(f"토큰 발급 실패: {response.text}")
            return None

    except Exception as e:
        logger.error(f"오류: {e}")
        return None


def test_api_endpoint(token, endpoint, params=None):
    """API 엔드포인트 테스트"""
    url = f"{BASE_URL}{endpoint}"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {token}",
    }

    logger.info(f"\n=== {endpoint} 테스트 ===")

    try:
        if params:
            response = requests.post(url, headers=headers, json=params)
        else:
            response = requests.get(url, headers=headers)

        logger.info(f"상태: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.info(
                f"응답: {json.dumps(result, indent=2, ensure_ascii=False)[:1000]}"
            )
            return result
        else:
            logger.error(f"실패: {response.text[:500]}")
            return None

    except Exception as e:
        logger.error(f"오류: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("키움 REST API 테스트")
    logger.info("=" * 60)

    # 1. 토큰 발급
    token = get_access_token()

    if not token:
        logger.error("토큰 발급 실패. 테스트 중단.")
        return

    logger.info(f"\n토큰: {token[:30]}...")

    # 2. API 테스트 (예상되는 엔드포인트들)
    # 키움 REST API 문서에 따라 엔드포인트가 다를 수 있음

    # 가능한 엔드포인트 테스트
    endpoints_to_try = [
        # 시세 관련
        ("/v1/trading/inquire-balance", None),  # 잔고 조회
        (
            "/v1/quotations/inquire-price",
            {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": "005930"},
        ),
        ("/v1/trading/inquire-daily-ccld", None),  # 일별 체결
        # 투자자별 매매동향 (추정)
        ("/v1/quotations/investor", {"fid_input_iscd": "005930"}),
        ("/v1/quotations/foreign-institution", {"fid_input_iscd": "005930"}),
    ]

    for endpoint, params in endpoints_to_try:
        test_api_endpoint(token, endpoint, params)

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    main()
