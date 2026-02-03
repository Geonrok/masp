# -*- coding: utf-8 -*-
"""
키움 REST API 엔드포인트 탐색
"""

import requests
import json
import os

# API 키
APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"
BASE_URL = "https://api.kiwoom.com"


def get_token():
    """토큰 발급"""
    url = f"{BASE_URL}/oauth2/token"
    response = requests.post(
        url,
        json={
            "grant_type": "client_credentials",
            "appkey": APPKEY,
            "secretkey": SECRETKEY,
        },
    )
    return response.json().get("token")


def test_endpoint(token, method, path, params=None, data=None):
    """엔드포인트 테스트"""
    url = f"{BASE_URL}{path}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {token}",
    }

    print(f"\n{'='*50}")
    print(f"{method} {path}")

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        else:
            response = requests.post(url, headers=headers, json=data)

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(
                f"SUCCESS! Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}"
            )
            return True
        elif response.status_code != 500:
            # 500이 아닌 다른 오류는 출력
            print(f"Response: {response.text[:300]}")

    except Exception as e:
        print(f"Error: {e}")

    return False


def main():
    print("키움 REST API 엔드포인트 탐색")
    print("=" * 60)

    token = get_token()
    print(f"Token: {token[:30]}...")

    # 가능한 엔드포인트 패턴들
    endpoints = [
        # 공식 문서 기반 추정
        ("GET", "/api/dostk/stkinfo", {"stk_cd": "005930"}),
        ("GET", "/api/dostk/chart", {"stk_cd": "005930", "chart_tp": "D"}),
        ("GET", "/api/dostk/investor", {"stk_cd": "005930"}),
        # TR 코드 기반
        ("POST", "/api/ka10001", {"stk_cd": "005930"}),
        ("POST", "/api/ka10008", {"stk_cd": "005930"}),
        ("POST", "/api/ka10081", {"stk_cd": "005930", "base_dt": "00000000"}),
        # 다른 형식
        (
            "GET",
            "/v1/quotations/inquire-price",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"},
        ),
        (
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"},
        ),
        # 기타
        ("GET", "/api/health", None),
        ("GET", "/health", None),
        ("GET", "/api/version", None),
    ]

    success_count = 0
    for method, path, params in endpoints:
        if test_endpoint(token, method, path, params):
            success_count += 1

    print(f"\n\n{'='*60}")
    print(f"테스트 완료: {success_count}/{len(endpoints)} 성공")


if __name__ == "__main__":
    main()
