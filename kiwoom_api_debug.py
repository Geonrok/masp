# -*- coding: utf-8 -*-
"""
키움 REST API 상세 디버그
"""

import requests
import json

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"
BASE_URL = "https://api.kiwoom.com"


def get_token():
    """토큰 발급"""
    url = f"{BASE_URL}/oauth2/token"
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    data = {
        "grant_type": "client_credentials",
        "appkey": APPKEY,
        "secretkey": SECRETKEY
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"토큰 응답: {response.json()}")
    return response.json()


def test_with_various_headers(token_data):
    """다양한 헤더 조합 테스트"""
    token = token_data.get('token')

    # 테스트할 엔드포인트 (ka10001: 주식현재가)
    test_cases = [
        # Case 1: Bearer 토큰만
        {
            "name": "Bearer 토큰만",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}"
            },
            "path": "/api/ka10001",
            "data": {"stk_cd": "005930"}
        },
        # Case 2: appkey + secretkey 헤더
        {
            "name": "appkey/secretkey 헤더",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "appkey": APPKEY,
                "appsecret": SECRETKEY
            },
            "path": "/api/ka10001",
            "data": {"stk_cd": "005930"}
        },
        # Case 3: tr_id 헤더 추가
        {
            "name": "tr_id 헤더 추가",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001"
            },
            "path": "/api/ka10001",
            "data": {"stk_cd": "005930"}
        },
        # Case 4: 다른 경로 형식
        {
            "name": "dostk 경로",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}"
            },
            "path": "/api/dostk/ka10001",
            "data": {"stk_cd": "005930"}
        },
        # Case 5: 쿼리 파라미터로
        {
            "name": "GET with params",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}"
            },
            "path": "/api/ka10001",
            "method": "GET",
            "params": {"stk_cd": "005930"}
        },
        # Case 6: custtype 헤더 (개인/법인 구분)
        {
            "name": "custtype 헤더",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "custtype": "P"  # P: 개인
            },
            "path": "/api/ka10001",
            "data": {"stk_cd": "005930"}
        },
        # Case 7: 한국투자증권 형식 참고
        {
            "name": "한투 형식 참고",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {token}",
                "appkey": APPKEY,
                "appsecret": SECRETKEY,
                "tr_id": "FHKST01010100"
            },
            "path": "/uapi/domestic-stock/v1/quotations/inquire-price",
            "params": {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"}
        },
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"테스트: {case['name']}")
        print(f"Path: {case['path']}")

        url = f"{BASE_URL}{case['path']}"
        method = case.get('method', 'POST')

        try:
            if method == "GET":
                response = requests.get(url, headers=case['headers'], params=case.get('params'))
            else:
                response = requests.post(url, headers=case['headers'], json=case.get('data'))

            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}")

            if response.status_code == 200:
                print("*** SUCCESS! ***")
                return case

        except Exception as e:
            print(f"Error: {e}")

    return None


def main():
    print("키움 REST API 상세 디버그")
    print("=" * 60)

    # 토큰 발급
    token_data = get_token()

    if not token_data.get('token'):
        print("토큰 발급 실패!")
        return

    # 다양한 헤더 테스트
    success = test_with_various_headers(token_data)

    if success:
        print(f"\n\n성공한 설정: {success['name']}")
    else:
        print("\n\n모든 테스트 실패")
        print("\n가능한 원인:")
        print("1. API 이용 신청 후 활성화까지 시간 필요")
        print("2. 키움 REST API 사이트에서 추가 설정 필요")
        print("3. 모의투자/실거래 서버 구분 필요")


if __name__ == "__main__":
    main()
