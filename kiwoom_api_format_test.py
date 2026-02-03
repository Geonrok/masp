# -*- coding: utf-8 -*-
"""
키움 REST API 파라미터 형식 테스트
- /api/dostk/ka10001 경로가 200 반환 확인됨
- API ID 파라미터 형식 찾기
"""

import requests
import json

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


def test_api_format(token):
    """다양한 API 파라미터 형식 테스트"""

    test_cases = [
        # Case 1: api_id를 body에 추가
        {
            "name": "api_id in body",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
            },
            "data": {"api_id": "ka10001", "stk_cd": "005930"},
        },
        # Case 2: tr_id를 body에 추가
        {
            "name": "tr_id in body",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
            },
            "data": {"tr_id": "ka10001", "stk_cd": "005930"},
        },
        # Case 3: api_id 헤더로
        {
            "name": "api_id header",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "api_id": "ka10001",
            },
            "data": {"stk_cd": "005930"},
        },
        # Case 4: tr_id 헤더로
        {
            "name": "tr_id header",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {"stk_cd": "005930"},
        },
        # Case 5: appkey도 헤더에 추가
        {
            "name": "appkey + api_id headers",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "appkey": APPKEY,
                "api_id": "ka10001",
            },
            "data": {"stk_cd": "005930"},
        },
        # Case 6: 키움 공식 문서 형식 추정 (input/output 구조)
        {
            "name": "input wrapper",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
            },
            "data": {"input": {"stk_cd": "005930"}},
        },
        # Case 7: tr_cd 필드
        {
            "name": "tr_cd in body",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
            },
            "data": {"tr_cd": "ka10001", "stk_cd": "005930"},
        },
        # Case 8: 대문자 TR_ID
        {
            "name": "TR_ID header",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "TR_ID": "ka10001",
            },
            "data": {"stk_cd": "005930"},
        },
        # Case 9: 다양한 파라미터명
        {
            "name": "stock_code field",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {"stock_code": "005930"},
        },
        # Case 10: scode 필드
        {
            "name": "scode field",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {"scode": "005930"},
        },
        # Case 11: 코드 없이 빈 요청
        {
            "name": "empty body with tr_id header",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {},
        },
        # Case 12: shcode (eBest 스타일)
        {
            "name": "shcode field",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {"shcode": "005930"},
        },
        # Case 13: 종목코드 형식 다르게
        {
            "name": "code field",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "data": {"code": "005930"},
        },
        # Case 14: query 파라미터 + POST
        {
            "name": "query params",
            "headers": {
                "Content-Type": "application/json;charset=UTF-8",
                "Authorization": f"Bearer {token}",
                "tr_id": "ka10001",
            },
            "params": {"stk_cd": "005930"},
            "data": {},
        },
    ]

    url = f"{BASE_URL}/api/dostk/ka10001"

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"테스트: {case['name']}")

        try:
            if "params" in case:
                response = requests.post(
                    url,
                    headers=case["headers"],
                    params=case["params"],
                    json=case["data"],
                )
            else:
                response = requests.post(
                    url, headers=case["headers"], json=case["data"]
                )

            print(f"Status: {response.status_code}")

            result = response.json()
            return_code = result.get("return_code")
            return_msg = result.get("return_msg", "")

            print(f"Return Code: {return_code}")
            print(f"Return Msg: {return_msg[:100]}")

            # return_code가 0이면 성공
            if return_code == 0:
                print("*** SUCCESS! ***")
                print(
                    f"Full Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}"
                )
                return case

            # 다른 오류 코드도 분석
            if return_code != 2:  # 2는 "API ID null" 오류
                print(f"** 다른 오류 코드 발견: {return_code} **")

        except Exception as e:
            print(f"Error: {e}")

    return None


def test_get_method(token):
    """GET 메서드로도 테스트"""
    print("\n" + "=" * 60)
    print("GET 메서드 테스트")
    print("=" * 60)

    test_cases = [
        # GET with query params
        {
            "name": "GET with stk_cd param",
            "headers": {"Authorization": f"Bearer {token}", "tr_id": "ka10001"},
            "params": {"stk_cd": "005930"},
        },
        {
            "name": "GET with api_id param",
            "headers": {"Authorization": f"Bearer {token}"},
            "params": {"api_id": "ka10001", "stk_cd": "005930"},
        },
    ]

    url = f"{BASE_URL}/api/dostk/ka10001"

    for case in test_cases:
        print(f"\n테스트: {case['name']}")
        try:
            response = requests.get(url, headers=case["headers"], params=case["params"])
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

            if response.status_code == 200:
                result = response.json()
                if result.get("return_code") == 0:
                    print("*** SUCCESS! ***")
                    return case
        except Exception as e:
            print(f"Error: {e}")

    return None


def main():
    print("키움 REST API 파라미터 형식 테스트")
    print("=" * 60)

    token = get_token()
    if not token:
        print("토큰 발급 실패!")
        return

    print(f"Token: {token[:30]}...")

    # POST 테스트
    success = test_api_format(token)

    if not success:
        # GET 테스트
        success = test_get_method(token)

    if success:
        print(f"\n\n{'='*60}")
        print(f"성공한 설정: {success['name']}")
        print(f"Headers: {success.get('headers', {})}")
        print(f"Data: {success.get('data', {})}")
    else:
        print(f"\n\n{'='*60}")
        print("모든 형식 테스트 실패")
        print("\n가능한 원인:")
        print("1. API 이용 권한이 아직 활성화되지 않음")
        print("2. 키움 REST API 개발자 페이지에서 추가 설정 필요")
        print("3. 특정 TR 코드 사용 권한 필요")


if __name__ == "__main__":
    main()
