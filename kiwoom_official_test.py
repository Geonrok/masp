# -*- coding: utf-8 -*-
"""
키움 REST API 공식 패키지(kiwoomRest) 테스트
"""

import asyncio

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"


async def main():
    print("=" * 60)
    print("키움 REST API 공식 패키지 테스트")
    print("=" * 60)

    api = None
    try:
        from kiwoomRest import KwRestApi

        # API 인스턴스 생성
        api = KwRestApi()
        print("API 인스턴스 생성 완료")

        # 로그인 (실거래서버)
        print("\n로그인 시도 중...")
        ret = await api.login(APPKEY, SECRETKEY, is_simulation=False)
        print(f"로그인 결과: {ret}")

        if not ret:
            print(f"로그인 실패: {api.last_message}")
            return

        print("로그인 성공!")

        # 1. 삼성전자 현재가 조회 (ka10001)
        print("\n=== 삼성전자 현재가 (ka10001) ===")
        try:
            response = await api.request("ka10001", {"stk_cd": "005930"})
            print(f"응답 코드: {response.return_code}")
            print(f"응답 메시지: {response.return_msg}")
            if response.return_code == 0 and response.body:
                print(f"데이터: {response.body}")
        except Exception as e:
            print(f"ka10001 오류: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)

        # 2. 에코프로비엠 현재가 (코스닥)
        print("\n=== 에코프로비엠 현재가 (247540) ===")
        try:
            response = await api.request("ka10001", {"stk_cd": "247540"})
            print(f"응답 코드: {response.return_code}")
            if response.return_code == 0 and response.body:
                print(f"데이터: {response.body}")
            else:
                print(f"메시지: {response.return_msg}")
        except Exception as e:
            print(f"에코프로비엠 오류: {e}")

        await asyncio.sleep(0.5)

        # 3. 외국인/기관 매매동향 (ka10008)
        print("\n=== 외국인/기관 매매동향 (ka10008) ===")
        try:
            response = await api.request("ka10008", {"stk_cd": "005930"})
            print(f"응답 코드: {response.return_code}")
            if response.return_code == 0 and response.body:
                print(f"데이터: {response.body}")
            else:
                print(f"메시지: {response.return_msg}")
        except Exception as e:
            print(f"ka10008 오류: {e}")

        await asyncio.sleep(0.5)

        # 4. 외국인/기관 종합 (ka10009)
        print("\n=== 외국인/기관 종합 (ka10009) ===")
        try:
            response = await api.request("ka10009", {"stk_cd": "005930"})
            print(f"응답 코드: {response.return_code}")
            if response.return_code == 0 and response.body:
                print(f"데이터: {response.body}")
            else:
                print(f"메시지: {response.return_msg}")
        except Exception as e:
            print(f"ka10009 오류: {e}")

        await asyncio.sleep(0.5)

        # 5. 일봉 데이터 (ka10081)
        print("\n=== 삼성전자 일봉 (ka10081) ===")
        try:
            response = await api.request("ka10081", {
                "stk_cd": "005930",
                "base_dt": "00000000",
                "upd_stkpc_tp": "1"
            })
            print(f"응답 코드: {response.return_code}")
            if response.return_code == 0 and response.body:
                data = response.body
                if isinstance(data, list) and len(data) > 0:
                    print(f"데이터 수: {len(data)}")
                    print(f"첫 번째: {data[0]}")
                elif isinstance(data, dict):
                    if 'output' in data:
                        output = data['output']
                        if isinstance(output, list) and len(output) > 0:
                            print(f"데이터 수: {len(output)}")
                            print(f"첫 번째: {output[0]}")
                        else:
                            print(f"output: {output}")
                    else:
                        print(f"데이터: {data}")
                else:
                    print(f"데이터: {data}")
            else:
                print(f"메시지: {response.return_msg}")
        except Exception as e:
            print(f"ka10081 오류: {e}")
            import traceback
            traceback.print_exc()

        # 6. 외국인기관 순매수 상위 (ka10131)
        print("\n=== 외국인/기관 순매수 상위 (ka10131) ===")
        try:
            response = await api.request("ka10131", {"mkt_tp": "0"})  # 0: 전체
            print(f"응답 코드: {response.return_code}")
            if response.return_code == 0 and response.body:
                print(f"데이터: {str(response.body)[:500]}")
            else:
                print(f"메시지: {response.return_msg}")
        except Exception as e:
            print(f"ka10131 오류: {e}")

    except ImportError as e:
        print(f"Import 오류: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 연결 종료
        if api:
            await api.close()
            print("\n연결 종료")

    print("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
