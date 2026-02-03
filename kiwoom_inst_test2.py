# -*- coding: utf-8 -*-
"""
키움 REST API - 기관/외국인 순위 TR 코드 테스트 (필수 파라미터 포함)
"""

import asyncio

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"


async def main():
    print("=" * 60)
    print("기관/외국인 순위 데이터 TR 코드 테스트")
    print("=" * 60)

    api = None
    try:
        from kiwoomRest import KwRestApi

        api = KwRestApi()
        ret = await api.login(APPKEY, SECRETKEY, is_simulation=False)
        if not ret:
            print(f"로그인 실패: {api.last_message}")
            return

        print("로그인 성공!\n")

        # 테스트할 TR 코드와 파라미터 조합
        test_cases = [
            # 외국인/기관 순매수 상위 (sort_tp 추가)
            (
                "ka10029",
                {"mrkt_tp": "0", "sort_tp": "1"},
                "외국인순매수상위 (전체, 순매수금액순)",
            ),
            ("ka10029", {"mrkt_tp": "1", "sort_tp": "1"}, "외국인순매수상위 (코스피)"),
            ("ka10029", {"mrkt_tp": "2", "sort_tp": "1"}, "외국인순매수상위 (코스닥)"),
            ("ka10030", {"mrkt_tp": "0", "sort_tp": "1"}, "기관순매수상위 (전체)"),
            ("ka10030", {"mrkt_tp": "1", "sort_tp": "1"}, "기관순매수상위 (코스피)"),
            ("ka10030", {"mrkt_tp": "2", "sort_tp": "1"}, "기관순매수상위 (코스닥)"),
            ("ka10031", {"mrkt_tp": "0", "sort_tp": "1"}, "외국인순매도상위 (전체)"),
            ("ka10032", {"mrkt_tp": "0", "sort_tp": "1"}, "기관순매도상위 (전체)"),
            # 외국인/기관 종합 (netslmt_tp 추가)
            (
                "ka10131",
                {"dt": "20260128", "mrkt_tp": "0", "netslmt_tp": "1"},
                "외국인/기관 순매수 상위",
            ),
            (
                "ka10131",
                {"dt": "20260128", "mrkt_tp": "0", "netslmt_tp": "2"},
                "외국인/기관 순매도 상위",
            ),
            # 프로그램매매 관련
            ("ka10027", {"mrkt_tp": "0"}, "프로그램 순매수 상위"),
        ]

        for tr_code, params, desc in test_cases:
            print(f"\n{'='*50}")
            print(f"[{tr_code}] {desc}")
            print(f"파라미터: {params}")

            try:
                response = await api.request(tr_code, params)
                print(f"응답코드: {response.return_code}")

                if response.return_code == 0 and response.body:
                    body = response.body
                    keys = list(body.keys())
                    print(f"응답 키: {keys[:15]}")

                    # 리스트 데이터 찾기
                    for key in keys:
                        if key not in ["return_code", "return_msg"] and isinstance(
                            body[key], list
                        ):
                            data = body[key]
                            print(f"  [{key}] 데이터 수: {len(data)}")
                            if len(data) > 0:
                                print(f"  첫 항목 키: {list(data[0].keys())[:10]}")
                                # 종목명과 순매수 정보 추출
                                first = data[0]
                                stk_nm = first.get("stk_nm", first.get("hname", "N/A"))
                                net_buy = first.get(
                                    "net_buy", first.get("netbuyqty", "N/A")
                                )
                                print(f"  샘플 - 종목: {stk_nm}, 순매수: {net_buy}")
                else:
                    print(f"메시지: {response.return_msg}")

            except Exception as e:
                print(f"오류: {e}")

            await asyncio.sleep(0.3)

    except Exception as e:
        print(f"오류: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if api:
            await api.close()
            print("\n연결 종료")


if __name__ == "__main__":
    asyncio.run(main())
