# -*- coding: utf-8 -*-
"""
키움 REST API - 기관/외국인 순위 데이터 (모든 필수 파라미터 포함)
"""

import asyncio
import pandas as pd
import os

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"

SAVE_PATH = "E:/투자/data/kiwoom_investor"


async def main():
    print("=" * 60)
    print("기관/외국인 순위 데이터 다운로드")
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

        # 테스트할 TR 코드와 파라미터 조합 (모든 필수 파라미터 포함)
        test_cases = [
            # 외국인 순매수 상위
            ("ka10029", {
                "mrkt_tp": "0",      # 시장구분 (0:전체, 1:코스피, 2:코스닥)
                "sort_tp": "1",      # 정렬구분 (1:순매수금액, 2:순매수수량)
                "trde_qty_cnd": "0", # 거래량조건 (0:전체, 1:1만이상, 2:5만이상...)
            }, "외국인순매수상위"),

            # 기관 순매수 상위
            ("ka10030", {
                "mrkt_tp": "0",
                "sort_tp": "1",
                "mang_stk_incls": "0",  # 관리종목포함 (0:미포함, 1:포함)
            }, "기관순매수상위"),

            # 외국인 순매도 상위
            ("ka10031", {
                "mrkt_tp": "0",
                "qry_tp": "1",       # 조회구분
                "sort_tp": "1",
            }, "외국인순매도상위"),

            # 기관 순매도 상위
            ("ka10032", {
                "mrkt_tp": "0",
                "sort_tp": "1",
                "mang_stk_incls": "0",
            }, "기관순매도상위"),

            # 외국인/기관 종합 순위
            ("ka10131", {
                "dt": "20260128",
                "mrkt_tp": "0",
                "netslmt_tp": "1",   # 순매수/순매도 (1:순매수, 2:순매도)
                "stk_inds_tp": "0",  # 주식/업종 (0:주식, 1:업종)
            }, "외국인/기관 순매수상위"),

            # 프로그램 순매수 상위
            ("ka10027", {
                "mrkt_tp": "0",
                "sort_tp": "1",
            }, "프로그램 순매수상위"),
        ]

        results = {}
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
                        if key not in ['return_code', 'return_msg'] and isinstance(body[key], list):
                            data = body[key]
                            print(f"  [{key}] 데이터 수: {len(data)}")
                            if len(data) > 0:
                                print(f"  첫 항목 키: {list(data[0].keys())}")
                                # 데이터 저장
                                df = pd.DataFrame(data)
                                filename = f"{SAVE_PATH}/{tr_code}_{desc}.csv"
                                df.to_csv(filename, index=False, encoding='utf-8-sig')
                                print(f"  저장: {filename}")
                                results[tr_code] = df
                else:
                    print(f"메시지: {response.return_msg}")

            except Exception as e:
                print(f"오류: {e}")

            await asyncio.sleep(0.3)

        # 결과 요약
        print("\n" + "=" * 60)
        print("다운로드 결과")
        print("=" * 60)
        print(f"성공: {len(results)}개 TR")
        for tr, df in results.items():
            print(f"  - {tr}: {len(df)}건")

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
