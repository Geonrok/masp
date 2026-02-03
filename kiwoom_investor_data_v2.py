# -*- coding: utf-8 -*-
"""
키움 REST API - 투자자별 매매동향 데이터 다운로드
공식 kiwoomRest 패키지 사용
"""

import asyncio
import os

import pandas as pd

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"

# 저장 경로
SAVE_PATH = "E:/투자/data/kiwoom_investor"
os.makedirs(SAVE_PATH, exist_ok=True)


async def download_stock_foreign_data(api, ticker, name):
    """종목별 외국인 보유/매매 데이터 (ka10008)"""
    print(f"\n{name} ({ticker}) 외국인 데이터 다운로드...")

    all_data = []

    try:
        # ka10008: 외국인 보유 현황 (stk_frgnr 키에 리스트 형태로 반환)
        response = await api.request("ka10008", {"stk_cd": ticker})

        if response.return_code == 0 and response.body:
            body = response.body
            if "stk_frgnr" in body:
                data = body["stk_frgnr"]
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
                    print(f"  첫 조회: {len(data)}건")

                    # 연속조회
                    cont_count = 0
                    while (
                        response.cont_yn == "Y"
                        and response.next_key
                        and cont_count < 50
                    ):
                        await asyncio.sleep(0.3)
                        response = await api.request(
                            "ka10008",
                            {"stk_cd": ticker},
                            cont_yn="Y",
                            next_key=response.next_key,
                        )
                        if (
                            response.return_code == 0
                            and response.body
                            and "stk_frgnr" in response.body
                        ):
                            new_data = response.body["stk_frgnr"]
                            if isinstance(new_data, list) and len(new_data) > 0:
                                all_data.extend(new_data)
                                cont_count += 1
                                print(
                                    f"  연속조회 {cont_count}: {len(new_data)}건 (누적 {len(all_data)}건)"
                                )
                            else:
                                break
                        else:
                            break

    except Exception as e:
        print(f"  오류: {e}")
        import traceback

        traceback.print_exc()

    if all_data:
        df = pd.DataFrame(all_data)
        df["ticker"] = ticker
        df["name"] = name
        filename = f"{SAVE_PATH}/{ticker}_foreign.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"  저장: {filename} ({len(df)}건)")
        return df

    print("  데이터 없음")
    return None


async def download_market_investor_ranking(api, ranking_type="foreign_buy"):
    """시장 투자자 순매수/순매도 상위"""
    tr_codes = {
        "foreign_buy": ("ka10029", "외국인순매수상위"),
        "inst_buy": ("ka10030", "기관순매수상위"),
        "foreign_sell": ("ka10031", "외국인순매도상위"),
        "inst_sell": ("ka10032", "기관순매도상위"),
    }

    if ranking_type not in tr_codes:
        return None

    tr_code, desc = tr_codes[ranking_type]
    print(f"\n{desc} 데이터 다운로드 ({tr_code})...")

    all_data = []
    # mrkt_tp 파라미터 사용 (mkt_tp가 아님)
    for mrkt_tp, mkt_name in [("0", "전체"), ("1", "코스피"), ("2", "코스닥")]:
        try:
            response = await api.request(tr_code, {"mrkt_tp": mrkt_tp})
            if response.return_code == 0 and response.body:
                body = response.body
                # 응답 키 확인
                data = None
                if "output" in body:
                    data = body["output"]
                else:
                    # 다른 키에서 데이터 찾기
                    for key in body.keys():
                        if key not in ["return_code", "return_msg"] and isinstance(
                            body[key], list
                        ):
                            data = body[key]
                            break

                if isinstance(data, list) and len(data) > 0:
                    for item in data:
                        item["market"] = mkt_name
                    all_data.extend(data)
                    print(f"  {mkt_name}: {len(data)}건")
                else:
                    print(f"  {mkt_name}: 데이터 없음 (키: {list(body.keys())})")
            else:
                print(f"  {mkt_name}: {response.return_msg}")
        except Exception as e:
            print(f"  {mkt_name} 오류: {e}")
        await asyncio.sleep(0.3)

    if all_data:
        df = pd.DataFrame(all_data)
        filename = f"{SAVE_PATH}/{ranking_type}_ranking.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"  저장: {filename} ({len(df)}건)")
        return df

    return None


async def download_daily_chart(api, ticker, name):
    """일봉 차트 데이터 (ka10081)"""
    print(f"\n{name} ({ticker}) 일봉 데이터 다운로드...")

    all_data = []

    try:
        response = await api.request(
            "ka10081",
            {"stk_cd": ticker, "base_dt": "00000000", "upd_stkpc_tp": "1"},  # 수정주가
        )

        if response.return_code == 0 and response.body:
            body = response.body
            # output 키 또는 다른 리스트 키에서 데이터 찾기
            data = None
            if "output" in body:
                data = body["output"]
            else:
                for key in body.keys():
                    if key not in ["return_code", "return_msg"] and isinstance(
                        body[key], list
                    ):
                        data = body[key]
                        break

            if isinstance(data, list) and len(data) > 0:
                all_data.extend(data)
                print(f"  첫 조회: {len(data)}건")

                # 연속조회
                cont_count = 0
                while response.cont_yn == "Y" and response.next_key and cont_count < 20:
                    await asyncio.sleep(0.3)
                    response = await api.request(
                        "ka10081",
                        {"stk_cd": ticker, "base_dt": "00000000", "upd_stkpc_tp": "1"},
                        cont_yn="Y",
                        next_key=response.next_key,
                    )
                    if response.return_code == 0 and response.body:
                        new_body = response.body
                        new_data = new_body.get("output")
                        if not new_data:
                            for key in new_body.keys():
                                if key not in [
                                    "return_code",
                                    "return_msg",
                                ] and isinstance(new_body[key], list):
                                    new_data = new_body[key]
                                    break
                        if isinstance(new_data, list) and len(new_data) > 0:
                            all_data.extend(new_data)
                            cont_count += 1
                            print(
                                f"  연속조회 {cont_count}: {len(new_data)}건 (누적 {len(all_data)}건)"
                            )
                        else:
                            break
                    else:
                        break

    except Exception as e:
        print(f"  오류: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df["ticker"] = ticker
        df["name"] = name
        filename = f"{SAVE_PATH}/{ticker}_daily.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"  저장: {filename} ({len(df)}건)")
        return df

    print("  데이터 없음")
    return None


async def main():
    print("=" * 60)
    print("키움 REST API - 투자자 데이터 다운로드")
    print("=" * 60)

    api = None
    try:
        from kiwoomRest import KwRestApi

        api = KwRestApi()
        print("API 인스턴스 생성 완료")

        # 로그인
        print("\n로그인 중...")
        ret = await api.login(APPKEY, SECRETKEY, is_simulation=False)
        if not ret:
            print(f"로그인 실패: {api.last_message}")
            return

        print("로그인 성공!")

        # 코스닥 주요 종목
        kosdaq_tickers = [
            ("247540", "에코프로비엠"),
            ("196170", "알테오젠"),
            ("263750", "펄어비스"),
            ("086520", "에코프로"),
            ("042700", "한미반도체"),
            ("028300", "HLB"),
            ("293490", "카카오게임즈"),
            ("091990", "셀트리온헬스케어"),
            ("068270", "셀트리온제약"),
            ("251270", "넷마블"),
        ]

        # 코스피 주요 종목
        kospi_tickers = [
            ("005930", "삼성전자"),
            ("000660", "SK하이닉스"),
            ("035720", "카카오"),
            ("005380", "현대차"),
            ("051910", "LG화학"),
            ("006400", "삼성SDI"),
            ("207940", "삼성바이오로직스"),
            ("373220", "LG에너지솔루션"),
        ]

        all_tickers = kosdaq_tickers + kospi_tickers

        # 1. 외국인 보유/매매 데이터 다운로드
        print("\n" + "=" * 60)
        print("외국인 보유/매매 데이터 다운로드 (ka10008)")
        print("=" * 60)

        foreign_data = {}
        for ticker, name in all_tickers:
            df = await download_stock_foreign_data(api, ticker, name)
            if df is not None:
                foreign_data[ticker] = df
            await asyncio.sleep(0.5)

        # 2. 투자자 순매수 상위 다운로드
        print("\n" + "=" * 60)
        print("투자자 순위 데이터 다운로드")
        print("=" * 60)

        for ranking_type in ["foreign_buy", "inst_buy", "foreign_sell", "inst_sell"]:
            await download_market_investor_ranking(api, ranking_type)
            await asyncio.sleep(0.5)

        # 3. 일봉 데이터 다운로드 (일부 종목만)
        print("\n" + "=" * 60)
        print("일봉 데이터 다운로드 (ka10081)")
        print("=" * 60)

        daily_data = {}
        for ticker, name in all_tickers[:5]:  # 처음 5개만
            df = await download_daily_chart(api, ticker, name)
            if df is not None:
                daily_data[ticker] = df
            await asyncio.sleep(0.5)

        # 결과 요약
        print("\n" + "=" * 60)
        print("다운로드 결과 요약")
        print("=" * 60)
        print(f"외국인 보유/매매 데이터: {len(foreign_data)}개 종목")
        print(f"일봉 데이터: {len(daily_data)}개 종목")
        print(f"저장 위치: {SAVE_PATH}")

        # 저장된 파일 목록
        if os.path.exists(SAVE_PATH):
            saved_files = os.listdir(SAVE_PATH)
            print(f"\n저장된 파일 ({len(saved_files)}개):")
            for f in sorted(saved_files):
                filepath = os.path.join(SAVE_PATH, f)
                size = os.path.getsize(filepath)
                print(f"  - {f} ({size:,} bytes)")

    except ImportError as e:
        print(f"Import 오류: {e}")
        print("pip install kiwoomRest 실행 필요")

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
