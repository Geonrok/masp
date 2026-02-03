# -*- coding: utf-8 -*-
"""
키움 REST API - 전체 데이터 다운로드 완료
"""

import asyncio
import os

import pandas as pd

APPKEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
SECRETKEY = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"

SAVE_PATH = "E:/투자/data/kiwoom_investor"
os.makedirs(SAVE_PATH, exist_ok=True)


async def download_daily_chart(api, ticker, name):
    """일봉 차트 데이터 (ka10081)"""
    filename = f"{SAVE_PATH}/{ticker}_daily.csv"

    # 이미 다운로드된 경우 스킵
    if os.path.exists(filename):
        print(f"  {name} ({ticker}): 이미 존재함, 스킵")
        return None

    print(f"  {name} ({ticker}) 일봉 다운로드...")

    all_data = []

    try:
        response = await api.request(
            "ka10081", {"stk_cd": ticker, "base_dt": "00000000", "upd_stkpc_tp": "1"}
        )

        if response.return_code == 0 and response.body:
            body = response.body
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

                # 연속조회
                cont_count = 0
                while response.cont_yn == "Y" and response.next_key and cont_count < 30:
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
                        else:
                            break
                    else:
                        break

    except Exception as e:
        print(f"    오류: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df["ticker"] = ticker
        df["name"] = name
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"    저장: {len(df)}건")
        return df

    print("    데이터 없음")
    return None


async def main():
    print("=" * 60)
    print("키움 REST API - 나머지 일봉 데이터 다운로드")
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

        # 외국인 데이터가 있는 종목 확인
        foreign_files = [
            f.replace("_foreign.csv", "")
            for f in os.listdir(SAVE_PATH)
            if f.endswith("_foreign.csv")
        ]
        daily_files = [
            f.replace("_daily.csv", "")
            for f in os.listdir(SAVE_PATH)
            if f.endswith("_daily.csv")
        ]

        # 일봉 데이터가 없는 종목
        missing_daily = set(foreign_files) - set(daily_files)
        print(f"외국인 데이터 있는 종목: {len(foreign_files)}개")
        print(f"일봉 데이터 있는 종목: {len(daily_files)}개")
        print(f"일봉 데이터 필요한 종목: {len(missing_daily)}개")

        # 종목명 매핑
        ticker_names = {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "035720": "카카오",
            "005380": "현대차",
            "051910": "LG화학",
            "006400": "삼성SDI",
            "207940": "삼성바이오로직스",
            "373220": "LG에너지솔루션",
            "247540": "에코프로비엠",
            "196170": "알테오젠",
            "263750": "펄어비스",
            "086520": "에코프로",
            "042700": "한미반도체",
            "028300": "HLB",
            "293490": "카카오게임즈",
            "091990": "셀트리온헬스케어",
            "068270": "셀트리온제약",
            "251270": "넷마블",
        }

        # 일봉 데이터 다운로드
        print("\n일봉 데이터 다운로드:")
        for ticker in sorted(missing_daily):
            name = ticker_names.get(ticker, ticker)
            await download_daily_chart(api, ticker, name)
            await asyncio.sleep(0.5)

        # 최종 결과
        print("\n" + "=" * 60)
        print("최종 데이터 현황")
        print("=" * 60)

        foreign_files = [f for f in os.listdir(SAVE_PATH) if f.endswith("_foreign.csv")]
        daily_files = [f for f in os.listdir(SAVE_PATH) if f.endswith("_daily.csv")]
        merged_files = [f for f in os.listdir(SAVE_PATH) if f.endswith("_merged.csv")]

        print(f"외국인 데이터: {len(foreign_files)}개 종목")
        print(f"일봉 데이터: {len(daily_files)}개 종목")
        print(f"병합 데이터: {len(merged_files)}개 종목")

        # 파일 목록
        print("\n저장된 파일:")
        all_files = os.listdir(SAVE_PATH)
        total_size = 0
        for f in sorted(all_files):
            filepath = os.path.join(SAVE_PATH, f)
            size = os.path.getsize(filepath)
            total_size += size
            print(f"  {f}: {size:,} bytes")
        print(f"\n총 용량: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

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
