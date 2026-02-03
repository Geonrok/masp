"""
일일 옵션 IV 및 투자자 데이터 수집기
===================================
매일 실행하여 옵션 IV, Put/Call Ratio, 투자자 수급 데이터를 누적 수집
Windows 작업 스케줄러에 등록하여 사용

사용법:
1. Windows 작업 스케줄러에서 새 작업 생성
2. 트리거: 매일 오후 4시 (장 마감 후)
3. 동작: python daily_option_iv_collector.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from ebest import OpenApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "E:/투자/data/kosdaq_futures/investor_data/collector.log", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"

SAVE_PATH = Path("E:/투자/data/kosdaq_futures/investor_data/daily")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


async def collect_option_iv(api):
    """옵션 IV 데이터 수집"""
    today = datetime.now().strftime("%Y%m%d")
    logger.info(f"옵션 IV 데이터 수집 ({today})")

    try:
        yyyymm = datetime.now().strftime("%Y%m")

        response = await api.request(
            "t2301",
            {
                "yyyymm": yyyymm,
                "cp": "1",
                "shcode": "",
            },
        )

        if response and hasattr(response, "body"):
            body = response.body

            result = {
                "date": today,
                "timestamp": datetime.now().isoformat(),
            }

            if "t2301OutBlock" in body:
                summary = body["t2301OutBlock"]
                result["call_iv"] = float(summary.get("cimpv", 0))
                result["put_iv"] = float(summary.get("pimpv", 0))
                result["hist_iv"] = float(summary.get("histimpv", 0))
                result["days_to_exp"] = int(summary.get("jandatecnt", 0))

            if "t2301OutBlock1" in body:
                call_data = body["t2301OutBlock1"]
                if isinstance(call_data, list):
                    result["call_volume"] = sum(
                        int(c.get("volume", 0)) for c in call_data
                    )

            if "t2301OutBlock2" in body:
                put_data = body["t2301OutBlock2"]
                if isinstance(put_data, list):
                    result["put_volume"] = sum(
                        int(p.get("volume", 0)) for p in put_data
                    )

            if result.get("call_volume", 0) > 0:
                result["put_call_ratio"] = result["put_volume"] / result["call_volume"]
            else:
                result["put_call_ratio"] = 0

            logger.info(
                f"  콜IV: {result.get('call_iv', 0):.2f}, 풋IV: {result.get('put_iv', 0):.2f}"
            )
            logger.info(f"  Put/Call Ratio: {result.get('put_call_ratio', 0):.3f}")

            return result

    except Exception as e:
        logger.error(f"옵션 IV 수집 오류: {e}")

    return None


async def collect_kosdaq_index(api):
    """코스닥150 지수 현재가"""
    today = datetime.now().strftime("%Y%m%d")
    logger.info(f"코스닥150 지수 수집 ({today})")

    try:
        # t1101 - 주식 현재가
        response = await api.request(
            "t1102",
            {
                "shcode": "Q500",  # 코스닥150 지수
            },
        )

        if response and hasattr(response, "body"):
            body = response.body
            if "t1102OutBlock" in body:
                data = body["t1102OutBlock"]
                return {
                    "date": today,
                    "price": float(data.get("price", 0)),
                    "change": float(data.get("change", 0)),
                    "volume": int(data.get("volume", 0)),
                }

    except Exception as e:
        logger.error(f"지수 수집 오류: {e}")

    return None


async def main():
    api = OpenApi()

    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error("로그인 실패")
        return

    logger.info("=" * 50)
    logger.info(f"일일 데이터 수집 시작: {datetime.now()}")
    logger.info("=" * 50)

    today = datetime.now().strftime("%Y%m%d")

    # 1. 옵션 IV 수집
    option_data = await collect_option_iv(api)
    await asyncio.sleep(0.5)

    # 2. 코스닥150 지수 수집
    index_data = await collect_kosdaq_index(api)

    # 결과 저장
    daily_data = {
        "date": today,
        "option": option_data,
        "index": index_data,
    }

    # 개별 파일 저장
    output_file = SAVE_PATH / f"daily_{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(daily_data, f, indent=2, ensure_ascii=False)

    # 누적 파일에 추가
    history_file = SAVE_PATH / "option_iv_history.csv"

    if option_data:
        new_row = pd.DataFrame([option_data])

        if history_file.exists():
            existing = pd.read_csv(history_file)
            # 중복 제거
            existing = existing[existing["date"] != today]
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        combined.to_csv(history_file, index=False, encoding="utf-8-sig")
        logger.info(f"히스토리 저장: {history_file} ({len(combined)}일)")

    logger.info("=" * 50)
    logger.info("수집 완료!")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
