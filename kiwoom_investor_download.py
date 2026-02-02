# -*- coding: utf-8 -*-
"""
키움 REST API로 투자자 데이터 다운로드
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta

# 환경 변수 설정 (import 전에 설정해야 함)
os.environ["KIWOOM_API_KEY"] = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
os.environ["KIWOOM_API_SECRET"] = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 저장 경로
SAVE_PATH = "E:/투자/data/kiwoom_investor_data"
os.makedirs(SAVE_PATH, exist_ok=True)


def test_connection():
    """연결 테스트"""
    logger.info("=" * 60)
    logger.info("키움 REST API 투자자 데이터 다운로드")
    logger.info("=" * 60)

    try:
        from kiwoom_rest_api.auth.token import get_access_token
        from kiwoom_rest_api.koreanstock.investor import (
            get_investor_trend,
            get_market_investor_trend
        )
        # 토큰 발급
        logger.info("토큰 발급 중...")
        token = get_access_token()
        logger.info(f"토큰 발급 성공: {token[:30]}...")

        return token, get_investor_trend, get_market_investor_trend

    except Exception as e:
        logger.error(f"초기화 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def download_investor_data(token, get_investor_trend, ticker, name, start_date, end_date):
    """종목별 투자자 데이터 다운로드"""
    try:
        logger.info(f"  {name} ({ticker}) 다운로드 중...")

        response = get_investor_trend(
            stock_code=ticker,
            period="D",
            start_date=start_date,
            end_date=end_date,
            access_token=token
        )

        if response and 'output1' in response:
            data = response['output1']
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(f"{SAVE_PATH}/{ticker}_investor.csv", index=False, encoding='utf-8-sig')
                logger.info(f"    저장 완료: {len(df)}일 데이터")
                return df

        elif response and 'body' in response:
            data = response['body']
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(f"{SAVE_PATH}/{ticker}_investor.csv", index=False, encoding='utf-8-sig')
                logger.info(f"    저장 완료: {len(df)}일 데이터")
                return df

        logger.info(f"    데이터 없음 (응답: {list(response.keys()) if response else 'None'})")

    except Exception as e:
        logger.error(f"    오류: {e}")

    return None


def download_market_investor_data(token, get_market_investor_trend, market, start_date, end_date):
    """시장 전체 투자자 데이터 다운로드"""
    try:
        market_name = "코스피" if market == "1" else "코스닥" if market == "2" else "전체"
        logger.info(f"  {market_name} 시장 투자자 데이터...")

        response = get_market_investor_trend(
            market_code=market,
            period="D",
            start_date=start_date,
            end_date=end_date,
            access_token=token
        )

        if response:
            logger.info(f"    응답 키: {list(response.keys())}")

            # output1 또는 body에서 데이터 추출
            data = response.get('output1') or response.get('body')
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(f"{SAVE_PATH}/market_{market}_investor.csv", index=False, encoding='utf-8-sig')
                logger.info(f"    저장 완료: {len(df)}일 데이터")
                return df

        logger.info("    데이터 없음")

    except Exception as e:
        logger.error(f"    오류: {e}")

    return None


def main():
    # 연결 테스트
    result = test_connection()

    if not result or not result[0]:
        logger.error("연결 실패. 종료.")
        return

    token, get_investor_trend, get_market_investor_trend = result

    if not token:
        logger.error("연결 실패. 종료.")
        return

    # 기간 설정
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    logger.info(f"기간: {start_date} ~ {end_date}")

    # 1. 시장 전체 투자자 데이터
    logger.info("\n=== 시장별 투자자 데이터 ===")
    for market in ["1", "2"]:  # 1:코스피, 2:코스닥
        download_market_investor_data(token, get_market_investor_trend, market, start_date, end_date)

    # 2. 코스닥 주요 종목
    logger.info("\n=== 코스닥 종목별 투자자 데이터 ===")
    kosdaq_tickers = [
        ('247540', '에코프로비엠'),
        ('196170', '알테오젠'),
        ('263750', '펄어비스'),
        ('091990', '셀트리온헬스케어'),
        ('293490', '카카오게임즈'),
        ('068270', '셀트리온제약'),
        ('028300', 'HLB'),
        ('042700', '한미반도체'),
        ('086520', '에코프로'),
        ('251270', '넷마블'),
    ]

    all_data = {}
    for ticker, name in kosdaq_tickers:
        df = download_investor_data(token, get_investor_trend, ticker, name, start_date, end_date)
        if df is not None:
            all_data[ticker] = df

        import time
        time.sleep(0.5)  # API 속도 제한

    # 3. 코스피 주요 종목
    logger.info("\n=== 코스피 종목별 투자자 데이터 ===")
    kospi_tickers = [
        ('005930', '삼성전자'),
        ('000660', 'SK하이닉스'),
        ('035720', '카카오'),
        ('005380', '현대차'),
        ('051910', 'LG화학'),
    ]

    for ticker, name in kospi_tickers:
        df = download_investor_data(token, get_investor_trend, ticker, name, start_date, end_date)
        if df is not None:
            all_data[ticker] = df

        import time
        time.sleep(0.5)

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("다운로드 결과")
    logger.info("=" * 60)
    logger.info(f"성공: {len(all_data)}개 종목")
    logger.info(f"저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    main()
