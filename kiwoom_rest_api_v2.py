# -*- coding: utf-8 -*-
"""
키움 REST API 테스트 v2
- kiwoom-rest-api 패키지 사용
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 키
API_KEY = "T0rWdlFRKTpNLbYMzQ-ofL_N0rV0-TLusbU1uGQ4ljE"
API_SECRET = "cS66TNBeJFrMAK9tX6PFPkFKukq_I36tb5ULFTWbxF4"


async def main():
    logger.info("=" * 60)
    logger.info("키움 REST API 테스트 v2")
    logger.info("=" * 60)

    try:
        from kiwoom_rest_api import KiwoomRestAPI

        # API 초기화
        api = KiwoomRestAPI(api_key=API_KEY, api_secret=API_SECRET)
        logger.info("API 초기화 완료")

        # 연결
        await api.connect()
        logger.info("연결 성공!")

        # 1. 삼성전자 현재가 조회 (ka10001)
        logger.info("\n=== 삼성전자 현재가 (ka10001) ===")
        try:
            response = await api.request("ka10001", {"stk_cd": "005930"})
            logger.info(f"응답 코드: {response.return_code}")
            logger.info(f"응답 메시지: {response.return_msg}")
            if response.body:
                logger.info(f"데이터: {response.body}")
        except Exception as e:
            logger.error(f"ka10001 오류: {e}")

        await asyncio.sleep(0.5)

        # 2. 삼성전자 일봉 차트 (ka10081)
        logger.info("\n=== 삼성전자 일봉 (ka10081) ===")
        try:
            response = await api.request("ka10081", {
                "stk_cd": "005930",
                "base_dt": "00000000",  # 현재일자
                "upd_stkpc_tp": "1"     # 수정주가 적용
            })
            logger.info(f"응답 코드: {response.return_code}")
            if response.body:
                data = response.body
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"데이터 수: {len(data)}")
                    logger.info(f"첫 번째: {data[0]}")
                else:
                    logger.info(f"데이터: {data}")
        except Exception as e:
            logger.error(f"ka10081 오류: {e}")

        await asyncio.sleep(0.5)

        # 3. 외국인 종목별 매매동향 (ka10008)
        logger.info("\n=== 외국인 매매동향 (ka10008) ===")
        try:
            response = await api.request("ka10008", {
                "stk_cd": "005930"
            })
            logger.info(f"응답 코드: {response.return_code}")
            if response.body:
                logger.info(f"데이터: {response.body}")
        except Exception as e:
            logger.error(f"ka10008 오류: {e}")

        await asyncio.sleep(0.5)

        # 4. 코스닥 종목 테스트 - 에코프로비엠 (247540)
        logger.info("\n=== 에코프로비엠 현재가 ===")
        try:
            response = await api.request("ka10001", {"stk_cd": "247540"})
            logger.info(f"응답 코드: {response.return_code}")
            if response.body:
                logger.info(f"데이터: {response.body}")
        except Exception as e:
            logger.error(f"에코프로비엠 오류: {e}")

        # 연결 종료
        await api.close()
        logger.info("\n연결 종료")

    except ImportError as e:
        logger.error(f"import 오류: {e}")
        logger.info("pip install kiwoom-rest-api 실행 필요")

    except Exception as e:
        logger.error(f"오류: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
