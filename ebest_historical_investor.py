"""
이베스트 - 히스토리컬 투자자 데이터 조회 시도
t1662, t1663 등 과거 데이터 TR 테스트
"""

import asyncio
from ebest import OpenApi
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"


async def test_tr(api, tr_code, params, description):
    """TR 코드 테스트"""
    logger.info(f"\n{'='*50}")
    logger.info(f"{description} ({tr_code})")
    logger.info(f"{'='*50}")
    logger.info(f"파라미터: {params}")

    try:
        response = await api.request(tr_code, params)

        if response and hasattr(response, 'body'):
            body = response.body
            logger.info(f"응답 키: {list(body.keys())}")

            result_data = []
            for key in body:
                if 'OutBlock' in key:
                    data = body[key]
                    if isinstance(data, list):
                        logger.info(f"{key}: {len(data)}개 데이터")
                        if len(data) > 0:
                            logger.info(f"  컬럼: {list(data[0].keys())}")
                            # 첫 번째 데이터에 유효한 값이 있는지 확인
                            first = data[0]
                            has_value = any(v for k, v in first.items()
                                           if v and v != '' and v != 0 and v != '0'
                                           and k not in ['rsp_cd', 'rsp_msg'])
                            if has_value:
                                logger.info(f"  첫 번째: {data[0]}")
                                if len(data) > 1:
                                    logger.info(f"  두 번째: {data[1]}")
                                result_data = data
                    elif isinstance(data, dict):
                        has_value = any(v for k, v in data.items()
                                       if v and v != '' and v != 0 and v != '0'
                                       and k not in ['rsp_cd', 'rsp_msg'])
                        if has_value:
                            logger.info(f"{key}: {data}")
                            result_data = [data]

            if 'rsp_msg' in body:
                logger.info(f"응답 메시지: {body.get('rsp_msg')}")

            return result_data

    except Exception as e:
        logger.error(f"오류: {e}")
        import traceback
        traceback.print_exc()

    return []


async def main():
    api = OpenApi()

    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error("로그인 실패")
        return

    logger.info("로그인 성공!")

    # 히스토리컬 투자자 데이터 TR들
    tests = [
        # 1. t1662 - 투자자별매매추이 (기간별)
        ("t1662", {
            "gubun": "0",  # 0:일
            "market": "1",  # 1:코스닥
            "sdate": "20260101",
            "edate": "20260128",
        }, "투자자별 매매추이 기간별"),

        # 2. t1663 - 투자자별매매추이 (종합)
        ("t1663", {
            "gubun": "1",  # 1:일
            "market": "1",  # 1:코스닥
            "upcode": "",
        }, "투자자별 매매추이 종합"),

        # 3. t1665 - 투자자매매종합 (기간)
        ("t1665", {
            "market": "1",  # 1:코스닥
            "upcode": "",
            "gubun": "1",  # 1:일별
            "sdate": "20260101",
            "edate": "20260128",
        }, "투자자매매종합 기간"),

        # 4. t1664 - 투자자매매종합 (일자)
        ("t1664", {
            "mgubun": "1",  # 1:코스닥
            "date": "",
        }, "투자자매매종합 일자"),

        # 5. t1771 - 외국인투자자 시장별 매매동향
        ("t1771", {
            "date": "",
            "gubun": "1",  # 1:코스닥
        }, "외국인 시장별 매매동향"),

        # 6. t1764 - 외국인 순매수 추이 (종목별)
        ("t1764", {
            "shcode": "247540",  # 에코프로비엠
            "gubun1": "0",  # 0:일
            "gubun2": "0",  # 0:순매수량
            "gubun3": "1",  # 1:외국인
            "fromdt": "20260101",
            "todt": "20260128",
        }, "외국인 순매수 추이 (에코프로비엠)"),

        # 7. t1702 - 외국인기관별종목별동향
        ("t1702", {
            "shcode": "247540",
            "frdt": "20260101",
            "todt": "20260128",
            "gubun": "0",  # 0:일간
            "type": "0",   # 0:금액
        }, "외국인기관별 종목동향 (에코프로비엠)"),

        # 8. 삼성전자로 테스트 (코스피)
        ("t1702", {
            "shcode": "005930",  # 삼성전자
            "frdt": "20260101",
            "todt": "20260128",
            "gubun": "0",
            "type": "0",
        }, "외국인기관별 종목동향 (삼성전자)"),

        # 9. t8413 - 일봉 차트
        ("t8413", {
            "shcode": "247540",
            "gubun": "2",  # 2:일
            "qrycnt": 100,
            "sdate": "20260101",
            "edate": "20260128",
            "comp_yn": "N",
        }, "일봉 차트 (에코프로비엠)"),

        # 10. t1305 - 기간별 시세
        ("t1305", {
            "shcode": "247540",
            "dwmcode": "1",  # 1:일
            "cnt": 100,
        }, "기간별 시세 (에코프로비엠)"),

        # 11. t4201 - 전체 투자자 매매동향
        ("t4201", {
            "shcode": "247540",
        }, "전체 투자자 매매동향"),

        # 12. t1752 - 종목별 외국인기관 매매추이
        ("t1752", {
            "shcode": "247540",
            "gubun": "0",  # 0:일
            "cts_date": "",
            "fromdt": "20260101",
            "todt": "20260128",
        }, "종목별 외국인기관 매매추이"),
    ]

    results = {}
    for tr_code, params, desc in tests:
        data = await test_tr(api, tr_code, params, desc)
        if data:
            results[f"{tr_code}_{desc}"] = data
        await asyncio.sleep(0.3)

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("결과 요약")
    logger.info("=" * 60)

    if results:
        for key, data in results.items():
            logger.info(f"✓ {key}: {len(data)}개 데이터")
    else:
        logger.info("유효한 데이터를 반환하는 TR이 없습니다.")

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
