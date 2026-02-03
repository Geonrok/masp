"""
Bithumb Connection Debugger
- MASP 프레임워크를 거치지 않고 pybithumb 직접 테스트
- 문제가 코드인지 API 키/네트워크인지 분리
"""

import os
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import pybithumb


def test_connection():
    print("=" * 60)
    print("=== Bithumb Connection Debugger ===")
    print("=" * 60)

    # 1. .env 로드
    load_dotenv()
    api_key = os.getenv("BITHUMB_API_KEY")
    sec_key = os.getenv("BITHUMB_SECRET_KEY")

    print("\n[1] 환경변수 확인")
    if not api_key:
        print("❌ BITHUMB_API_KEY 미설정")
        return
    if not sec_key:
        print("❌ BITHUMB_SECRET_KEY 미설정")
        return

    print(f"✅ API Key 로드됨 (길이: {len(api_key)})")
    print(f"✅ Secret Key 로드됨 (길이: {len(sec_key)})")

    # 공백/따옴표 체크
    if api_key != api_key.strip():
        print("⚠️ API Key에 공백 포함")
    if sec_key != sec_key.strip():
        print("⚠️ Secret Key에 공백 포함")
    if '"' in api_key or "'" in api_key:
        print("⚠️ API Key에 따옴표 포함")
    if '"' in sec_key or "'" in sec_key:
        print("⚠️ Secret Key에 따옴표 포함")

    # 2. IP 확인
    print("\n[2] 현재 IP 확인")
    try:
        import requests

        my_ip = requests.get("https://api.ipify.org", timeout=5).text
        print(f"📡 현재 공인 IP: {my_ip}")
        print("   → Bithumb API 설정에서 이 IP가 허용되어 있는지 확인하세요")
    except Exception as e:
        print(f"❌ IP 확인 실패: {e}")

    # 3. pybithumb 버전 확인
    print("\n[3] pybithumb 버전")
    try:
        version = getattr(pybithumb, "__version__", "unknown")
        print(f"📦 pybithumb 버전: {version}")
        print("   ⚠️ 주의: pybithumb 1.0.21은 2021년 버전")
        print("   ⚠️ Bithumb API 2.0 (2024년 JWT 방식)과 호환되지 않을 수 있음")
    except Exception as e:
        print(f"❌ 버전 확인 실패: {e}")

    # 4. pybithumb 연결 테스트
    print("\n[4] pybithumb 연결 테스트")
    try:
        bithumb = pybithumb.Bithumb(api_key, sec_key)
        print("✅ Bithumb 객체 생성 성공")

        # 4a. 공개 API 테스트 (인증 불필요)
        print("\n[4a] 공개 API 테스트 (현재가 조회)")
        try:
            price = pybithumb.get_current_price("BTC")
            print(f"✅ BTC 현재가: {price:,.0f} KRW")
        except Exception as e:
            print(f"❌ 현재가 조회 실패: {e}")

        # 4b. 비공개 API 테스트 (인증 필요 - 잔고 조회)
        print("\n[4b] 비공개 API 테스트 (잔고 조회)")
        try:
            balance = bithumb.get_balance("BTC")
            if balance is None:
                print("❌ 잔고 조회 실패: None 반환")
                print("   → API 키 또는 IP 설정 문제 가능성")
            else:
                print(f"✅ BTC 잔고: {balance}")
                print("   → API 키 인증 성공!")
        except Exception as e:
            print(f"❌ 잔고 조회 실패: {e}")
            print("   → Invalid Apikey: API 키 타입/권한/IP 확인 필요")

        # 4c. 주문 테스트 (소액)
        print("\n[4c] 주문 테스트 (0.00001 BTC ≈ 1,400원)")
        print("   ⚠️ 이 테스트는 실제 주문을 시도합니다")
        confirm = input("   진행하시겠습니까? (y/n): ").strip().lower()

        if confirm == "y":
            try:
                result = bithumb.buy_market_order("BTC", 0.00001)
                if result is None:
                    print("❌ 주문 실패: None 반환")
                else:
                    print(f"✅ 주문 결과: {result}")
            except Exception as e:
                print(f"❌ 주문 실패: {e}")
        else:
            print("   주문 테스트 스킵")

    except Exception as e:
        print(f"❌ Bithumb 객체 생성 실패: {e}")

    # 5. 결론
    print("\n" + "=" * 60)
    print("=== 진단 결론 ===")
    print("=" * 60)
    print("""
잔고 조회 성공 + 주문 실패 → Write 권한 없음
잔고 조회 실패 → API 키 인증 실패
  → 가능한 원인:
     1. API 키 타입 불일치 (v1.2.0용 vs API 2.0/JWT용)
     2. 권한 부족 (Read only)
     3. IP 불일치
     4. 키 미활성화/만료

권장 조치:
  1. Bithumb API 관리에서 키 타입 확인 (v1.2.0인지 2.0인지)
  2. v1.2.0 키를 새로 발급받아 재시도
  3. IP 허용 목록 확인
  4. 권한에 Write(주문) 포함 확인
""")


if __name__ == "__main__":
    test_connection()
