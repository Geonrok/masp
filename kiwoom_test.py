# -*- coding: utf-8 -*-
"""
키움 API 연결 테스트
- 32-bit Python 필수
- 키움 OpenAPI+ 설치 및 로그인 필요
"""

import sys

print(f"Python: {sys.version}")
print(f"Architecture: {'64-bit' if sys.maxsize > 2**32 else '32-bit'}")

if sys.maxsize > 2**32:
    print("\n[ERROR] 64-bit Python입니다. 키움 API는 32-bit Python이 필요합니다.")
    print("32-bit Python 경로: C:\\Python311-32\\python.exe")
    sys.exit(1)

print("\n32-bit Python 확인됨. 키움 API 테스트 시작...")

try:
    from pykiwoom.kiwoom import Kiwoom

    print("[OK] pykiwoom import 성공")
except ImportError as e:
    print(f"[FAIL] pykiwoom import 실패: {e}")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import QApplication

    print("[OK] PyQt5 import 성공")
except ImportError as e:
    print(f"[FAIL] PyQt5 import 실패: {e}")
    sys.exit(1)

# QApplication 생성 (필수)
app = QApplication(sys.argv)

print("\n키움 API 초기화 중...")
try:
    kiwoom = Kiwoom()
    print("[OK] Kiwoom 객체 생성 성공")
except Exception as e:
    print(f"[FAIL] Kiwoom 객체 생성 실패: {e}")
    print("\n가능한 원인:")
    print("1. 키움 OpenAPI+ 모듈이 설치되지 않음")
    print("2. 키움증권 로그인이 필요함")
    sys.exit(1)

# 로그인 시도
print("\n로그인 시도 중... (키움 로그인 창이 열립니다)")
try:
    kiwoom.CommConnect(block=True)

    # 로그인 상태 확인
    state = kiwoom.GetConnectState()
    if state == 1:
        print("[OK] 로그인 성공!")

        # 계좌 정보
        accounts = kiwoom.GetLoginInfo("ACCNO")
        print(f"계좌번호: {accounts}")

        user_id = kiwoom.GetLoginInfo("USER_ID")
        print(f"사용자 ID: {user_id}")

        server_type = kiwoom.GetLoginInfo("GetServerGubun")
        print(f"서버: {'모의투자' if server_type == '1' else '실거래'}")
    else:
        print(f"[FAIL] 로그인 실패 (상태: {state})")

except Exception as e:
    print(f"[FAIL] 로그인 오류: {e}")

print("\n테스트 완료!")
