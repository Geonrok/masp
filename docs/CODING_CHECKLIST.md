# MASP 코딩 체크리스트

## 모든 작업 공통 (필수)
- [ ] 모든 함수에 타입 힌트 (`def func(x: int) -> str:`)
- [ ] None/빈값 처리 (early return 또는 기본값)
- [ ] 테스트 파일 생성 (새 함수당 최소 1개 테스트)
- [ ] import 정리 (사용하지 않는 import 제거)

## 수학/금융 연산
- [ ] 0으로 나누기 방어 (`if denominator == 0: return default`)
- [ ] NaN/Inf 방어 (`math.isfinite(result)`)
- [ ] 소수점 정밀도 확인

## Streamlit UI
- [ ] 빈 데이터 → `st.info()` 표시
- [ ] session_state key 충돌 방지 (prefix 사용: `th_`, `bt_` 등)
- [ ] 필터 변경 시 상태 리셋 로직
- [ ] 목록 정렬 (`sorted()` 적용)
- [ ] subheader/label 선행 공백 없음

## Plotly 차트
- [ ] `template="plotly_dark"` 설정
- [ ] 적절한 height/margin 설정
- [ ] 색상: 수익=#00C853, 손실=#FF5252

## 커밋 전 확인
- [ ] pytest 전체 통과
- [ ] 새 테스트 개수 확인
- [ ] 불필요한 print/debug 코드 제거