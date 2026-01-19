# Automated Code Review Workflow

MASP 프로젝트의 자동화된 코드 리뷰 워크플로우 가이드.

## Overview

코드 작성 후 커밋 전 `codex review --uncommitted` 명령으로 자동 코드 리뷰를 수행합니다.

## Workflow

```
Code → __init__.py Export → pytest → codex review → Fix Issues → Commit
```

### Step-by-Step

1. **코드 작성**
   - 새 컴포넌트/기능 구현
   - 기존 코드 수정

2. **Export 추가** (해당 시)
   ```python
   # services/dashboard/components/__init__.py
   from services.dashboard.components.new_component import render_new_component
   ```

3. **테스트 실행**
   ```bash
   python -m pytest tests/dashboard/test_new_component.py -v --tb=short
   ```

4. **Codex Review**
   ```bash
   cd "E:\투자\Multi-Asset Strategy Platform"
   codex review --uncommitted
   ```

5. **이슈 수정** (발견 시)
   - P1 (Critical): 즉시 수정 필수
   - P2 (Important): 수정 권장
   - P3 (Minor): 선택적 수정

6. **커밋**
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

## Review Application Policy

| 분류 | 적용 기준 |
|------|----------|
| **필수보강** (P1) | 항상 적용 |
| **권장보강** (P2) | 항상 적용 |
| **선택보강** (P3) | 명명/스타일만 스킵 |

**이유**: 완성도 우선. 나중에 다시 손대지 않기 위해.

## Priority Levels

### P1 - Critical (필수보강)
- 보안 취약점
- 데이터 손실 위험
- 런타임 에러 가능성
- 심각한 로직 오류

**예시:**
- SQL Injection 취약점
- 인증/인가 누락
- Null pointer dereference
- 무한 루프 가능성

### P2 - Important (권장보강)
- 잠재적 버그
- 성능 이슈
- 엣지 케이스 미처리
- API 설계 문제

**예시:**
- Timezone 처리 오류
- 빈 리스트/None 처리 누락
- 비효율적 알고리즘
- 부적절한 에러 핸들링

### P3 - Minor (선택보강)
- 코드 스타일
- 네이밍 컨벤션
- 문서화 부족
- 리팩토링 제안

**예시:**
- 변수명 개선 제안
- 주석 추가 권장
- 함수 분리 제안
- 타입 힌트 추가

## Common Patterns

### Datetime Timezone Handling

```python
# Bad: TypeError when comparing mixed tz-aware/naive
if dt1 < dt2:  # Raises if mixed

# Good: Safe comparison with fallback
def _safe_datetime_compare(dt1, dt2):
    try:
        if dt1 < dt2: return -1
        elif dt1 > dt2: return 1
        return 0
    except TypeError:
        # Fallback: strip tzinfo
        dt1_naive = dt1.replace(tzinfo=None) if dt1.tzinfo else dt1
        dt2_naive = dt2.replace(tzinfo=None) if dt2.tzinfo else dt2
        ...
```

### Percentage Formatting

```python
# For returns/PnL (signed)
def _format_percent(value):
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"  # +15.50%, -8.25%

# For MDD/Volatility/Win Rate (unsigned)
def _format_plain_percent(value):
    return f"{value:.2f}%"  # 22.50%, 8.50%
```

### Filter Fallback Pattern

```python
def _filter_by_criteria(items, criteria, allow_fallback=True):
    filtered = [item for item in items if matches(item, criteria)]

    if filtered:
        return filtered, False  # (result, used_fallback)
    elif allow_fallback:
        return items, True  # Return all with warning flag
    else:
        return [], False  # Return empty
```

### Deterministic Demo Data

```python
# Use fixed reference date for demo mode
_DEMO_REFERENCE_DATE = datetime(2026, 1, 1, 12, 0, 0)

def _get_demo_data():
    base_time = _DEMO_REFERENCE_DATE
    return [
        Item(timestamp=base_time - timedelta(minutes=1), ...),
        Item(timestamp=base_time - timedelta(minutes=5), ...),
    ]
```

## Troubleshooting

### Review 결과가 비어있을 때
- 변경 사항이 staging 되어 있는지 확인
- `git status` 로 modified 파일 확인

### P2 이슈가 계속 발생할 때
- 근본 원인 분석 (edge case 누락 등)
- 테스트 케이스 추가로 커버리지 확보

### codex 명령 실패 시
- API 키 설정 확인
- 네트워크 연결 확인
- `codex --version` 으로 설치 확인

## Integration with CI/CD

```yaml
# .github/workflows/review.yml (예시)
name: Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Codex Review
        run: codex review --uncommitted
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## References

- [Codex CLI Documentation](https://github.com/openai/codex)
- [MASP Coding Checklist](./CODING_CHECKLIST.md)
