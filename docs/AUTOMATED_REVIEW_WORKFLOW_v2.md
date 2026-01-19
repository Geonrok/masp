# MASP 자동 검수 워크플로우

## 개요
Claude Code Opus 4.5와 Codex CLI를 연동한 자동 코드 리뷰 시스템

## 아키텍처
```
┌─────────────────────────────────────┐
│  Claude Code Opus 4.5               │
│  - 계획 수립                         │
│  - 코드 작성                         │
│  - 테스트 작성                       │
└─────────────────┬───────────────────┘
                  │ 자동 호출
                  ▼
┌─────────────────────────────────────┐
│  codex review --uncommitted         │
│  - 비대화형 코드 리뷰               │
│  - 필수/권장/선택 보강 출력          │
└─────────────────┬───────────────────┘
                  │ 결과 분석
                  ▼
┌─────────────────────────────────────┐
│  Claude Code                        │
│  - 검수 결과 적용                    │
│  - 재테스트                         │
│  - 통과 시 커밋                     │
└─────────────────────────────────────┘
```

## 사전 요구사항

### 1. Claude Code 설치
```powershell
npm install -g @anthropic-ai/claude-code
```

### 2. Codex CLI 설치
```powershell
npm install -g @openai/codex
```

### 3. 인증 설정
```powershell
# Claude Code OAuth 토큰 (영구 설정)
[Environment]::SetEnvironmentVariable("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-...", "User")

# Codex는 별도 로그인
codex login
```

## 워크플로우 상세

### Phase 1: 코드 작성
1. Claude Code가 요구사항 분석
2. 컴포넌트 코드 작성 (예: `component.py`)
3. 테스트 코드 작성 (예: `test_component.py`)
4. `__init__.py` export 추가

### Phase 2: 로컬 테스트
```powershell
pytest tests/dashboard/test_component.py -v
pytest --tb=short -q  # 전체 테스트
```

### Phase 3: 자동 검수
```powershell
cd "E:\투자\Multi-Asset Strategy Platform"
codex review --uncommitted
```

### Phase 4: 검수 결과 적용

#### 적용 원칙
| 구분 | 적용 여부 | 기준 |
|------|----------|------|
| 필수보강 | ✅ 항상 | - |
| 권장보강 | ✅ 적용 | 런타임 예외, 데이터 오염, UX 일관성, 보안, 메모리 누수, 에러 핸들링 |
| 권장보강 | ❌ 스킵 | 명명 변경, 코드 정리, 스타일, 성능 최적화 (병목 아닌 경우) |

#### 판단 기준
> "프로덕션에서 장애나 손실이 발생할 수 있는가?"

### Phase 5: 커밋
```powershell
git add <files>
git commit -m "Add component description"
```

## Claude Code 프롬프트 템플릿

### 새 컴포넌트 작업 시작
```
## [Phase]-[번호] 작업: [컴포넌트명]

### 워크플로우 (전체 자동화)
1. 코드 작성 → 파일 저장
2. __init__.py에 export 추가
3. pytest 실행 → 전체 통과 확인
4. codex review --uncommitted 실행
5. 검수 결과 분석 및 적용
6. 재테스트 → 통과 시 커밋

### 검수 적용 원칙
- 필수보강: 항상 적용
- 권장보강: 런타임 안정성/UX 관련만 적용
- 명명/스타일 관련: 스킵

작업 시작해줘.
```

## 실행 예시

### 성공 케이스
```
Claude Code: 코드 작성 완료, pytest 38 passed
Claude Code: codex review --uncommitted 실행
Codex: 통과, 권장보강 2건 (timezone 관련)
Claude Code: 권장보강 적용 (런타임 안정성)
Claude Code: pytest 42 passed
Claude Code: git commit "Add component"
```

### 수정 필요 케이스
```
Claude Code: 코드 작성 완료, pytest 35 passed
Claude Code: codex review --uncommitted 실행
Codex: 필수보강 1건 (f-string 인코딩 깨짐)
Claude Code: 필수보강 적용
Claude Code: pytest 35 passed
Claude Code: codex review --uncommitted 재실행
Codex: 통과
Claude Code: git commit "Add component"
```

## 장점

1. **품질 향상**: 다른 AI의 시각으로 코드 검증
2. **자동화**: 수동 복사/붙여넣기 제거
3. **일관성**: 동일한 검수 기준 적용
4. **속도**: 검수 루프 자동 반복

## 제한사항

1. Codex CLI 사용량은 GPT 웹과 한도 공유
2. `codex review`는 uncommitted 변경사항만 검토
3. 복잡한 아키텍처 리뷰는 수동 검토 권장

## 관련 파일

- `E:\AI_Review\scripts\Extract-ReviewFiles.ps1` - 수동 검수용 스크립트
- `E:\AI_Review\templates\GPT_REVIEW_TEMPLATE.md` - 검수 템플릿
- `docs/CODING_CHECKLIST.md` - 코딩 체크리스트

---

## 병렬 듀얼 리뷰 시스템 (v2.0)

### 1. 개요
Codex CLI와 Gemini CLI를 동시에 실행하여 상호 보완적인 코드 리뷰를 수행합니다.

### 2. 사용법 예시
```powershell
# 기본 실행 (Uncommitted Changes)
.\scripts\review-parallel.ps1

# Staged 파일 검토
.\scripts\review-parallel.ps1 -Target --staged

# 특정 파일 검토
.\scripts\review-parallel.ps1 -Target "services/market_data.py" -Quiet
```

### 3. 출력 파일 구조
```
review-results/
├── codex_review_20260119_120000.md   (Codex 상세 결과)
├── gemini_review_20260119_120000.md  (Gemini 상세 결과)
└── review_summary_20260119_120000.md (통합 요약 리포트)
```

### 4. AI별 검토 초점
| 구분 | Codex | Gemini |
|------|-------|--------|
| **주 강점** | 구문 정확성, 라이브러리 사용법 | 보안 취약점, 아키텍처, 엣지 케이스 |
| **검토 영역** | 타입 힌트, PEP 8, API 사용 | 비즈니스 로직, 데이터 흐름, 예외 처리 |
| **피드백 스타일** | 구체적인 코드 제안 | 하이 레벨 분석 및 리스크 경고 |

### 5. 피드백 충돌 시 우선순위 규칙
1. **보안 이슈:** Gemini 의견 우선 (보수적 접근)
2. **코드 스타일:** Codex 의견 우선 (표준 준수)
3. **로직/성능:** 두 AI가 모두 지적한 경우 **필수 수정 (P1)**

### 6. 워크플로우 다이어그램
```
                 ┌──────────────────┐
                 │  Code Changes    │
                 └────────┬─────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │ Codex Review  │           │ Gemini Review │
    │ (Syntax/API)  │           │ (Security/Biz)│
    └───────┬───────┘           └───────┬───────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
                 ┌──────────────────┐
                 │  Summary Report  │
                 │ (Merge Findings) │
                 └──────────────────┘
```

### 7. 기대 효과
- **시간 단축:** 병렬 실행으로 리뷰 시간 50% 절감
- **품질 향상:** 상호 보완적인 관점(Syntax vs Logic)으로 사각지대 제거
- **안정성 강화:** Gemini의 보안/아키텍처 중심 리뷰로 리스크 최소화