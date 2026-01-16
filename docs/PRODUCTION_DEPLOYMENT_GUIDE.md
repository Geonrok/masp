# MASP Production Deployment Guide

## 1. 필수 환경 변수
```bash
# .env.production
MASP_HEALTH_HOST=0.0.0.0          # K8s: Pod IP 권장
MASP_HEALTH_PORT=8080
MASP_HEALTH_STRICT=1              # ⚠️ 운영 필수
MASP_ENABLE_METRICS=1             # Prometheus 활성화
MASP_HEARTBEAT_SEC=60             # 운영: 60초 권장
MASP_HEARTBEAT_LOG_LEVEL=DEBUG    # 운영: DEBUG 권장
```

## 2. MASP_HEALTH_STRICT=1 권장 이유

| 모드 | strict=0 (기본) | strict=1 (권장) |
|------|-----------------|-----------------|
| Bind 실패 시 | 경고 로그 후 계속 실행 | RuntimeError, 즉시 종료 |
| 모니터링 | Health 불능, 스케줄러만 동작 | 조기 실패로 문제 즉시 감지 |
| K8s 동작 | Pod 정상으로 오인 | Pod 재시작, 자동 복구 |

### ⚠️ strict=0의 위험성

**시나리오: 포트 8080이 이미 사용 중**

- `strict=0`: 스케줄러 정상 동작, Health 엔드포인트 불능
  → K8s probe 실패 → 트래픽 차단
  → 스케줄러는 "정상"으로 보임 (모니터링 블라인드)

- `strict=1`: 즉시 RuntimeError → Pod 재시작 → 정상화

## 3. 배포 체크리스트

- [ ] `MASP_HEALTH_STRICT=1` 설정
- [ ] `MASP_ENABLE_METRICS=1` 설정
- [ ] K8s probe 설정 완료
- [ ] NetworkPolicy 적용
- [ ] Prometheus ServiceMonitor 설정

## 4. PromQL 모니터링 쿼리
```promql
# Scheduler 상태
masp_scheduler_running

# Heartbeat 빈도
rate(masp_scheduler_heartbeat_total[5m])

# Uptime
masp_scheduler_uptime_seconds

# Active Exchanges
masp_scheduler_active_exchanges
```