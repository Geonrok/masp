# MASP Kubernetes Health Probe Configuration Guide

## Overview
MASP Health Server는 기본적으로 `127.0.0.1:8080`에 바인딩됩니다.
Kubernetes 환경에서 probe 접근을 위해 아래 설정이 필요합니다.

## 1. 환경 변수 설정

### Option A: Pod IP 바인딩 (권장)
```yaml
env:
  - name: MASP_HEALTH_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
  - name: MASP_HEALTH_PORT
    value: "8080"
  - name: MASP_HEALTH_STRICT
    value: "1"
```

### Option B: 모든 인터페이스 바인딩
```yaml
env:
  - name: MASP_HEALTH_HOST
    value: "0.0.0.0"
```

> ⚠️ Option B 사용 시 NetworkPolicy 필수

## 2. Kubernetes Probe 설정
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: masp-scheduler
spec:
  containers:
    - name: scheduler
      image: masp:latest
      ports:
        - containerPort: 8080
          name: health
      livenessProbe:
        httpGet:
          path: /health/live
          port: health
        initialDelaySeconds: 10
        periodSeconds: 30
        failureThreshold: 3
      readinessProbe:
        httpGet:
          path: /health/ready
          port: health
        initialDelaySeconds: 5
        periodSeconds: 10
        failureThreshold: 3
      startupProbe:
        httpGet:
          path: /health
          port: health
        initialDelaySeconds: 0
        periodSeconds: 5
        failureThreshold: 30
```

## 3. NetworkPolicy 예제
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: masp-health-access
spec:
  podSelector:
    matchLabels:
      app: masp-scheduler
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - port: 8080
      protocol: TCP
```

## 4. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| Probe timeout | Host 바인딩 불일치 | `MASP_HEALTH_HOST` 설정 |
| 503 on ready | Scheduler 미초기화 | `initialDelaySeconds` 증가 |
| Connection refused | 포트 충돌 | `MASP_HEALTH_PORT` 변경 |
