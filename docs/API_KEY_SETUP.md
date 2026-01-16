# MASP API Key Setup Guide

## Method 1: KeyManager (Recommended for Production)

KeyManager encrypts API keys and stores them securely.

### Prerequisites
```bash
# MASP_MASTER_KEY must be a 32-byte base64 string
$env:MASP_MASTER_KEY = "your-secure-master-key-base64-encoded"
```

### Register keys via API (If endpoint is enabled)

Note: Endpoint availability depends on your MASP deployment configuration.
If unavailable, use environment variables for development or consult your KeyManager setup.
```bash
curl -X POST http://localhost:8000/api/v1/keys/upbit \
  -H "X-MASP-ADMIN-TOKEN: your-admin-token" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key", "secret_key": "your-secret-key"}'
```

## Method 2: Environment Variables (Dev/Test Only)

For development, set environment variables and start the server.
```powershell
$env:UPBIT_API_KEY = "your-api-key"
$env:UPBIT_SECRET_KEY = "your-secret-key"
```

Warning: Environment variables are not encrypted. Use KeyManager for production.

## Enable LIVE Mode

LIVE mode is disabled by default, even if keys exist.
To enable LIVE mode explicitly:
```powershell
$env:MASP_DASHBOARD_LIVE = "1"
```

## Security Checklist
- [ ] No API keys hardcoded in source
- [ ] `.env` is in `.gitignore`
- [ ] MASP_MASTER_KEY is strong and protected
