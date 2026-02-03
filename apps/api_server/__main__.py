"""
API Server - Entry point.

Usage:
    python -m apps.api_server
    python -m apps.api_server --port 8080
    python -m apps.api_server --ssl-cert cert.pem --ssl-key key.pem
"""

import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Strategy Platform API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=os.getenv("API_SSL_CERTFILE"),
        help="Path to SSL certificate file (.pem)",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=os.getenv("API_SSL_KEYFILE"),
        help="Path to SSL private key file (.pem)",
    )
    parser.add_argument(
        "--ssl-key-password",
        type=str,
        default=os.getenv("API_SSL_KEYFILE_PASSWORD"),
        help="Password for encrypted private key",
    )

    args = parser.parse_args()

    # Build uvicorn configuration
    uvicorn_config = {
        "app": "apps.api_server.main:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
    }

    # Check for SSL configuration
    ssl_enabled = False
    if args.ssl_cert and args.ssl_key:
        cert_path = Path(args.ssl_cert)
        key_path = Path(args.ssl_key)

        if cert_path.exists() and key_path.exists():
            uvicorn_config["ssl_certfile"] = str(cert_path.resolve())
            uvicorn_config["ssl_keyfile"] = str(key_path.resolve())
            if args.ssl_key_password:
                uvicorn_config["ssl_keyfile_password"] = args.ssl_key_password
            ssl_enabled = True
        else:
            print(f"Warning: SSL certificate or key file not found")
            if not cert_path.exists():
                print(f"  Certificate: {args.ssl_cert} (not found)")
            if not key_path.exists():
                print(f"  Key: {args.ssl_key} (not found)")

    protocol = "https" if ssl_enabled else "http"
    print(f"Starting API server at {protocol}://{args.host}:{args.port}")
    print(f"Dashboard: {protocol}://{args.host}:{args.port}/")
    print(f"API docs: {protocol}://{args.host}:{args.port}/docs")

    if ssl_enabled:
        print(f"SSL enabled with certificate: {args.ssl_cert}")

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
