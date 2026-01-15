"""
API Server - Entry point.

Usage:
    python -m apps.api_server
    python -m apps.api_server --port 8080
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Strategy Platform API Server")
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
    
    args = parser.parse_args()
    
    print(f"Starting API server at http://{args.host}:{args.port}")
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "apps.api_server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()


