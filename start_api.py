#!/usr/bin/env python3
"""
UNR-Explainer API 启动脚本
"""

import uvicorn
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="启动 UNR-Explainer API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口") 
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNR-Explainer API 服务启动")
    print("=" * 60)
    print(f"监听地址: http://{args.host}:{args.port}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name()}")
    print("API 文档: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

if __name__ == "__main__":
    main() 