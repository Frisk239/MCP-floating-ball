#!/usr/bin/env python3
"""
MCP Floating Ball - 控制台测试脚本

快速启动AI助手的测试脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.console_agent import main

if __name__ == "__main__":
    print("🚀 启动 MCP Floating Ball AI助手...")
    print("如果这是第一次运行，请确保已经配置了正确的API密钥")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        sys.exit(1)