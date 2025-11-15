#!/usr/bin/env python3
"""
MCP Floating Ball - 环境设置脚本

自动检查和配置项目运行环境
"""

import sys
import os
from pathlib import Path
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python版本过低，需要Python 3.11或更高版本")
        print(f"   当前版本: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查必要的依赖"""
    print("\n🔍 检查依赖包...")

    required_packages = [
        "openai",
        "pydantic",
        "loguru",
        "httpx",
        "asyncio"
    ]

    optional_packages = [
        ("dashscope", "阿里云DashScope服务"),
        ("opencv-python", "图像处理"),
        ("pyaudio", "音频处理"),
        ("numpy", "数值计算"),
        ("rich", "美化输出")
    ]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - 缺失")
            missing_required.append(package)

    print("\n📦 可选依赖:")
    for package, description in optional_packages:
        try:
            # 处理包名可能和导入名不同的情况
            import_name = package.replace("-", "_").replace("opencv-python", "cv2")
            __import__(import_name)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️  {package} - {description} - 缺失（可选）")
            missing_optional.append(package)

    if missing_required:
        print(f"\n❌ 缺少必要的依赖包: {', '.join(missing_required)}")
        print("请运行: pip install -r requirements/base.txt")
        return False

    if missing_optional:
        print(f"\n⚠️  缺少可选依赖包: {', '.join(missing_optional)}")
        print("部分功能可能无法使用，运行: pip install -r requirements/base.txt")

    return True

def check_config():
    """检查配置文件"""
    print("\n🔧 检查配置文件...")

    env_file = Path(".env")
    config_dir = Path("config")

    if not env_file.exists():
        print("❌ .env配置文件不存在")
        if (config_dir / ".env.example").exists():
            print("💡 发现配置模板，正在复制...")
            import shutil
            shutil.copy(config_dir / ".env.example", env_file)
            print("✅ .env配置文件已创建，请编辑添加您的API密钥")
        else:
            print("❌ 配置模板文件也不存在")
            return False
    else:
        print("✅ .env配置文件存在")

    # 检查API密钥配置
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_keys = {
            "MOONSHOT_API_KEY": "月之暗面Kimi",
            "ALIBABA_CLOUD_ACCESS_KEY_ID": "阿里云DashScope",
            "METASO_API_KEY": "秘塔AI搜索"
        }

        configured_services = []
        missing_services = []

        for key, service in api_keys.items():
            value = os.getenv(key)
            if value and value != f"your_{key.lower()}_here":
                configured_services.append(service)
            else:
                missing_services.append(service)

        if configured_services:
            print(f"✅ 已配置服务: {', '.join(configured_services)}")

        if missing_services:
            print(f"⚠️  未配置服务: {', '.join(missing_services)}")
            print("   请在.env文件中添加相应的API密钥")

    except ImportError:
        print("⚠️  python-dotenv未安装，无法检查API密钥配置")

    return True

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")

    directories = [
        "data/logs",
        "data/cache",
        "data/temp",
    ]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ 创建目录: {directory}")
        else:
            print(f"   ✅ 目录已存在: {directory}")

def check_platform():
    """检查平台兼容性"""
    print(f"\n💻 平台信息: {platform.system()} {platform.release()}")

    if platform.system() == "Windows":
        print("✅ Windows平台兼容性良好")
        print("💡 某些功能可能需要管理员权限")
    elif platform.system() == "Linux":
        print("✅ Linux平台兼容性良好")
    elif platform.system() == "Darwin":
        print("✅ macOS平台兼容性良好")
    else:
        print(f"⚠️  未经测试的平台: {platform.system()}")

def run_basic_test():
    """运行基础功能测试"""
    print("\n🧪 运行基础功能测试...")

    try:
        # 测试核心配置加载
        sys.path.insert(0, str(Path.cwd()))
        from src.core.config import get_settings

        print("   ✅ 配置系统加载成功")

        # 测试日志系统
        from src.core.logging import get_logger
        test_logger = get_logger("setup_test")
        test_logger.info("日志系统测试成功")
        print("   ✅ 日志系统正常")

        # 测试工具系统
        from src.tools.registry import tool_registry
        print(f"   ✅ 工具注册系统正常，已注册 {len(tool_registry)} 个工具")

        print("✅ 基础功能测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 基础功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 MCP Floating Ball 环境设置向导")
    print("=" * 50)

    all_checks_passed = True

    # 检查Python版本
    if not check_python_version():
        all_checks_passed = False

    # 检查依赖
    if not check_dependencies():
        all_checks_passed = False

    # 创建目录
    create_directories()

    # 检查配置
    if not check_config():
        all_checks_passed = False

    # 检查平台
    check_platform()

    # 基础功能测试
    if not run_basic_test():
        all_checks_passed = False

    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 环境设置完成！")
        print("\n📝 下一步:")
        print("1. 编辑 .env 文件，添加您的API密钥")
        print("2. 运行: python scripts/test_console.py")
        print("3. 开始使用AI助手！")
        print("\n📖 帮助:")
        print("- 输入 /help 查看使用帮助")
        print("- 输入 /status 查看服务状态")
    else:
        print("❌ 环境设置存在问题，请根据上述提示解决")
        print("\n🔧 建议:")
        print("1. 确保Python版本 >= 3.11")
        print("2. 运行: pip install -r requirements/base.txt")
        print("3. 配置API密钥")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())