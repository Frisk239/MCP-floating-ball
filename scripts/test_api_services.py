#!/usr/bin/env python3
"""
MCP Floating Ball - API服务测试脚本

专门测试AI服务提供商的连接和基本功能。
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class APIServiceTester:
    """API服务测试器"""

    def __init__(self):
        self.test_results = []

    async def test_kimi_service(self):
        """测试Kimi服务"""
        print("🤖 测试月之暗面Kimi服务...")

        try:
            from src.ai.orchestrator import chat, ai_orchestrator

            # 检查Kimi服务是否可用
            if "kimi" not in [p.value for p in ai_orchestrator.providers.keys()]:
                print("   ❌ Kimi服务未配置或未初始化")
                return False

            # 测试对话功能
            messages = [
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ]

            print("   📝 发送测试消息...")
            response = await chat(messages, max_tokens=100)

            if response and response.get("content"):
                print("   ✅ Kimi对话服务正常")
                print(f"   📄 回复: {response['content'][:100]}...")
                return True
            else:
                print("   ❌ Kimi对话服务响应异常")
                return False

        except Exception as e:
            print(f"   ❌ Kimi服务测试失败: {e}")
            return False

    async def test_metaso_service(self):
        """测试秘塔搜索服务"""
        print("🔍 测试秘塔AI搜索服务...")

        try:
            from src.ai.orchestrator import search, ai_orchestrator

            # 检查秘塔服务是否可用
            if "metaso" not in [p.value for p in ai_orchestrator.providers.keys()]:
                print("   ❌ 秘塔搜索服务未配置或未初始化")
                return False

            # 测试搜索功能
            print("   🔍 执行测试搜索...")
            result = await search("Python编程", max_results=3)

            if result and result.get("results"):
                print("   ✅ 秘塔搜索服务正常")
                print(f"   📊 找到 {result.get('total_results', 0)} 个结果")
                if result.get("results"):
                    first_result = result["results"][0]
                    print(f"   📄 示例结果: {first_result.get('title', '无标题')}")
                return True
            else:
                print("   ❌ 秘塔搜索服务响应异常")
                return False

        except Exception as e:
            print(f"   ❌ 秘塔搜索服务测试失败: {e}")
            return False

    async def test_dashscope_service(self):
        """测试DashScope服务"""
        print("🗣️ 测试阿里云DashScope服务...")

        try:
            from src.ai.orchestrator import ai_orchestrator

            # 检查DashScope服务是否可用
            if "dashscope" not in [p.value for p in ai_orchestrator.providers.keys()]:
                print("   ❌ DashScope服务未配置或未初始化")
                return False

            # 测试服务信息
            provider = ai_orchestrator.get_provider("dashscope")
            if hasattr(provider, 'get_service_info'):
                service_info = provider.get_service_info()
                print("   ✅ DashScope服务配置正常")
                print(f"   📊 支持的服务: ASR={service_info.get('supports_asr')}, TTS={service_info.get('supports_tts')}, Vision={service_info.get('supports_vision')}")
                return True
            else:
                print("   ❌ DashScope服务信息获取失败")
                return False

        except Exception as e:
            print(f"   ❌ DashScope服务测试失败: {e}")
            return False

    async def test_health_check(self):
        """测试健康检查"""
        print("🏥 执行系统健康检查...")

        try:
            from src.ai.orchestrator import ai_orchestrator

            health_status = await ai_orchestrator.health_check()

            if health_status:
                print("   ✅ 健康检查完成")
                print(f"   📊 整体状态: {health_status.get('overall_status', 'unknown')}")

                providers_status = health_status.get('providers', {})
                for provider, status in providers_status.items():
                    status_emoji = "✅" if status.get("status") == "healthy" else "❌"
                    print(f"   {status_emoji} {provider.upper()}: {status.get('status', 'unknown')}")

                return health_status.get('overall_status') == "healthy"
            else:
                print("   ❌ 健康检查失败")
                return False

        except Exception as e:
            print(f"   ❌ 健康检查异常: {e}")
            return False

    async def test_console_agent(self):
        """测试控制台代理基本功能"""
        print("💻 测试控制台代理...")

        try:
            from src.interfaces.console_agent import ConsoleAgent

            # 创建控制台代理实例
            agent = ConsoleAgent()

            # 测试初始化（但不运行主循环）
            print("   🔧 初始化控制台代理...")
            await agent.initialize()

            print("   ✅ 控制台代理初始化成功")
            print("   📊 对话历史长度: 0 (新会话)")

            # 清理
            await ai_orchestrator.close()

            return True

        except Exception as e:
            print(f"   ❌ 控制台代理测试失败: {e}")
            return False

    async def test_basic_conversation(self):
        """测试基本对话功能"""
        print("💬 测试基本对话功能...")

        try:
            from src.ai.orchestrator import chat

            # 测试简单对话
            test_questions = [
                "你好",
                "什么是AI？",
                "1+1等于几？"
            ]

            success_count = 0
            for i, question in enumerate(test_questions, 1):
                print(f"   📝 问题 {i}: {question}")

                messages = [
                    {"role": "user", "content": question}
                ]

                try:
                    response = await chat(messages, max_tokens=50)
                    if response and response.get("content"):
                        print(f"   ✅ 回复 {i}: {response['content'][:50]}...")
                        success_count += 1
                    else:
                        print(f"   ❌ 回复 {i}: 无响应内容")
                except Exception as e:
                    print(f"   ❌ 回复 {i}: {e}")

            success_rate = success_count / len(test_questions)
            print(f"   📊 对话成功率: {success_rate*100:.1f}% ({success_count}/{len(test_questions)})")

            return success_rate >= 0.6  # 至少60%成功率

        except Exception as e:
            print(f"   ❌ 基本对话测试失败: {e}")
            return False

    async def test_search_functionality(self):
        """测试搜索功能"""
        print("🔍 测试搜索功能...")

        try:
            from src.ai.orchestrator import search

            test_queries = [
                "Python教程",
                "机器学习",
                "AI助手"
            ]

            success_count = 0
            for i, query in enumerate(test_queries, 1):
                print(f"   🔍 搜索 {i}: {query}")

                try:
                    result = await search(query, max_results=3)
                    if result and result.get("results"):
                        result_count = len(result["results"])
                        print(f"   ✅ 搜索 {i}: 找到 {result_count} 个结果")
                        success_count += 1
                    else:
                        print(f"   ❌ 搜索 {i}: 无结果")
                except Exception as e:
                    print(f"   ❌ 搜索 {i}: {e}")

            success_rate = success_count / len(test_queries)
            print(f"   📊 搜索成功率: {success_rate*100:.1f}% ({success_count}/{len(test_queries)})")

            return success_rate >= 0.6  # 至少60%成功率

        except Exception as e:
            print(f"   ❌ 搜索功能测试失败: {e}")
            return False

    def print_summary(self, results: Dict[str, bool]):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("🧪 API服务测试摘要")
        print("="*60)

        total_tests = len(results)
        successful_tests = sum(results.values())
        failed_tests = total_tests - successful_tests

        print(f"📊 测试总数: {total_tests}")
        print(f"✅ 成功: {successful_tests}")
        print(f"❌ 失败: {failed_tests}")
        print(f"📈 成功率: {(successful_tests/total_tests*100):.1f}%")

        print("\n📋 详细结果:")
        for test_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}")

        if failed_tests == 0:
            print("\n🎉 所有API服务测试通过！系统准备就绪。")
            print("\n💡 下一步:")
            print("   • 运行: python scripts/test_console.py")
            print("   • 开始使用AI助手进行对话")
        else:
            print("\n⚠️  部分API服务存在问题。")
            print("\n🔧 建议:")
            print("   • 检查API密钥配置是否正确")
            print("   • 确认网络连接正常")
            print("   • 查看具体错误信息")

    async def run_all_tests(self):
        """运行所有API服务测试"""
        print("🚀 开始 MCP Floating Ball API服务测试")
        print("="*60)

        results = {}

        # 基础服务测试
        print("\n🔧 基础服务测试")
        print("-" * 30)

        results["控制台代理"] = await self.test_console_agent()
        results["健康检查"] = await self.test_health_check()

        # AI服务测试
        print("\n🤖 AI服务测试")
        print("-" * 30)

        results["Kimi服务"] = await self.test_kimi_service()
        results["秘塔搜索服务"] = await self.test_metaso_service()
        results["DashScope服务"] = await self.test_dashscope_service()

        # 功能测试
        print("\n💬 功能测试")
        print("-" * 30)

        results["基本对话"] = await self.test_basic_conversation()
        results["搜索功能"] = await self.test_search_functionality()

        # 打印摘要
        self.print_summary(results)

        return all(results.values())


async def main():
    """主函数"""
    tester = APIServiceTester()

    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ API服务测试失败: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))