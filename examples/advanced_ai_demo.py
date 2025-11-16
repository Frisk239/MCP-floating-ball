"""
MCP Floating Ball - 高级AI系统演示

展示如何使用高级AI控制器的各种功能。
"""

import asyncio
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.assistant.advanced_ai_controller import get_advanced_ai_controller, ControllerMode
from src.assistant.intelligent_learner import LearningMode
from src.core.logging import get_logger

logger = get_logger("advanced_ai_demo")


class AdvancedAIDemo:
    """高级AI演示类"""

    def __init__(self):
        self.controller = get_advanced_ai_controller()
        self.logger = get_logger(self.__class__.__name__)

    async def run_demo(self):
        """运行完整演示"""
        try:
            print("🚀 MCP Floating Ball 高级AI系统演示")
            print("=" * 60)

            # 1. 启动系统
            print("\n1️⃣ 启动高级AI控制器...")
            success = await self.controller.start()
            if not success:
                print("❌ 启动失败")
                return

            print("✅ 高级AI控制器启动成功")

            # 2. 显示系统状态
            await self.show_system_status()

            # 3. 智能任务执行演示
            await self.demo_intelligent_task_execution()

            # 4. 多模型分析演示
            await self.demo_multi_model_analysis()

            # 5. 高级融合分析演示
            await self.demo_advanced_fusion()

            # 6. 工作流推荐演示
            await self.demo_workflow_recommendations()

            # 7. 性能监控演示
            await self.demo_performance_monitoring()

            # 8. 异常检测演示
            await self.demo_anomaly_detection()

            # 9. 学习洞察演示
            await self.demo_learning_insights()

            # 10. 系统管理演示
            await self.demo_system_management()

            print("\n🎉 演示完成！")

        except Exception as e:
            self.logger.error(f"演示失败: {e}")
            print(f"❌ 演示失败: {e}")

        finally:
            # 清理资源
            print("\n🛑 停止系统...")
            await self.controller.stop()
            print("✅ 系统已停止")

    async def show_system_status(self):
        """显示系统状态"""
        print("\n2️⃣ 系统状态检查...")
        status = await self.controller.get_system_status()

        print(f"📊 控制器状态: {'运行中' if status['controller']['is_running'] else '已停止'}")
        print(f"🎯 运行模式: {status['controller']['mode']}")
        print(f"⏱️ 运行时间: {status['controller']['uptime_seconds']:.1f}秒")
        print(f"📈 总请求数: {status['statistics']['total_requests']}")
        print(f"✅ 成功请求: {status['statistics']['successful_requests']}")
        print(f"❌ 失败请求: {status['statistics']['failed_requests']}")
        print(f"📊 成功率: {status['statistics']['success_rate']}%")

        print("\n🔧 服务状态:")
        for service_name, health in status['services'].items():
            status_icon = "✅" if health['status'] == 'running' else "❌"
            print(f"   {status_icon} {service_name}: {health['status']} (响应时间: {health['response_time']:.3f}s)")

    async def demo_intelligent_task_execution(self):
        """演示智能任务执行"""
        print("\n3️⃣ 智能任务执行演示...")

        tasks = [
            "分析这个Python代码的性能问题",
            "帮我写一个数据处理脚本",
            "解释机器学习中的过拟合概念",
            "推荐一些提高编程效率的方法"
        ]

        for i, task in enumerate(tasks, 1):
            print(f"\n   任务 {i}: {task}")
            result = await self.controller.intelligent_task_execution(task)

            if result['success']:
                print(f"   ✅ 执行成功")
                print(f"   🤖 使用模型: {result['model_used']}")
                print(f"   📊 置信度: {result['confidence']:.2f}")
                print(f"   ⏱️ 执行时间: {result['execution_time']:.2f}秒")
                print(f"   📝 结果: {result['result'][:100]}...")
            else:
                print(f"   ❌ 执行失败: {result.get('error', '未知错误')}")

    async def demo_multi_model_analysis(self):
        """演示多模型分析"""
        print("\n4️⃣ 多模型分析演示...")

        prompt = "请分析人工智能在医疗健康领域的应用前景和挑战"
        print(f"   分析主题: {prompt}")

        result = await self.controller.multi_model_analysis(prompt, ["kimi", "dashscope"])

        if result['success']:
            print(f"   ✅ 多模型分析成功")
            print(f"   🔗 参与模型: {', '.join(result['contributing_models'])}")
            print(f"   🎯 融合方法: {result['fusion_method']}")
            print(f"   📊 置信度: {result['confidence']:.2f}")
            print(f"   📝 融合结果: {result['fused_result'][:150]}...")
        else:
            print(f"   ❌ 多模型分析失败: {result.get('error', '未知错误')}")

    async def demo_advanced_fusion(self):
        """演示高级融合分析"""
        print("\n5️⃣ 高级融合分析演示...")

        prompt = "分析区块链技术在供应链管理中的优势和风险"
        print(f"   分析主题: {prompt}")

        result = await self.controller.advanced_fusion_analysis(prompt, "comprehensive")

        if result['success']:
            print(f"   ✅ 高级融合分析成功")
            print(f"   🔗 参与模型: {', '.join(result['contributing_models'])}")
            print(f"   🧠 融合方法: {result['fusion_method']}")
            print(f"   🔍 分析深度: {result['analysis_depth']}")
            print(f"   📊 置信度: {result['confidence']:.2f}")
            print(f"   📝 融合结果: {result['fused_output'][:150]}...")
        else:
            print(f"   ❌ 高级融合分析失败: {result.get('error', '未知错误')}")

    async def demo_workflow_recommendations(self):
        """演示工作流推荐"""
        print("\n6️⃣ 工作流推荐演示...")

        user_id = "demo_user"
        context = {
            "current_task": "数据分析",
            "skill_level": "intermediate",
            "preferences": ["automation", "efficiency"]
        }

        print(f"   用户ID: {user_id}")
        print(f"   上下文: {context}")

        result = await self.controller.workflow_recommendation(user_id, context)

        if result['success']:
            print(f"   ✅ 工作流推荐成功")
            print(f"   📋 推荐数量: {len(result['recommendations'])}")

            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"   {i}. {rec['workflow_name']}")
                print(f"      置信度: {rec['confidence']:.2f}")
                print(f"      推荐类型: {rec['recommendation_type']}")
                print(f"      说明: {rec['explanation']}")
                print()
        else:
            print(f"   ❌ 工作流推荐失败: {result.get('error', '未知错误')}")

    async def demo_performance_monitoring(self):
        """演示性能监控"""
        print("\n7️⃣ 性能监控演示...")

        # 获取性能摘要
        result = await self.controller.performance_optimization("summary")

        if result['success']:
            print(f"   ✅ 性能监控数据获取成功")
            summary = result['performance_summary']
            print(f"   📊 监控状态: {summary['monitoring_status']}")
            print(f"   📈 总指标数: {summary['total_metrics']}")
            print(f"   🚨 活跃告警数: {summary['active_alerts']}")
            print(f"   🔧 优化次数: {summary['optimization_count']}")

            # 显示当前指标
            print(f"\n   📊 当前指标:")
            for metric_name, metric_data in summary['current_metrics'].items():
                print(f"      {metric_name}: {metric_data['current']:.2f}{metric_data['unit']}")
        else:
            print(f"   ❌ 性能监控失败: {result.get('error', '未知错误')}")

    async def demo_anomaly_detection(self):
        """演示异常检测"""
        print("\n8️⃣ 异常检测演示...")

        # 模拟一些用户行为
        for i in range(5):
            await self.controller.anomaly_detector.add_user_event(
                user_id="demo_user",
                event_type="task_execution",
                action=f"action_{i}",
                context={"complexity": i * 0.2},
                duration=10 + i * 5,
                success=i % 4 != 0  # 75%成功率
            )

        # 获取异常分析
        result = await self.controller.anomaly_analysis(1)  # 最近1小时

        if result['success']:
            print(f"   ✅ 异常检测分析成功")
            summary = result['anomaly_summary']
            print(f"   📊 检测总数: {summary['total_detections']}")
            print(f"   🚨 高危异常: {summary['critical_detections']}")
            print(f"   🧠 ML模型状态: {'已训练' if summary['ml_model_trained'] else '未训练'}")
            print(f"   👥 受影响用户: {summary['affected_users']}")

            # 显示异常类型分布
            if summary['type_distribution']:
                print(f"\n   📊 异常类型分布:")
                for anomaly_type, count in summary['type_distribution'].items():
                    print(f"      {anomaly_type}: {count}")
        else:
            print(f"   ❌ 异常检测失败: {result.get('error', '未知错误')}")

    async def demo_learning_insights(self):
        """演示学习洞察"""
        print("\n9️⃣ 学习洞察演示...")

        # 模拟一些学习交互
        interactions = [
            ("task_completion", "完成了数据分析任务", True, 15.0, 0.9),
            ("error_handling", "遇到了配置错误", False, 5.0, 0.3),
            ("workflow_usage", "使用了自动化工作流", True, 25.0, 0.8),
            ("feature_discovery", "发现了新功能", True, 10.0, 0.95)
        ]

        for interaction_type, content, success, duration, satisfaction in interactions:
            await self.controller.intelligent_learner.record_interaction(
                interaction_type=interaction_type,
                content=content,
                context={"demo": True},
                outcome="success" if success else "failure",
                duration=duration,
                satisfaction_score=satisfaction
            )

        # 获取学习洞察
        result = await self.controller.learning_insights("demo_user")

        if result['success']:
            print(f"   ✅ 学习洞察获取成功")
            stats = result['learning_statistics']
            print(f"   📊 总交互数: {stats['total_interactions']}")
            print(f"   📈 成功率: {stats['success_rate']:.2%}")
            print(f"   ⏱️ 平均持续时间: {stats['average_duration']:.1f}秒")
            print(f"   😊 平均满意度: {stats['average_satisfaction']:.2f}")
            print(f"   🧠 学习模式: {stats['learning_mode']}")

            # 显示推荐
            recommendations = result['recommendations']
            if recommendations:
                print(f"\n   💡 个性化推荐 ({len(recommendations)}条):")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"      {i}. {rec['title']}")
                    print(f"         {rec['description']}")
                    print(f"         优先级: {rec['priority']}")
        else:
            print(f"   ❌ 学习洞察获取失败: {result.get('error', '未知错误')}")

    async def demo_system_management(self):
        """演示系统管理"""
        print("\n🔟 系统管理演示...")

        # 模式切换演示
        modes = [ControllerMode.NORMAL, ControllerMode.PERFORMANCE, ControllerMode.LEARNING]

        for mode in modes:
            print(f"\n   切换到 {mode.value} 模式...")
            success = await self.controller.set_mode(mode)
            if success:
                print(f"   ✅ 模式切换成功")

                # 执行健康检查
                health_result = await self.controller.execute_command("health_check")
                if health_result['success']:
                    print(f"   📊 健康检查通过，服务数: {len(health_result['health_status'])}")
            else:
                print(f"   ❌ 模式切换失败")

        # 执行管理命令演示
        commands = [
            ("status", {}, "获取系统状态"),
            ("optimize", {"type": "auto"}, "性能优化"),
            ("insights", {}, "获取学习洞察")
        ]

        print(f"\n   🎛️ 管理命令演示:")
        for cmd in commands:
            if isinstance(cmd, tuple) and len(cmd) == 3:
                command, params, desc = cmd
            elif isinstance(cmd, tuple) and len(cmd) == 2:
                command, desc = cmd
                params = {}
            else:
                command = cmd
                desc = cmd
                params = {}

            print(f"\n   执行命令: {command} ({desc})")
            result = await self.controller.execute_command(command, params)

            if result['success']:
                print(f"   ✅ 命令执行成功")
                if 'statistics' in result:
                    stats = result['statistics']
                    print(f"      请求统计: 成功 {stats['successful_requests']}/{stats['total_requests']}")
            else:
                print(f"   ❌ 命令执行失败: {result.get('error', '未知错误')}")


async def main():
    """主函数"""
    try:
        demo = AdvancedAIDemo()
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n💥 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎬 启动MCP Floating Ball高级AI系统演示")
    print("请确保已安装所有依赖包...")

    # 运行演示
    asyncio.run(main())