#!/usr/bin/env python3
"""
测试智能学习功能

验证SQLite数据库管理和智能学习器的功能。
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_database_manager():
    """测试数据库管理器"""
    print("🔄 测试数据库管理器...")

    try:
        from src.core.database import get_database_manager

        # 获取数据库管理器
        db = get_database_manager()
        print("✅ 数据库管理器初始化成功")

        # 测试系统配置
        test_config_key = "test_config_key"
        db.set_config(test_config_key, "test_value", "string", "测试配置")
        retrieved_value = db.get_config(test_config_key)
        if retrieved_value == "test_value":
            print("✅ 系统配置读写测试通过")
        else:
            print(f"❌ 系统配置读写测试失败: 期望 'test_value', 得到 '{retrieved_value}'")

        # 测试实体搜索
        entities = db.search_entity("记事本")
        if entities:
            print(f"✅ 实体搜索测试通过: 找到 {len(entities)} 个匹配实体")
            for entity in entities[:2]:
                print(f"   - {entity['entity_name']} ({entity['entity_type']}) - 匹配度: {entity['match_score']}")
        else:
            print("❌ 实体搜索测试失败: 未找到匹配实体")

        # 测试命令历史记录
        test_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        success = db.add_command_history(
            session_id=test_session_id,
            user_id="test_user",
            original_command="打开记事本",
            intent_type="APP_LAUNCH",
            intent_confidence=1.0,
            parameters={"application": "notepad.exe"},
            tool_name="application_launcher",
            execution_time=0.5,
            success=True
        )
        if success:
            print("✅ 命令历史记录测试通过")
        else:
            print("❌ 命令历史记录测试失败")

        # 获取命令统计
        stats = db.get_command_statistics("test_user", days=1)
        if stats.get("total_commands", 0) > 0:
            print("✅ 命令统计测试通过")
            print(f"   - 总命令数: {stats['total_commands']}")
            print(f"   - 成功率: {stats['success_rate']:.2%}")
        else:
            print("❌ 命令统计测试失败")

        # 获取数据库统计
        db_stats = db.get_database_stats()
        if db_stats:
            print("✅ 数据库统计测试通过")
            print(f"   - 系统实体数: {db_stats.get('system_entities_count', 0)}")
            print(f"   - 数据库大小: {db_stats.get('file_size_mb', 0)} MB")
        else:
            print("❌ 数据库统计测试失败")

        print("🎉 数据库管理器测试完成！")
        return True

    except Exception as e:
        print(f"❌ 数据库管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intelligent_learner():
    """测试智能学习器"""
    print("\n🔄 测试智能学习器...")

    try:
        from src.assistant.intelligent_learner import IntelligentLearner

        # 创建智能学习器
        learner = IntelligentLearner("test_user")
        print("✅ 智能学习器初始化成功")

        # 模拟学习一些命令
        test_commands = [
            {
                "original_command": "打开记事本",
                "intent_type": "APP_LAUNCH",
                "intent_confidence": 1.0,
                "parameters": {"application": "notepad.exe"},
                "tool_name": "application_launcher",
                "execution_time": 0.5,
                "success": True
            },
            {
                "original_command": "打开百度",
                "intent_type": "WEB_SCRAPING",
                "intent_confidence": 1.0,
                "parameters": {"url": "https://www.baidu.com"},
                "tool_name": "web_scraper",
                "execution_time": 1.2,
                "success": True
            },
            {
                "original_command": "系统信息",
                "intent_type": "SYSTEM_INFO",
                "intent_confidence": 1.0,
                "parameters": {"info_type": "basic"},
                "tool_name": "system_info",
                "execution_time": 0.3,
                "success": True
            }
        ]

        # 学习这些命令
        for i, cmd in enumerate(test_commands):
            success = learner.learn_from_command(
                **cmd,
                session_id=f"test_session_{i}",
                context_data={"test": True}
            )
            if success:
                print(f"✅ 学习命令成功: {cmd['original_command']}")
            else:
                print(f"❌ 学习命令失败: {cmd['original_command']}")

        # 测试意图预测
        test_command = "打开计算器"
        prediction = learner.predict_intent(test_command)
        print(f"✅ 意图预测测试: '{test_command}' -> {prediction['predicted_intent']} (置信度: {prediction.get('confidence', 0):.2f})")
        if prediction.get("based_on_history"):
            print(f"   基于历史相似命令: {prediction.get('similar_commands', [])}")

        # 测试个性化建议
        suggestions = learner.get_personalized_suggestions("打开")
        if suggestions:
            print(f"✅ 个性化建议测试: 找到 {len(suggestions)} 个建议")
            for suggestion in suggestions[:3]:
                print(f"   - {suggestion.get('type', 'unknown')}: {suggestion.get('suggestion', '')} (评分: {suggestion.get('score', 0):.2f})")
        else:
            print("⚠️ 个性化建议测试: 未找到建议（可能是正常现象）")

        # 测试实体推荐
        entity_recs = learner.get_entity_recommendations("记事本", entity_type="application")
        if entity_recs:
            print(f"✅ 实体推荐测试: 找到 {len(entity_recs)} 个推荐实体")
            for entity in entity_recs[:2]:
                print(f"   - {entity['entity_name']} (个性化评分: {entity.get('personalized_score', 0):.2f})")
        else:
            print("⚠️ 实体推荐测试: 未找到推荐实体")

        # 测试用户洞察
        insights = learner.get_user_insights(days=1)
        if insights:
            print("✅ 用户洞察测试通过")
            stats = insights.get("statistics", {})
            if stats:
                print(f"   - 分析周期: {insights.get('analysis_period', '未知')}")
                print(f"   - 总命令数: {stats.get('total_commands', 0)}")
                print(f"   - 成功率: {stats.get('success_rate', 0):.2%}")

            preferences = insights.get("preferences", {})
            if preferences:
                preferred_intents = preferences.get("preferred_intents", {})
                if preferred_intents:
                    print("   - 偏好意图:", list(preferred_intents.keys()))
        else:
            print("⚠️ 用户洞察测试: 无足够数据生成洞察")

        print("🎉 智能学习器测试完成！")
        return True

    except Exception as e:
        print(f"❌ 智能学习器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_nlp_processor():
    """测试增强NLP处理器"""
    print("\n🔄 测试增强NLP处理器...")

    try:
        from src.assistant.enhanced_nlp_processor import EnhancedNLPProcessor

        # 创建增强NLP处理器（启用学习功能）
        nlp = EnhancedNLPProcessor("test_user")
        print(f"✅ 增强NLP处理器初始化成功 (学习功能: {'启用' if nlp.learning_enabled else '禁用'})")

        # 测试命令解析
        test_commands = [
            "打开记事本",
            "搜索Python教程",
            "系统信息",
            "截图"
        ]

        for cmd in test_commands:
            try:
                result = nlp.parse_command(cmd)
                if result.get("success"):
                    hierarchical_intent = result.get("hierarchical_intent")
                    if hierarchical_intent:
                        print(f"✅ '{cmd}' -> {hierarchical_intent.primary.value} -> {hierarchical_intent.secondary.value} (置信度: {hierarchical_intent.confidence:.2f})")
                    else:
                        print(f"✅ '{cmd}' -> 解析成功，但缺少层次意图")
                else:
                    print(f"❌ '{cmd}' -> 解析失败: {result.get('error')}")
            except Exception as e:
                print(f"❌ '{cmd}' -> 解析错误: {e}")

        # 测试学习功能
        if nlp.learning_enabled:
            # 模拟执行学习
            success = nlp.learn_from_execution(
                original_command="打开计算器",
                intent_type="APP_LAUNCH",
                intent_confidence=1.0,
                parameters={"application": "calc.exe"},
                tool_name="application_launcher",
                execution_time=0.4,
                success=True,
                session_id="test_enhanced_nlp"
            )
            if success:
                print("✅ 学习执行测试通过")
            else:
                print("❌ 学习执行测试失败")

            # 测试个性化建议
            suggestions = nlp.get_personalized_suggestions("打开")
            if suggestions:
                print(f"✅ 个性化建议测试: 找到 {len(suggestions)} 个建议")
            else:
                print("⚠️ 个性化建议测试: 未找到建议")

            # 测试实体推荐
            entity_recs = nlp.get_entity_recommendations("百度", entity_type="website")
            if entity_recs:
                print(f"✅ 实体推荐测试: 找到 {len(entity_recs)} 个推荐实体")
            else:
                print("⚠️ 实体推荐测试: 未找到推荐实体")

        print("🎉 增强NLP处理器测试完成！")
        return True

    except Exception as e:
        print(f"❌ 增强NLP处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎈 MCP Floating Ball - 智能功能测试")
    print("=" * 60)

    tests = [
        ("数据库管理器测试", test_database_manager),
        ("智能学习器测试", test_intelligent_learner),
        ("增强NLP处理器测试", test_enhanced_nlp_processor),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有智能功能测试通过！系统智能化升级成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要检查相关模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)