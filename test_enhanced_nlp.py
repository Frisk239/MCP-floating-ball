#!/usr/bin/env python3
"""
测试增强NLP处理器
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_nlp():
    """测试增强NLP处理器的意图识别"""
    try:
        from src.assistant.enhanced_nlp_processor import EnhancedNLPProcessor

        nlp = EnhancedNLPProcessor()

        # 测试用例
        test_cases = [
            "打开记事本",           # 应用启动
            "打开百度",             # 网页导航
            "打开百度搜索人工智能",   # 复合命令
            "搜索Python教程",       # 搜索
            "福州天气怎么样",       # 天气查询
            "截图",                # 截图
            "识别图片中的文字",     # OCR
            "查看系统信息",         # 系统信息
            "启动微信",             # 应用启动
            "访问谷歌",             # 网页导航
        ]

        print("🔄 测试增强NLP处理器...")
        print("=" * 80)

        for test_input in test_cases:
            try:
                print(f"\n📝 输入: {test_input}")
                result = nlp.parse_command(test_input)

                if result.get("success"):
                    commands = result.get("commands", [])
                    if commands:
                        cmd = commands[0]
                        hierarchical_intent = result.get("hierarchical_intent")

                        print(f"✅ 一级意图: {hierarchical_intent.primary.value}")
                        print(f"✅ 二级意图: {hierarchical_intent.secondary.value}")
                        print(f"✅ 动作: {hierarchical_intent.action}")
                        print(f"✅ 参数: {hierarchical_intent.parameters}")
                        print(f"✅ 置信度: {hierarchical_intent.confidence:.2f}")
                        print(f"✅ 映射意图: {cmd.intent_type.name}")
                    else:
                        print("❌ 没有解析到命令")
                else:
                    print(f"❌ 解析失败: {result.get('error')}")

            except Exception as e:
                print(f"❌ 测试出错: {e}")

        print("\n" + "=" * 80)
        print("🎉 增强NLP处理器测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_nlp()