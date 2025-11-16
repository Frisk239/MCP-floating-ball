#!/usr/bin/env python3
"""快速测试意图识别"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_intent_recognition():
    try:
        from src.assistant.nlp_processor import NLPProcessor

        nlp = NLPProcessor()

        test_commands = [
            "打开记事本",
            "截图",
            "系统信息",
            "搜索Python教程",
            "识别文字"
        ]

        print("🔄 测试意图识别...")
        for cmd in test_commands:
            try:
                result = nlp.parse_command(cmd)
                if result.get("success") and result.get("commands"):
                    command_obj = result["commands"][0]
                    intent_name = command_obj.intent_type.name
                    confidence = command_obj.confidence
                    print(f"✅ '{cmd}' -> {intent_name} (置信度: {confidence:.2f})")
                else:
                    print(f"❌ '{cmd}' -> 解析失败")
            except Exception as e:
                print(f"❌ '{cmd}' -> 错误: {e}")

        print("🎉 意图识别测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intent_recognition()