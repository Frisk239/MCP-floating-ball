"""
修复导入问题的脚本
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

def check_file(filepath):
    """检查文件是否存在导入问题"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"检查文件: {filepath}")

        # 检查是否有未定义的类引用
        imports_to_add = []

        if 'AIModelResponse' in content and 'class AIModelResponse' not in content:
            imports_to_add.append('AIModelResponse')

        if 'FusionStrategy' in content and 'class FusionStrategy' not in content:
            imports_to_add.append('FusionStrategy')

        if 'TaskRequirements' in content and 'class TaskRequirements' not in content:
            imports_to_add.append('TaskRequirements')

        if imports_to_add:
            print(f"  需要添加: {imports_to_add}")
        else:
            print("  没有发现明显的导入问题")

        return imports_to_add

    except Exception as e:
        print(f"  检查失败: {e}")
        return []

def fix_ai_orchestrator():
    """修复 ai_orchestrator.py"""
    filepath = 'src/assistant/ai_orchestrator.py'

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否需要添加缺失的类定义
        missing_classes = []

        if 'class AIModelResponse' not in content and 'AIModelResponse' in content:
            missing_classes.append('AIModelResponse')

        if 'class FusionStrategy' not in content and 'FusionStrategy' in content:
            missing_classes.append('FusionStrategy')

        if 'class TaskRequirements' not in content and 'TaskRequirements' in content:
            missing_classes.append('TaskRequirements')

        if missing_classes:
            print(f"在 {filepath} 中需要添加以下类定义: {missing_classes}")

            # 在文件开头添加缺失的类定义
            new_content = content

            # 添加缺失的导入
            if 'from enum import Enum' not in new_content:
                new_content = 'from enum import Enum\nfrom typing import Dict, List, Any, Optional\nfrom dataclasses import dataclass, field\nfrom datetime import datetime\n\n' + new_content

            # 添加缺失的类定义
            classes_to_add = []

            if 'AIModelResponse' in missing_classes:
                classes_to_add.append('''
@dataclass
class AIModelResponse:
    """AI模型响应"""
    model_id: str
    model_output: str
    confidence: Optional[float] = None
    execution_time: Optional[float] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
''')

            if 'FusionStrategy' in missing_classes:
                classes_to_add.append('''
class FusionStrategy(Enum):
    """融合策略"""
    CONFIDENCE_BASED = "confidence_based"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    EXPERT_WEIGHTED = "expert_weighted"
    CONDORCET = "condorcet"
    BORDA_COUNT = "borda_count"
    DEMPSTER_SHAFER = "dempster_shafer"
    BAYESIAN_FUSION = "bayesian_fusion"
    NEURAL_ENSEMBLE = "neural_ensemble"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    CONFLICT_RESOLUTION = "conflict_resolution"
    HIERARCHICAL_FUSION = "hierarchical_fusion"
''')

            if 'TaskRequirements' in missing_classes:
                classes_to_add.append('''
@dataclass
class TaskRequirements:
    """任务需求"""
    task_type: str
    complexity: str = "medium"
    priority: str = "normal"
    domain: Optional[str] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
''')

            # 在第一个dataclass之前插入这些类定义
            lines = new_content.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('@dataclass'):
                    insert_index = i
                    break

            if insert_index > 0:
                # 插入类定义
                class_definitions = '\n'.join(classes_to_add)
                lines.insert(insert_index, class_definitions)
                new_content = '\n'.join(lines)

            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ 已修复 {filepath}")
        else:
            print(f"✅ {filepath} 不需要修复")

    except Exception as e:
        print(f"❌ 修复 {filepath} 失败: {e}")

def main():
    """主函数"""
    print("🔧 MCP Floating Ball - 修复导入问题")
    print("=" * 50)

    # 检查主要文件
    files_to_check = [
        'src/assistant/ai_orchestrator.py',
        'src/assistant/model_fusion.py',
        'src/assistant/task_dispatcher.py',
        'src/assistant/ab_testing.py'
    ]

    for filepath in files_to_check:
        if os.path.exists(filepath):
            missing_imports = check_file(filepath)
            if missing_imports and 'ai_orchestrator.py' in filepath:
                fix_ai_orchestrator()
        else:
            print(f"⚠️  文件不存在: {filepath}")

    print("\n🎯 修复完成！请重新运行演示脚本。")

if __name__ == "__main__":
    main()