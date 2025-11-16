"""
MCP Floating Ball - 预定义智能工作流

提供一系列开箱即用的智能工作流模板。
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.core.workflow import (
    WorkflowDefinition, WorkflowStage, WorkflowTask, WorkflowTrigger,
    TriggerType, get_workflow_engine
)


class PredefinedWorkflows:
    """预定义工作流集合"""

    @staticmethod
    def create_smart_document_processor() -> WorkflowDefinition:
        """智能文档处理工作流"""
        return WorkflowDefinition(
            id="smart_document_processor",
            name="智能文档处理",
            description="自动捕获屏幕内容，识别文字，并进行智能分析和处理",
            tags=["文档", "OCR", "智能分析"],
            version="1.0.0",
            author="system",
            variables={
                "save_to_file": True,
                "output_format": "markdown",
                "language": "zh-CN"
            },
            triggers=[
                WorkflowTrigger(
                    trigger_type=TriggerType.VOICE_COMMAND,
                    config={"patterns": ["处理文档", "分析文档", "智能处理"]}
                ),
                WorkflowTrigger(
                    trigger_type=TriggerType.MANUAL,
                    config={}
                )
            ],
            stages=[
                WorkflowStage(
                    id="capture_screen",
                    name="屏幕捕获",
                    tasks=[
                        WorkflowTask(
                            id="screen_shot",
                            name="截取屏幕",
                            tool_name="screen_capture",
                            parameters={"save_to_file": "{{variables.save_to_file}}"},
                            timeout=10.0
                        )
                    ]
                ),
                WorkflowStage(
                    id="text_recognition",
                    name="文字识别",
                    depends_on=["capture_screen"],
                    tasks=[
                        WorkflowTask(
                            id="ocr_process",
                            name="OCR识别",
                            tool_name="ocr_engine",
                            parameters={
                                "image_path": "{{result.screen_shot.result.file_path}}",
                                "language": "{{variables.language}}"
                            },
                            retry_policy={"max_attempts": 3, "delay": 2}
                        )
                    ]
                ),
                WorkflowStage(
                    id="content_analysis",
                    name="内容分析",
                    depends_on=["text_recognition"],
                    tasks=[
                        WorkflowTask(
                            id="analyze_text",
                            name="文本分析",
                            tool_name="text_operations",
                            parameters={
                                "operation": "analyze",
                                "text": "{{result.ocr_process.result.text}}",
                                "analysis_type": "comprehensive"
                            }
                        ),
                        WorkflowTask(
                            id="extract_keywords",
                            name="提取关键词",
                            tool_name="text_operations",
                            parameters={
                                "operation": "extract_keywords",
                                "text": "{{result.ocr_process.result.text}}",
                                "max_keywords": 10
                            },
                            parallel_group="analysis"
                        ),
                        WorkflowTask(
                            id="generate_summary",
                            name="生成摘要",
                            tool_name="text_operations",
                            parameters={
                                "operation": "summarize",
                                "text": "{{result.ocr_process.result.text}}",
                                "max_length": 200
                            },
                            parallel_group="analysis"
                        )
                    ],
                    parallel_execution=True
                ),
                WorkflowStage(
                    id="format_output",
                    name="格式化输出",
                    depends_on=["content_analysis"],
                    conditions=[
                        {"type": "variable_equals", "variable": "save_to_file", "value": True}
                    ],
                    tasks=[
                        WorkflowTask(
                            id="format_markdown",
                            name="格式化为Markdown",
                            tool_name="text_operations",
                            parameters={
                                "operation": "format",
                                "text": "{{result.analyze_text.result.processed_text}}",
                                "format": "markdown",
                                "include_metadata": True
                            }
                        ),
                        WorkflowTask(
                            id="save_file",
                            name="保存文件",
                            tool_name="text_operations",
                            parameters={
                                "operation": "save",
                                "content": "{{result.format_markdown.result.formatted_text}}",
                                "filename": f"processed_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            }
                        )
                    ]
                )
            ],
            error_handling={
                "retry_policy": {"max_attempts": 2, "delay": 5},
                "fallback_actions": ["notify_user", "save_error_log"]
            }
        )

    @staticmethod
    def create_smart_web_researcher() -> WorkflowDefinition:
        """智能网络研究工作流"""
        return WorkflowDefinition(
            id="smart_web_researcher",
            name="智能网络研究",
            description="基于查询内容进行深度网络搜索，收集、整理和分析信息",
            tags=["搜索", "研究", "信息收集"],
            version="1.0.0",
            author="system",
            variables={
                "max_results": 10,
                "include_images": True,
                "analysis_depth": "comprehensive"
            },
            triggers=[
                WorkflowTrigger(
                    trigger_type=TriggerType.VOICE_COMMAND,
                    config={"patterns": ["搜索", "研究", "查找资料"]}
                )
            ],
            stages=[
                WorkflowStage(
                    id="search_initiation",
                    name="搜索启动",
                    tasks=[
                        WorkflowTask(
                            id="parse_query",
                            name="解析查询",
                            tool_name="text_operations",
                            parameters={
                                "operation": "parse_query",
                                "text": "{{variables.query}}",
                                "extract_entities": True
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="web_search",
                    name="网络搜索",
                    depends_on=["search_initiation"],
                    tasks=[
                        WorkflowTask(
                            id="search_web",
                            name="网页搜索",
                            tool_name="multi_search",
                            parameters={
                                "query": "{{result.parse_query.result.cleaned_query}}",
                                "max_results": "{{variables.max_results}}",
                                "search_engines": ["baidu", "google", "bing"],
                                "include_images": "{{variables.include_images}}"
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="content_collection",
                    name="内容收集",
                    depends_on=["web_search"],
                    tasks=[
                        WorkflowTask(
                            id="scrape_pages",
                            name="抓取页面",
                            tool_name="web_scraper",
                            parameters={
                                "urls": "{{result.search_web.result.urls[:5]}}",  # 抓取前5个结果
                                "extract_main_content": True,
                                "remove_ads": True
                            },
                            retry_policy={"max_attempts": 3, "delay": 1}
                        )
                    ]
                ),
                WorkflowStage(
                    id="information_analysis",
                    name="信息分析",
                    depends_on=["content_collection"],
                    tasks=[
                        WorkflowTask(
                            id="extract_key_info",
                            name="提取关键信息",
                            tool_name="text_operations",
                            parameters={
                                "operation": "extract_information",
                                "content": "{{result.scrape_pages.result.content}}",
                                "analysis_depth": "{{variables.analysis_depth}}"
                            }
                        ),
                        WorkflowTask(
                            id="verify_sources",
                            name="验证信源",
                            tool_name="text_operations",
                            parameters={
                                "operation": "verify_sources",
                                "sources": "{{result.search_web.result.urls}}",
                                "content": "{{result.scrape_pages.result.content}}"
                            },
                            parallel_group="analysis"
                        ),
                        WorkflowTask(
                            id="sentiment_analysis",
                            name="情感分析",
                            tool_name="text_operations",
                            parameters={
                                "operation": "sentiment_analysis",
                                "content": "{{result.scrape_pages.result.content}}"
                            },
                            parallel_group="analysis"
                        )
                    ],
                    parallel_execution=True
                ),
                WorkflowStage(
                    id="generate_report",
                    name="生成报告",
                    depends_on=["information_analysis"],
                    tasks=[
                        WorkflowTask(
                            id="compile_report",
                            name="编译报告",
                            tool_name="text_operations",
                            parameters={
                                "operation": "compile_report",
                                "search_query": "{{variables.query}}",
                                "key_information": "{{result.extract_key_info.result.information}}",
                                "sources": "{{result.search_web.result.urls}}",
                                "sentiment": "{{result.sentiment_analysis.result.sentiment}}",
                                "format": "markdown"
                            }
                        ),
                        WorkflowTask(
                            id="save_report",
                            name="保存报告",
                            tool_name="text_operations",
                            parameters={
                                "operation": "save",
                                "content": "{{result.compile_report.result.report}}",
                                "filename": f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            }
                        )
                    ]
                )
            ]
        )

    @staticmethod
    def create_system_maintenance() -> WorkflowDefinition:
        """系统维护工作流"""
        return WorkflowDefinition(
            id="system_maintenance",
            name="系统维护",
            description="定期执行系统清理、优化和健康检查",
            tags=["维护", "优化", "健康检查"],
            version="1.0.0",
            author="system",
            triggers=[
                WorkflowTrigger(
                    trigger_type=TriggerType.SCHEDULE,
                    config={"cron": "0 2 * * 0"}  # 每周日凌晨2点
                )
            ],
            stages=[
                WorkflowStage(
                    id="system_info",
                    name="系统信息收集",
                    tasks=[
                        WorkflowTask(
                            id="get_system_info",
                            name="获取系统信息",
                            tool_name="system_info",
                            parameters={"info_type": "detailed"}
                        )
                    ]
                ),
                WorkflowStage(
                    id="cleanup_operations",
                    name="清理操作",
                    depends_on=["system_info"],
                    tasks=[
                        WorkflowTask(
                            id="cleanup_temp_files",
                            name="清理临时文件",
                            tool_name="file_operations",
                            parameters={
                                "operation": "cleanup_temp",
                                "older_than_days": 7
                            }
                        ),
                        WorkflowTask(
                            id="cleanup_logs",
                            name="清理日志",
                            tool_name="file_operations",
                            parameters={
                                "operation": "cleanup_logs",
                                "older_than_days": 30,
                                "keep_error_logs": True
                            },
                            parallel_group="cleanup"
                        ),
                        WorkflowTask(
                            id="database_maintenance",
                            name="数据库维护",
                            tool_name="database_operations",
                            parameters={
                                "operation": "maintenance",
                                "vacuum": True,
                                "optimize": True
                            },
                            parallel_group="cleanup"
                        )
                    ],
                    parallel_execution=True
                ),
                WorkflowStage(
                    id="health_check",
                    name="健康检查",
                    depends_on=["cleanup_operations"],
                    tasks=[
                        WorkflowTask(
                            id="check_disk_space",
                            name="检查磁盘空间",
                            tool_name="system_info",
                            parameters={"check_type": "disk_space"}
                        ),
                        WorkflowTask(
                            id="check_memory_usage",
                            name="检查内存使用",
                            tool_name="system_info",
                            parameters={"check_type": "memory"}
                        ),
                        WorkflowTask(
                            id="check_services",
                            name="检查服务状态",
                            tool_name="system_info",
                            parameters={"check_type": "services"}
                        )
                    ],
                    parallel_execution=True
                ),
                WorkflowStage(
                    id="optimization",
                    name="系统优化",
                    depends_on=["health_check"],
                    conditions=[
                        {"type": "task_success", "task_id": "check_disk_space"}
                    ],
                    tasks=[
                        WorkflowTask(
                            id="optimize_startup",
                            name="优化启动项",
                            tool_name="system_info",
                            parameters={
                                "operation": "optimize_startup",
                                "disable_unnecessary": True
                            }
                        ),
                        WorkflowTask(
                            id="defragment_disks",
                            name="磁盘整理",
                            tool_name="system_info",
                            parameters={
                                "operation": "defragment",
                                "condition": "{{result.check_disk_space.result.fragmentation > 10}}"
                            }
                        )
                    ]
                )
            ]
        )

    @staticmethod
    def create_voice_assistant_automation() -> WorkflowDefinition:
        """语音助手自动化工作流"""
        return WorkflowDefinition(
            id="voice_assistant_automation",
            name="语音助手自动化",
            description="智能语音助手，支持多轮对话和任务自动化",
            tags=["语音", "助手", "自动化"],
            version="1.0.0",
            author="system",
            triggers=[
                WorkflowTrigger(
                    trigger_type=TriggerType.VOICE_COMMAND,
                    config={"patterns": ["助手", "AI助手", "智能助手"]}
                )
            ],
            stages=[
                WorkflowStage(
                    id="voice_input",
                    name="语音输入",
                    tasks=[
                        WorkflowTask(
                            id="activate_voice",
                            name="激活语音识别",
                            tool_name="voice_activation",
                            parameters={
                                "wake_word": "助手",
                                "timeout": 10,
                                "continuous": True
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="speech_recognition",
                    name="语音识别",
                    depends_on=["voice_input"],
                    tasks=[
                        WorkflowTask(
                            id="recognize_speech",
                            name="识别语音",
                            tool_name="asr_engine",
                            parameters={
                                "audio_data": "{{result.activate_voice.result.audio_data}}",
                                "language": "zh-CN",
                                "enhanced": True
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="intent_processing",
                    name="意图处理",
                    depends_on=["speech_recognition"],
                    tasks=[
                        WorkflowTask(
                            id="parse_command",
                            name="解析命令",
                            tool_name="nlp_processor",
                            parameters={
                                "text": "{{result.recognize_speech.result.text}}",
                                "context": {"conversation_id": "{{variables.conversation_id}}"}
                            }
                        ),
                        WorkflowTask(
                            id="get_personalized_suggestions",
                            name="获取个性化建议",
                            tool_name="intelligent_learner",
                            parameters={
                                "current_command": "{{result.recognize_speech.result.text}}",
                                "context": "{{variables.conversation_context}}"
                            },
                            parallel_group="processing"
                        )
                    ],
                    parallel_execution=True
                ),
                WorkflowStage(
                    id="task_execution",
                    name="任务执行",
                    depends_on=["intent_processing"],
                    tasks=[
                        WorkflowTask(
                            id="execute_command",
                            name="执行命令",
                            tool_name="command_handler",
                            parameters={
                                "command": "{{result.recognize_speech.result.text}}",
                                "intent": "{{result.parse_command.result.intent}}",
                                "parameters": "{{result.parse_command.result.parameters}}"
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="response_generation",
                    name="响应生成",
                    depends_on=["task_execution"],
                    tasks=[
                        WorkflowTask(
                            id="generate_response",
                            name="生成响应",
                            tool_name="nlp_processor",
                            parameters={
                                "operation": "generate_response",
                                "command_result": "{{result.execute_command.result}}",
                                "suggestions": "{{result.get_personalized_suggestions.result}}",
                                "tone": "friendly"
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="voice_output",
                    name="语音输出",
                    depends_on=["response_generation"],
                    tasks=[
                        WorkflowTask(
                            id="text_to_speech",
                            name="文本转语音",
                            tool_name="tts_engine",
                            parameters={
                                "text": "{{result.generate_response.result.response_text}}",
                                "voice": "female",
                                "speed": 1.0,
                                "emotion": "neutral"
                            }
                        ),
                        WorkflowTask(
                            id="play_audio",
                            name="播放音频",
                            tool_name="audio_player",
                            parameters={
                                "audio_data": "{{result.text_to_speech.result.audio_data}}",
                                "volume": 0.8
                            }
                        )
                    ]
                ),
                WorkflowStage(
                    id="learning_update",
                    name="学习更新",
                    depends_on=["voice_output"],
                    tasks=[
                        WorkflowTask(
                            id="update_learning_model",
                            name="更新学习模型",
                            tool_name="intelligent_learner",
                            parameters={
                                "operation": "learn_from_interaction",
                                "original_command": "{{result.recognize_speech.result.text}}",
                                "intent_type": "{{result.parse_command.result.intent_type}}",
                                "execution_result": "{{result.execute_command.result}}",
                                "user_feedback": "{{variables.user_feedback}}"
                            }
                        )
                    ]
                )
            ]
        )

    @classmethod
    def register_all_workflows(cls):
        """注册所有预定义工作流"""
        engine = get_workflow_engine()

        workflows = [
            cls.create_smart_document_processor(),
            cls.create_smart_web_researcher(),
            cls.create_system_maintenance(),
            cls.create_voice_assistant_automation()
        ]

        for workflow in workflows:
            success = engine.register_definition(workflow)
            if success:
                print(f"✅ 工作流注册成功: {workflow.name} ({workflow.id})")
            else:
                print(f"❌ 工作流注册失败: {workflow.name} ({workflow.id})")

        return len(workflows)


def setup_predefined_workflows():
    """设置预定义工作流"""
    try:
        count = PredefinedWorkflows.register_all_workflows()
        print(f"🎉 预定义工作流设置完成，共注册 {count} 个工作流")
        return True
    except Exception as e:
        print(f"❌ 预定义工作流设置失败: {e}")
        return False


if __name__ == "__main__":
    # 测试工作流注册
    setup_predefined_workflows()