"""
MCP Floating Ball - 数据库管理器

提供SQLite数据库操作，用于存储用户模式、系统实体映射和命令历史。
"""

import sqlite3
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from src.core.logging import get_logger
from src.core.exceptions import MCPFloatingBallError

logger = get_logger("database")


class DatabaseError(MCPFloatingBallError):
    """数据库操作错误"""
    pass


class DatabaseManager:
    """SQLite数据库管理器"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库管理器

        Args:
            db_path: 数据库文件路径，默认为项目根目录下的data/mcp_floating_ball.db
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent.parent / "data" / "mcp_floating_ball.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self.logger = get_logger(self.__class__.__name__)

        # 初始化数据库
        self._initialize_database()
        self.logger.info(f"数据库管理器初始化完成: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地数据库连接"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row  # 使结果可以按列名访问
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
        return self._local.connection

    def _initialize_database(self):
        """初始化数据库表结构"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 用户模式表 - 存储用户使用习惯和偏好
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    pattern_type TEXT NOT NULL,  -- 'command_preference', 'time_preference', 'context_preference'
                    pattern_data TEXT NOT NULL,  -- JSON格式存储模式数据
                    frequency INTEGER DEFAULT 1,  -- 使用频率
                    confidence REAL DEFAULT 0.5,  -- 置信度
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, pattern_type, pattern_data)
                )
            """)

            # 系统实体映射表 - 存储应用程序、网站、系统命令的映射
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,  -- 'application', 'website', 'system_command', 'file_type'
                    entity_name TEXT NOT NULL,  -- 实体名称（如"记事本"、"百度"）
                    entity_data TEXT NOT NULL,  -- JSON格式存储实体数据
                    aliases TEXT,  -- 别名列表，JSON格式
                    category TEXT,  -- 分类（如"办公"、"娱乐"、"开发"）
                    tags TEXT,  -- 标签，JSON格式
                    popularity INTEGER DEFAULT 0,  -- 受欢迎程度
                    success_rate REAL DEFAULT 1.0,  -- 成功率
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entity_type, entity_name)
                )
            """)

            # 命令历史表 - 记录所有命令执行历史
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT DEFAULT 'default',
                    original_command TEXT NOT NULL,  -- 原始命令
                    intent_type TEXT NOT NULL,  -- 识别的意图类型
                    intent_confidence REAL,  -- 意图置信度
                    parameters TEXT,  -- 参数，JSON格式
                    tool_name TEXT,  -- 使用的工具名称
                    execution_time REAL,  -- 执行时间（秒）
                    success BOOLEAN DEFAULT TRUE,  -- 是否成功
                    error_message TEXT,  -- 错误信息
                    context_data TEXT,  -- 上下文数据，JSON格式
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 为命令历史表创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp
                ON command_history (user_id, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_intent_type
                ON command_history (intent_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON command_history (session_id)
            """)

            # 学习数据表 - 存储机器学习相关的数据
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default',
                    learning_type TEXT NOT NULL,  -- 'intent_correction', 'parameter_optimization', 'sequence_learning'
                    input_data TEXT NOT NULL,  -- 输入数据，JSON格式
                    output_data TEXT NOT NULL,  -- 输出数据，JSON格式
                    feedback_score REAL,  -- 反馈评分
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 为学习数据表创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_learning
                ON learning_data (user_id, learning_type)
            """)

            # 系统配置表 - 存储系统级配置
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,
                    config_value TEXT NOT NULL,
                    config_type TEXT DEFAULT 'string',  -- 'string', 'integer', 'real', 'boolean', 'json'
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建触发器：自动更新updated_at字段
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_user_patterns_timestamp
                AFTER UPDATE ON user_patterns
                FOR EACH ROW
                BEGIN
                    UPDATE user_patterns SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_system_entities_timestamp
                AFTER UPDATE ON system_entities
                FOR EACH ROW
                BEGIN
                    UPDATE system_entities SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_system_config_timestamp
                AFTER UPDATE ON system_config
                FOR EACH ROW
                BEGIN
                    UPDATE system_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)

            conn.commit()
            self.logger.info("数据库表结构初始化完成")

            # 初始化默认数据
            self._initialize_default_data()

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"数据库初始化失败: {e}")

    def _initialize_default_data(self):
        """初始化默认数据"""
        try:
            self._initialize_default_entities()
            self._initialize_default_config()
        except Exception as e:
            logger.warning(f"初始化默认数据失败: {e}")

    def _initialize_default_entities(self):
        """初始化默认系统实体"""
        default_entities = [
            # 应用程序
            {
                "entity_type": "application",
                "entity_name": "记事本",
                "entity_data": json.dumps({
                    "executable": "notepad.exe",
                    "path": "C:\\Windows\\System32\\notepad.exe",
                    "description": "Windows记事本程序"
                }),
                "aliases": json.dumps(["notepad", "笔记本", "文本编辑器"]),
                "category": "办公",
                "tags": json.dumps(["文本编辑", "简单", "系统自带"])
            },
            {
                "entity_type": "application",
                "entity_name": "计算器",
                "entity_data": json.dumps({
                    "executable": "calc.exe",
                    "path": "C:\\Windows\\System32\\calc.exe",
                    "description": "Windows计算器程序"
                }),
                "aliases": json.dumps(["calculator", "calc"]),
                "category": "工具",
                "tags": json.dumps(["计算", "数学", "系统自带"])
            },
            {
                "entity_type": "application",
                "entity_name": "画图",
                "entity_data": json.dumps({
                    "executable": "mspaint.exe",
                    "path": "C:\\Windows\\System32\\mspaint.exe",
                    "description": "Windows画图程序"
                }),
                "aliases": json.dumps(["paint", "绘图", "mspaint"]),
                "category": "创意",
                "tags": json.dumps(["绘图", "图像编辑", "简单"])
            },
            # 网站
            {
                "entity_type": "website",
                "entity_name": "百度",
                "entity_data": json.dumps({
                    "url": "https://www.baidu.com",
                    "description": "百度搜索引擎",
                    "search_url": "https://www.baidu.com/s?wd={query}"
                }),
                "aliases": json.dumps(["baidu", "百度搜索"]),
                "category": "搜索",
                "tags": json.dumps(["搜索", "中文", "国内"])
            },
            {
                "entity_type": "website",
                "entity_name": "谷歌",
                "entity_data": json.dumps({
                    "url": "https://www.google.com",
                    "description": "谷歌搜索引擎",
                    "search_url": "https://www.google.com/search?q={query}"
                }),
                "aliases": json.dumps(["google", "谷歌搜索", "gg"]),
                "category": "搜索",
                "tags": json.dumps(["搜索", "英文", "国际"])
            },
            {
                "entity_type": "website",
                "entity_name": "知乎",
                "entity_data": json.dumps({
                    "url": "https://www.zhihu.com",
                    "description": "知乎问答社区"
                }),
                "aliases": json.dumps(["zhihu"]),
                "category": "社区",
                "tags": json.dumps(["问答", "知识", "中文"])
            }
        ]

        for entity in default_entities:
            self.add_entity(**entity)

    def _initialize_default_config(self):
        """初始化默认系统配置"""
        default_configs = [
            {
                "config_key": "database_version",
                "config_value": "1.0.0",
                "config_type": "string",
                "description": "数据库版本号"
            },
            {
                "config_key": "learning_enabled",
                "config_value": "true",
                "config_type": "boolean",
                "description": "是否启用学习功能"
            },
            {
                "config_key": "pattern_threshold",
                "config_value": "3",
                "config_type": "integer",
                "description": "模式识别阈值"
            }
        ]

        for config in default_configs:
            self.set_config(config["config_key"], config["config_value"],
                          config["config_type"], config["description"])

    # ==================== 用户模式管理 ====================

    def add_user_pattern(self, user_id: str, pattern_type: str, pattern_data: Dict[str, Any],
                        frequency: int = 1, confidence: float = 0.5) -> bool:
        """添加用户模式"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO user_patterns
                (user_id, pattern_type, pattern_data, frequency, confidence, last_used)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                user_id, pattern_type, json.dumps(pattern_data), frequency, confidence
            ))

            conn.commit()
            self.logger.debug(f"用户模式已添加: {user_id} - {pattern_type}")
            return True

        except Exception as e:
            logger.error(f"添加用户模式失败: {e}")
            return False

    def get_user_patterns(self, user_id: str = "default", pattern_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取用户模式"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM user_patterns WHERE user_id = ?"
            params = [user_id]

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            query += " ORDER BY frequency DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "pattern_type": row["pattern_type"],
                    "pattern_data": json.loads(row["pattern_data"]),
                    "frequency": row["frequency"],
                    "confidence": row["confidence"],
                    "last_used": row["last_used"],
                    "created_at": row["created_at"]
                })

            return result

        except Exception as e:
            logger.error(f"获取用户模式失败: {e}")
            return []

    def update_user_pattern_frequency(self, pattern_id: int, increment: int = 1) -> bool:
        """更新用户模式使用频率"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_patterns
                SET frequency = frequency + ?, last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (increment, pattern_id))

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"更新用户模式频率失败: {e}")
            return False

    # ==================== 系统实体管理 ====================

    def add_entity(self, entity_type: str, entity_name: str, entity_data: Dict[str, Any],
                  aliases: Optional[List[str]] = None, category: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> bool:
        """添加系统实体"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO system_entities
                (entity_type, entity_name, entity_data, aliases, category, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_type, entity_name, json.dumps(entity_data),
                json.dumps(aliases or []), category, json.dumps(tags or [])
            ))

            conn.commit()
            self.logger.debug(f"系统实体已添加: {entity_type} - {entity_name}")
            return True

        except Exception as e:
            logger.error(f"添加系统实体失败: {e}")
            return False

    def get_entities(self, entity_type: Optional[str] = None, category: Optional[str] = None,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        """获取系统实体"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM system_entities WHERE 1=1"
            params = []

            if entity_type:
                query += " AND entity_type = ?"
                params.append(entity_type)

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY popularity DESC, success_rate DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "entity_type": row["entity_type"],
                    "entity_name": row["entity_name"],
                    "entity_data": json.loads(row["entity_data"]),
                    "aliases": json.loads(row["aliases"]),
                    "category": row["category"],
                    "tags": json.loads(row["tags"]),
                    "popularity": row["popularity"],
                    "success_rate": row["success_rate"],
                    "last_used": row["last_used"],
                    "created_at": row["created_at"]
                })

            return result

        except Exception as e:
            logger.error(f"获取系统实体失败: {e}")
            return []

    def search_entity(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索系统实体"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            base_query = """
                SELECT *,
                       CASE
                           WHEN entity_name LIKE ? THEN 3
                           WHEN aliases LIKE ? THEN 2
                           WHEN tags LIKE ? THEN 1
                           ELSE 0
                       END as match_score
                FROM system_entities
                WHERE (entity_name LIKE ? OR aliases LIKE ? OR tags LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"]

            if entity_type:
                base_query += " AND entity_type = ?"
                params.append(entity_type)

            base_query += " ORDER BY match_score DESC, popularity DESC LIMIT 20"

            cursor.execute(base_query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                if row["match_score"] > 0:  # 只返回有匹配的结果
                    result.append({
                        "id": row["id"],
                        "entity_type": row["entity_type"],
                        "entity_name": row["entity_name"],
                        "entity_data": json.loads(row["entity_data"]),
                        "aliases": json.loads(row["aliases"]),
                        "category": row["category"],
                        "tags": json.loads(row["tags"]),
                        "popularity": row["popularity"],
                        "success_rate": row["success_rate"],
                        "match_score": row["match_score"]
                    })

            return result

        except Exception as e:
            logger.error(f"搜索系统实体失败: {e}")
            return []

    def update_entity_popularity(self, entity_id: int, success: bool = True) -> bool:
        """更新实体受欢迎程度和成功率"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 先获取当前的受欢迎程度
            cursor.execute("SELECT popularity FROM system_entities WHERE id = ?", (entity_id,))
            result = cursor.fetchone()

            if not result:
                return False

            current_popularity = result[0] or 0

            # 更新受欢迎程度
            cursor.execute("""
                UPDATE system_entities
                SET popularity = popularity + 1,
                    success_rate = CASE
                        WHEN ? THEN (success_rate * ? + 1) / (popularity + 1)
                        ELSE (success_rate * ?) / (popularity + 1)
                    END,
                    last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (success, current_popularity, current_popularity, entity_id))

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"更新实体受欢迎程度失败: {e}")
            return False

    # ==================== 命令历史管理 ====================

    def add_command_history(self, session_id: str, user_id: str, original_command: str,
                           intent_type: str, intent_confidence: float, parameters: Dict[str, Any],
                           tool_name: str, execution_time: float, success: bool = True,
                           error_message: Optional[str] = None, context_data: Optional[Dict[str, Any]] = None) -> bool:
        """添加命令历史记录"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO command_history
                (session_id, user_id, original_command, intent_type, intent_confidence,
                 parameters, tool_name, execution_time, success, error_message, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, user_id, original_command, intent_type, intent_confidence,
                json.dumps(parameters), tool_name, execution_time, success, error_message,
                json.dumps(context_data or {})
            ))

            conn.commit()
            self.logger.debug(f"命令历史已添加: {original_command} -> {intent_type}")
            return True

        except Exception as e:
            logger.error(f"添加命令历史失败: {e}")
            return False

    def get_command_history(self, user_id: str = "default", limit: int = 1000,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取命令历史"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM command_history WHERE user_id = ?"
            params = [user_id]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "original_command": row["original_command"],
                    "intent_type": row["intent_type"],
                    "intent_confidence": row["intent_confidence"],
                    "parameters": json.loads(row["parameters"]),
                    "tool_name": row["tool_name"],
                    "execution_time": row["execution_time"],
                    "success": row["success"],
                    "error_message": row["error_message"],
                    "context_data": json.loads(row["context_data"]),
                    "timestamp": row["timestamp"]
                })

            return result

        except Exception as e:
            logger.error(f"获取命令历史失败: {e}")
            return []

    def get_command_statistics(self, user_id: str = "default", days: int = 7) -> Dict[str, Any]:
        """获取命令使用统计"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            start_date = datetime.now() - timedelta(days=days)

            # 总体统计
            cursor.execute("""
                SELECT
                    COUNT(*) as total_commands,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_commands,
                    AVG(execution_time) as avg_execution_time,
                    AVG(intent_confidence) as avg_confidence
                FROM command_history
                WHERE user_id = ? AND timestamp >= ?
            """, (user_id, start_date))

            overall_stats = cursor.fetchone()

            # 意图类型统计
            cursor.execute("""
                SELECT intent_type, COUNT(*) as count,
                       AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                FROM command_history
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY intent_type
                ORDER BY count DESC
            """, (user_id, start_date))

            intent_stats = []
            for row in cursor.fetchall():
                intent_stats.append({
                    "intent_type": row["intent_type"],
                    "count": row["count"],
                    "success_rate": row["success_rate"]
                })

            # 每日使用趋势
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as commands
                FROM command_history
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (user_id, start_date))

            daily_trend = []
            for row in cursor.fetchall():
                daily_trend.append({
                    "date": row["date"],
                    "commands": row["commands"]
                })

            return {
                "period_days": days,
                "total_commands": overall_stats["total_commands"] or 0,
                "successful_commands": overall_stats["successful_commands"] or 0,
                "success_rate": (overall_stats["successful_commands"] / overall_stats["total_commands"]) if overall_stats["total_commands"] > 0 else 0,
                "avg_execution_time": overall_stats["avg_execution_time"] or 0,
                "avg_confidence": overall_stats["avg_confidence"] or 0,
                "intent_statistics": intent_stats,
                "daily_trend": daily_trend
            }

        except Exception as e:
            logger.error(f"获取命令统计失败: {e}")
            return {}

    # ==================== 学习数据管理 ====================

    def add_learning_data(self, user_id: str, learning_type: str, input_data: Dict[str, Any],
                         output_data: Dict[str, Any], feedback_score: Optional[float] = None) -> bool:
        """添加学习数据"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO learning_data
                (user_id, learning_type, input_data, output_data, feedback_score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id, learning_type, json.dumps(input_data),
                json.dumps(output_data), feedback_score
            ))

            conn.commit()
            self.logger.debug(f"学习数据已添加: {user_id} - {learning_type}")
            return True

        except Exception as e:
            logger.error(f"添加学习数据失败: {e}")
            return False

    def get_learning_data(self, user_id: str = "default", learning_type: Optional[str] = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """获取学习数据"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM learning_data WHERE user_id = ?"
            params = [user_id]

            if learning_type:
                query += " AND learning_type = ?"
                params.append(learning_type)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "learning_type": row["learning_type"],
                    "input_data": json.loads(row["input_data"]),
                    "output_data": json.loads(row["output_data"]),
                    "feedback_score": row["feedback_score"],
                    "created_at": row["created_at"]
                })

            return result

        except Exception as e:
            logger.error(f"获取学习数据失败: {e}")
            return []

    # ==================== 系统配置管理 ====================

    def set_config(self, key: str, value: str, config_type: str = "string",
                  description: Optional[str] = None) -> bool:
        """设置系统配置"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO system_config (config_key, config_value, config_type, description)
                VALUES (?, ?, ?, ?)
            """, (key, value, config_type, description))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"设置系统配置失败: {e}")
            return False

    def get_config(self, key: str, default_value: Any = None) -> Any:
        """获取系统配置"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT config_value, config_type FROM system_config WHERE config_key = ?", (key,))
            row = cursor.fetchone()

            if row:
                value = row["config_value"]
                config_type = row["config_type"]

                # 根据类型转换值
                if config_type == "integer":
                    return int(value)
                elif config_type == "real":
                    return float(value)
                elif config_type == "boolean":
                    return value.lower() in ("true", "1", "yes", "on")
                elif config_type == "json":
                    return json.loads(value)
                else:
                    return value
            else:
                return default_value

        except Exception as e:
            logger.error(f"获取系统配置失败: {e}")
            return default_value

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有系统配置"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM system_config")
            rows = cursor.fetchall()

            result = {}
            for row in rows:
                key = row["config_key"]
                value = self.get_config(key)
                result[key] = value

            return result

        except Exception as e:
            logger.error(f"获取所有系统配置失败: {e}")
            return {}

    # ==================== 数据库维护 ====================

    def cleanup_old_data(self, days: int = 90) -> bool:
        """清理旧数据"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            # 清理旧的命令历史
            cursor.execute("DELETE FROM command_history WHERE timestamp < ?", (cutoff_date,))
            deleted_history = cursor.rowcount

            # 清理旧的学习数据
            cursor.execute("DELETE FROM learning_data WHERE created_at < ?", (cutoff_date,))
            deleted_learning = cursor.rowcount

            conn.commit()
            self.logger.info(f"清理完成: 删除了 {deleted_history} 条历史记录和 {deleted_learning} 条学习记录")
            return True

        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            return False

    def vacuum_database(self) -> bool:
        """压缩数据库"""
        try:
            conn = self._get_connection()
            conn.execute("VACUUM")
            self.logger.info("数据库压缩完成")
            return True

        except Exception as e:
            logger.error(f"数据库压缩失败: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            stats = {}

            # 表大小统计
            tables = ["user_patterns", "system_entities", "command_history", "learning_data", "system_config"]
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()["count"]

            # 数据库文件大小
            if self.db_path.exists():
                stats["file_size_bytes"] = self.db_path.stat().st_size
                stats["file_size_mb"] = round(stats["file_size_bytes"] / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return {}

    def close(self):
        """关闭数据库连接"""
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                delattr(self._local, 'connection')
                self.logger.debug("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")


# 全局数据库管理器实例
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


def get_database() -> DatabaseManager:
    """获取数据库管理器实例（别名）"""
    return get_database_manager()


def close_database():
    """关闭全局数据库管理器"""
    global _database_manager
    if _database_manager:
        _database_manager.close()
        _database_manager = None


# 导出
__all__ = [
    "DatabaseManager",
    "DatabaseError",
    "get_database_manager",
    "get_database",
    "close_database"
]