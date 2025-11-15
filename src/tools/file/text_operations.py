"""
MCP Floating Ball - 文本操作工具

提供各种文本处理功能，包括读取、写入、搜索、替换、格式化等。
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import csv

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from ...core.logging import get_logger
from ...core.exceptions import ToolError
from ..base import BaseTool, ToolMetadata, ToolCategory, ParameterType, ToolParameter

logger = get_logger(__name__)


class TextOperationsTool(BaseTool):
    """文本操作工具"""

    def __init__(self):
        """初始化文本操作工具"""
        super().__init__()
        self.logger = get_logger("tool.text_operations")

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="text_operations",
            display_name="文本操作工具",
            description="文本操作工具，提供文件读写、文本搜索、替换、格式化等功能",
            category=ToolCategory.FILE,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["file", "text", "editor", "search", "replace"],
            parameters=[
                ToolParameter(
                    name="operation",
                    type=ParameterType.STRING,
                    description="操作类型",
                    required=True,
                    enum=["read", "write", "append", "search", "replace", "count", "format", "split", "merge", "validate", "extract", "convert", "statistics"]
                ),
                ToolParameter(
                    name="file_path",
                    type=ParameterType.STRING,
                    description="文件路径",
                    required=False
                ),
                ToolParameter(
                    name="content",
                    type=ParameterType.STRING,
                    description="文本内容",
                    required=False
                ),
                ToolParameter(
                    name="search_text",
                    type=ParameterType.STRING,
                    description="搜索文本",
                    required=False
                ),
                ToolParameter(
                    name="replace_text",
                    type=ParameterType.STRING,
                    description="替换文本",
                    required=False
                ),
                ToolParameter(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="文件编码",
                    required=False,
                    default="utf-8"
                )
            ],
            examples=["读取文件内容", "在文件中搜索文本", "替换文件中的文本", "统计文件字数"]
        )

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="operation",
                type=ParameterType.STRING.value,
                description="操作类型",
                required=True,
                choices=[
                    "read", "write", "append", "search", "replace",
                    "count", "format", "split", "merge", "validate",
                    "extract", "convert", "statistics"
                ],
                examples=["read", "search", "replace"]
            ),
            ToolParameter(
                name="file_path",
                type=ParameterType.STRING.value,
                description="文件路径",
                required=False,
                examples=["/path/to/file.txt", "C:\\Users\\document.txt"]
            ),
            ToolParameter(
                name="content",
                type=ParameterType.STRING.value,
                description="文本内容",
                required=False,
                examples=["Hello World", "这是要写入的内容"]
            ),
            ToolParameter(
                name="search_text",
                type=ParameterType.STRING.value,
                description="搜索文本",
                required=False,
                examples=["error", "TODO", "function"]
            ),
            ToolParameter(
                name="replace_text",
                type=ParameterType.STRING.value,
                description="替换文本",
                required=False,
                examples=["fixed", "DONE", "method"]
            ),
            ToolParameter(
                name="encoding",
                type=ParameterType.STRING.value,
                description="文件编码",
                required=False,
                default="utf-8",
                examples=["utf-8", "gbk", "latin-1"]
            ),
            ToolParameter(
                name="pattern",
                type=ParameterType.STRING.value,
                description="正则表达式模式",
                required=False,
                examples=[r"\d{4}-\d{2}-\d{2}", r"email: \w+@\w+\.\w+"]
            ),
            ToolParameter(
                name="case_sensitive",
                type=ParameterType.BOOLEAN.value,
                description="是否区分大小写",
                required=False,
                default=False
            ),
            ToolParameter(
                name="files",
                type=ParameterType.ARRAY.value,
                description="文件列表（用于批量操作）",
                required=False,
                examples=[["file1.txt", "file2.txt"]]
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行文本操作

        Args:
            operation: 操作类型
            file_path: 文件路径
            content: 文本内容
            search_text: 搜索文本
            replace_text: 替换文本
            encoding: 文件编码
            pattern: 正则表达式模式
            case_sensitive: 是否区分大小写
            files: 文件列表

        Returns:
            操作结果
        """
        try:
            operation = kwargs.get("operation", "")
            if not operation:
                raise ToolError("操作类型不能为空")

            self.logger.info("执行文本操作", operation=operation, kwargs=kwargs)

            start_time = time.time()

            # 根据操作类型执行相应的方法
            if operation == "read":
                result = self._read_file(kwargs)
            elif operation == "write":
                result = self._write_file(kwargs)
            elif operation == "append":
                result = self._append_file(kwargs)
            elif operation == "search":
                result = self._search_text(kwargs)
            elif operation == "replace":
                result = self._replace_text(kwargs)
            elif operation == "count":
                result = self._count_text(kwargs)
            elif operation == "format":
                result = self._format_text(kwargs)
            elif operation == "split":
                result = self._split_file(kwargs)
            elif operation == "merge":
                result = self._merge_files(kwargs)
            elif operation == "validate":
                result = self._validate_file(kwargs)
            elif operation == "extract":
                result = self._extract_text(kwargs)
            elif operation == "convert":
                result = self._convert_encoding(kwargs)
            elif operation == "statistics":
                result = self._get_file_statistics(kwargs)
            else:
                raise ToolError(f"不支持的操作类型: {operation}")

            execution_time = time.time() - start_time

            self.logger.info(
                "文本操作完成",
                operation=operation,
                success=result.get("success", False),
                execution_time=execution_time
            )

            result["execution_time"] = execution_time
            return result

        except Exception as e:
            error_msg = f"文本操作失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0
            }

    def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """读取文件内容"""
        file_path = params.get("file_path", "")
        encoding = params.get("encoding", "utf-8")
        lines = params.get("lines", None)  # 读取指定行数

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            # 自动检测编码
            if encoding == "auto" and CHARDET_AVAILABLE:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # 读取文件
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                if lines:
                    content_lines = []
                    for i, line in enumerate(f):
                        if i >= lines:
                            break
                        content_lines.append(line.rstrip('\n\r'))
                    content = "\n".join(content_lines)
                else:
                    content = f.read()

            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "content": content,
                "encoding": encoding,
                "file_size": file_size,
                "line_count": content.count('\n') + 1 if content else 0,
                "word_count": len(content.split()),
                "char_count": len(content),
                "message": f"成功读取文件: {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"读取文件失败: {e}"
            }

    def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """写入文件内容"""
        file_path = params.get("file_path", "")
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")
        backup = params.get("backup", False)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if content is None:
            raise ToolError("文件内容不能为None")

        try:
            # 创建备份
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(file_path, backup_path)
                backup_info = {"backup_path": backup_path}
            else:
                backup_info = {}

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 写入文件
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(str(content))

            file_size = os.path.getsize(file_path)

            result = {
                "success": True,
                "file_path": file_path,
                "encoding": encoding,
                "file_size": file_size,
                "content_length": len(str(content)),
                "line_count": str(content).count('\n') + 1,
                "message": f"成功写入文件: {file_path}"
            }

            result.update(backup_info)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"写入文件失败: {e}"
            }

    def _append_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """追加内容到文件"""
        file_path = params.get("file_path", "")
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")
        newline = params.get("newline", True)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            # 文件不存在，创建新文件
            return self._write_file(params)

        try:
            with open(file_path, 'a', encoding=encoding) as f:
                if newline:
                    f.write(f"\n{content}")
                else:
                    f.write(str(content))

            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "file_path": file_path,
                "encoding": encoding,
                "file_size": file_size,
                "appended_length": len(str(content)),
                "message": f"成功追加内容到文件: {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"追加内容失败: {e}"
            }

    def _search_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """在文件中搜索文本"""
        file_path = params.get("file_path", "")
        search_text = params.get("search_text", "")
        pattern = params.get("pattern", "")
        case_sensitive = params.get("case_sensitive", False)
        encoding = params.get("encoding", "utf-8")
        max_results = params.get("max_results", 100)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not search_text and not pattern:
            raise ToolError("搜索文本或正则表达式模式不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            results = []
            total_matches = 0

            # 执行搜索
            if pattern:
                # 正则表达式搜索
                flags = 0 if case_sensitive else re.IGNORECASE
                matches = list(re.finditer(pattern, content, flags))
                total_matches = len(matches)

                for i, match in enumerate(matches):
                    if i >= max_results:
                        break

                    line_num = content[:match.start()].count('\n') + 1
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(content)

                    line_text = content[line_start:line_end]
                    results.append({
                        "line": line_num,
                        "column": match.start() - line_start + 1,
                        "match": match.group(),
                        "line_text": line_text,
                        "context": match.start()
                    })
            else:
                # 普通文本搜索
                search_content = content if case_sensitive else content.lower()
                search_term = search_text if case_sensitive else search_text.lower()

                start_pos = 0
                while True:
                    pos = search_content.find(search_term, start_pos)
                    if pos == -1 or len(results) >= max_results:
                        break

                    line_num = content[:pos].count('\n') + 1
                    line_start = content.rfind('\n', 0, pos) + 1
                    line_end = content.find('\n', pos)
                    if line_end == -1:
                        line_end = len(content)

                    line_text = content[line_start:line_end]
                    results.append({
                        "line": line_num,
                        "column": pos - line_start + 1,
                        "match": search_text,
                        "line_text": line_text,
                        "context": pos
                    })

                    total_matches += 1
                    start_pos = pos + 1

            return {
                "success": True,
                "file_path": file_path,
                "search_text": search_text,
                "pattern": pattern,
                "case_sensitive": case_sensitive,
                "total_matches": total_matches,
                "returned_matches": len(results),
                "results": results,
                "message": f"搜索完成，找到 {total_matches} 个匹配"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"搜索失败: {e}"
            }

    def _replace_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """替换文件中的文本"""
        file_path = params.get("file_path", "")
        search_text = params.get("search_text", "")
        replace_text = params.get("replace_text", "")
        pattern = params.get("pattern", "")
        case_sensitive = params.get("case_sensitive", False)
        encoding = params.get("encoding", "utf-8")
        backup = params.get("backup", True)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not search_text and not pattern:
            raise ToolError("搜索文本或正则表达式模式不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            # 创建备份
            if backup:
                backup_path = f"{file_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(file_path, backup_path)
            else:
                backup_path = None

            # 读取文件内容
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            original_content = content
            replacement_count = 0

            # 执行替换
            if pattern:
                # 正则表达式替换
                flags = 0 if case_sensitive else re.IGNORECASE
                new_content, replacement_count = re.subn(pattern, replace_text, content, flags=flags)
            else:
                # 普通文本替换
                if case_sensitive:
                    new_content = content.replace(search_text, replace_text)
                    replacement_count = content.count(search_text)
                else:
                    # 不区分大小写的替换
                    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                    new_content, replacement_count = pattern.subn(replace_text, content)

            # 写入新内容
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(new_content)

            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "file_path": file_path,
                "search_text": search_text,
                "replace_text": replace_text,
                "pattern": pattern,
                "case_sensitive": case_sensitive,
                "replacement_count": replacement_count,
                "original_size": len(original_content),
                "new_size": len(new_content),
                "backup_path": backup_path,
                "message": f"替换完成，共替换 {replacement_count} 处"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"替换失败: {e}"
            }

    def _count_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """统计文本内容"""
        file_path = params.get("file_path", "")
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")

        if file_path:
            # 从文件读取
            result = self._read_file(params)
            if not result["success"]:
                return result
            content = result["content"]

        if not content:
            content = ""

        # 统计信息
        char_count = len(content)
        char_count_no_spaces = len(content.replace(" ", ""))
        word_count = len(content.split())
        line_count = content.count('\n') + 1
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

        # 字符统计
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        numbers = len(re.findall(r'\d', content))
        spaces = len(re.findall(r'\s', content))

        return {
            "success": True,
            "statistics": {
                "char_count": char_count,
                "char_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "line_count": line_count,
                "paragraph_count": paragraph_count,
                "chinese_chars": chinese_chars,
                "english_chars": english_chars,
                "numbers": numbers,
                "spaces": spaces
            },
            "message": "文本统计完成"
        }

    def _format_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """格式化文本"""
        content = params.get("content", "")
        format_type = params.get("format_type", "normalize")
        file_path = params.get("file_path", "")

        if file_path:
            # 从文件读取
            result = self._read_file(params)
            if not result["success"]:
                return result
            content = result["content"]

        if not content:
            raise ToolError("文本内容不能为空")

        try:
            if format_type == "normalize":
                # 标准化：去除多余空格和换行
                formatted = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # 多个空行变两个
                formatted = re.sub(r'[ \t]+', ' ', formatted)  # 多个空格变一个
                formatted = formatted.strip()  # 去除首尾空白
            elif format_type == "remove_extra_spaces":
                # 去除多余空格
                formatted = re.sub(r' +', ' ', content)
            elif format_type == "remove_extra_lines":
                # 去除多余空行
                formatted = re.sub(r'\n\s*\n+', '\n\n', content)
            elif format_type == "json_pretty":
                # JSON格式化
                try:
                    json_data = json.loads(content)
                    formatted = json.dumps(json_data, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    raise ToolError("内容不是有效的JSON格式")
            elif format_type == "sort_lines":
                # 排序行
                lines = content.split('\n')
                formatted = '\n'.join(sorted(lines))
            elif format_type == "unique_lines":
                # 去重行
                lines = content.split('\n')
                seen = set()
                unique_lines = []
                for line in lines:
                    if line not in seen:
                        seen.add(line)
                        unique_lines.append(line)
                formatted = '\n'.join(unique_lines)
            else:
                raise ToolError(f"不支持的格式类型: {format_type}")

            return {
                "success": True,
                "original_content": content,
                "formatted_content": formatted,
                "format_type": format_type,
                "original_length": len(content),
                "formatted_length": len(formatted),
                "message": f"文本格式化完成: {format_type}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"文本格式化失败: {e}"
            }

    def _split_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """分割文件"""
        file_path = params.get("file_path", "")
        split_type = params.get("split_type", "lines")
        split_value = params.get("split_value", 1000)
        output_dir = params.get("output_dir", os.path.dirname(file_path))
        encoding = params.get("encoding", "utf-8")

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            # 读取文件
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()

            output_files = []
            base_name = Path(file_path).stem

            if split_type == "lines":
                # 按行数分割
                chunk_size = int(split_value)
                for i in range(0, len(lines), chunk_size):
                    chunk = lines[i:i + chunk_size]
                    output_file = os.path.join(output_dir, f"{base_name}_part_{i//chunk_size + 1}.txt")
                    with open(output_file, 'w', encoding=encoding) as f:
                        f.writelines(chunk)
                    output_files.append(output_file)
            elif split_type == "size":
                # 按文件大小分割（字节）
                max_size = int(split_value)
                current_size = 0
                current_chunk = []
                part_num = 1

                for line in lines:
                    line_size = len(line.encode(encoding))
                    if current_size + line_size > max_size and current_chunk:
                        # 写入当前块
                        output_file = os.path.join(output_dir, f"{base_name}_part_{part_num}.txt")
                        with open(output_file, 'w', encoding=encoding) as f:
                            f.writelines(current_chunk)
                        output_files.append(output_file)

                        # 开始新块
                        current_chunk = [line]
                        current_size = line_size
                        part_num += 1
                    else:
                        current_chunk.append(line)
                        current_size += line_size

                # 写入最后一块
                if current_chunk:
                    output_file = os.path.join(output_dir, f"{base_name}_part_{part_num}.txt")
                    with open(output_file, 'w', encoding=encoding) as f:
                        f.writelines(current_chunk)
                    output_files.append(output_file)

            return {
                "success": True,
                "original_file": file_path,
                "split_type": split_type,
                "split_value": split_value,
                "total_lines": len(lines),
                "output_files": output_files,
                "files_created": len(output_files),
                "message": f"文件分割完成，创建了 {len(output_files)} 个文件"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"文件分割失败: {e}"
            }

    def _merge_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个文件"""
        files = params.get("files", [])
        output_file = params.get("output_file", "")
        encoding = params.get("encoding", "utf-8")
        separator = params.get("separator", "\n")

        if not files:
            raise ToolError("文件列表不能为空")

        if not output_file:
            raise ToolError("输出文件路径不能为空")

        try:
            merged_content = []
            total_size = 0

            for file_path in files:
                if not os.path.exists(file_path):
                    self.logger.warning(f"文件不存在，跳过: {file_path}")
                    continue

                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                    merged_content.append(content)
                    total_size += len(content)

            # 写入合并后的文件
            final_content = separator.join(merged_content)
            with open(output_file, 'w', encoding=encoding) as f:
                f.write(final_content)

            output_size = os.path.getsize(output_file)

            return {
                "success": True,
                "input_files": files,
                "output_file": output_file,
                "encoding": encoding,
                "files_processed": len([f for f in files if os.path.exists(f)]),
                "total_input_size": total_size,
                "output_size": output_size,
                "message": f"文件合并完成，处理了 {len([f for f in files if os.path.exists(f)])} 个文件"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"文件合并失败: {e}"
            }

    def _validate_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证文件格式"""
        file_path = params.get("file_path", "")
        validation_type = params.get("validation_type", "encoding")
        encoding = params.get("encoding", "utf-8")

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            validation_result = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "validation_type": validation_type
            }

            if validation_type == "encoding":
                # 验证文件编码
                if CHARDET_AVAILABLE:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    validation_result.update({
                        "detected_encoding": detected['encoding'],
                        "confidence": detected['confidence'],
                        "is_valid": True
                    })

                    # 尝试用检测到的编码读取
                    try:
                        with open(file_path, 'r', encoding=detected['encoding']) as f:
                            f.read()
                        validation_result["readable"] = True
                    except:
                        validation_result["readable"] = False
                else:
                    validation_result["error"] = "chardet模块不可用"

            elif validation_type == "json":
                # 验证JSON格式
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    json.loads(content)
                validation_result.update({
                    "is_valid": True,
                    "format": "JSON"
                })

            elif validation_type == "csv":
                # 验证CSV格式
                with open(file_path, 'r', encoding=encoding) as f:
                    csv.Sniffer().sniff(f.read(1024))
                validation_result.update({
                    "is_valid": True,
                    "format": "CSV"
                })

            else:
                raise ToolError(f"不支持的验证类型: {validation_type}")

            validation_result["success"] = True
            validation_result["message"] = "文件验证通过"

            return validation_result

        except json.JSONDecodeError as e:
            return {
                "success": False,
                "validation_type": validation_type,
                "error": str(e),
                "message": "JSON格式验证失败"
            }
        except Exception as e:
            return {
                "success": False,
                "validation_type": validation_type,
                "error": str(e),
                "message": f"文件验证失败: {e}"
            }

    def _extract_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取特定文本"""
        file_path = params.get("file_path", "")
        extract_type = params.get("extract_type", "lines")
        pattern = params.get("pattern", "")
        start_line = params.get("start_line", 1)
        end_line = params.get("end_line", None)
        encoding = params.get("encoding", "utf-8")

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()

            extracted_content = []
            extracted_lines = []

            if extract_type == "lines":
                # 提取指定行
                start = max(1, int(start_line)) - 1
                end = int(end_line) if end_line else len(lines)
                end = min(end, len(lines))

                extracted_lines = lines[start:end]
                extracted_content = [line.rstrip('\n\r') for line in extracted_lines]

            elif extract_type == "pattern":
                # 按模式提取
                if not pattern:
                    raise ToolError("提取模式不能为空")

                regex = re.compile(pattern)
                for line in lines:
                    match = regex.search(line)
                    if match:
                        extracted_lines.append(line)
                        extracted_content.append(match.group(1) if match.groups() else match.group())

            elif extract_type == "non_empty":
                # 提取非空行
                for line in lines:
                    if line.strip():
                        extracted_lines.append(line)
                        extracted_content.append(line.strip())

            else:
                raise ToolError(f"不支持的提取类型: {extract_type}")

            return {
                "success": True,
                "file_path": file_path,
                "extract_type": extract_type,
                "extracted_count": len(extracted_content),
                "total_lines": len(lines),
                "extracted_content": extracted_content,
                "message": f"文本提取完成，提取了 {len(extracted_content)} 项"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"文本提取失败: {e}"
            }

    def _convert_encoding(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """转换文件编码"""
        file_path = params.get("file_path", "")
        target_encoding = params.get("target_encoding", "utf-8")
        source_encoding = params.get("source_encoding", "auto")
        output_file = params.get("output_file", "")
        backup = params.get("backup", True)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            # 自动检测源编码
            if source_encoding == "auto" and CHARDET_AVAILABLE:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                result = chardet.detect(raw_data)
                source_encoding = result['encoding']

            # 读取原文件
            with open(file_path, 'r', encoding=source_encoding, errors='replace') as f:
                content = f.read()

            # 设置输出文件路径
            if not output_file:
                path = Path(file_path)
                output_file = str(path.parent / f"{path.stem}_converted{path.suffix}")

            # 创建备份
            if backup:
                backup_path = f"{file_path}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(file_path, backup_path)
            else:
                backup_path = None

            # 写入新编码文件
            with open(output_file, 'w', encoding=target_encoding) as f:
                f.write(content)

            return {
                "success": True,
                "input_file": file_path,
                "output_file": output_file,
                "source_encoding": source_encoding,
                "target_encoding": target_encoding,
                "backup_path": backup_path,
                "message": f"编码转换成功: {source_encoding} -> {target_encoding}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"编码转换失败: {e}"
            }

    def _get_file_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取文件详细统计信息"""
        file_path = params.get("file_path", "")
        encoding = params.get("encoding", "utf-8")
        analyze_content = params.get("analyze_content", True)

        if not file_path:
            raise ToolError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise ToolError(f"文件不存在: {file_path}")

        try:
            file_stat = os.stat(file_path)
            base_stats = {
                "file_path": file_path,
                "size_bytes": file_stat.st_size,
                "size_human": self._format_file_size(file_stat.st_size),
                "created_time": file_stat.st_ctime,
                "modified_time": file_stat.st_mtime,
                "accessed_time": file_stat.st_atime
            }

            if analyze_content:
                # 分析文件内容
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()

                content_stats = {
                    "line_count": content.count('\n') + 1,
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "empty_lines": content.count('\n\n') + 1,
                    "encoding": encoding,
                    "encoding_detected": None,
                    "language_hint": self._detect_language(content)
                }

                # 检测实际编码
                if CHARDET_AVAILABLE:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    content_stats["encoding_detected"] = detected['encoding']

                base_stats.update({"content": content_stats})

            return {
                "success": True,
                "statistics": base_stats,
                "message": "文件统计信息获取成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"获取文件统计失败: {e}"
            }

    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.6:
            return "chinese"
        elif english_ratio > 0.6:
            return "english"
        elif chinese_ratio > 0.1 and english_ratio > 0.1:
            return "mixed"
        else:
            return "unknown"


# 注册工具
from ..registry import tool_registry
tool_registry.register(TextOperationsTool())