"""
MCP Floating Ball - 多搜索引擎工具

提供集成多个搜索引擎的功能，包括Google、百度、必应等。
"""

import time
import json
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from ...core.logging import get_logger
from ...core.exceptions import ToolError
from ..base import BaseTool, ToolMetadata, ToolCategory, ParameterType, ToolParameter

logger = get_logger(__name__)


class MultiSearchTool(BaseTool):
    """多搜索引擎工具"""

    def __init__(self):
        """初始化多搜索引擎工具"""
        super().__init__()
        self.logger = get_logger("tool.multi_search")

        # 搜索引擎配置
        self.search_engines = {
            "google": {
                "name": "Google",
                "base_url": "https://www.google.com/search",
                "params": {"q": "{query}", "num": "{num_results}"},
                "results_selector": "div.g",
                "title_selector": "h3",
                "link_selector": "a",
                "snippet_selector": "div.VwiC3b"
            },
            "baidu": {
                "name": "百度",
                "base_url": "https://www.baidu.com/s",
                "params": {"wd": "{query}", "rn": "{num_results}"},
                "results_selector": "div.result",
                "title_selector": "h3.t",
                "link_selector": "a",
                "snippet_selector": "div.c-abstract"
            },
            "bing": {
                "name": "Bing",
                "base_url": "https://www.bing.com/search",
                "params": {"q": "{query}", "count": "{num_results}"},
                "results_selector": "li.b_algo",
                "title_selector": "h2",
                "link_selector": "a",
                "snippet_selector": "p"
            },
            "duckduckgo": {
                "name": "DuckDuckGo",
                "base_url": "https://html.duckduckgo.com/html/",
                "params": {"q": "{query}"},
                "results_selector": "div.result",
                "title_selector": "a.result__a",
                "link_selector": "a.result__a",
                "snippet_selector": "a.result__snippet"
            }
        }

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="multi_search",
            display_name="多搜索引擎",
            description="多搜索引擎集成工具，支持Google、百度、必应等多个搜索引擎",
            category=ToolCategory.NETWORK,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["search", "google", "baidu", "bing", "network"],
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="搜索关键词",
                    required=True
                ),
                ToolParameter(
                    name="engines",
                    type=ParameterType.ARRAY,
                    description="搜索引擎列表",
                    required=False,
                    enum=["google", "baidu", "bing", "duckduckgo"],
                    default=["google"]
                ),
                ToolParameter(
                    name="num_results",
                    type=ParameterType.INTEGER,
                    description="每个引擎返回的结果数量",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="language",
                    type=ParameterType.STRING,
                    description="搜索语言",
                    required=False,
                    default="zh-cn"
                ),
                ToolParameter(
                    name="region",
                    type=ParameterType.STRING,
                    description="搜索地区",
                    required=False,
                    default="cn"
                )
            ],
            examples=["搜索Python教程", "使用多个搜索引擎查找AI资料", "限制每个引擎返回5个结果"]
        )

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="query",
                type=ParameterType.STRING.value,
                description="搜索关键词",
                required=True,
                examples=["Python编程教程", "机器学习", "AI助手开发"]
            ),
            ToolParameter(
                name="engines",
                type=ParameterType.ARRAY.value,
                description="搜索引擎列表",
                required=False,
                choices=["google", "baidu", "bing", "duckduckgo"],
                default=["google"],
                examples=[["google", "baidu"]]
            ),
            ToolParameter(
                name="num_results",
                type=ParameterType.INTEGER.value,
                description="每个引擎返回的结果数量",
                required=False,
                default=10,
                examples=[5, 10, 20]
            ),
            ToolParameter(
                name="language",
                type=ParameterType.STRING.value,
                description="搜索语言",
                required=False,
                default="zh-cn",
                examples=["zh-cn", "en-us", "ja"]
            ),
            ToolParameter(
                name="region",
                type=ParameterType.STRING.value,
                description="搜索区域",
                required=False,
                default="cn",
                examples=["cn", "us", "jp"]
            ),
            ToolParameter(
                name="safe_search",
                type=ParameterType.STRING.value,
                description="安全搜索级别",
                required=False,
                choices=["off", "moderate", "strict"],
                default="moderate"
            ),
            ToolParameter(
                name="time_range",
                type=ParameterType.STRING.value,
                description="时间范围",
                required=False,
                choices=["day", "week", "month", "year"],
                examples=["day", "week", "month"]
            ),
            ToolParameter(
                name="use_cache",
                type=ParameterType.BOOLEAN.value,
                description="是否使用缓存",
                required=False,
                default=False
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行多搜索引擎查询

        Args:
            query: 搜索关键词
            engines: 搜索引擎列表
            num_results: 结果数量
            language: 语言
            region: 区域
            safe_search: 安全搜索
            time_range: 时间范围
            use_cache: 是否使用缓存

        Returns:
            搜索结果
        """
        try:
            if not REQUESTS_AVAILABLE:
                raise ToolError("requests模块不可用，请安装: pip install requests")

            query = kwargs.get("query", "")
            engines = kwargs.get("engines", ["google"])
            num_results = kwargs.get("num_results", 10)
            language = kwargs.get("language", "zh-cn")
            region = kwargs.get("region", "cn")
            safe_search = kwargs.get("safe_search", "moderate")
            time_range = kwargs.get("time_range", "")
            use_cache = kwargs.get("use_cache", False)

            if not query:
                raise ToolError("搜索关键词不能为空")

            if not engines:
                raise ToolError("搜索引擎列表不能为空")

            self.logger.info(
                "开始多搜索引擎查询",
                query=query,
                engines=engines,
                num_results=num_results
            )

            start_time = time.time()

            # 执行搜索
            search_results = {}
            total_results = 0
            successful_engines = 0

            for engine in engines:
                try:
                    engine_config = self.search_engines.get(engine)
                    if not engine_config:
                        self.logger.warning(f"不支持的搜索引擎: {engine}")
                        continue

                    results = self._search_single_engine(
                        engine, query, engine_config, {
                            "num_results": num_results,
                            "language": language,
                            "region": region,
                            "safe_search": safe_search,
                            "time_range": time_range,
                            "use_cache": use_cache
                        }
                    )

                    search_results[engine] = results
                    total_results += len(results.get("results", []))
                    successful_engines += 1

                except Exception as e:
                    self.logger.error(f"搜索引擎 {engine} 查询失败: {e}")
                    search_results[engine] = {
                        "success": False,
                        "error": str(e),
                        "results": []
                    }

            execution_time = time.time() - start_time

            # 合并和排序结果
            merged_results = self._merge_search_results(search_results)

            self.logger.info(
                "多搜索引擎查询完成",
                query=query,
                successful_engines=successful_engines,
                total_results=total_results,
                execution_time=execution_time
            )

            return {
                "success": successful_engines > 0,
                "query": query,
                "engines": engines,
                "successful_engines": successful_engines,
                "total_results": total_results,
                "search_results": search_results,
                "merged_results": merged_results,
                "execution_time": execution_time,
                "message": f"搜索完成，{successful_engines}/{len(engines)} 个引擎成功，共获得 {total_results} 个结果"
            }

        except Exception as e:
            error_msg = f"多搜索引擎查询失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0
            }

    def _search_single_engine(
        self,
        engine_name: str,
        query: str,
        engine_config: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """搜索单个搜索引擎"""
        try:
            # 构建请求参数
            params = {}
            for param_key, param_value in engine_config["params"].items():
                if "{query}" in param_value:
                    params[param_key] = param_value.replace("{query}", query)
                elif "{num_results}" in param_value:
                    params[param_key] = param_value.replace("{num_results}", str(options["num_results"]))
                else:
                    params[param_key] = param_value

            # 添加通用参数
            if options.get("language"):
                params["hl"] = options["language"]
            if options.get("region"):
                params["gl"] = options["region"]
            if options.get("safe_search"):
                params["safe"] = options["safe_search"]
            if options.get("time_range"):
                time_range_map = {
                    "day": "d",
                    "week": "w",
                    "month": "m",
                    "year": "y"
                }
                if engine_name in ["google", "bing"]:
                    params["tbs"] = f"qdr:{time_range_map.get(options['time_range'], 'd')}"

            # 构建请求头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            # 发送请求
            response = requests.get(
                engine_config["base_url"],
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            # 解析结果
            results = self._parse_search_results(
                response.text,
                engine_config,
                options["num_results"]
            )

            return {
                "success": True,
                "engine": engine_name,
                "query": query,
                "results": results,
                "result_count": len(results)
            }

        except requests.RequestException as e:
            return {
                "success": False,
                "engine": engine_name,
                "error": f"网络请求失败: {e}",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "engine": engine_name,
                "error": f"搜索失败: {e}",
                "results": []
            }

    def _parse_search_results(
        self,
        html_content: str,
        engine_config: Dict[str, Any],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """解析搜索结果HTML"""
        results = []

        try:
            if not BS4_AVAILABLE:
                # 使用正则表达式解析（备用方案）
                return self._parse_with_regex(html_content, engine_config, max_results)

            soup = BeautifulSoup(html_content, 'html.parser')

            # 查找结果容器
            result_containers = soup.select(engine_config["results_selector"])

            for container in result_containers[:max_results]:
                try:
                    # 提取标题
                    title_elem = container.select_one(engine_config["title_selector"])
                    title = title_elem.get_text(strip=True) if title_elem else ""

                    # 提取链接
                    link_elem = container.select_one(engine_config["link_selector"])
                    link = link_elem.get("href", "") if link_elem else ""

                    # 清理链接（移除重定向）
                    if link.startswith("/url?"):
                        # Google重定向链接
                        import urllib.parse as urlparse
                        parsed = urlparse.parse_qs(urlparse.urlparse(link).query)
                        if "q" in parsed:
                            link = parsed["q"][0]

                    # 提取描述
                    snippet_elem = container.select_one(engine_config["snippet_selector"])
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if title and link:
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet,
                            "engine": engine_config.get("name", "Unknown"),
                            "position": len(results) + 1
                        })

                except Exception as e:
                    self.logger.warning(f"解析单个结果失败: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"解析搜索结果失败: {e}")
            # 尝试备用解析方法
            return self._parse_with_regex(html_content, engine_config, max_results)

        return results

    def _parse_with_regex(
        self,
        html_content: str,
        engine_config: Dict[str, Any],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """使用正则表达式解析搜索结果（备用方案）"""
        results = []

        try:
            # 简单的链接和标题提取正则表达式
            link_pattern = r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, html_content, re.IGNORECASE)

            for i, (link, title) in enumerate(matches[:max_results]):
                title = re.sub(r'<[^>]+>', '', title).strip()
                link = link.strip()

                if title and link and not link.startswith("#") and "javascript:" not in link:
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": "",
                        "engine": engine_config.get("name", "Unknown"),
                        "position": i + 1
                    })

        except Exception as e:
            self.logger.error(f"正则表达式解析失败: {e}")

        return results

    def _merge_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """合并多个搜索引擎的结果"""
        merged = []
        seen_links = set()

        for engine_name, engine_result in search_results.items():
            if not engine_result.get("success"):
                continue

            for result in engine_result.get("results", []):
                link = result.get("link", "")

                # 去重（基于链接）
                if link and link not in seen_links:
                    seen_links.add(link)
                    merged.append(result)

        # 按位置排序（简单策略）
        merged.sort(key=lambda x: x.get("position", 999))

        return merged[:50]  # 限制合并后的结果数量

    def get_engine_info(self) -> Dict[str, Any]:
        """获取搜索引擎信息"""
        return {
            "engines": self.search_engines,
            "total_engines": len(self.search_engines),
            "dependencies": {
                "requests": REQUESTS_AVAILABLE,
                "beautifulsoup4": BS4_AVAILABLE,
                "feedparser": FEEDPARSER_AVAILABLE
            }
        }

    def search_images(
        self,
        query: str,
        engines: List[str] = None,
        num_results: int = 10,
        image_size: str = "medium"
    ) -> Dict[str, Any]:
        """
        搜索图片

        Args:
            query: 搜索关键词
            engines: 搜索引擎列表
            num_results: 结果数量
            image_size: 图片大小

        Returns:
            图片搜索结果
        """
        try:
            if not engines:
                engines = ["google", "bing"]

            image_results = {}

            for engine in engines:
                try:
                    if engine == "google":
                        results = self._search_google_images(query, num_results)
                    elif engine == "bing":
                        results = self._search_bing_images(query, num_results)
                    else:
                        continue

                    image_results[engine] = results

                except Exception as e:
                    self.logger.error(f"{engine} 图片搜索失败: {e}")
                    image_results[engine] = {"success": False, "error": str(e), "results": []}

            return {
                "success": True,
                "query": query,
                "image_results": image_results,
                "message": "图片搜索完成"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "图片搜索失败"
            }

    def _search_google_images(self, query: str, num_results: int) -> Dict[str, Any]:
        """搜索Google图片"""
        # 这里可以实现Google图片搜索
        # 由于需要处理复杂的JavaScript和反爬虫，这里提供一个基础框架
        return {
            "success": True,
            "engine": "google_images",
            "results": [],
            "message": "Google图片搜索需要特殊处理，当前仅提供框架"
        }

    def _search_bing_images(self, query: str, num_results: int) -> Dict[str, Any]:
        """搜索Bing图片"""
        # 这里可以实现Bing图片搜索
        return {
            "success": True,
            "engine": "bing_images",
            "results": [],
            "message": "Bing图片搜索需要特殊处理，当前仅提供框架"
        }

    def search_news(
        self,
        query: str,
        engines: List[str] = None,
        num_results: int = 10,
        time_range: str = "week"
    ) -> Dict[str, Any]:
        """
        搜索新闻

        Args:
            query: 搜索关键词
            engines: 搜索引擎列表
            num_results: 结果数量
            time_range: 时间范围

        Returns:
            新闻搜索结果
        """
        try:
            if not engines:
                engines = ["google", "bing"]

            news_results = {}

            for engine in engines:
                try:
                    if engine == "google":
                        results = self._search_google_news(query, num_results, time_range)
                    elif engine == "bing":
                        results = self._search_bing_news(query, num_results)
                    else:
                        continue

                    news_results[engine] = results

                except Exception as e:
                    self.logger.error(f"{engine} 新闻搜索失败: {e}")
                    news_results[engine] = {"success": False, "error": str(e), "results": []}

            return {
                "success": True,
                "query": query,
                "news_results": news_results,
                "message": "新闻搜索完成"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "新闻搜索失败"
            }

    def _search_google_news(self, query: str, num_results: int, time_range: str) -> Dict[str, Any]:
        """搜索Google新闻"""
        # Google新闻搜索实现
        return {
            "success": True,
            "engine": "google_news",
            "results": [],
            "message": "Google新闻搜索需要特殊处理，当前仅提供框架"
        }

    def _search_bing_news(self, query: str, num_results: int) -> Dict[str, Any]:
        """搜索Bing新闻"""
        # Bing新闻搜索实现
        return {
            "success": True,
            "engine": "bing_news",
            "results": [],
            "message": "Bing新闻搜索需要特殊处理，当前仅提供框架"
        }

    def suggest_queries(self, partial_query: str, engine: str = "google") -> Dict[str, Any]:
        """
        获取搜索建议

        Args:
            partial_query: 部分搜索关键词
            engine: 搜索引擎

        Returns:
            搜索建议
        """
        try:
            if not REQUESTS_AVAILABLE:
                raise ToolError("requests模块不可用")

            suggestions = []

            # 不同搜索引擎的建议API
            if engine == "google":
                suggestions = self._get_google_suggestions(partial_query)
            elif engine == "baidu":
                suggestions = self._get_baidu_suggestions(partial_query)
            elif engine == "bing":
                suggestions = self._get_bing_suggestions(partial_query)

            return {
                "success": True,
                "partial_query": partial_query,
                "engine": engine,
                "suggestions": suggestions,
                "count": len(suggestions)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestions": []
            }

    def _get_google_suggestions(self, query: str) -> List[str]:
        """获取Google搜索建议"""
        try:
            url = "http://suggestqueries.google.com/complete/search"
            params = {
                "client": "firefox",
                "q": query,
                "hl": "zh-cn"
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                # Google返回的是JSON格式
                import json
                data = json.loads(response.text)
                return data[1] if len(data) > 1 else []
        except Exception as e:
            self.logger.error(f"获取Google建议失败: {e}")

        return []

    def _get_baidu_suggestions(self, query: str) -> List[str]:
        """获取百度搜索建议"""
        try:
            url = "https://suggestion.baidu.com/su"
            params = {
                "wd": query,
                "p": "3",
                "cb": "window.baidu.sug"
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                # 百度返回的是JSONP格式，需要解析
                import json
                content = response.text
                if content.startswith('window.baidu.sug('):
                    content = content[19:-1]  # 移除JSONP包装
                    data = json.loads(content)
                    return data.get('s', [])
        except Exception as e:
            self.logger.error(f"获取百度建议失败: {e}")

        return []

    def _get_bing_suggestions(self, query: str) -> List[str]:
        """获取Bing搜索建议"""
        try:
            url = "https://api.bing.com/qsonhs.aspx"
            params = {
                "form": "REDIR",
                "mkt": "zh-CN",
                "query": query
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                import json
                data = json.loads(response.text)
                if "AS" in data and "Results" in data["AS"]:
                    return [item["Txt"] for item in data["AS"]["Results"]]
        except Exception as e:
            self.logger.error(f"获取Bing建议失败: {e}")

        return []


# 注册工具
from ..registry import tool_registry
tool_registry.register(MultiSearchTool())