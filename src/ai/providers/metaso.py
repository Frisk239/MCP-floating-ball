"""
MCP Floating Ball - 秘塔AI搜索服务提供商

实现与秘塔AI搜索API的集成，提供增强的搜索功能。
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
import httpx

from ...core.config import MetasoSettings
from ...core.logging import get_logger
from ...core.exceptions import APIError, AIServiceError, AuthenticationError

logger = get_logger(__name__)


class MetasoProvider:
    """秘塔AI搜索服务提供商"""

    def __init__(self, config: MetasoSettings):
        """
        初始化秘塔AI搜索服务提供商

        Args:
            config: 秘塔配置
        """
        self.config = config
        self.base_url = "https://api.metaso.cn/v1"
        self.logger = get_logger(f"ai.provider.metaso")

        # 验证API密钥
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """验证API密钥"""
        if not self.config.api_key:
            raise AuthenticationError("秘塔API密钥不能为空", provider="metaso")

        self.logger.info("秘塔API密钥验证成功")

    def search(
        self,
        query: str,
        search_type: str = "web",
        max_results: int = 10,
        language: str = "zh",
        region: str = "cn",
        safe_search: str = "moderate",
        include_images: bool = False,
        include_videos: bool = False,
        include_news: bool = False
    ) -> Dict[str, Any]:
        """
        执行搜索

        Args:
            query: 搜索查询
            search_type: 搜索类型（web、image、video、news等）
            max_results: 最大结果数量
            language: 语言代码
            region: 地区代码
            safe_search: 安全搜索级别
            include_images: 是否包含图片结果
            include_videos: 是否包含视频结果
            include_news: 是否包含新闻结果

        Returns:
            搜索结果

        Raises:
            APIError: 搜索失败
            AuthenticationError: 认证失败
        """
        try:
            self.logger.info(
                "开始执行搜索",
                query=query,
                search_type=search_type,
                max_results=max_results
            )

            start_time = time.time()

            # 构建请求参数
            params = {
                "q": query,
                "type": search_type,
                "count": max_results,
                "lang": language,
                "region": region,
                "safe": safe_search,
                "include_images": include_images,
                "include_videos": include_videos,
                "include_news": include_news
            }

            # 发送请求
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MCP-Floating-Ball/1.0"
            }

            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

            execution_time = time.time() - start_time

            # 处理响应
            result = {
                "query": query,
                "search_type": search_type,
                "total_results": data.get("total_results", 0),
                "results": data.get("results", []),
                "images": data.get("images", []) if include_images else [],
                "videos": data.get("videos", []) if include_videos else [],
                "news": data.get("news", []) if include_news else [],
                "related_searches": data.get("related_searches", []),
                "execution_time": execution_time,
                "language": language,
                "region": region
            }

            self.logger.info(
                "搜索成功",
                total_results=result["total_results"],
                execution_time=execution_time
            )

            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"秘塔API认证失败: {e.response.text}",
                    provider="metaso"
                )
            elif e.response.status_code == 429:
                raise APIError(
                    f"秘塔API速率限制: {e.response.text}",
                    provider="metaso"
                )
            else:
                raise APIError(
                    f"秘塔API请求失败: {e.response.status_code} - {e.response.text}",
                    provider="metaso"
                )
        except Exception as e:
            error_msg = f"搜索失败: {e}"
            self.logger.error(error_msg)
            raise APIError(error_msg, provider="metaso")

    async def search_async(
        self,
        query: str,
        search_type: str = "web",
        max_results: int = 10,
        language: str = "zh",
        region: str = "cn",
        safe_search: str = "moderate",
        include_images: bool = False,
        include_videos: bool = False,
        include_news: bool = False
    ) -> Dict[str, Any]:
        """
        异步执行搜索

        Args:
            query: 搜索查询
            search_type: 搜索类型
            max_results: 最大结果数量
            language: 语言代码
            region: 地区代码
            safe_search: 安全搜索级别
            include_images: 是否包含图片结果
            include_videos: 是否包含视频结果
            include_news: 是否包含新闻结果

        Returns:
            搜索结果
        """
        try:
            self.logger.info(
                "开始执行异步搜索",
                query=query,
                search_type=search_type,
                max_results=max_results
            )

            start_time = time.time()

            # 构建请求参数
            params = {
                "q": query,
                "type": search_type,
                "count": max_results,
                "lang": language,
                "region": region,
                "safe": safe_search,
                "include_images": include_images,
                "include_videos": include_videos,
                "include_news": include_news
            }

            # 发送异步请求
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MCP-Floating-Ball/1.0"
            }

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

            execution_time = time.time() - start_time

            # 处理响应
            result = {
                "query": query,
                "search_type": search_type,
                "total_results": data.get("total_results", 0),
                "results": data.get("results", []),
                "images": data.get("images", []) if include_images else [],
                "videos": data.get("videos", []) if include_videos else [],
                "news": data.get("news", []) if include_news else [],
                "related_searches": data.get("related_searches", []),
                "execution_time": execution_time,
                "language": language,
                "region": region
            }

            self.logger.info(
                "异步搜索成功",
                total_results=result["total_results"],
                execution_time=execution_time
            )

            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"秘塔API认证失败: {e.response.text}",
                    provider="metaso"
                )
            elif e.response.status_code == 429:
                raise APIError(
                    f"秘塔API速率限制: {e.response.text}",
                    provider="metaso"
                )
            else:
                raise APIError(
                    f"秘塔API请求失败: {e.response.status_code} - {e.response.text}",
                    provider="metaso"
                )
        except Exception as e:
            error_msg = f"异步搜索失败: {e}"
            self.logger.error(error_msg)
            raise APIError(error_msg, provider="metaso")

    def search_websites(
        self,
        query: str,
        websites: List[str],
        max_results_per_site: int = 5
    ) -> Dict[str, Any]:
        """
        在指定网站中搜索

        Args:
            query: 搜索查询
            websites: 网站列表
            max_results_per_site: 每个网站的最大结果数

        Returns:
            搜索结果
        """
        try:
            self.logger.info(
                "开始在指定网站搜索",
                query=query,
                websites=websites,
                max_results_per_site=max_results_per_site
            )

            start_time = time.time()

            # 构建站点特定的搜索查询
            site_queries = []
            for site in websites:
                site_query = f"site:{site} {query}"
                site_queries.append(site_query)

            # 并发执行搜索
            results = {}
            total_results = 0

            for i, site_query in enumerate(site_queries):
                try:
                    site_result = self.search(
                        query=site_query,
                        max_results=max_results_per_site
                    )

                    website = websites[i]
                    results[website] = {
                        "results": site_result["results"],
                        "total_results": site_result["total_results"]
                    }
                    total_results += site_result["total_results"]

                except Exception as e:
                    self.logger.warning(f"网站 {websites[i]} 搜索失败: {e}")
                    results[websites[i]] = {
                        "results": [],
                        "total_results": 0,
                        "error": str(e)
                    }

            execution_time = time.time() - start_time

            final_result = {
                "query": query,
                "websites": websites,
                "site_results": results,
                "total_results": total_results,
                "execution_time": execution_time
            }

            self.logger.info(
                "多网站搜索成功",
                websites_count=len(websites),
                total_results=total_results,
                execution_time=execution_time
            )

            return final_result

        except Exception as e:
            error_msg = f"多网站搜索失败: {e}"
            self.logger.error(error_msg)
            raise APIError(error_msg, provider="metaso")

    async def search_websites_async(
        self,
        query: str,
        websites: List[str],
        max_results_per_site: int = 5
    ) -> Dict[str, Any]:
        """
        异步在指定网站中搜索

        Args:
            query: 搜索查询
            websites: 网站列表
            max_results_per_site: 每个网站的最大结果数

        Returns:
            搜索结果
        """
        try:
            self.logger.info(
                "开始异步在指定网站搜索",
                query=query,
                websites=websites,
                max_results_per_site=max_results_per_site
            )

            start_time = time.time()

            # 构建站点特定的搜索查询
            site_queries = []
            for site in websites:
                site_query = f"site:{site} {query}"
                site_queries.append(site_query)

            # 并发执行异步搜索
            tasks = []
            for site_query in site_queries:
                task = self.search_async(
                    query=site_query,
                    max_results=max_results_per_site
                )
                tasks.append(task)

            site_results = await asyncio.gather(*tasks, return_exceptions=True)

            results = {}
            total_results = 0

            for i, site_result in enumerate(site_results):
                website = websites[i]

                if isinstance(site_result, Exception):
                    self.logger.warning(f"网站 {website} 搜索失败: {site_result}")
                    results[website] = {
                        "results": [],
                        "total_results": 0,
                        "error": str(site_result)
                    }
                else:
                    results[website] = {
                        "results": site_result["results"],
                        "total_results": site_result["total_results"]
                    }
                    total_results += site_result["total_results"]

            execution_time = time.time() - start_time

            final_result = {
                "query": query,
                "websites": websites,
                "site_results": results,
                "total_results": total_results,
                "execution_time": execution_time
            }

            self.logger.info(
                "异步多网站搜索成功",
                websites_count=len(websites),
                total_results=total_results,
                execution_time=execution_time
            )

            return final_result

        except Exception as e:
            error_msg = f"异步多网站搜索失败: {e}"
            self.logger.error(error_msg)
            raise APIError(error_msg, provider="metaso")

    def get_search_suggestions(
        self,
        query: str,
        language: str = "zh"
    ) -> List[str]:
        """
        获取搜索建议

        Args:
            query: 搜索查询
            language: 语言代码

        Returns:
            搜索建议列表
        """
        try:
            self.logger.info("获取搜索建议", query=query)

            params = {
                "q": query,
                "lang": language,
                "type": "suggestions"
            }

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "MCP-Floating-Ball/1.0"
            }

            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(
                    f"{self.base_url}/suggestions",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

            suggestions = data.get("suggestions", [])
            self.logger.info("获取搜索建议成功", suggestions_count=len(suggestions))

            return suggestions

        except Exception as e:
            self.logger.error(f"获取搜索建议失败: {e}")
            return []

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务信息字典
        """
        return {
            "provider": "metaso",
            "base_url": self.base_url,
            "supports_web_search": True,
            "supports_image_search": True,
            "supports_video_search": True,
            "supports_news_search": True,
            "supports_site_search": True,
            "supports_suggestions": True,
            "timeout": self.config.timeout
        }

    async def test_connection(self) -> bool:
        """
        测试连接

        Returns:
            连接是否成功
        """
        try:
            # 执行一个简单的搜索测试
            result = await self.search_async("test", max_results=1)
            return bool(result.get("results"))
        except Exception as e:
            self.logger.error(f"秘塔搜索连接测试失败: {e}")
            return False


# 导出的类和函数
__all__ = [
    "MetasoProvider",
]