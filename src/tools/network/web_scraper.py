"""
MCP Floating Ball - 网页内容提取工具

提供网页抓取和内容提取功能，支持文本、链接、图片等多种内容提取。
"""

import time
import json
import re
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path

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

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from ...core.logging import get_logger
from ...core.exceptions import ToolError
from ..base import BaseTool, ToolMetadata, ToolCategory, ParameterType, ToolParameter

logger = get_logger(__name__)


class WebScraperTool(BaseTool):
    """网页内容提取工具"""

    def __init__(self):
        """初始化网页内容提取工具"""
        super().__init__()
        self.logger = get_logger("tool.web_scraper")

        # 请求头配置
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="web_scraper",
            display_name="网页内容提取",
            description="网页内容提取工具，支持文本、链接、图片等内容提取和批量抓取",
            category=ToolCategory.NETWORK,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["web", "scraper", "content", "html", "parser"],
            parameters=[
                ToolParameter(
                    name="url",
                    type=ParameterType.STRING,
                    description="目标网页URL",
                    required=True
                ),
                ToolParameter(
                    name="operation",
                    type=ParameterType.STRING,
                    description="操作类型",
                    required=True,
                    enum=["extract_content", "extract_links", "extract_images", "extract_forms", "get_metadata", "screenshot", "bulk_scrape"]
                ),
                ToolParameter(
                    name="selector",
                    type=ParameterType.STRING,
                    description="CSS选择器",
                    required=False
                ),
                ToolParameter(
                    name="output_format",
                    type=ParameterType.STRING,
                    description="输出格式",
                    required=False,
                    enum=["json", "text", "markdown", "html"],
                    default="json"
                ),
                ToolParameter(
                    name="use_selenium",
                    type=ParameterType.BOOLEAN,
                    description="是否使用Selenium处理动态内容",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="wait_time",
                    type=ParameterType.INTEGER,
                    description="等待时间（秒）",
                    required=False,
                    default=5
                )
            ],
            examples=["提取网页正文内容", "获取页面所有链接", "下载网页中的图片", "截取网页截图"]
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
                    "extract_content", "extract_links", "extract_images",
                    "extract_tables", "extract_forms", "extract_metadata",
                    "scrape_page", "batch_scrape", "rss_feed", "sitemap"
                ],
                examples=["extract_content", "extract_links", "scrape_page"]
            ),
            ToolParameter(
                name="url",
                type=ParameterType.STRING.value,
                description="网页URL",
                required=False,
                examples=["https://www.example.com", "https://news.sina.com.cn"]
            ),
            ToolParameter(
                name="urls",
                type=ParameterType.ARRAY.value,
                description="URL列表（用于批量操作）",
                required=False,
                examples=[["https://site1.com", "https://site2.com"]]
            ),
            ToolParameter(
                name="selector",
                type=ParameterType.STRING.value,
                description="CSS选择器",
                required=False,
                examples=[".content", "div.article", "#main"]
            ),
            ToolParameter(
                name="format",
                type=ParameterType.STRING.value,
                description="输出格式",
                required=False,
                choices=["text", "html", "json", "markdown"],
                default="text",
                examples=["text", "json", "markdown"]
            ),
            ToolParameter(
                name="use_selenium",
                type=ParameterType.BOOLEAN.value,
                description="是否使用Selenium（用于动态内容）",
                required=False,
                default=False
            ),
            ToolParameter(
                name="wait_time",
                type=ParameterType.INTEGER.value,
                description="等待时间（秒）",
                required=False,
                default=5,
                examples=[3, 5, 10]
            ),
            ToolParameter(
                name="save_to_file",
                type=ParameterType.BOOLEAN.value,
                description="是否保存到文件",
                required=False,
                default=False
            ),
            ToolParameter(
                name="output_dir",
                type=ParameterType.STRING.value,
                description="输出目录",
                required=False,
                examples=["./output", "/path/to/output"]
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行网页内容提取

        Args:
            operation: 操作类型
            url: 网页URL
            urls: URL列表
            selector: CSS选择器
            format: 输出格式
            use_selenium: 是否使用Selenium
            wait_time: 等待时间
            save_to_file: 是否保存到文件
            output_dir: 输出目录

        Returns:
            提取结果
        """
        try:
            if not REQUESTS_AVAILABLE:
                raise ToolError("requests模块不可用，请安装: pip install requests")

            operation = kwargs.get("operation", "")
            if not operation:
                raise ToolError("操作类型不能为空")

            self.logger.info("开始网页内容提取", operation=operation, kwargs=kwargs)

            start_time = time.time()

            # 根据操作类型执行相应的方法
            if operation == "extract_content":
                result = self._extract_content(kwargs)
            elif operation == "extract_links":
                result = self._extract_links(kwargs)
            elif operation == "extract_images":
                result = self._extract_images(kwargs)
            elif operation == "extract_tables":
                result = self._extract_tables(kwargs)
            elif operation == "extract_forms":
                result = self._extract_forms(kwargs)
            elif operation == "extract_metadata":
                result = self._extract_metadata(kwargs)
            elif operation == "scrape_page":
                result = self._scrape_page(kwargs)
            elif operation == "batch_scrape":
                result = self._batch_scrape(kwargs)
            elif operation == "rss_feed":
                result = self._extract_rss_feed(kwargs)
            elif operation == "sitemap":
                result = self._extract_sitemap(kwargs)
            else:
                raise ToolError(f"不支持的操作类型: {operation}")

            execution_time = time.time() - start_time

            self.logger.info(
                "网页内容提取完成",
                operation=operation,
                success=result.get("success", False),
                execution_time=execution_time
            )

            result["execution_time"] = execution_time
            return result

        except Exception as e:
            error_msg = f"网页内容提取失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0
            }

    def _extract_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页内容"""
        url = params.get("url", "")
        selector = params.get("selector", "")
        output_format = params.get("format", "text")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                content = self._get_content_with_selenium(url, selector, wait_time)
            else:
                content = self._get_content_with_requests(url, selector)

            if not content:
                raise ToolError("无法获取网页内容")

            # 根据输出格式处理内容
            if output_format == "text":
                formatted_content = self._format_as_text(content)
            elif output_format == "html":
                formatted_content = str(content)
            elif output_format == "json":
                formatted_content = self._format_as_json(content)
            elif output_format == "markdown":
                formatted_content = self._format_as_markdown(content)
            else:
                formatted_content = self._format_as_text(content)

            return {
                "success": True,
                "url": url,
                "format": output_format,
                "content": formatted_content,
                "content_length": len(formatted_content),
                "selector": selector,
                "use_selenium": use_selenium,
                "message": f"成功提取网页内容: {url}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"内容提取失败: {e}"
            }

    def _extract_links(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页链接"""
        url = params.get("url", "")
        selector = params.get("selector", "a")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                soup = self._get_soup_with_selenium(url, wait_time)
            else:
                soup = self._get_soup_with_requests(url)

            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取链接
            links = []
            base_url = url

            link_elements = soup.select(selector) if selector else soup.find_all("a")
            for link_elem in link_elements:
                try:
                    href = link_elem.get("href", "")
                    text = link_elem.get_text(strip=True)
                    title = link_elem.get("title", "")

                    if href:
                        # 处理相对URL
                        absolute_url = urljoin(base_url, href)
                        parsed_url = urlparse(absolute_url)

                        link_info = {
                            "url": absolute_url,
                            "text": text,
                            "title": title,
                            "domain": parsed_url.netloc,
                            "is_internal": parsed_url.netloc == urlparse(base_url).netloc,
                            "is_anchor": href.startswith("#"),
                            "is_mailto": href.startswith("mailto:"),
                            "is_tel": href.startswith("tel:")
                        }

                        links.append(link_info)
                except Exception as e:
                    self.logger.warning(f"处理链接失败: {e}")
                    continue

            # 去重
            seen_urls = set()
            unique_links = []
            for link in links:
                if link["url"] not in seen_urls:
                    seen_urls.add(link["url"])
                    unique_links.append(link)

            return {
                "success": True,
                "url": url,
                "links": unique_links,
                "total_links": len(unique_links),
                "internal_links": len([l for l in unique_links if l["is_internal"]]),
                "external_links": len([l for l in unique_links if not l["is_internal"]]),
                "anchor_links": len([l for l in unique_links if l["is_anchor"]]),
                "message": f"成功提取链接: {len(unique_links)} 个"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"链接提取失败: {e}"
            }

    def _extract_images(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页图片"""
        url = params.get("url", "")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)
        download_images = params.get("download_images", False)
        output_dir = params.get("output_dir", "./images")

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                soup = self._get_soup_with_selenium(url, wait_time)
            else:
                soup = self._get_soup_with_requests(url)

            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取图片
            images = []
            base_url = url

            img_elements = soup.find_all("img")
            for img_elem in img_elements:
                try:
                    src = img_elem.get("src", "")
                    alt = img_elem.get("alt", "")
                    title = img_elem.get("title", "")

                    if src:
                        # 处理相对URL
                        absolute_url = urljoin(base_url, src)
                        parsed_url = urlparse(absolute_url)

                        image_info = {
                            "url": absolute_url,
                            "alt": alt,
                            "title": title,
                            "domain": parsed_url.netloc,
                            "file_type": self._get_image_file_type(absolute_url)
                        }

                        # 尝试获取图片尺寸
                        if img_elem.get("width") or img_elem.get("height"):
                            image_info["width"] = img_elem.get("width")
                            image_info["height"] = img_elem.get("height")

                        images.append(image_info)

                        # 下载图片
                        if download_images:
                            self._download_image(absolute_url, output_dir)

                except Exception as e:
                    self.logger.warning(f"处理图片失败: {e}")
                    continue

            return {
                "success": True,
                "url": url,
                "images": images,
                "total_images": len(images),
                "downloaded": len(images) if download_images else 0,
                "message": f"成功提取图片: {len(images)} 个"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"图片提取失败: {e}"
            }

    def _extract_tables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页表格"""
        url = params.get("url", "")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                soup = self._get_soup_with_selenium(url, wait_time)
            else:
                soup = self._get_soup_with_requests(url)

            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取表格
            tables = []
            table_elements = soup.find_all("table")

            for i, table_elem in enumerate(table_elements):
                try:
                    table_data = []
                    rows = table_elem.find_all("tr")

                    for row in rows:
                        row_data = []
                        # 处理表头
                        headers = row.find_all("th")
                        if headers:
                            row_data = [header.get_text(strip=True) for header in headers]
                        else:
                            # 处理普通单元格
                            cells = row.find_all(["td", "th"])
                            row_data = [cell.get_text(strip=True) for cell in cells]

                        if row_data:
                            table_data.append(row_data)

                    if table_data:
                        tables.append({
                            "index": i,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "data": table_data
                        })

                except Exception as e:
                    self.logger.warning(f"处理表格失败: {e}")
                    continue

            return {
                "success": True,
                "url": url,
                "tables": tables,
                "total_tables": len(tables),
                "message": f"成功提取表格: {len(tables)} 个"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"表格提取失败: {e}"
            }

    def _extract_forms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页表单"""
        url = params.get("url", "")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                soup = self._get_soup_with_selenium(url, wait_time)
            else:
                soup = self._get_soup_with_requests(url)

            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取表单
            forms = []
            form_elements = soup.find_all("form")

            for i, form_elem in enumerate(form_elements):
                try:
                    form_info = {
                        "index": i,
                        "action": form_elem.get("action", ""),
                        "method": form_elem.get("method", "get").lower(),
                        "fields": []
                    }

                    # 提取表单字段
                    inputs = form_elem.find_all(["input", "select", "textarea"])
                    for input_elem in inputs:
                        field_info = {
                            "tag": input_elem.name,
                            "type": input_elem.get("type", "text"),
                            "name": input_elem.get("name", ""),
                            "id": input_elem.get("id", ""),
                            "value": input_elem.get("value", ""),
                            "placeholder": input_elem.get("placeholder", ""),
                            "required": input_elem.has_attr("required"),
                            "disabled": input_elem.has_attr("disabled")
                        }

                        if input_elem.name == "select":
                            # 处理选择框选项
                            options = input_elem.find_all("option")
                            field_info["options"] = [
                                {
                                    "value": opt.get("value", ""),
                                    "text": opt.get_text(strip=True),
                                    "selected": opt.has_attr("selected")
                                }
                                for opt in options
                            ]

                        form_info["fields"].append(field_info)

                    forms.append(form_info)

                except Exception as e:
                    self.logger.warning(f"处理表单失败: {e}")
                    continue

            return {
                "success": True,
                "url": url,
                "forms": forms,
                "total_forms": len(forms),
                "message": f"成功提取表单: {len(forms)} 个"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"表单提取失败: {e}"
            }

    def _extract_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网页元数据"""
        url = params.get("url", "")
        use_selenium = params.get("use_selenium", False)
        wait_time = params.get("wait_time", 5)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if use_selenium:
                soup = self._get_soup_with_selenium(url, wait_time)
            else:
                soup = self._get_soup_with_requests(url)

            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取元数据
            metadata = {}

            # 基本元标签
            title_elem = soup.find("title")
            if title_elem:
                metadata["title"] = title_elem.get_text(strip=True)

            # Meta标签
            meta_tags = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content = meta.get("content")
                if name and content:
                    meta_tags[name] = content

            metadata["meta"] = meta_tags

            # 重要的元数据
            important_meta = [
                "description", "keywords", "author", "robots", "viewport",
                "og:title", "og:description", "og:image", "og:url",
                "twitter:title", "twitter:description", "twitter:image"
            ]

            for meta_name in important_meta:
                if meta_name in meta_tags:
                    metadata[meta_name.replace(":", "_")] = meta_tags[meta_name]

            # 链接标签
            links = {}
            for link in soup.find_all("link"):
                rel = link.get("rel", [])
                if isinstance(rel, list):
                    rel = " ".join(rel)
                href = link.get("href")
                if rel and href:
                    links[rel] = href

            metadata["links"] = links

            # 语言和编码
            html_elem = soup.find("html")
            if html_elem:
                metadata["lang"] = html_elem.get("lang")
                metadata["dir"] = html_elem.get("dir")

            # 结构化数据
            structured_data = []
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    structured_data.append(data)
                except:
                    continue

            if structured_data:
                metadata["structured_data"] = structured_data

            return {
                "success": True,
                "url": url,
                "metadata": metadata,
                "message": "成功提取网页元数据"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"元数据提取失败: {e}"
            }

    def _scrape_page(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """抓取完整页面"""
        url = params.get("url", "")
        output_format = params.get("format", "json")
        save_to_file = params.get("save_to_file", False)
        output_dir = params.get("output_dir", "./output")

        if not url:
            raise ToolError("URL不能为空")

        try:
            # 获取网页内容
            soup = self._get_soup_with_requests(url)
            if not soup:
                raise ToolError("无法获取网页内容")

            # 提取各种内容
            page_data = {
                "url": url,
                "timestamp": time.time(),
                "title": soup.title.get_text(strip=True) if soup.title else "",
                "content": {
                    "text": soup.get_text(strip=True),
                    "html": str(soup)
                },
                "links": self._extract_links({"url": url})["links"],
                "images": self._extract_images({"url": url})["images"],
                "metadata": self._extract_metadata({"url": url})["metadata"]
            }

            # 保存到文件
            if save_to_file:
                output_path = self._save_page_data(page_data, output_dir, output_format)

            return {
                "success": True,
                "url": url,
                "page_data": page_data,
                "output_path": output_path if save_to_file else None,
                "message": "成功抓取完整页面"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"页面抓取失败: {e}"
            }

    def _batch_scrape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """批量抓取网页"""
        urls = params.get("urls", [])
        operation = params.get("operation", "extract_content")
        save_to_file = params.get("save_to_file", False)
        output_dir = params.get("output_dir", "./output")

        if not urls:
            raise ToolError("URL列表不能为空")

        results = []
        successful_count = 0

        for url in urls:
            try:
                # 复制参数并设置当前URL
                scrape_params = params.copy()
                scrape_params["url"] = url
                scrape_params["operation"] = operation

                # 执行抓取
                if operation == "scrape_page":
                    result = self._scrape_page(scrape_params)
                elif operation == "extract_content":
                    result = self._extract_content(scrape_params)
                elif operation == "extract_links":
                    result = self._extract_links(scrape_params)
                else:
                    result = {"success": False, "error": f"不支持的操作: {operation}"}

                result["url"] = url
                results.append(result)

                if result.get("success"):
                    successful_count += 1

            except Exception as e:
                results.append({
                    "success": False,
                    "url": url,
                    "error": str(e)
                })

        return {
            "success": successful_count > 0,
            "total_urls": len(urls),
            "successful_count": successful_count,
            "failed_count": len(urls) - successful_count,
            "results": results,
            "message": f"批量抓取完成: {successful_count}/{len(urls)} 成功"
        }

    def _extract_rss_feed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取RSS/Atom feed"""
        url = params.get("url", "")
        max_items = params.get("max_items", 20)

        if not url:
            raise ToolError("URL不能为空")

        try:
            if not FEEDPARSER_AVAILABLE:
                raise ToolError("feedparser模块不可用，请安装: pip install feedparser")

            feed = feedparser.parse(url)

            if feed.bozo:
                return {
                    "success": False,
                    "error": f"Feed解析错误: {feed.bozo_exception}",
                    "message": "RSS/Atom feed解析失败"
                }

            # 提取feed信息
            feed_info = {
                "title": feed.feed.get("title", ""),
                "description": feed.feed.get("description", ""),
                "link": feed.feed.get("link", ""),
                "language": feed.feed.get("language", ""),
                "updated": feed.feed.get("updated", ""),
                "author": feed.feed.get("author", ""),
                "entries": []
            }

            # 提取条目
            for entry in feed.entries[:max_items]:
                entry_info = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "description": entry.get("description", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "updated": entry.get("updated", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.get("term", "") for tag in entry.get("tags", [])]
                }

                feed_info["entries"].append(entry_info)

            return {
                "success": True,
                "url": url,
                "feed_info": feed_info,
                "total_entries": len(feed_info["entries"]),
                "message": f"成功提取RSS feed: {len(feed_info['entries'])} 个条目"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"RSS feed提取失败: {e}"
            }

    def _extract_sitemap(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """提取网站地图"""
        url = params.get("url", "")
        max_urls = params.get("max_urls", 1000)

        if not url:
            raise ToolError("URL不能为空")

        try:
            soup = self._get_soup_with_requests(url)
            if not soup:
                raise ToolError("无法获取网站地图")

            urls = []

            # 提取URL
            url_elements = soup.find_all("url")
            for url_elem in url_elements[:max_urls]:
                try:
                    loc_elem = url_elem.find("loc")
                    if loc_elem:
                        url_info = {
                            "url": loc_elem.get_text(strip=True),
                            "lastmod": url_elem.find("lastmod").get_text(strip=True) if url_elem.find("lastmod") else "",
                            "changefreq": url_elem.find("changefreq").get_text(strip=True) if url_elem.find("changefreq") else "",
                            "priority": url_elem.find("priority").get_text(strip=True) if url_elem.find("priority") else ""
                        }
                        urls.append(url_info)
                except Exception as e:
                    continue

            return {
                "success": True,
                "sitemap_url": url,
                "urls": urls,
                "total_urls": len(urls),
                "message": f"成功提取网站地图: {len(urls)} 个URL"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"网站地图提取失败: {e}"
            }

    # 辅助方法
    def _get_content_with_requests(self, url: str, selector: str = "") -> Any:
        """使用requests获取内容"""
        response = requests.get(url, headers=self.default_headers, timeout=30)
        response.raise_for_status()

        if not BS4_AVAILABLE:
            return response.text

        soup = BeautifulSoup(response.text, 'html.parser')
        if selector:
            return soup.select_one(selector)
        return soup

    def _get_content_with_selenium(self, url: str, selector: str = "", wait_time: int = 5) -> Any:
        """使用Selenium获取内容"""
        if not SELENIUM_AVAILABLE:
            raise ToolError("selenium模块不可用，请安装: pip install selenium")

        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=chrome_options)
        try:
            driver.get(url)
            time.sleep(wait_time)

            if selector:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                return BeautifulSoup(element.get_attribute("outerHTML"), 'html.parser')
            else:
                return BeautifulSoup(driver.page_source, 'html.parser')
        finally:
            driver.quit()

    def _get_soup_with_requests(self, url: str) -> Any:
        """使用requests获取BeautifulSoup对象"""
        if not BS4_AVAILABLE:
            raise ToolError("beautifulsoup4模块不可用，请安装: pip install beautifulsoup4")

        response = requests.get(url, headers=self.default_headers, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')

    def _get_soup_with_selenium(self, url: str, wait_time: int = 5) -> Any:
        """使用Selenium获取BeautifulSoup对象"""
        if not BS4_AVAILABLE:
            raise ToolError("beautifulsoup4模块不可用，请安装: pip install beautifulsoup4")

        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=chrome_options)
        try:
            driver.get(url)
            time.sleep(wait_time)
            return BeautifulSoup(driver.page_source, 'html.parser')
        finally:
            driver.quit()

    def _format_as_text(self, content: Any) -> str:
        """格式化为纯文本"""
        if hasattr(content, 'get_text'):
            return content.get_text(strip=True)
        elif isinstance(content, str):
            return content
        else:
            return str(content)

    def _format_as_json(self, content: Any) -> str:
        """格式化为JSON"""
        if isinstance(content, str):
            return content
        else:
            return json.dumps(str(content), ensure_ascii=False, indent=2)

    def _format_as_markdown(self, content: Any) -> str:
        """格式化为Markdown"""
        if hasattr(content, 'get_text'):
            text = content.get_text()
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        # 简单的Markdown转换
        return text.replace('\n', '\n\n')

    def _get_image_file_type(self, url: str) -> str:
        """获取图片文件类型"""
        path = urlparse(url).path.lower()
        if path.endswith(('.jpg', '.jpeg')):
            return 'jpg'
        elif path.endswith('.png'):
            return 'png'
        elif path.endswith('.gif'):
            return 'gif'
        elif path.endswith('.webp'):
            return 'webp'
        elif path.endswith('.svg'):
            return 'svg'
        else:
            return 'unknown'

    def _download_image(self, url: str, output_dir: str):
        """下载图片"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"image_{int(time.time())}.jpg"

            filepath = os.path.join(output_dir, filename)

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

        except Exception as e:
            self.logger.error(f"下载图片失败 {url}: {e}")

    def _save_page_data(self, page_data: Dict[str, Any], output_dir: str, format_type: str) -> str:
        """保存页面数据到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        url = page_data["url"]
        filename = re.sub(r'[^\w\-_\.]', '_', urlparse(url).netloc + urlparse(url).path)
        if not filename:
            filename = f"page_{int(time.time())}"

        if format_type == "json":
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)
        elif format_type == "html":
            filepath = os.path.join(output_dir, f"{filename}.html")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_data["content"]["html"])
        else:
            filepath = os.path.join(output_dir, f"{filename}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_data["content"]["text"])

        return filepath


# 注册工具
from ..registry import tool_registry
tool_registry.register(WebScraperTool())