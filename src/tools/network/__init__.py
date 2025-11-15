"""
MCP Floating Ball - 网络工具模块

包含网络相关的工具：
- multi_search: 多搜索引擎
- web_scraper: 网页内容提取
"""

from .multi_search import MultiSearchTool
from .web_scraper import WebScraperTool

__all__ = [
    "MultiSearchTool",
    "WebScraperTool"
]