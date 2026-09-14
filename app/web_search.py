"""
网络搜索工具：知识库没有相关内容时，从公开网络检索资料

- 优先 Tavily API（专为 RAG/LLM 设计，返回干净正文，需 TAVILY_API_KEY）
- 降级 DuckDuckGo（免费无需 Key，需 pip install duckduckgo-search）
- 两者都不可用时返回 []，调用方降级为原"知识库未找到"提示
"""
import os
import logging
import httpx

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


def web_search(query, max_results=5):
    """
    搜索公开网络，返回 [{title, url, content, type:"web"}]
    失败时返回 []（不抛异常，调用方自行降级）
    """
    # 优先 Tavily
    if TAVILY_API_KEY:
        try:
            results = _tavily_search(query, max_results)
            if results:
                logger.info(f"网络搜索(Tavily)：{query[:30]}... → {len(results)}条")
                return results
        except Exception as e:
            logger.warning(f"Tavily 搜索失败，降级 DuckDuckGo: {e}")

    # 降级 DuckDuckGo
    try:
        results = _ddg_search(query, max_results)
        if results:
            logger.info(f"网络搜索(DuckDuckGo)：{query[:30]}... → {len(results)}条")
            return results
    except ImportError:
        logger.warning("未安装 duckduckgo-search，跳过网络搜索（pip install duckduckgo-search）")
    except Exception as e:
        logger.warning(f"DuckDuckGo 搜索失败: {e}")

    return []


def _tavily_search(query, max_results):
    """Tavily 搜索：专为 RAG 设计，返回页面正文摘要"""
    resp = httpx.post(
        TAVILY_URL,
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "type": "web",
        }
        for r in data.get("results", [])
    ]


def _ddg_search(query, max_results):
    """DuckDuckGo 搜索：免费无需 Key，延迟导入（未安装时不报错）"""
    from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
                "type": "web",
            })
    return results
