"""
语义缓存：相同/相似问题直接返回缓存答案，省 API 费用+提速
支持 Redis 存储（生产级），连不上 Redis 时自动降级到内存缓存

v2 改进（P0）：
1. embedding 由调用方传入，缓存判定与向量检索复用同一次 embedding 调用
2. 写入前查重：相似问题直接更新，避免缓存无限膨胀
3. 容量上限 + TTL 过期，控制 Redis 单 key 体积与扫描成本
"""
import os
import json
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 智谱 embedding 客户端
embedding_client = OpenAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

CACHE_THRESHOLD = 0.9       # 相似度阈值
CACHE_TTL_SECONDS = 24 * 3600  # 缓存项 24 小时过期
MAX_CACHE_SIZE = 200        # 容量上限，超出淘汰最旧
REDIS_KEY = "kuberag:semantic_cache"  # Redis 里的缓存 key

# ===== Redis 连接（带降级）=====
_redis_client = None
_memory_cache = []  # 降级用的内存缓存

def _get_redis():
    """获取 Redis 连接，失败返回 None"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()  # 测试连接
        print("[缓存] Redis 连接成功，使用 Redis 语义缓存")
        return _redis_client
    except Exception as e:
        print(f"[缓存] Redis 连接失败，降级到内存缓存：{e}")
        _redis_client = None
        return None

def _get_all_cache():
    """从 Redis 或内存获取所有缓存项"""
    r = _get_redis()
    if r:
        try:
            data = r.get(REDIS_KEY)
            return json.loads(data) if data else []
        except Exception:
            return []
    return _memory_cache

def _save_all_cache(cache_list):
    """保存所有缓存项到 Redis 或内存"""
    r = _get_redis()
    if r:
        try:
            r.set(REDIS_KEY, json.dumps(cache_list))
        except Exception:
            pass
    else:
        global _memory_cache
        _memory_cache = cache_list

# ===== 治理：TTL 过期 + 容量上限 =====
def _prune(cache_list, now=None):
    """剔除已过期的缓存项"""
    now = now or time.time()
    cutoff = now - CACHE_TTL_SECONDS
    return [item for item in cache_list if item.get("ts", 0) > cutoff]

def _trim(cache_list):
    """超过容量上限时，按写入时间淘汰最旧的"""
    if len(cache_list) > MAX_CACHE_SIZE:
        cache_list.sort(key=lambda item: item.get("ts", 0), reverse=True)
        return cache_list[:MAX_CACHE_SIZE]
    return cache_list

# ===== 核心函数 =====
def cosine_similarity(a, b):
    """余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_cached_answer(question_emb):
    """
    检查缓存：如果有相似问题，返回 (answer, sources, similarity)
    没有则返回 None
    question_emb 由调用方计算后传入，避免重复 embedding 调用
    """
    cache_list = _prune(_get_all_cache())
    for item in cache_list:
        sim = cosine_similarity(question_emb, item["embedding"])
        if sim >= CACHE_THRESHOLD:
            return item["answer"], item["sources"], sim

    return None

def save_to_cache(question_emb, question, answer, sources):
    """
    存入缓存。写入前先查重：与已有条目相似度 >= 阈值时直接更新，
    避免同一问题反复追加导致缓存无限膨胀
    """
    cache_list = _prune(_get_all_cache())
    # 统一转 list：无论调用方传 list 还是 numpy 数组都可 JSON 序列化
    new_item = {
        "question": question,
        "embedding": list(question_emb),
        "answer": answer,
        "sources": sources,
        "ts": time.time(),
    }

    # 查重更新
    for i, item in enumerate(cache_list):
        sim = cosine_similarity(question_emb, item["embedding"])
        if sim >= CACHE_THRESHOLD:
            cache_list[i] = new_item
            _save_all_cache(_trim(cache_list))
            return

    # 追加新条目
    cache_list.append(new_item)
    _save_all_cache(_trim(cache_list))
