"""
语义缓存：相同/相似问题直接返回缓存答案，省 API 费用+提速
支持 Redis 存储（生产级），连不上 Redis 时自动降级到内存缓存
"""
import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 智谱 embedding 客户端
embedding_client = OpenAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

CACHE_THRESHOLD = 0.9  # 相似度阈值
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

# ===== 核心函数 =====
def cosine_similarity(a, b):
    """余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_cached_answer(question):
    """
    检查缓存：如果有相似问题，返回 (answer, sources, similarity)
    没有则返回 None
    """
    resp = embedding_client.embeddings.create(model="embedding-3", input=question)
    question_emb = resp.data[0].embedding

    cache_list = _get_all_cache()
    for item in cache_list:
        sim = cosine_similarity(question_emb, item["embedding"])
        if sim >= CACHE_THRESHOLD:
            return item["answer"], item["sources"], sim

    return None

def save_to_cache(question, answer, sources):
    """存入缓存"""
    resp = embedding_client.embeddings.create(model="embedding-3", input=question)
    question_emb = resp.data[0].embedding

    cache_list = _get_all_cache()
    cache_list.append({
        "question": question,
        "embedding": question_emb,
        "answer": answer,
        "sources": sources
    })
    _save_all_cache(cache_list)