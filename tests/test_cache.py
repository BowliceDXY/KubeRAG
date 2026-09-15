"""语义缓存：余弦相似度纯函数测试"""
import math

import numpy as np

from app.cache import cosine_similarity


def test_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == 1.0


def test_orthogonal_vectors():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_similar_vectors_exceed_cache_threshold():
    # 夹角余弦 0.95，应高于缓存命中阈值 0.9
    a = np.array([1.0, 0.0])
    b = np.array([0.95, math.sqrt(1 - 0.95**2)])
    assert cosine_similarity(a, b) == 0.95
    assert cosine_similarity(a, b) > 0.9
