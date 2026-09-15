"""RRF 融合排序纯函数测试"""
from app.rag import rrf_fuse


def _item(text, source="a.pdf", page=1):
    return {"text": text, "metadata": {"source": source, "page": page}}


def test_fusion_merges_both_lists():
    vec = [_item(f"v{i}") for i in range(3)]
    bm = [_item(f"b{i}") for i in range(3)]
    fused = rrf_fuse(vec, bm, top_k=5)
    # 两个列表共 6 个不同文本，融合后取 top5
    assert len(fused) == 5


def test_common_item_ranks_first():
    common = _item("common")
    vec = [common, _item("v1"), _item("v2")]
    bm = [_item("b1"), _item("b2"), common]
    fused = rrf_fuse(vec, bm, top_k=5)
    # 同时在两个列表排名靠前的文本，融合后应排第一
    assert fused[0]["text"] == "common"


def test_dedup_by_text():
    a = _item("dup")
    b = _item("dup")
    fused = rrf_fuse([a], [b], top_k=5)
    texts = [f["text"] for f in fused]
    assert len(texts) == len(set(texts)) == 1
