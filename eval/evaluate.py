"""
RAG 检索质量评测脚本。
对比四种检索策略的 HitRate@k 和 MRR：
  1. BM25 关键词检索
  2. 向量语义检索
  3. 混合检索（BM25 + 向量 + RRF 融合）
  4. 混合检索 + LLM 重排序
输出: eval/results.json + 控制台报告
"""
import sys, os, json, time
sys.path.insert(0, r"D:\KubeRAG")
os.chdir(r"D:\KubeRAG")
from dotenv import load_dotenv
load_dotenv()

from app.rag import (
    collection, bm25_search, hybrid_search, rerank,
    _get_question_embedding, logger
)
import logging
logging.getLogger().setLevel(logging.WARNING)  # 评测时静默 info 日志

TESTSET = r"D:\KubeRAG\eval\testset.jsonl"
OUTPUT = r"D:\KubeRAG\eval\results.json"
TOP_K = 5


def vector_search(question, q_emb, top_k=5):
    """纯向量检索，返回 [{id, text, metadata, distance}]"""
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else 0,
        })
    return items


def chunk_matches(result, expected_text, expected_source):
    """
    判断检索结果是否为目标文档块。
    用文本前缀匹配（前80字符）+ 来源文件名匹配，比 ID 重建更稳健。
    """
    text = result.get("text", "")
    meta = result.get("metadata", {})
    source = meta.get("source", "")

    # 来源必须匹配（文件名部分）
    expected_basename = os.path.basename(expected_source) if expected_source else ""
    result_basename = os.path.basename(source) if source else ""
    if expected_basename and result_basename and expected_basename != result_basename:
        return False

    # 文本前缀匹配（前80字符，去除空白）
    expected_prefix = "".join(expected_text[:80].split())
    result_prefix = "".join(text[:80].split())
    if expected_prefix and result_prefix:
        return expected_prefix in result_prefix or result_prefix in expected_prefix

    return False


def evaluate_strategy(name, retrieve_fn, questions):
    """对一种检索策略计算 HitRate@k 和 MRR"""
    hits_at = {3: 0, 5: 0}
    reciprocal_ranks = []
    details = []

    for item in questions:
        q = item["question"]
        expected_text = item.get("chunk_text", "")
        expected_source = item.get("expected_source", "")
        results = retrieve_fn(q)

        # 找目标块在结果中的排名（1-based）
        rank = None
        for i, r in enumerate(results):
            if chunk_matches(r, expected_text, expected_source):
                rank = i + 1
                break

        if rank:
            if rank <= 3:
                hits_at[3] += 1
            if rank <= 5:
                hits_at[5] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append({
            "question": q[:50],
            "expected": os.path.basename(expected_source)[:25],
            "rank": rank,
            "found": rank is not None,
        })

    n = len(questions)
    return {
        "strategy": name,
        "hit_rate_at_3": round(hits_at[3] / n, 4),
        "hit_rate_at_5": round(hits_at[5] / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "details": details,
    }


def main():
    # 加载测试题集
    questions = []
    with open(TESTSET, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    print(f"加载测试题: {len(questions)} 道\n")

    # 策略 1: BM25
    print("[1/4] 评测 BM25 检索...", flush=True)
    def bm25_fn(q):
        return bm25_search(q, top_k=TOP_K)
    r_bm25 = evaluate_strategy("BM25", bm25_fn, questions)

    # 策略 2: 向量
    print("[2/4] 评测向量检索...", flush=True)
    def vec_fn(q):
        emb = _get_question_embedding(q)
        return vector_search(q, emb, top_k=TOP_K)
    r_vec = evaluate_strategy("Vector", vec_fn, questions)

    # 策略 3: 混合检索
    print("[3/4] 评测混合检索（BM25+Vector+RRF）...", flush=True)
    def hybrid_fn(q):
        emb = _get_question_embedding(q)
        return hybrid_search(q, top_k=10, q_emb=emb)[:TOP_K]
    r_hybrid = evaluate_strategy("Hybrid(RRF)", hybrid_fn, questions)

    # 策略 4: 混合 + LLM 重排序
    print("[4/4] 评测混合检索 + LLM 重排序（较慢，每道题一次 LLM 调用）...", flush=True)
    def rerank_fn(q):
        emb = _get_question_embedding(q)
        raw = hybrid_search(q, top_k=10, q_emb=emb)
        return rerank(q, raw, top_k=TOP_K)
    r_rerank = evaluate_strategy("Hybrid+Rerank", rerank_fn, questions)

    # 输出报告
    all_results = [r_bm25, r_vec, r_hybrid, r_rerank]

    print("\n" + "=" * 65)
    print(f"{'检索策略':<20} {'HitRate@3':>10} {'HitRate@5':>10} {'MRR':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['strategy']:<20} {r['hit_rate_at_3']:>10.2%} {r['hit_rate_at_5']:>10.2%} {r['mrr']:>10.4f}")
    print("=" * 65)

    # 逐题明细（只打印未命中的）
    print("\n未命中题目明细（混合+重排序）:")
    for d in r_rerank["details"]:
        if not d["found"]:
            print(f"  ✗ [{d['expected']}] {d['question']}")

    # 保存结果
    save_data = {r["strategy"]: {k: v for k, v in r.items() if k != "details"} for r in all_results}
    save_data["_testset_size"] = len(questions)
    save_data["_details"] = {r["strategy"]: r["details"] for r in all_results}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存 → {OUTPUT}")


if __name__ == "__main__":
    main()
