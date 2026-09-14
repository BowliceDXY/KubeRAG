"""
从知识库文档块自动生成 RAG 测试题集。
对每个采样的文档块，让 LLM 生成一个"该块能回答的问题"，作为检索评测的 ground truth。
输出: eval/testset.jsonl
"""
import sys, os, json, random
sys.path.insert(0, r"D:\KubeRAG")
os.chdir(r"D:\KubeRAG")
from dotenv import load_dotenv
load_dotenv()

from app.rag import collection, llm_client

NUM_QUESTIONS = 20
OUTPUT = r"D:\KubeRAG\eval\testset.jsonl"

def sample_chunks(n=20):
    """从 ChromaDB 采样 n 个块，尽量覆盖不同文档来源"""
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    # 按来源分组
    by_source = {}
    for i, meta in enumerate(metas):
        src = meta.get("source", "unknown")
        by_source.setdefault(src, []).append(i)

    print(f"知识库共 {len(ids)} 个块，来自 {len(by_source)} 个文档")
    for src, indices in by_source.items():
        print(f"  {src}: {len(indices)} 块")

    # 均匀采样：每个来源分配配额
    sampled = []
    per_source = max(1, n // len(by_source))
    for src, indices in by_source.items():
        k = min(per_source, len(indices))
        sampled.extend(random.sample(indices, k))
    # 补齐到 n
    remaining = [i for i in range(len(ids)) if i not in sampled]
    random.shuffle(remaining)
    while len(sampled) < n and remaining:
        sampled.append(remaining.pop())

    return [(ids[i], docs[i], metas[i]) for i in sampled[:n]]


def generate_question(chunk_text, source, page):
    """让 LLM 根据文档块生成一个可回答的问题"""
    prompt = f"""请根据以下文档片段，生成一个用户可能会问的问题，要求该问题的答案就包含在这个片段中。
只输出问题本身，不要解释，不要加引号。

【文档来源】{source} 第{page}页
【文档片段】
{chunk_text[:500]}

请生成一个问题："""

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    question = resp.choices[0].message.content.strip()
    # 清理可能的引号
    question = question.strip('"').strip("'").strip("「」").strip("『』")
    return question


def main():
    random.seed(42)  # 可复现
    chunks = sample_chunks(NUM_QUESTIONS)

    testset = []
    for i, (chunk_id, text, meta) in enumerate(chunks):
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        print(f"[{i+1}/{NUM_QUESTIONS}] 生成问题: {source} 第{page}页 ...", end=" ", flush=True)
        try:
            question = generate_question(text, source, page)
            print(f"→ {question[:40]}")
            testset.append({
                "question": question,
                "expected_id": chunk_id,
                "expected_source": source,
                "expected_page": page,
                "chunk_text": text[:300],
            })
        except Exception as e:
            print(f"失败: {e}")

    # 保存
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in testset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n已生成 {len(testset)} 道测试题 → {OUTPUT}")


if __name__ == "__main__":
    main()
