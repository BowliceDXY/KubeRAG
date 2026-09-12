import chromadb
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import pathlib
import logging
import jieba
from rank_bm25 import BM25Okapi
import httpx
from app.cache import get_cached_answer, save_to_cache

load_dotenv()

# ===== 路径配置 =====
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
UPLOAD_PATH = str(PROJECT_ROOT / "data" / "uploads")

# ===== 日志配置 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 两个客户端：智谱管“向量”,DeepSeek管“回答”
embed_client = OpenAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用持久化存储：数据存到本地文件夹，服务重启也不丢
db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db.get_or_create_collection(name="ai_docs")

# ===== BM25 混合检索需要的全局存储 =====
all_texts = [] # 所有文档块的文本（用于 BM25 关键词检索）
all_metadatas = [] # 对应的元数据（文件名、页码）
bm25_index = None # BM25 索引对象（延迟构建）

def build_bm25_index():
    """基于 all_texts 构建 BM25 索引（中文用 jieba 分词）"""
    global bm25_index
    if not all_texts:
        logger.warning("没有文档，跳过 BM25 索引构建")
        return
    tokenized_corpus = [list(jieba.cut(text)) for text in all_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 索引构建完成，共 {len(all_texts)} 个文档块")

def load_existing_docs():
    """服务重启时，从 ChromaDB 加载已有文档，重建 BM25 索引"""
    global all_texts, all_metadatas
    existing = collection.get(include=["documents", "metadatas"])
    if existing["documents"]:
        all_texts = existing["documents"]
        all_metadatas = existing["metadatas"]
        build_bm25_index()
        logger.info(f"从 ChromaDB 加载了 {len(all_texts)} 个已有文档块")

# 服务启动时自动加载已有文档
load_existing_docs()

def build_vector_store(pdf_path):
    """
    输入 PDF 路径 → 切片 → 向量化 → 存入 Chroma，同时更新 BM25 索引
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    metadatas = [
        {"source": os.path.basename(pdf_path), "page": c.metadata.get("page", 0) + 1}
        for c in chunks
    ]

    # 向量化
    vectors = []
    BATCH =60
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        try:
            resp = embed_client.embeddings.create(model="embedding-3", input=batch)
            vectors.extend([item.embedding for item in resp.data])
            logger.info(f"向量化进度 {len(vectors)}/{len(texts)}")
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            raise

    file_name = os.path.basename(pdf_path)
    ids = [f"{file_name}_{i}" for i in range(len(texts))]

    # 存入 Chroma
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    # ===== 新增：追加到 BM25 的全局存储，并重建索引 =====
    all_texts.extend(texts)
    all_metadatas.extend(metadatas)
    build_bm25_index()

    logger.info(f"入库完成：{file_name}，共 {len(texts)} 块")
    return len(texts)

def bm25_search(query, top_k=10):
    """BM25 关键词检索：擅长精确匹配（术语、编号、专有名词）"""
    if bm25_index is None:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "text": all_texts[idx],
            "metadata": all_metadatas[idx],
            "bm25_score": float(scores[idx])
        })
    return results

def vector_search(query, top_k=10):
    """向量语义检索：擅长理解意思相似但用词不同的问题"""
    q_resp = embed_client.embeddings.create(model="embedding-3", input=query)
    q_vec = q_resp.data[0].embedding
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
        })
    return output

def rrf_fuse(vector_results, bm25_results, top_k=5, rrf_k=60):
    """
    RRF (Reciprocal Rank Fusion) 融合排序
    公式: score = 1 / (k + rank)
    两个检索结果按排名各自打分，求和后排序
    优点：不需要关心向量相似度和 BM25 分数的量纲差异
    """
    scores = {}
    for rank, item in enumerate(vector_results):
        key = item["text"]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, item in enumerate(bm25_results):
        key = item["text"]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)

    # 找回对应的文本和元数据
    result_map = {}
    for item in vector_results + bm25_results:
        result_map[item["text"]] = item

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for text, score in sorted_items[:top_k]:
        if text in result_map:
            fused.append({
                "text": text,
                "metadata": result_map[text]["metadata"],
                "rrf_score": score
            })
    return fused

def hybrid_search(question, top_k=5):
    """
    混合检索：向量检索 + BM25 关键词检索 → RRF 融合
    比纯向量检索更准：能同时命中"意思相似"和"关键词精确匹配"
    """
    vec_results = vector_search(question, top_k=10)
    bm_results = bm25_search(question, top_k=10)
    fused = rrf_fuse(vec_results, bm_results, top_k=top_k)
    logger.info(f"混合检索：向量{len(vec_results)}条 + BM25{len(bm_results)}条 → 融合后{len(fused)}条")
    return fused

def rerank(query, documents, top_k=5):
    """
    LLM-based 重排序：让 DeepSeek 判断每段文本和问题的相关性，按相关性排序
    不依赖外部 rerank API，用已有的 DeepSeek 即可
    如果调用失败，自动降级返回原结果
    """
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents

    try:
        # 构造让 LLM 打分的 prompt
        docs_text = "\n".join([
            f"[{i+1}] {doc['text'][:200]}"
            for i, doc in enumerate(documents)
        ])
        prompt = f"""你是一个相关性评估专家。请判断以下文档片段与用户问题的相关程度。

【用户问题】{query}

【候选文档】
{docs_text}

请按相关性从高到低排序，只返回编号，格式如：3,1,5,2,4（取前{top_k}个最相关的）。不要解释。"""

        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()

        # 解析返回的编号
        import re
        indices = [int(x) - 1 for x in re.findall(r'\d+', answer)]
        indices = [i for i in indices if 0 <= i < len(documents)]

        # 按 LLM 排序的顺序重组结果
        reranked = [documents[i] for i in indices[:top_k]]
        # 补全（防止 LLM 返回编号不全）
        if len(reranked) < top_k:
            for i in range(len(documents)):
                if i not in indices and len(reranked) < top_k:
                    reranked.append(documents[i])

        logger.info(f"LLM重排序完成：{len(documents)}条 → {len(reranked)}条")
        return reranked[:top_k]
    except Exception as e:
        logger.warning(f"重排序失败，降级使用混合检索结果: {e}")
        return documents[:top_k]

def ask(question):
    """输入问题 → 混合检索 → 重排序 → DeepSeek 基于检索结果回答 → 返回答案+出处"""
    # 先查语义缓存
    cached = get_cached_answer(question)
    if cached:
        answer, sources, sim = cached
        logger.info(f"缓存命中：问题={question[:30]}...，相似度={sim:.2f}")
        return {"answer": answer, "sources": sources}
    raw_results = hybrid_search(question, top_k=10)
    results = rerank(question, raw_results, top_k=5)

    if not results:
        return {
            "answer": "抱歉，知识库中没有找到相关内容，请先上传文档。",
            "sources": []
        }

    context_parts = []
    sources = []
    for i, item in enumerate(results):
        context_parts.append(
            f"[资料{i+1}]（来自 {item['metadata']['source']} 第{item['metadata']['page']}页）\n{item['text']}"
        )
        sources.append({
            "file": item["metadata"]["source"],
            "page": item["metadata"]["page"],
            "content": item["text"][:100]
        })

    context = "\n\n".join(context_parts)

    prompt = f"""请根据以下文档回答用户问题。如果文档中没有提到相关信息，请如实说明。
回答时请在关键结论后用 [资料X] 标注引用来源。

【文档内容】
{context}

【用户问题】
{question}

请基于文档内容回答："""

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是个专业的AI助手，请严格基于提供的文档内容回答，不要编造。"},
            {"role": "user", "content": prompt}
        ]
    )

    answer = resp.choices[0].message.content
    logger.info(f"问答完成：问题={question[:30]}...，来源数={len(sources)}")

    # 存入语义缓存
    save_to_cache(question, answer, sources)

    return {
        "answer": answer,
        "sources": sources
    }

def ask_stream(question, history=None):
    """
    流式问答：逐字返回答案（像 ChatGPT 一样一个字一个字蹦出来）
    history: 对话历史，格式 [{"role": "user"/"assistant", "content": "..."}]
    返回：生成器，逐字 yield
    """
    # 检索 + 重排序（和 ask 一样）
    results = hybrid_search(question, top_k=10)
    results = rerank(question, results, top_k=5)

    if not results:
        yield "抱歉，知识库中没有找到相关内容，请先上传文档。"
        return

    # 组装上下文
    context_parts = []
    for i, item in enumerate(results):
        context_parts.append(
            f"[资料{i+1}]（来自 {item['metadata']['source']} 第{item['metadata']['page']}页）\n{item['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""请根据以下文档回答用户问题。如果文档中没有提到相关信息，请如实说明。
回答时请在关键结论后用 [资料X] 标注引用来源。

【文档内容】
{context}

【用户问题】
{question}

请基于文档内容回答："""

    # 构造消息（包含历史对话）
    messages = [
        {"role": "system", "content": "你是个专业的AI助手，请严格基于提供的文档内容回答，不要编造。"}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 流式调用 DeepSeek（stream=True）
    stream = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    logger.info(f"流式问答完成：问题={question[:30]}...")