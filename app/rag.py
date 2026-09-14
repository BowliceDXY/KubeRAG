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
from app.web_search import web_search

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

# 两个客户端：智谱管"向量",DeepSeek管"回答"
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

def build_vector_store(pdf_path, source_name=None):
    """
    输入 PDF 路径 → 切片 → 向量化 → 存入 Chroma，同时更新 BM25 索引
    source_name: 展示用的可读文件名（默认取路径 basename）
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    display_name = source_name or os.path.basename(pdf_path)
    metadatas = [
        {"source": display_name, "page": c.metadata.get("page", 0) + 1}
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

    # ===== 追加到 BM25 的全局存储，并重建索引 =====
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

def vector_search(query, top_k=10, q_emb=None):
    """
    向量语义检索：擅长理解意思相似但用词不同的问题
    q_emb 可由调用方传入（与缓存判定共用同一次 embedding），为空时自行计算
    """
    if q_emb is None:
        q_resp = embed_client.embeddings.create(model="embedding-3", input=query)
        q_emb = q_resp.data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
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

def hybrid_search(question, top_k=5, q_emb=None):
    """
    混合检索：向量检索 + BM25 关键词检索 → RRF 融合
    比纯向量检索更准：能同时命中"意思相似"和"关键词精确匹配"
    """
    vec_results = vector_search(question, top_k=10, q_emb=q_emb)
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

# ===== 多轮对话支持 =====

def _trim_history(history, max_messages=6, max_chars=2000):
    """
    上下文窗口管理：限制带入的消息条数与总字符数，超出从最旧开始丢弃。
    避免历史随轮数线性膨胀、最终超出模型上下文窗口
    """
    if not history:
        return []
    trimmed = list(history)
    while len(trimmed) > max_messages or sum(len(m.get("content", "")) for m in trimmed) > max_chars:
        if len(trimmed) <= 2:  # 至少保留最近一轮
            break
        trimmed.pop(0)
    return trimmed

def rewrite_query(question, history):
    """
    多轮追问改写：结合历史把指代词（"它""这个方案""第三页"等）补全为
    一段可以独立检索文档的完整问句。改写失败时降级返回原问题。
    """
    if not history or len(history) < 2:
        return question
    try:
        history_text = "\n".join(
            f"{m['role']}: {m.get('content', '')[:100]}" for m in history[-4:]
        )
        prompt = f"""你是查询改写助手。根据对话历史，把用户最新问题改写成一段可以独立用于文档检索的完整问句。
如果最新问题本身已经很完整，则原样返回。只输出改写后的问句，不要解释。

【对话历史】
{history_text}

【用户最新问题】
{question}

改写后的问题："""
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and rewritten != question:
            logger.info(f"追问改写：{question[:30]}... → {rewritten[:30]}...")
            return rewritten
    except Exception as e:
        logger.warning(f"查询改写失败，使用原问题: {e}")
    return question

def _get_question_embedding(question):
    """计算问题向量（缓存判定与向量检索共用，避免重复调用 embedding API）"""
    resp = embed_client.embeddings.create(model="embedding-3", input=question)
    return resp.data[0].embedding

def _build_context(results):
    """把检索结果组装成 prompt 上下文 + 引用列表"""
    context_parts = []
    sources = []
    for i, item in enumerate(results):
        context_parts.append(
            f"[资料{i+1}]（来自 {item['metadata']['source']} 第{item['metadata']['page']}页）\n{item['text']}"
        )
        sources.append({
            "file": item["metadata"]["source"],
            "page": item["metadata"]["page"],
            "content": item["text"][:100],
            "type": "kb",
        })
    return "\n\n".join(context_parts), sources


def _build_web_context(results):
    """把网络搜索结果组装成 prompt 上下文 + 引用列表（标注为网络来源）"""
    context_parts = []
    sources = []
    for i, item in enumerate(results):
        context_parts.append(
            f"[资料{i+1}]（来自网络：{item['title']} - {item['url']}）\n{item['content']}"
        )
        sources.append({
            "file": item["title"],
            "page": item["url"],
            "content": item["content"][:100],
            "type": "web",
        })
    return "\n\n".join(context_parts), sources


# 知识库相关性阈值（用于判断是否降级到网络搜索）
KB_DISTANCE_THRESHOLD = 0.80  # cosine distance：最近文档距离 > 此值 → 知识库无语义相关内容
KB_BM25_THRESHOLD = 10.0      # BM25 最高分 > 此值 → 关键词强匹配，仍走知识库（兜底）


def _check_kb_answerable(question, context):
    """
    LLM 可回答性检查：判断知识库上下文是否包含回答该问题的具体信息。
    解决"同领域但无答案"问题——向量距离近不代表文档里真有答案。
    返回 True（可回答）或 False（不可回答，应降级网络）。
    """
    if not context:
        return False
    try:
        prompt = f"""请判断以下文档内容是否包含回答用户问题所需的具体信息。
只回答"是"或"否"，不要解释。

【用户问题】{question}

【文档内容】
{context[:2000]}

是否包含回答该问题的具体信息？（是/否）"""
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        answer = resp.choices[0].message.content.strip()
        is_answerable = "是" in answer
        logger.info(f"可回答性检查：问题='{question[:25]}...' → {'可回答(知识库)' if is_answerable else '不可回答(降级网络)'}")
        return is_answerable
    except Exception as e:
        logger.warning(f"可回答性检查失败，默认使用知识库: {e}")
        return True  # 检查失败时保守走知识库，避免误降级


def _retrieve(question, q_emb):
    """
    统一检索入口：知识库优先，无相关内容时自动降级到网络搜索。
    返回 (context, sources, source_type)，source_type 为 "kb" / "web" / "none"。

    三级降级策略：
      1. 快速路由：向量距离 > 0.80 且 BM25 < 10 → 直接联网（省一次知识库检索）
      2. 知识库检索 + LLM 可回答性检查：文档能回答 → 用知识库；不能回答 → 降级联网
      3. 网络也无结果 → 返回 none
    """
    # 信号1：向量距离（轻量查 1 条，取最近距离）
    try:
        vec_probe = collection.query(query_embeddings=[q_emb], n_results=1)
        min_distance = vec_probe["distances"][0][0] if vec_probe.get("distances") else 999.0
    except Exception:
        min_distance = 999.0

    # 信号2：BM25 最高分
    bm_results = bm25_search(question, top_k=5)
    max_bm25 = max((r["bm25_score"] for r in bm_results), default=0.0)

    # 第一级：明显不相关 → 直接联网，省掉知识库检索和重排序
    if min_distance > KB_DISTANCE_THRESHOLD and max_bm25 < KB_BM25_THRESHOLD:
        logger.info(
            f"检索路由：问题='{question[:25]}...' 向量距离={min_distance:.3f} "
            f"BM25最高={max_bm25:.1f} → 直接降级网络"
        )
        web_results = web_search(question, max_results=5)
        if web_results:
            context, sources = _build_web_context(web_results)
            return context, sources, "web"
        return "", [], "none"

    # 第二级：知识库检索 + 重排序
    logger.info(
        f"检索路由：问题='{question[:25]}...' 向量距离={min_distance:.3f} "
        f"BM25最高={max_bm25:.1f} → 知识库检索"
    )
    raw_results = hybrid_search(question, top_k=10, q_emb=q_emb)
    results = rerank(question, raw_results, top_k=5)

    if not results:
        # 知识库检索为空 → 降级联网
        web_results = web_search(question, max_results=5)
        if web_results:
            context, sources = _build_web_context(web_results)
            return context, sources, "web"
        return "", [], "none"

    context, sources = _build_context(results)

    # 第三级：LLM 可回答性检查——文档里到底有没有这个问题的答案
    if not _check_kb_answerable(question, context):
        logger.info(f"知识库无法回答，降级网络搜索：{question[:30]}...")
        web_results = web_search(question, max_results=5)
        if web_results:
            context, sources = _build_web_context(web_results)
            return context, sources, "web"
        # 网络也没结果时，仍返回知识库内容（至少有相关文档）
        return context, sources, "kb"

    return context, sources, "kb"

def ask(question):
    """
    非流式问答：ask_stream 的薄封装，收集所有流式事件后一次性返回。
    检索、缓存、降级、可回答性检查等全部逻辑复用 ask_stream，保证两接口行为一致。
    """
    answer = ""
    sources = []
    for event in ask_stream(question):
        if event["type"] == "delta":
            answer += event["content"]
        elif event["type"] == "sources":
            sources = event["sources"]
    return {"answer": answer, "sources": sources}

def ask_stream(question, history=None):
    """
    流式问答：逐字返回答案。
    以事件 dict 形式 yield：
      {"type": "delta",  "content": str}        答案增量
      {"type": "sources", "sources": [...]}      引用来源（答案结束后）
    流程：查缓存（命中直接返回）→ 追问改写 → 混合检索 → 重排 → 流式生成 → 写缓存
    """
    # 一次 embedding：缓存判定 + 向量检索共用
    q_emb = _get_question_embedding(question)

    # 先查语义缓存：命中则直接返回缓存答案，不调用 LLM
    cached = get_cached_answer(q_emb)
    if cached:
        answer, sources, sim = cached
        logger.info(f"[流式] 缓存命中：问题={question[:30]}...，相似度={sim:.2f}")
        yield {"type": "delta", "content": answer}
        yield {"type": "sources", "sources": sources}
        return

    # 上下文窗口管理 + 多轮追问改写
    history = _trim_history(history)
    search_query = rewrite_query(question, history)

    # 改写后若查询词变化，向量检索用改写后的 embedding；否则复用缓存判定的 embedding
    search_emb = q_emb
    if search_query != question:
        search_emb = _get_question_embedding(search_query)

    context, sources, source_type = _retrieve(search_query, search_emb)

    if source_type == "none":
        yield {"type": "delta", "content": "抱歉，知识库和网络搜索都没有找到相关内容。你可以尝试换个问法，或先上传相关文档。"}
        yield {"type": "sources", "sources": []}
        return

    # 根据来源类型选择 prompt
    if source_type == "web":
        prompt = f"""请根据以下网络公开资料回答用户问题。回答时注明信息来自网络公开资料，并在关键结论后用 [资料X] 标注来源。如果资料中没有相关信息，请如实说明。

【网络公开资料】
{context}

【用户问题】
{question}

请回答："""
        system_content = "你是个专业的AI助手。回答基于网络公开资料，注明来源，不要编造。"
    else:
        prompt = f"""请根据以下文档回答用户问题。如果文档中没有提到相关信息，请如实说明。
回答时请在关键结论后用 [资料X] 标注引用来源。

【文档内容】
{context}

【用户问题】
{question}

请基于文档内容回答："""
        system_content = "你是个专业的AI助手，请严格基于提供的文档内容回答，不要编造。"

    # 构造消息（历史只含纯问答对，不含完整 prompt，避免膨胀）
    messages = [
        {"role": "system", "content": system_content}
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

    full_answer = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_answer += token
            yield {"type": "delta", "content": token}

    # 写入语义缓存
    save_to_cache(q_emb, question, full_answer, sources)

    logger.info(f"流式问答完成：问题={question[:30]}...，来源={source_type}，来源数={len(sources)}")
    yield {"type": "sources", "sources": sources}
