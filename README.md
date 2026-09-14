# KubeRAG - 云原生智能知识库问答平台

基于 RAG（检索增强生成）技术的智能文档问答系统，支持 PDF 文档上传、混合检索、流式输出、多轮对话、语义缓存，采用 Go 网关 + Python 后端的云原生架构。

## 技术栈

| 类别 | 技术 | 说明 |
|---|---|---|
| 后端语言 | Python 3.11 | RAG 核心逻辑 |
| Web 框架 | FastAPI | 高性能异步 API |
| 网关 | Go 1.23+ / net/http | 反向代理、令牌桶限流、日志中间件 |
| 大模型 | DeepSeek-chat | 问答生成、重排序、查询改写 |
| Embedding | 智谱 embedding-3 | 文本向量化 |
| 向量数据库 | ChromaDB | 向量存储与检索 |
| 关键词检索 | BM25 (rank_bm25) + jieba | 中文分词 + 关键词检索 |
| 文档处理 | PyPDF + LangChain TextSplitter | PDF 解析与文本切分 |
| 关系数据库 | SQLite | 对话历史持久化 |
| 缓存 | 语义缓存（Redis + 内存降级） | embedding 相似度匹配，相似问题秒回 |
| 前端 | HTML + CSS + JavaScript | 聊天界面，支持流式输出 |
| 容器化 | Docker + docker-compose | 一键部署 |

## 功能特性

- **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 融合排序，兼顾精确匹配和语义理解
- **LLM 重排序**：用大模型对检索结果进行相关性重排，提升答案准确度
- **多轮对话**：SQLite 持久化对话历史 + 上下文窗口裁剪 + 追问改写（指代补全），避免上下文膨胀
- **语义缓存**：embedding 余弦相似度 > 0.9 直接返回缓存答案，带容量上限与 TTL 过期，节省 API 费用
- **流式输出**：SSE 事件流逐字返回，答案结束后下发引用来源
- **Go 网关**：令牌桶限流（每 IP 每秒 10 请求）、请求日志、反向代理、空闲 IP 自动清理
- **Agent 工具调用**：大模型自主决定调用知识库查询 / 计算器 / 时间 / 网络搜索工具（AST 白名单安全求值）
- **联网搜索降级**：知识库无关键词匹配时自动调用网络搜索（Tavily API 优先，DuckDuckGo 免费降级），用公开资料回答并标注可点击来源链接
- **Docker 部署**：docker-compose 一键启动后端 + 网关 + Redis

## 联网搜索降级机制

当用户问题在知识库中没有关键词匹配（BM25 最高分为 0）时，系统自动降级到网络搜索，用公开资料回答：

```
用户提问 → BM25 关键词检查
              ├─ 有匹配 → 混合检索（BM25 + 向量 + RRF + 重排序）→ 基于文档回答
              └─ 无匹配 → 网络搜索（Tavily 优先 → DuckDuckGo 降级）→ 基于公开资料回答，标注来源链接
```

- **Tavily API**（推荐）：专为 RAG 设计，返回干净页面正文，需 `TAVILY_API_KEY`，免费额度 1000 次/月
- **DuckDuckGo**（降级）：免费无需 Key，需 `pip install duckduckgo-search`
- 两者均不可用时，返回"知识库和网络搜索都没有找到相关内容"
- Agent 模式下，大模型可自主决定调用 `web_search` 工具

## 检索质量评测

内置自动化评测体系（`eval/`），从知识库文档自动生成测试题集，对比四种检索策略的 **HitRate@k** 和 **MRR**：

| 检索策略 | HitRate@3 | HitRate@5 | MRR |
|---|---|---|---|
| BM25 关键词检索 | 90.0% | 95.0% | 0.843 |
| 向量语义检索 | 75.0% | 75.0% | 0.692 |
| 混合检索（BM25+向量+RRF） | 80.0% | 85.0% | 0.738 |
| **混合检索 + LLM 重排序** | **95.0%** | **95.0%** | **0.875** |

**关键发现**：
- LLM 重排序将 MRR 从 0.738 提升至 0.875（+18.6%），是效果最好的策略
- 纯 RRF 融合反而低于 BM25 单独使用（向量检索噪音稀释了关键词强信号），重排序器修复了该问题
- 测试集：20 道题，覆盖全部 5 个文档，由 LLM 基于文档块自动生成

运行评测：
```bash
# 生成测试题集（从知识库采样，LLM 生成问题）
python eval/generate_testset.py

# 运行评测（输出 HitRate@k / MRR + 逐题明细）
python eval/evaluate.py
```

## 项目架构

```
┌────────────┐    ┌───────────────────────────┐    ┌──────────────────────────┐
│  浏览器     │───▶│  Go 网关 :8080             │───▶│  Python FastAPI :9003    │
│ (index.html)│    │  ├─ 限流中间件（每IP 10/s）│    │  ├─ /upload   PDF入库     │
└────────────┘    │  ├─ 日志中间件             │    │  ├─ /ask      普通问答    │
                  │  └─ 反向代理（流式透传）    │    │  ├─ /ask/stream SSE流式  │
                  └───────────────────────────┘    │  └─ /agent/ask Agent问答  │
                                                   └──────────┬───────────────┘
                                                              │
                         ┌────────────────────────────────────┼──────────────────────────┐
                         ▼                                    ▼                          ▼
                 ┌──────────────┐                    ┌──────────────┐           ┌──────────────┐
                 │ ChromaDB     │                    │ Redis        │           │ SQLite       │
                 │ 向量存储/检索  │                    │ 语义缓存      │           │ 对话历史      │
                 └──────────────┘                    └──────────────┘           └──────────────┘
```

## 项目结构

```
KubeRAG/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI 入口，API 路由
│   ├── rag.py        # RAG 核心：混合检索、重排序、追问改写、流式问答
│   ├── db.py         # SQLite 对话历史持久化
│   ├── agent.py      # Agent 工具调用
│   ├── cache.py      # 语义缓存（Redis + 内存降级，容量/TTL 治理）
│   └── web_search.py # 联网搜索（Tavily 优先 + DuckDuckGo 降级）
├── eval/
│   ├── generate_testset.py # 从知识库自动生成测试题集
│   ├── evaluate.py        # 检索质量评测（HitRate@k / MRR）
│   ├── testset.jsonl      # 测试题集（20题）
│   └── results.json       # 评测结果
├── k8s/
│   ├── namespace.yaml     # 命名空间
│   ├── configmap.yaml     # 非敏感配置
│   ├── secret.yaml        # API Key（部署前替换）
│   ├── redis.yaml         # Redis 部署 + PVC
│   ├── backend.yaml       # FastAPI 后端部署 + PVC + 探针
│   ├── gateway.yaml       # Go 网关部署
│   ├── hpa.yaml           # 自动扩缩容
│   ├── ingress.yaml       # 外部访问入口
│   ├── kustomization.yaml # Kustomize 一键部署
│   └── README.md          # K8s 部署指南
├── gateway/
│   ├── main.go       # Go 网关：限流、日志、反向代理
│   ├── go.mod
│   └── Dockerfile
├── static/
│   └── index.html    # 前端聊天页面
├── data/
│   ├── uploads/      # 上传的 PDF 文件
│   ├── chroma_db/    # ChromaDB 向量数据
│   └── chat_history.db # SQLite 对话历史
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## 快速开始

### 环境要求

- Python 3.11+
- Go 1.23+（可选，仅网关需要）
- Docker（可选，容器化部署）

### 本地运行

1. **克隆项目并安装依赖**
   ```bash
   cd KubeRAG
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

2. **配置环境变量**
   复制 `.env.example` 为 `.env`，填入 DeepSeek 和智谱的 API Key

3. **启动 Python 后端**
   ```bash
   uvicorn app.main:app --port 9003
   ```

4. **启动 Go 网关（可选）**
   ```bash
   cd gateway
   go run main.go
   # 网关默认转发到 http://127.0.0.1:9003，可用环境变量覆盖：
   # BACKEND_URL=http://127.0.0.1:9003 GATEWAY_PORT=:8080 go run main.go
   ```

5. **访问**
   - 前端页面：http://127.0.0.1:9003/
   - API 文档：http://127.0.0.1:9003/docs
   - 通过网关访问：http://127.0.0.1:8080/

### Docker 部署

```bash
# 先准备 .env（DEEPSEEK_API_KEY / BIGMODEL_API_KEY）
docker-compose up --build -d
```

- 后端：http://127.0.0.1:9003/
- 网关：http://127.0.0.1:8080/（自动转发到后端）

### Kubernetes 部署

完整 K8s manifest 在 `k8s/` 目录，支持一键部署：

```bash
# 配置 API Key（替换 secret.yaml 中的占位值，或用 kubectl 创建）
kubectl create secret generic kuberag-secrets -n kuberag \
  --from-literal=DEEPSEEK_API_KEY=your-key \
  --from-literal=BIGMODEL_API_KEY=your-key \
  --from-literal=TAVILY_API_KEY=your-key

# 一键部署（Kustomize）
kubectl apply -k k8s/

# 验证
kubectl get all -n kuberag
kubectl get hpa -n kuberag

# 访问（端口转发）
kubectl port-forward svc/gateway -n kuberag 8080:8080
# 浏览器打开 http://localhost:8080
```

**K8s 部署特性**：
- 后端 + 网关各 2 副本起步，RollingUpdate 零停机发布
- HPA 自动扩缩（后端 2-5 副本，CPU 70% / 内存 80% 触发）
- ConfigMap 管理非敏感配置，Secret 管理 API Key
- PVC 持久化 ChromaDB 向量数据、上传文件、对话历史
- Liveness / Readiness 探针，preStop 优雅退出
- Ingress 支持 SSE 流式输出（关闭 proxy buffering）
- 详细部署指南见 [k8s/README.md](k8s/README.md)

### API 接口

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/health` | 健康检查 |
| GET | `/` | 前端页面 |
| GET | `/sessions` | 列出所有对话会话（按最近活跃倒序） |
| GET | `/history/{session_id}` | 获取某个会话的历史消息（页面刷新时恢复聊天记录） |
| POST | `/upload` | 上传 PDF 文档（仅 .pdf，≤50MB） |
| POST | `/ask` | 普通问答，返回 `{"answer": str, "sources": [...]}` |
| POST | `/ask/stream` | 流式问答（SSE 事件流） |
| POST | `/agent/ask` | Agent 问答（大模型自主调用工具） |

#### SSE 事件格式（/ask/stream）

```
data: {"type": "delta", "content": "答"}      # 答案增量，逐字返回
data: {"type": "delta", "content": "案"}
data: {"type": "sources", "sources": [...]}   # 引用来源（答案结束后发送一次）
data: {"type": "done"}                        # 本轮结束
```
