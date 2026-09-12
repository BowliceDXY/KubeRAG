# KubeRAG - 云原生智能知识库问答平台

基于 RAG（检索增强生成）技术的智能文档问答系统，支持 PDF 文档上传、混合检索、流式输出、多轮对话、语义缓存，采用 Go 网关 + Python 后端的云原生架构。

## 技术栈

| 类别 | 技术 | 说明 |
|---|---|---|
| 后端语言 | Python 3.11 | RAG 核心逻辑 |
| Web 框架 | FastAPI | 高性能异步 API |
| 网关 | Go 1.27 + net/http | 反向代理、令牌桶限流、日志中间件 |
| 大模型 | DeepSeek-chat | 问答生成 |
| Embedding | 智谱 embedding-3 | 文本向量化 |
| 向量数据库 | ChromaDB | 向量存储与检索 |
| 关键词检索 | BM25 (rank_bm25) + jieba | 中文分词 + 关键词检索 |
| 文档处理 | PyPDF + LangChain TextSplitter | PDF 解析与文本切分 |
| 关系数据库 | SQLite | 对话历史持久化 |
| 缓存 | 语义缓存（内存版） | embedding 相似度匹配，相似问题秒回 |
| 前端 | HTML + CSS + JavaScript | 聊天界面，支持流式输出 |
| 容器化 | Docker + docker-compose | 一键部署 |

## 功能特性

- **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 融合排序，兼顾精确匹配和语义理解
- **LLM 重排序**：用大模型对检索结果进行相关性重排，提升答案准确度
- **流式输出**：基于 SSE 的逐字返回，提升用户体验
- **多轮对话**：SQLite 持久化对话历史，支持上下文理解
- **语义缓存**：embedding 余弦相似度 > 0.9 直接返回缓存答案，节省 API 费用
- **Go 网关**：令牌桶限流（每 IP 每秒 10 请求）、请求日志、反向代理
- **前端界面**：左侧边栏（上传 PDF、新建对话、历史列表），右侧聊天区域（流式输出、参考来源）
- **Docker 部署**：docker-compose 一键启动

## 项目架构


## 项目结构
KubeRAG/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI 入口，API 路由
│ ├── rag.py # RAG 核心：混合检索、重排序、问答
│ ├── db.py # SQLite 对话历史持久化
│ └── cache.py # 语义缓存
├── gateway/
│ ├── main.go # Go 网关：限流、日志、反向代理
│ └── go.mod
├── static/
│ └── index.html # 前端聊天页面
├── data/
│ ├── uploads/ # 上传的 PDF 文件
│ ├── chroma_db/ # ChromaDB 向量数据
│ └── chat_history.db # SQLite 对话历史
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md


## 快速开始

### 环境要求

- Python 3.11+
- Go 1.21+（可选，仅网关需要）
- Docker（可选，容器化部署）

### 本地运行

1. **克隆项目并安装依赖**
cd KubeRAG
python -m venv venv
#### Windows
venv\Scripts\activate
#### macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

2.**配置环境变量：**
复制 .env.example 为 .env，填入 DeepSeek 和智谱的 API Key

3.**启动 Python 后端**
uvicorn app.main:app --port 9003

4.**启动 Go 网关（可选）**
cd gateway
go run main.go

5.**访问**
前端页面：http://127.0.0.1:9003/
API 文档：http://127.0.0.1:9003/docs
通过网关访问：http://127.0.0.1:8080/

6.**Docker 部署:**
docker-compose up --build -d
访问 http://127.0.0.1:8000/

7.**API接口**

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/health` | 健康检查 |
| GET | `/` | 前端页面 |
| POST | `/upload` | 上传 PDF 文档 |
| POST | `/ask` | 普通问答（返回答案 + 出处） |
| POST | `/ask/stream` | 流式问答（SSE 逐字返回） |