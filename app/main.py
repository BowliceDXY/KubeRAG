import json
import uuid
import pathlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app import rag, db, agent

load_dotenv()  # 读取 .env 文件
app = FastAPI()
db.init_db()

# 挂载前端静态页面
static_dir = str(pathlib.Path(__file__).parent.parent / "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 统一上传目录（与 rag.py 的 UPLOAD_PATH 一致，位于 data/ 卷内，容器重建不丢数据）
UPLOAD_DIR = pathlib.Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


class AskRequest(BaseModel):
    question: str


class StreamAskRequest(BaseModel):
    question: str
    session_id: str = "default"


class AgentRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(static_dir) / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 文件类型校验（仅 PDF）
    original_name = file.filename or "upload.pdf"
    ext = pathlib.Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="文件超过 50MB 限制")
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 防路径穿越 + 防重名：存储名用 uuid 重命名，原文件名仅作为展示元数据
    stored_name = f"{uuid.uuid4().hex}{ext}"
    pdf_path = UPLOAD_DIR / stored_name
    pdf_path.write_bytes(content)

    # 建库：切片 → 向量化 → 存 Chroma（元数据中记录可读的原文件名）
    display_name = pathlib.Path(original_name).name
    count = rag.build_vector_store(str(pdf_path), source_name=display_name)
    return {"message": f"上传成功，已入库{count}块", "stored_name": stored_name}


@app.post("/ask")
def ask(req: AskRequest):
    """普通问答：返回 {"answer": str, "sources": [...]}"""
    return rag.ask(req.question)


def _sse_event(data: dict) -> str:
    """把事件 dict 序列化为 SSE 帧"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
async def ask_stream_endpoint(req: StreamAskRequest):
    """
    流式问答：SSE 事件流，事件类型：
      delta   -> 答案增量（逐字返回）
      sources -> 引用来源（答案结束后发送一次）
      done    -> 本轮结束
    """
    question = req.question.strip()
    session_id = req.session_id
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    def generate():
        history = db.get_history(session_id)
        full_answer = ""
        sources = []
        for event in rag.ask_stream(question, history=history):
            if event["type"] == "delta":
                full_answer += event["content"]
                yield _sse_event({"type": "delta", "content": event["content"]})
            elif event["type"] == "sources":
                sources = event["sources"]
                yield _sse_event({"type": "sources", "sources": sources})
        # 历史只存纯问答对（不存带上下文的完整 prompt），避免多轮膨胀
        db.save_message(session_id, "user", question)
        db.save_message(session_id, "assistant", full_answer)
        yield _sse_event({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/agent/ask")
def agent_ask_endpoint(request: AgentRequest):
    """Agent 问答：大模型自主决定调用工具"""
    return agent.agent_ask(request.question)


@app.get("/sessions")
def list_sessions():
    """列出所有对话会话（按最近活跃倒序），用于页面加载时恢复左侧对话列表"""
    return {"sessions": db.list_sessions()}


@app.get("/history/{session_id}")
def get_session_history(session_id: str):
    """获取某个会话的历史消息（按时间正序，最多 50 条）"""
    messages = db.get_history(session_id, limit=50)
    return {"session_id": session_id, "messages": messages}
