import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from app import rag
from app import db
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
from app import agent

load_dotenv() #读取.env文件
app = FastAPI()
db.init_db()
# 挂载前端静态页面
static_dir = str(pathlib.Path(__file__).parent.parent / "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(static_dir) / "index.html"))

class ChatRequest(BaseModel):
    question: str

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的中文技术助手，回答要简洁、准确，不要废话"},
            {"role": "user", "content": req.question}
        ]
    )
    return {"answer": response.choices[0].message.content}
    print(response)


class TranslateRequest(BaseModel):
    text: str
@app.post("/translate")
def translate(req: TranslateRequest):
    text = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [{"role": "system", "content": "你是一个翻译助手，把用户输入翻译成英文，只输出翻译结果，不要多余解释"},
                    {"role": "user", "content": req.text}]
    )
    return {"translation": text.choices[0].message.content}

# RAG接口
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 保存上传的文件到本地
    os.makedirs("./uploads", exist_ok=True)  # 创建 uploads 文件夹存上传文件；`exist_ok=True` = 文件夹已存在也不报错
    pdf_path = f"./uploads/{file.filename}" # f-string 拼出完整路径（uploads / 原文件名.pdf）
    with open(pdf_path,"wb") as f: # **with 语句 = 用完自动关闭文件（不关会占用文件句柄，是经典 bug 来源） `"wb"`：二进制写模式（PDF 是二进制文件，不能用文本模式）
        f.write(await file.read()) # 把上传内容写入本地
    # 建库：切片 → 向量化 → 存 Chroma
    count = rag.build_vector_store(pdf_path) # 交给 rag.py 完成 "切片→向量化→存库"，返回块数
    return {"message": f"上传成功，已入库{count}块"}

class AskRequest(BaseModel):
    question: str

class StreamAskRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
def ask(req: AskRequest):
    answer = rag.ask(req.question)
    return {"answer": answer}

@app.post("/ask/stream")
async def ask_stream_endpoint(req: StreamAskRequest):
    """流式问答：答案逐字返回，支持多轮对话"""
    question = req.question
    session_id = req.session_id

    if not question:
        return {"error": "question 不能为空"}

    def generate():
        full_answer = ""
        history = db.get_history(session_id)
        for token in rag.ask_stream(question, history=history):
            full_answer += token
            yield token
        # 保存到 SQLite
        db.save_message(session_id, "user", question)
        db.save_message(session_id, "assistant", full_answer)

    return StreamingResponse(generate(), media_type="text/plain")

class AgentRequest(BaseModel):
    question: str

@app.post("/agent/ask")
def agent_ask_endpoint(request: AgentRequest):
    """Agent 问答：大模型自主决定调用工具"""
    result = agent.agent_ask(request.question)
    return result