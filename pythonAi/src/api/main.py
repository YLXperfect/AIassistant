# src/api/main.py
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import re
import uuid
import asyncio
from typing import Optional
from fastapi import Request  
from src.Agent.RAG_chain import RAGEngineLCEL
from src.Agent.agentCore import create_ai_agent, get_api_key, get_memory_as_langchain_messages
from src.Agent.memory import ConversationMemory
from fastapi.responses import FileResponse
from docx import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader,UnstructuredWordDocumentLoader
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
rag_engine = None
agent = None
# 注意：生产环境应按会话管理memory，这里简单使用全局仅用于演示
memory = ConversationMemory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, agent
    logger.info("正在初始化 RAG 引擎...")
    try:
        rag_engine = RAGEngineLCEL(
            persist_directory="./chroma_db",
            docs_path="./knowledge_base"
        )
        logger.info("✅ RAG 引擎初始化成功")
        
        # 初始化Agent
        api_key = get_api_key()
        agent = create_ai_agent(api_key, rag_engine)
        logger.info("✅ Agent 初始化成功")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise e
    yield
    logger.info("API 服务正在关闭")

app = FastAPI(
    title="简历润色助手 API",
    description="提供基于Agent的简历问答与润色服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 数据模型 ----------


class PolishRequest(BaseModel):
    text: str
    style: str = "professional"


@app.post("/ask")
async def ask_endpoint(
    raw_request: Request,  # 新增，用于手动读取 JSON
    file: UploadFile = File(None),
    question: str = Form(None)
):
    """
    统一的问答接口：
    - 无文件时，手动从请求体读取 JSON { "question": "..." }
    - 有文件时，从表单字段获取 question
    """
    logger.info(f"=== /ask 请求参数 ===")
    logger.info(f"file: {file}")
    logger.info(f"question (form): {question}")
    logger.info(f"===================")

    global memory
    if agent is None or rag_engine is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    user_input = ""
    file_text = ""
    has_file = file is not None and file.filename is not None

    if has_file:
        # ---------- 文件上传处理 ----------
        if not question:
            raise HTTPException(status_code=400, detail="请提供问题")
        user_input = question.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 验证文件类型（略，保持不变）
        # ... 文件处理代码 ...
        # 提取文件文本到 file_text
        # 验证文件类型
        allowed_extensions = ['.txt', '.pdf', '.docx', '.md']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

        # 保存临时文件并提取文本
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            file_text = await extract_text_from_file(tmp_path, file_ext)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            await file.close()
    else:
        # ---------- 纯文本 JSON 处理 ----------
        try:
            body = await raw_request.json()  # 手动解析 JSON
        except Exception as e:
            logger.error(f"JSON 解析失败: {e}")
            raise HTTPException(status_code=400, detail="无效的 JSON 格式")

        user_input = body.get("question", "").strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="字段 'question' 不能为空")

    # ---------- 构造 Agent 消息 ----------
    if file_text:
        agent_message = f"用户上传了文件，内容如下：\n{file_text}\n\n用户问题：{user_input}"
    else:
        agent_message = user_input

    # 添加到记忆并调用 Agent（保持不变）
    memory.add_to_memory('user', agent_message)
    langchain_messages = get_memory_as_langchain_messages(memory)

    try:
        input_dict = {"messages": langchain_messages}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.invoke, input_dict)

    # 提取答案（可能需要根据返回结构调整）
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, dict) and 'messages' in response:
        # 如果返回的是包含消息的字典，取最后一条消息的内容
            last_message = response['messages'][-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            answer = str(response)
        memory.add_to_memory('assistant', answer)
        return {"question": user_input, "answer": answer}
    except Exception as e:
        logger.error(f"Agent调用失败: {e}", exc_info=True)  # 打印详细错误栈
        raise HTTPException(status_code=500, detail="处理失败")
# ---------- 润色下载接口（不变）----------
@app.post("/download_docx", response_class=FileResponse)
async def download_docx(request: PolishRequest, background_tasks: BackgroundTasks):
    """直接接收润色后的文本，生成Word文档并返回"""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    doc = Document()
    doc.add_heading('润色后的文本', level=1)
    for para in request.text.split('\n'):
        if para.strip():
            doc.add_paragraph(para.strip())

    file_name = f"润色结果_{uuid.uuid4().hex[:8]}.docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name

    background_tasks.add_task(os.unlink, tmp_path)

    return FileResponse(
        path=tmp_path,
        filename=file_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )



async def extract_text_from_file(file_path: str, file_ext: str) -> str:
    """
    从文件中提取文本内容的辅助函数。
    复用你之前在 RAGEngineLCEL 中实现的逻辑思路。
    """
    text = ""
    try:
        if file_ext == '.pdf':
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
        
        elif file_ext == '.txt' or file_ext == '.md':
            # 直接读取文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        elif file_ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path, mode="single")
            docs = loader.load()
            text = docs[0].page_content
        
        # 应用你之前实现的文本清洗函数（如果有）
        cleaned_text = _clean_document_text(text)  # 如果是静态方法
        # 或者直接复制清洗逻辑过来
        
        return cleaned_text
    except Exception as e:
        logger.error(f"从文件 {file_path} 提取文本失败: {e}")
        raise


def _clean_document_text(text):
        """清洗文档文本的辅助函数"""
        # 1. 去除多余的空行（将连续两个以上的换行替换为两个换行）
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # 2. 去除行首行尾的空白
        lines = [line.strip() for line in text.split('\n')]
        # 3. 过滤掉空行后重新组合（根据需求选择是否保留空行分隔）
        cleaned_lines = [line for line in lines if line]  # 完全去掉空行
        # 或者保留一个空行作为段落分隔：cleaned_text = '\n\n'.join(cleaned_lines)
        cleaned_text = '\n'.join(cleaned_lines)  # 简单连接
        return cleaned_text