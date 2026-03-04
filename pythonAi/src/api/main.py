# src/api/main.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，以便能够导入 src 下的模块
# 假设项目根目录是当前文件所在目录的父目录的父目录（即 pythonAi/）
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))  # 插入到最前面优先搜索

from fastapi import FastAPI, HTTPException, UploadFile, File  ,Form,BackgroundTasks
import shutil
from pathlib import Path
from typing import Annotated
import tempfile  #临时文件管理
from typing import List



from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import re

from src.AiAgent.RAGLearn import RAGEngineLCEL



from fastapi.responses import FileResponse  # 用于返回文件
from docx import Document  # 今天学习知识点：python-docx 核心类
import uuid

#fastapi 搭建
# 加载 .env 文件中的环境变量（例如 ZHIPUAI_API_KEY）
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    logger.info("正在初始化 RAG 引擎...")
    try:
        rag_engine = RAGEngineLCEL(
            persist_directory="./chroma_db",
            docs_path="./knowledge_base"
        )
        #测试问题， 测试服务启动
        # test_answer = rag_engine.query_document("STAR法则是什么？")
        # logger.info(f"RAG 引擎初始化成功，测试回答片段: {test_answer[:50]}...")
    except Exception as e:
        logger.error(f"RAG 引擎初始化失败: {e}")
        raise e
    logger.info("API 服务已启动")
    yield
    logger.info("API 服务正在关闭")

# 创建一个FastAPI应用实例
app = FastAPI(
    title="简历润色助手 API",
    description="提供基于RAG的简历问答与润色服务",
    version="1.0.0",
    lifespan=lifespan
)


# 定义请求和响应的数据模型
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    source: str = "knowledge_base"  # 可选项，标识答案来源

# 全局变量，用于存储 RAG 引擎实例（在启动时初始化）
rag_engine = None


#端点
#@app.get("/") 告诉 FastAPI：当有 HTTP GET 请求访问根路径 / 时，就执行下面的函数。
@app.get("/")
async def root():
    return {"message": "你好！简历润色助手API已启动。"}


#@app.post("/echo") 则对应：当有 HTTP POST 请求访问路径 /echo 时，执行下面的函数。
@app.post("/echo")
async def echo(text: str):
    return {"received": text}

#向模型提问的方法
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    接收用户问题，调用 RAG 链返回答案。
    """
    # 检查 RAG 引擎是否已初始化
    if rag_engine is None:
        logger.error("RAG 引擎未初始化")
        raise HTTPException(status_code=503, detail="服务未就绪，请稍后重试")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"收到问题: {question}")

    try:
        # 调用你的 RAG 引擎的 query_document 方法
        answer = rag_engine.query_document(question)
        logger.info(f"生成答案: {answer[:100]}...")  # 只记录前100字符
    except Exception as e:
        logger.error(f"处理问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

    return AskResponse(question=question, answer=answer)


#上传文件 
#返回文件大小
@app.post("/files/") 
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

#返回文件名
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}



# 添加文件上传相关的数据模型
class UploadResponse(BaseModel):
    """文件上传后的响应模型"""
    filename: str
    content_preview: str  # 提取文本的预览
    message: str
    num_characters: int   # 提取的文本长度

#上传接收文件
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):  # 今天学习的核心知识点1：UploadFile 和 File
    #UploadFile， FastAPI 提供的文件上传专用类型。它比单纯的 bytes 更强大，包含文件的元数据（如文件名 filename、内容类型 content_type）以及异步读写方法（如 read()、write()、seek()），适合处理大文件  (...) 代表参数是必须的   
    #File(...) 是 FastAPI 的依赖项（依赖函数），用于明确告诉 FastAPI：这个参数应该从请求的表单数据中获取，而不是从 JSON 请求体或其他地方。
    # 这里的 ...（三个点）是 Python 的 Ellipsis 对象，表示这个参数是必需的。如果客户端没有上传文件，FastAPI 会自动返回 422 错误。
    # 你也可以给它设置默认值，例如 File(None) 表示文件可选，但通常上传文件都是必填的
    """
    上传简历文件（支持 .txt, .pdf, .docx），提取文本内容并返回预览。
    """
    logger.info(f"接收到上传文件: {file.filename}")

    # 1. 验证文件类型
    allowed_extensions = ['.txt', '.pdf', '.docx', '.md']  # 支持你的常见格式
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，请上传 {allowed_extensions} 格式"
        )

    # 2. 保存上传的文件到临时目录
    try:
        # ：临时文件管理
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            # 将上传的内容写入临时文件
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        logger.info(f"文件已保存到临时路径: {tmp_path}")

        # 3. 提取文本内容
        # 今天学习的知识点3：复用已有的文档加载逻辑
        
        text_content = await extract_text_from_file(tmp_path, file_ext)
        
        # 4. 可选：将文件内容添加到临时知识库（后续步骤）
        # 目前只返回提取的文本预览

        # 5. 返回响应
        preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
        return UploadResponse(
            filename=file.filename,
            content_preview=preview,
            message="文件上传成功，文本已提取",
            num_characters=len(text_content)
        )

    except Exception as e:
        logger.error(f"处理上传文件时出错: {e}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        # 6. 清理临时文件（今天学习的知识点4：资源清理）  只保留一个上传文件
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"临时文件已删除: {tmp_path}")
        # 关闭原始文件句柄
        await file.close()


from langchain_community.document_loaders import UnstructuredWordDocumentLoader,PyPDFLoader


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

#新增提问 + 文件上传接口
@app.post("/upload-ask")
async def upload_ask(file:UploadFile=File(...),question:str = Form(...)):
    """
    上传文件并针对文件内容提问。
    结合知识库和上传文件的内容生成答案。
    """
        # 1. 基本验证
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 2. 验证文件类型
    allowed_extensions = ['.txt', '.pdf', '.docx', '.md']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

    # 3. 保存上传文件到临时位置并提取文本
    tmp_path = None
    file_text = ""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        logger.info(f"文件已保存到临时路径: {tmp_path}")

        # 提取文本（复用昨天的 extract_text_from_file 函数）
        file_text = await extract_text_from_file(tmp_path, file_ext)
        logger.info(f"提取文本长度: {len(file_text)} 字符")

        # 4. 结合文件文本和知识库生成答案
        # 将文件文本分割后，与知识库一起检索，融合上下文。，调用 RAG 链
        answer = await answer_with_file_context(question, file_text, rag_engine)
        
        # 5. 返回响应
        return {
            "question": question,
            "answer": answer,
            "file_processed": file.filename,
            "file_preview": file_text[:200] + "..." if len(file_text) > 200 else file_text
        }

    except Exception as e:
        logger.error(f"处理上传问答时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        # 6. 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        await file.close()

async def answer_with_file_context(question: str, file_text: str, rag_engine) -> str:
    """
    混合检索方案：将文件文本分割、向量化后，与知识库一起检索。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    

    # 1. 分割文件文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    file_docs = text_splitter.create_documents([file_text])
    
    # 2. 为文件块创建临时向量库（使用内存存储，无需持久化）
    # 这里使用 Chroma 的 from_documents 并设置 persist_directory=None 使其在内存中运行
    file_vectorstore = Chroma.from_documents(
        documents=file_docs,
        embedding=rag_engine.embeddings,
        persist_directory=None  # 内存模式
    )
    
    # 3. 从文件向量库检索相关块
    file_retriever = file_vectorstore.as_retriever(search_kwargs={"k": 2})
    # file_results = file_retriever._get_relevant_documents(question)
    file_results = await file_retriever.ainvoke(question)
    file_context = "\n".join([doc.page_content for doc in file_results])
    
    # 4. 从知识库检索相关块
    kb_results = rag_engine.vectorstore.similarity_search(question, k=3)
    kb_context = "\n".join([doc.page_content for doc in kb_results])
    
    # 5. 组合上下文
    combined_context = f"用户上传文件相关内容：\n{file_context}\n\n知识库相关内容：\n{kb_context}"
    
    # 6. 构建提示词
    prompt = f"""根据以下上下文回答问题。如果上下文中没有相关信息，请说不知道。
            上下文：
            {combined_context}
            问题：{question}
            回答："""
    response = rag_engine.llm.invoke(prompt)
    return response.content

#定义润色请求
class PolishRequest(BaseModel):
    text: str                           # 需要润色的原文
    style: str = "professional"          # 润色风格，默认专业风格

#润色接口
async def polish_text_with_llm(original_text: str, style: str = "professional") -> str:
    """
    使用 LLM 润色文本。
    学习点：构建风格化的提示词，调用 LLM。
    """
    if not original_text or not original_text.strip():
        raise HTTPException(status_code=400, detail="输入文本不能为空")

    # 根据风格选择不同的提示前缀
    style_prompts = {
        "professional": "请用专业、正式的语气润色以下文本，保持原意但提升表达质量，使语言更精炼、逻辑更清晰：\n\n",
        "concise": "请用简洁、精炼的语言重写以下文本，去除冗余词汇，保留核心信息：\n\n",
        "friendly": "请用友好、亲切的语气润色以下文本，使其更易读、更有亲和力：\n\n",
    }
    prompt = style_prompts.get(style, style_prompts["professional"]) + original_text

    try:
        # 调用 LLM（注意：rag_engine.llm 是同步的，但 FastAPI 会在线程池中运行它）
        response = rag_engine.llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"润色文本时出错: {e}")
        raise HTTPException(status_code=500, detail=f"润色失败: {str(e)}")




@app.post("/polish_download_clean", response_class=FileResponse)
async def polish_download_with_cleanup(request: PolishRequest, background_tasks: BackgroundTasks):
    """
    带自动清理的润色下载端点。
    学习点：BackgroundTasks 在响应后执行清理。
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    polished_text = await polish_text_with_llm(request.text, request.style)

    doc = Document()
    doc.add_heading('润色后的文本', level=1)
    for para in polished_text.split('\n'):
        if para.strip():
            doc.add_paragraph(para.strip())

    file_name = f"生成后的简历_{uuid.uuid4().hex[:8]}.docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name

    # 在后台任务中添加删除临时文件的操作
    background_tasks.add_task(os.unlink, tmp_path)

    return FileResponse(
        path=tmp_path,
        filename=file_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )