# src/api/main.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，以便能够导入 src 下的模块
# 假设项目根目录是当前文件所在目录的父目录的父目录（即 pythonAi/）
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))  # 插入到最前面优先搜索

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging

from src.AiAgent.RAGLearn import RAGEngineLCEL

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
        test_answer = rag_engine.query_document("STAR法则是什么？")
        logger.info(f"RAG 引擎初始化成功，测试回答片段: {test_answer[:50]}...")
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


# # 定义一个最简单的GET根路径接口，用于测试服务是否启动
# @app.get("/")
# async def root():
#     """
#     根路径，返回欢迎信息，确认API服务正常运行。
#     """
#     return {"message": "你好！简历润色助手API已启动。"}

# # 如果你希望测试一个简单的POST接口，可以再加一个
# @app.post("/echo")
# async def echo(text: str):
#     """
#     简单的回声接口，用于测试POST请求。
#     """
#     return {"received": text}





@app.get("/")
async def root():
    return {"message": "你好！简历润色助手API已启动。"}

@app.post("/echo")
async def echo(text: str):
    return {"received": text}

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