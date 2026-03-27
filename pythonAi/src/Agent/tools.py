'''
langchain工具模版
@tool
def 工具名(参数1: 类型, 参数2: 类型) -> 返回类型:
    """
    清晰描述工具功能的文档字符串。
    模型完全依赖这个描述来决定是否调用此工具。
    
    Args:
        参数1: 参数说明。
        参数2: 参数说明。
        
    Returns:
        返回结果的说明。
        
    示例:
        可以给出调用示例。
    """
    # 1. 在这里编写核心逻辑（计算、查询API等）
    # 2. 处理可能发生的错误
    # 3. 关键：用 return 返回一个字符串结果
    return "格式良好的结果字符串"
'''

"""
langchain工具模版
"""
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader  
from langchain_chroma import Chroma
import math
import requests
from datetime import datetime
import pytz   
import os
import time
import logging
from src.Agent.DocumetLoader import DocumentProcessor  # 导入统一的文档处理器

logger = logging.getLogger(__name__)

# 全局变量，用于注入RAG引擎（在初始化Agent时设置）
_rag_engine = None
_llm = None
_doc_processor = None  # 添加文档处理器实例

def init_tools(rag_engine, llm):
    """在创建Agent前调用，注入RAG引擎实例与Agent LLM实例"""
    global _rag_engine, _llm, _doc_processor
    _rag_engine = rag_engine
    _llm = llm
    # 创建文档处理器实例（传入rag_engine以便使用其embeddings）
    _doc_processor = DocumentProcessor(rag_engine=rag_engine)
    logger.info("工具已注入RAG引擎、LLM和文档处理器")

# ========== 基础工具 ==========

@tool
def search_Weather(city:str )->str:
    """在网上查询天气
    Args:
        city: 城市
    """
    try:
        url = f"http://wttr.in/{city}?format=%C+%t"  # 免费API，无需key
        response = requests.get(url)
        return response.text.strip() if response.status_code == 200 else "无法获取天气"
    except:
        return "天气查询失败"

@tool("domath", description="计算输入的数学表达式")
def calculator(expression: str) -> str:
    """精确计算数学表达式，支持幂运算（如2^5或2**5）。输入纯表达式字符串。"""
    safe_expression = expression.replace("^", "**")
    try:
        result = eval(safe_expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except:
        return "计算错误，请检查表达式"

@tool
def get_current_time(city:str="北京")->str:
    """根据传入的城市获取城市当前时间"""
    try:
        tz = pytz.timezone({
            "北京": "Asia/Shanghai",
            "香港": "Asia/Hong_Kong",
            "纽约": "America/New_York"
        }.get(city, "Asia/Shanghai"))
        return datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S %Z")
    except:
        return "城市不支持"

# ========== 知识库相关工具（使用RAGEngine）==========

@tool
def smart_document_qa(question: str) -> str:
    """
    从简历知识库中查询相关信息。
    当用户询问关于简历写作、STAR法则、量化成果、项目描述等问题时使用。
    """
    if _rag_engine is None:
        return "知识库未就绪"
    try:
        # 使用RAGEngine的get_context方法检索知识库
        answer = _rag_engine.get_context(question, k=6, search_type="mmr", llm=_llm)
        logger.info(f"调用了smart_document_qa工具，问题: {question[:50]}...")
        return answer
    except Exception as e:
        logger.error(f"知识库查询失败: {e}")
        return f"查询知识库时出错: {str(e)}"

# ========== 上传文件处理工具（使用DocumentProcessor）==========

@tool
def extract_relevant_chunks(file_text: str, question: str, k: int = 4) -> str:
    """
    从用户上传的文档文本中抽取与问题最相关的片段（文档块）。
    
    使用场景：
    - 用户上传简历/文档，并要求“基于文档内容进行润色/改写/总结/提取要点”
    
    Args:
        file_text: 用户上传文件解析出的纯文本（可能较长）
        question: 用户意图/任务描述（如“请按STAR法则润色这段经历”）
        k: 返回的相关块数量（默认4）
    """
    if _rag_engine is None or _doc_processor is None:
        return "服务未就绪"
    
    try:
        # 1. 使用DocumentProcessor处理上传文本
        # 传入元数据，标记为上传文件
        docs = _doc_processor.process_for_upload(
            text=file_text,
            metadata={"source": "uploaded_file", "type": "user_upload"}
        )
        
        if not docs:
            return "文件内容为空或处理后无有效内容"
        
        # 2. 为上传文本建立临时向量库（使用RAGEngine的embeddings）
        tmp_vs = Chroma.from_documents(
            documents=docs,
            embedding=_rag_engine.embeddings,
            persist_directory=None  # 不持久化
        )
        
        # 3. 检索相关片段
        retriever = tmp_vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": int(k),
                "fetch_k": max(12, int(k) * 3),
                "lambda_mult": 0.5
            }
        )
        
        rel_docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in rel_docs if getattr(d, "page_content", None)])
        
        return f"（以下为从上传文档中抽取的相关片段）\n{context}" if context.strip() else "未抽取到有效片段"
        
    except Exception as e:
        logger.error(f"抽取上传文档片段失败: {e}")
        return f"抽取失败: {str(e)}"

@tool(return_direct=True)
def polish_text(text: str, style: str = "professional", rules_context: str = "", file_context: str = "") -> str:
    """
    润色文本（结合知识库规则 + 上传文档相关片段）。
    
    支持风格：professional（专业）、concise（简洁）、friendly（友好）。
    当有 rules_context / file_context 时，会优先遵循其中的规则与事实。
    """
    if _llm is None:
        return "服务未就绪"
    
    style_prompts = {
        "professional": "请用专业、正式的语气润色以下文本，保持原意但提升表达质量，使语言更精炼、逻辑更清晰：\n\n",
        "concise": "请用简洁、精炼的语言重写以下文本，去除冗余词汇，保留核心信息：\n\n",
        "friendly": "请用友好、亲切的语气润色以下文本，使其更易读、更有亲和力：\n\n",
    }
    
    prompt = (
        style_prompts.get(style, style_prompts["professional"])
        + "\n【必须遵循的简历写作规则（来自知识库）】\n"
        + (rules_context.strip() or "（无）")
        + "\n\n【与本次任务相关的上传文档片段（仅供参考，不要编造未出现的信息）】\n"
        + (file_context.strip() or "（无）")
        + "\n\n【需要润色的原文】\n"
        + text
    )
    
    start = time.time()
    try:
        response = _llm.invoke(prompt)
        elapsed = time.time() - start
        logger.info(f"LLM 润色成功调用耗时: {elapsed:.2f} 秒")
        return response.content
    except Exception as e:
        logger.error(f"润色失败: {e}")
        elapsed = time.time() - start
        return f"润色失败: {str(e)}"



if __name__ == "__main__":
    
    print(search_Weather.name)
    print(search_Weather.args)
    print(search_Weather.description)

    print(calculator.name)
#测试调用， 用invoke
    print(search_Weather.invoke({'city':"成都"}))