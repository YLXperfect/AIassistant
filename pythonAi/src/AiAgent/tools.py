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


from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader  

import math
import requests
from datetime import datetime
import pytz   
import os

# pytz 是Python第三方库（全名Python Time Zone），专门处理全球时区（包括夏令时自动调整）。



# @tool 把函数变成了一个具有标准化接口的“工具对象”
@tool
def search_Weather(city:str )->str:
    #描述， 告诉大模型这个工具是做什么的
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
    





# 自定义工具    自定义工具名  可省略  ;  description :工具描述  args_schema : 参数描述  用到哪个写哪个
#eval() python内置函数， 把字符串当成Python代码执行，并返回计算结果
'''
第二个参数：{"builtins": {}}（globals字典）最关键的安全措施！
默认eval能访问所有内置函数（如open、import、exec）。
这里把__builtins__设成空字典{}：完全禁用内置函数。
效果：恶意代码如__import__('os')会直接报NameError，无法执行。
这叫“禁用内置”沙盒，防止代码注入攻击。

第三个参数：{"math": math}（locals字典）提供有限的本地变量/模块。
这里只允许用math模块   提前（import math）。
所以表达式能用math.sqrt、math.pi、math.sin等，但不能用其他（如os、sys）。
'''

@tool("domath",description="计算输入的数学表达式")
def calculator(expression: str) -> str:
    # 当让模型进行幂运算时出错，Python的运算符陷阱：^ 在Python是按位异或（XOR），不是幂运算
    """精确计算数学表达式，支持幂运算（如2^5或2**5）。输入纯表达式字符串。"""
    safe_expression = expression.replace("^","**")
    result = eval(safe_expression, {"__builtins__": {}}, {"math": math})


    try:
        
        return str(result)
    except:
        return "计算错误，请检查表达式"


@tool
def get_current_time(city:str="北京")->str:
    ''' 根据传入的城市获取城市当前时间'''
    try:
        tz = pytz.timezone({
            "北京": "Asia/Shanghai",
            "香港": "Asia/Hong_Kong",
            "纽约": "America/New_York"
        }.get(city, "Asia/Shanghai"))
        return datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S %Z")
    except:
        return "城市不支持"
        
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings


        

# 全局向量库
vectorstore = None
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",  # 智谱最新嵌入模型，免费好用
    api_key=os.getenv("ZHIPUAI_API_KEY"),  # 和 LLM 同 key
    )
@tool
def query_document(fileName:str,question:str)->str:
    """向量检索本地TXT或PDF文档，并根据问题回答。
    参数:
        filename: 文件名（放在项目根目录，支持 .txt 或 .pdf）
        question: 用户关于文档的问题
    """
    global vectorstore

    try:
        file_path = os.path.join(os.getcwd(), fileName)  #构建绝对路径
        if not os.path.exists(file_path):
            return f"文件 {fileName} 不存在。"
        
        
        # 根据文件类型选择加载器（LangChain 标准文档加载）

        if fileName.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif fileName.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            return "不支持的文件格式，只支持 .txt 或 .pdf"

        print(f"【工具调试】正在读取文件: {file_path}")  # 加这行看控制台

        # docs = loader.load()
        # content = "\n".join([doc.page_content for doc in docs])
        # 简单 RAG：把文档内容 + 问题给模型回答（初步版）  可能文档太长，影响另外的文件读取  token太长
        # return f"文档内容（前1000字）：{content[:1000]}...\n\n根据文档回答问题 '{question}' 的答案是："


        #只截取前500字
        # short_content = content[:500] + "..." if len(content) > 500 else content
        # print(f"【工具调试】读取成功，内容长度: {len(content)}")


        #读取文件
        docs = loader.load()
        #文件分块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80) #chunk_size 切成300字的块，  重叠80字
        splits = text_splitter.split_documents(docs)

        # 构建/更新向量库  第一次创建， 后续添加
        if vectorstore is None:
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        else:
            vectorstore.add_documents(splits)

        # 检索 
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) ## k=5：返回 5 个最相似片段
        relevant_docs = retriever.invoke(question)# 用问题向量检索
        
        context = "\n".join([doc.page_content for doc in relevant_docs])

        return f"相关文档片段：\n{context}\n\n请根据以上内容回答问题 '{question}'："



    
    except Exception as e:
        return f" RAG 失败: {str(e)}"



        


@tool("sayHello",description="say hello to you!")
def greeting(name:str)->str:
    
    return print(f"你好{name}")


if __name__ == "__main__":
    print(greeting.invoke("张三"))
    print(search_Weather.name)
    print(search_Weather.args)
    print(search_Weather.description)

    print(calculator.name)
#测试调用， 用invoke
    print(search_Weather.invoke({'city':"成都"}))