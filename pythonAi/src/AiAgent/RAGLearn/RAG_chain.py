from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.retrieval import create_retrieval_chain  # 新版 RAG 链
from langchain_classic.retrievers import MultiQueryRetriever  # 可选多查询
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGEngineLCEL:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPUAI_API_KEY"))

        self.llm = ChatZhipuAI(model="glm-4-flash", temperature=0.0, api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

        # RAG 链
        template = """根据以下上下文精准回答问题，只使用上下文信息，不要编造：
        上下文：
        {context}

        问题：{question}

        回答："""
        prompt = ChatPromptTemplate.from_template(template)
        # 高级检索器（MMR + 多查询）
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
            ),
            llm=self.llm
        )

        self.rag_chain = (
            #第一个字典：准备输入，从输入中提取出 “question” 和检索到的 “context
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt   #将上一步的输出（一个包含context和question的dict）传入提示模板
            | self.llm
            | StrOutputParser() #解析大模型的输出为字符串
        )
        '''{“context”: self.retriever, “question”: RunnablePassthrough()}：这是一个 RunnableParallel 的简写。它接收一个输入（即用户问题），然后并行执行两个操作：
        retriever：接收问题，检索出相关文档，结果赋值给 context。
        RunnablePassthrough()：简单地让原问题通过，赋值给 question。
        这一步的输出是一个字典：{“context”: “检索到的文本”, “question”: “原始问题”}。
        | prompt：管道符将上一步的输出字典传递给提示模板 self.prompt。模板会按照定义，将 context 和 question 填入指定位置，生成完整的提示词字符串。
        | llm：将提示词字符串传递给大语言模型。
        | StrOutputParser()：将大模型的复杂响应对象解析为纯文本字符串
        '''
    def query_document(self,question:str):
        return self.rag_chain.invoke(question)
    # def add_document(self,file_name:str):




# 测试
# question = "jianli.pdf 里我的联系方式是什么？"

# answer = rag_chain.invoke(question)
# print(f"最终回答: {answer}")