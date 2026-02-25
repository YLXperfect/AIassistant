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


    def debug_inspect_vectorstore(self, keyword: str = "工作经历", k: int = 20):
        """调试：直接查看向量库中与关键词相关的所有文档块"""
        # 1. 直接用相似度搜索获取原始块和分数
        docs_with_score = self.vectorstore.similarity_search_with_score(keyword, k=k)
    
        print(f"\n🔍【向量库诊断】关键词: '{keyword}'")
        print(f"检索到 {len(docs_with_score)} 个相关块\n")
    
        for i, (doc, score) in enumerate(docs_with_score):
            print(f"--- 块 {i+1} (相似度: {score:.4f}) ---")
            print(f"内容预览: {doc.page_content[:150].replace(chr(10), ' ')}")
            print(f"元数据: {doc.metadata}")
            print(f"完整内容长度: {len(doc.page_content)} 字符")
            print()
    
        return docs_with_score




# 测试
# question = "jianli.pdf 里我的联系方式是什么？"

# answer = rag_chain.invoke(question)
# print(f"最终回答: {answer}")




# 使用示例
if __name__ == "__main__":
    engine = RAGEngineLCEL()
    # 诊断所有包含“工作”的块
    engine.debug_inspect_vectorstore("工作经历 2018 2020")