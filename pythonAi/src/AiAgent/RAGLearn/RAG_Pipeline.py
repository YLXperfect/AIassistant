
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate
import os


class RAGEngine:
    def __init__(self, persist_directory="./chroma_db"):
        # 初始化向量库、LLM、检索器（代码同之前任务）
        self.embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        self.llm = ChatZhipuAI(model="glm-4", api_key=os.getenv("ZHIPUAI_API_KEY"), temperature=0.1)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # 定义提示词
        prompt_template = """基于以下已知信息回答问题。如果信息不足，请说无法回答。
        已知信息：
        {context}
        问题：
        {question}
        请用中文回答："""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # 创建链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
    
    def query(self, question: str) -> str:
        """对外提供的查询接口"""
        result = self.qa_chain.invoke({"query": question})
        return result['result']

if  __name__=="__main__":
    engine = RAGEngine()
    question = "jianli.pdf 里我的联系方式是什么？"
    result = engine.query(question=question)
    print(f"回答是:{result}")


