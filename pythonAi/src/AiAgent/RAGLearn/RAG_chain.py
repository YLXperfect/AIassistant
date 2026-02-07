from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.retrieval import create_retrieval_chain  # 新版 RAG 链
from langchain_classic.retrievers import MultiQueryRetriever  # 可选多查询
import os

# 嵌入 + LLM
embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPUAI_API_KEY"))
llm = ChatZhipuAI(model="glm-4-flash", temperature=0.0, api_key=os.getenv("ZHIPUAI_API_KEY"))

# 向量库（复用你的 chroma_db）
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 高级检索器（MMR + 多查询）
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
    ),
    llm=llm
)

# RAG 链
template = """根据以下上下文精准回答问题，只使用上下文信息，不要编造：

上下文：
{context}

问题：{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 测试
question = "jianli.pdf 里我的联系方式是什么？"
answer = rag_chain.invoke(question)
print(f"最终回答: {answer}")