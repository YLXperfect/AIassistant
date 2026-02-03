# 使用LangChain的Embeddings功能   嵌入模型  

from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader  

from loadDucoment import documentLoad

import os
        

# 全局向量库
vectorstore = None

#嵌入模型
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",  
    api_key=os.getenv("ZHIPUAI_API_KEY"),  # 和 LLM 同 key
    max_retries=3,
    request_timeout=10,
    )   

text = '''我是股神
很爱学习
爱炒股'''

embedding = embeddings.embed_query(text)  #相同的文本， 转化之后是相同的
# print(f"embedding==: {embedding}")   # 打印嵌入结果
# print(f"embedding.shape==: {len(embedding)}")  # 打印嵌入结果的维度  2048  embedding-3 模型初始化


load_test = documentLoad()
docs = load_test.query_document_pdf("jianli.pdf")

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,      # 每个chunk的大小
            chunk_overlap=50,    # 重叠的大小
            length_function=len, # 计算长度的函数
            separators=[
                "\n\n",        # 空行（章节间隔）
                "\n●", "\n## ", "\n# ", # 捕获简历中的章节标题
                "。", "！", "？",        # 句子结束
                "；", "，",              # 分句
                " ",                     
                ""
            ],   #pdf的分割优化
             )

chunks = text_splitter.split_documents(docs)

# 创建向量库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"

    # collection_metadata={
    #     "hnsw:space": "cosine",           相似度计算方法
    #     "hnsw:construction_ef": 200,      # 构建时的性能参数
    #     "hnsw:search_ef": 100,            # 搜索时的性能参数
    #     "hnsw:M": 16,                      # 每个节点的连接数
    # }
)

# 简单相似度检索测试
query = "这个人有哪些编程技能，在哪家公司工作过？"
results = vectorstore.similarity_search(query, k=4)



print("\n检索结果：")
for i, doc in enumerate(results):
    print(f"【第{i+1}块】: {doc.page_content}")
    content = doc.page_content.replace('\n', ' ').strip()
    print(f"   {content[:120]}...")