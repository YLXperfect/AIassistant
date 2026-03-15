#检索方式学习 ，混合检索， 两步检索，agentic rag


from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough ,RunnableLambda,RunnableBranch
from langchain_classic.retrievers import MultiQueryRetriever,EnsembleRetriever,ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank

from clean_data import clean_md_to_df,clean_pdf


from langchain_core.documents import Document
import os

from langchain_community.document_loaders import PyPDFLoader,UnstructuredMarkdownLoader,TextLoader,UnstructuredWordDocumentLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter,MarkdownHeaderTextSplitter
from typing import List, Dict, Any, Optional

ApiKey = os.getenv("ZHIPUAI_API_KEY")


# 知识库文件配置表（新增文件只需在这里加一项）
KNOWLEDGE_FILES_CONFIG: List[Dict[str, Any]] = [
    {
        "path": "knowledge_base/rules.md",
        "metadata": {"source": "rules.md", "type": "rule", "category": "general"}
    },
    {
        "path": "knowledge_base/star.md",
        "metadata": {"source": "star.md", "type": "rule", "category": "star_method"}
    },
    {
        "path": "knowledge_base/template.md",
        "metadata": {"source": "template.md", "type": "template", "category": "template_education"}
    },
    {
        "path": "knowledge_base/简历模版.docx",
        "metadata": {"source": "简历模版.docx", "type": "template", "category": "template"}
    },
    
]

# Chroma 持久化目录（建议每次测试时可删除旧目录重新构建）
CHROMA_PERSIST_DIR = "./chroma_db"

# src/AiAgentDeep/RAG_ways.py
# Day9 主文件：实现混合检索 + 重排序 + metadata 过滤
# 目标：优化知识库检索质量，特别是针对简历规则、STAR 方法、模板的精准召回




def load_documents_with_metadata() -> List[Document]:
    """
    Day9 优化：使用配置表加载文档 + 统一添加 metadata
    支持 .md 和 .pdf 文件
    """
    all_docs = []
    
    for config in KNOWLEDGE_FILES_CONFIG:
        path = config["path"]
        base_metadata = config["metadata"]
        
        if not os.path.exists(path):
            print(f"警告：文件不存在，跳过 -> {path}")
            continue
        
        # 根据文件类型选择 loader  导入之后根据分类进行数据清洗
        if path.lower().endswith((".md", ".markdown")):
            loader = UnstructuredMarkdownLoader(path)
            raw_docs = loader.load()
            cleaned_docs = clean_md_to_df(raw_docs)
        elif path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            
            raw_docs = loader.load()
            cleaned_docs = clean_pdf(raw_docs)
        else:
            print(f"不支持的文件格式，跳过 -> {path}")
            continue
        
        # docs = loader.load()
        docs = cleaned_docs
        
        # 为该文件的所有文档片段添加 metadata
        for doc in docs:
            doc.metadata.update(base_metadata)  # 使用 update 保留 loader 自带信息（如 page）
        
        all_docs.extend(docs)
    
    # 文档分割（chunk 大小和重叠可根据实际效果调整）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(all_docs)
    print(f"总共加载并切分出 {len(split_docs)} 个文档片段")
    return split_docs


def build_vectorstore(docs: List[Document]) -> Chroma:
    """构建或加载 Chroma 向量库"""
    embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPUAI_API_KEY"))
    
    # 如果目录已存在，可选择加载旧的；这里演示每次重建（生产环境可改为增量）
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("加载已有 Chroma 数据库...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="resume_rules"
        )
    else:
        print("创建新的 Chroma 向量库...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="resume_rules"
        )
    
    return vectorstore


class RAGPipeline:
    def __init__(self):
        self.docs = load_documents_with_metadata()
        self.vectorstore = build_vectorstore(self.docs)
        self.embeddings = ZhipuAIEmbeddings()
        self.llm = ChatZhipuAI(model="glm-4.6v", temperature=0.0, api_key=os.getenv("ZHIPUAI_API_KEY"))
        
        # Day9 核心：构建混合检索器
        self._build_retrievers()
        
        # RAG 提示模板（可复用 Day8 的高级提示）
        self.prompt = ChatPromptTemplate.from_template(
            """根据以下上下文精准回答问题，只使用上下文信息，不要编造或添加外部知识。
            如果上下文不足以回答，请直接说“根据现有资料无法完整回答”。
            
            上下文：
            {context}
            
            问题：{question}
            
            回答："""
        )
    
    def _build_retrievers(self):
        """Day9 核心：混合检索 + rerank"""
        # 语义检索器 语义检索召回6块
        # semantic_retriever = self.vectorstore.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
        # )
        #增加问题改写
        semantic_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
            ),
            include_original=True,
            llm = self.llm
        )
        
        
        # 关键词检索器（BM25）  关键词检索召回6个块
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 6
        
        # 混合检索（Ensemble）      EnsembleRetriever 组合检索器 权重分配
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.65, 0.35]  
        )
        
        # 重排序（rerank）   拿到两种检索召回的文档块， 通过rerank模型对召回的所有文档块打分， 最后返回最终的文档块
        #rerank 模型（这里是 ms-marco-MiniLM-L-12-v2）是一个小型交叉编码器（cross-encoder），它同时看 query + 文档全文，打出的分数更接近人类判断的相关性
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=4)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.ensemble_retriever
        )
    
    def get_retriever(self, metadata_filter: Optional[Dict] = None):
        """支持 metadata 过滤的 retriever"""
        if metadata_filter:
            # 注意：EnsembleRetriever 本身不支持 filter，需要在子 retriever 上分别加
            # 这里简化演示：只对语义部分加 filter（BM25 不支持 filter）
            filtered_semantic = self.vectorstore.as_retriever(
                search_kwargs={"k": 6, "filter": metadata_filter}
            )
            return EnsembleRetriever(
                retrievers=[filtered_semantic, BM25Retriever.from_documents(self.docs, k=6)],
                weights=[0.65, 0.35]
            )
        return self.compression_retriever

        
    
    def create_qa_chain(self, chain_type: str = "stuff", metadata_filter: Optional[Dict] = None):
        retriever = self.get_retriever(metadata_filter)
        
        if chain_type == "stuff":
            chain = (
                {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                 "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            return chain
        
        raise NotImplementedError("Day9 重点实现混合检索，高级链型待 Day8 代码迁移")
    
    def query(self, question: str, metadata_filter: Optional[Dict] = None) -> str:
        chain = self.create_qa_chain(metadata_filter=metadata_filter)
        return chain.invoke(question)
        


# ===================== 测试与使用示例 =====================
if __name__ == "__main__":
    rag = RAGPipeline()
    
    test_queries = [
        "STAR 方法怎么写简历经历？",
        "如何量化简历中的工作成果？",
        "教育背景模板怎么写？"
    ]
    
    # 普通查询
    print("\n=== 无过滤查询 ===\n")
    for q in test_queries:
        print(f"问题：{q}")
        print("回答：", rag.query(q))
        print("-" * 60)

    docs = rag.compression_retriever.invoke(test_queries)
    print(f"检索到 {len(docs)} 个文档片段")
    
    # 只召回 rule 类型的文档
    # print("\n=== 只召回 type='rule' 的文档 ===\n")
    # rule_filter = {"type": "rule"}
    # for q in test_queries:
    #     print(f"问题：{q}")
    #     print("回答（仅规则）：", rag.query(q, metadata_filter=rule_filter))
    #     print("-" * 60)