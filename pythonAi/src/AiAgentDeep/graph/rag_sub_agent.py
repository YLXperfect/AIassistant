"""
Day 23: RAG 子 Agent（检索质量优化版）
优化：提高相关性 + 增加调试信息
"""

import sys
import os
from typing import List

# ====================== 路径修复 ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
aiagentdeep_dir = os.path.dirname(current_dir)
sys.path.insert(0, aiagentdeep_dir)

from RAGLearn.AdvancedRAG import RAGEngineLCEL

class RAGSubAgent:
    def __init__(self):
        print("🔄 [RAGSubAgent] 初始化...")
        
        knowledge_base_path = os.path.join(aiagentdeep_dir, "RAGLearn", "knowledge_base")
        chroma_db_path = os.path.join(aiagentdeep_dir, "RAGLearn", "chroma_db")
        
        self.rag_engine = RAGEngineLCEL(
            persist_directory=chroma_db_path,
            docs_path=knowledge_base_path
        )
        
        self.vectorstore = self.rag_engine.vectorstore
        
        print(f"✅ RAGSubAgent 初始化完成")
        print(f"   向量库文档数量: {self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else '未知'}")

    def retrieve_rules(self, query: str, k: int = 10) -> List[str]:
        """优化后的检索方法"""
        print(f"🔍 [RAG] 检索查询: {query}")

        # 优化检索参数
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,                    # 返回更多候选
                "fetch_k": 30,             # 初始召回更多
                "lambda_mult": 0.7         # 提高多样性权重（0.5~0.8 之间调整）
            }
        )

        docs = retriever.invoke(query)
        
        rules = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content and len(content) > 30:
                rules.append(content)
                print(f"   [{i+1}] 相似度块长度: {len(content)} | 预览: {content[:100]}...")

        print(f"✅ 检索完成，返回 {len(rules)} 条规则")
        return rules


# ====================== 测试 ======================
if __name__ == "__main__":
    print("=== Day 23 RAGSubAgent 检索优化测试 ===\n")
    agent = RAGSubAgent()
    
    # 重点测试 STAR 相关查询
    test_queries = [
        "STAR 方法如何用于简历项目描述",
        "简历中如何使用 STAR 法则",
        "STAR 法则详解",
        "简历项目经历如何用 STAR 写"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"查询: {query}")
        rules = agent.retrieve_rules(query, k=10)
        
        print(f"\n最终返回的前 4 条规则：")
        for i, rule in enumerate(rules[:4], 1):
            print(f"{i}. {rule[:180]}{'...' if len(rule) > 180 else ''}")