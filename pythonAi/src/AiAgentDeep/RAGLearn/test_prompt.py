# src/AiAgentDeep/test_prompts.py
# Day10 测试脚本：对比不同 Prompt 版本的效果

from RAG_ways import RAGPipeline  
from prompt import (
    PROMPT_FEW_SHOT_WITH_CONTEXT,
    PROMPT_COT,
    PROMPT_BASIC,
    BASE_SYSTEM_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser

rag = RAGPipeline()
llm = rag.llm

def test_prompt_variant(prompt_template, user_experience, query_for_context=None):
    """测试单个 Prompt 变体"""
    # 先用 RAG 召回上下文（默认用规则过滤）
    context_retriever = rag.get_retriever(metadata_filter={"type": "rule"})
    context_docs = context_retriever.invoke(query_for_context or user_experience)
    context = "\n\n".join(doc.page_content for doc in context_docs[:4])  # Top-4

    chain = (
        {"context": lambda _: context, "user_experience": lambda _: user_experience}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    print(f"\n[测试经历] {user_experience}")
    print(f"[召回上下文摘要] {context[:300]}...")
    print(f"[输出] {chain.invoke({})}")
    print("-" * 80)

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        "负责公司微信公众号运营和内容发布",
        "参与前端页面开发与优化",
        "组织员工培训活动",
        "维护服务器和网络安全",
    ]

    variants = [
        ("=== 基础版 Prompt ===", PROMPT_BASIC, {}),
        ("\n=== Few-shot + 上下文版 ===", PROMPT_FEW_SHOT_WITH_CONTEXT, {}),
        (
            "\n=== CoT 多步思考版 ===",
            PROMPT_COT,
            {"query_for_context": "STAR 方法 量化 简历优化规则"},
        ),
    ]

    for title, prompt, kwargs in variants:
        print(title)
        for experience in test_cases:
            test_prompt_variant(prompt, experience, **kwargs)