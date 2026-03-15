# data_cleaning.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import re
import pandas as pd

def clean_pdf(file_path: str) -> list[Document]:
    """Step 1: 加载 PDF
    Step 2: 立即清洗（Day11核心：加载后第一时间去噪）
    返回清洗后的 Document 列表，供后续 chunking 使用"""
    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()  # 原始加载

    cleaned_docs = []
    for doc in raw_docs:
        text = doc.page_content
        # 清洗逻辑（当天重点）
        text = re.sub(r'\n{3,}', '\n\n', text)          # 移除过多空行
        text = re.sub(r'\s+', ' ', text)                # 压缩多空格
        text = text.strip()                             # 去前后空白
        # 可加更多：移除页码、页眉（如 r'Page \d+'）

        if len(text) > 50:  # 过滤太短的无用片段
            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata  # 保留源信息
                )
            )
    return cleaned_docs





def clean_md_to_df(docs: list[Document]) -> list[Document]:
    """
    输入：已加载的 Markdown Document 列表
    输出：清洗 + 结构化后的 Document 列表（或根据需要返回 DataFrame 后转回 Document）
    """
    if not docs:
        return []

    # 合并所有文档内容（MD 文件通常 loader 会按段/页切，但常需要整体处理）
    full_text = "\n\n".join(doc.page_content for doc in docs)

    # 分割成规则块（假设用 ## 或其他标题分隔，根据你的 rules.md/star.md 实际格式调整）
    sections = re.split(r'^##\s+', full_text, flags=re.MULTILINE)
    if len(sections) <= 1:
        sections = re.split(r'^\s*###?\s+', full_text, flags=re.MULTILINE)  # 备选分隔

    data = []
    for i, sec in enumerate(sections):
        if not sec.strip():
            continue
        lines = sec.strip().split('\n')
        rule_type = lines[0].strip() if lines else f"section_{i}"
        description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        # 简单清洗
        description = re.sub(r'\s+', ' ', description).strip()
        
        data.append({
            'rule_type': rule_type,
            'description': description,
            'source': docs[0].metadata.get('source', 'unknown') if docs else 'unknown'
        })

    df = pd.DataFrame(data)

    cleaned_docs = []
    for _, row in df.iterrows():
        content = f"{row['rule_type']}\n{row['description']}"
        cleaned_docs.append(
            Document(
                page_content=content,
                metadata={
                    "rule_type": row['rule_type'],
                    "source": row['source'],
                    "cleaned": True
                }
            )
        )

    print(f"MD 清洗：原始 {len(docs)} → 结构化后 {len(cleaned_docs)} 个 Document")
    return cleaned_docs