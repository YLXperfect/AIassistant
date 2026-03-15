#RAG 第二步文档分割


# src/AiAgentDeep/RAGLearn/text_splitting_demo.py
import os
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter
import tiktoken  # 用于token计数

class TextSplitterDemo:
    """文本分割器演示类"""
    
    def __init__(self):
        # 创建不同分割器实例
        self.splitters = self._create_splitters()
        
        # 示例文档（用于演示）
        self.sample_doc = self._create_sample_document()
    
    def _create_sample_document(self) -> str:
        """创建示例文档"""
        return """# RAG系统学习指南

## 第一章：什么是RAG？

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。它通过从外部知识库检索相关信息，然后基于这些信息生成回答。

主要优势包括：
1. 提高准确性
2. 减少幻觉
3. 可引用来源

## 第二章：文档处理流程

文档处理是RAG的基础，包括以下步骤：

### 2.1 文档加载
从各种来源加载文档，包括：
- 文本文件 (.txt, .md)
- PDF文档
- 网页内容
- 数据库记录

### 2.2 文本分割
将长文档分割成较小的chunks，保持语义完整性。

### 2.3 向量化
将文本转换为向量表示，用于相似度计算。

### 2.4 向量存储
将向量存储在专门的数据库中，支持快速检索。

## 第三章：最佳实践

### 3.1 分割策略
选择合适的chunk大小：
- 小chunk（200-500字符）：适合事实查询
- 中chunk（500-1000字符）：适合概念解释
- 大chunk（1000-2000字符）：适合复杂分析

### 3.2 重叠设置
适当的重叠（10-20%）可以保持上下文连贯性。

### 3.3 语义边界
尽量在自然边界处分割，如段落、标题处。

## 总结

RAG系统是构建智能问答系统的关键技术。掌握文档处理流程是成功的第一步。

更多信息请参考官方文档。
"""
    
    def _create_splitters(self) -> Dict[str, Any]:
        """创建各种文本分割器"""
        
        # 1. 递归字符分割器（最常用）
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,      # 每个chunk的大小
            chunk_overlap=50,    # 重叠的大小
            length_function=len, # 计算长度的函数
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "] # 分割符
        )
        
        # 2. 字符分割器（简单按字符数）
        character_splitter = CharacterTextSplitter(
            separator="\n\n",  # 按双换行分割
            chunk_size=300,    # 每个chunk的大小
            chunk_overlap=50,  # 重叠的大小
            length_function=len # 计算长度的函数
        )
        
        # 3. Markdown分割器（按标题分割）
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=300,      # 每个chunk的大小
            chunk_overlap=50     # 重叠的大小
        )
        
        # 4. Token分割器（按token数）
        # 需要先定义token计数函数
        # def tiktoken_len(text):
        #     """使用tiktoken计算token数"""
        #     encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4使用的编码
        #     return len(encoding.encode(text))
        
        # token_splitter = TokenTextSplitter(
        #     chunk_size=100,      # token数
        #     chunk_overlap=20,    # 重叠的大小
        #     encoding_name="cl100k_base", # 编码名称
        #     disallowed_special=() # 不允许的特殊token
        # )
        
        return {
            'recursive': recursive_splitter,
            'character': character_splitter,
            'markdown': markdown_splitter,
            # 'token': token_splitter
        }
    
    def demo_basic_splitting(self):
        """演示基础分割"""
        print("🔧 基础分割演示")
        print("=" * 60)
        
        # 使用递归分割器
        chunks = self.splitters['markdown'].split_text(self.sample_doc)
        
        print(f"原始文档长度: {len(self.sample_doc)} 字符")
        print(f"分割后chunks数量: {len(chunks)}")
        print("\n分割结果:")
        
        for i, chunk in enumerate(chunks):
            print(f"\n📄 Chunk #{i+1} ({len(chunk)} 字符):")
            print("-" * 40)
            # 显示前150个字符
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            print(preview)
        
        return chunks
    
    def compare_splitters(self):
        """比较不同分割器的效果"""
        print("\n🔄 分割器对比")
        print("=" * 60)
        
        results = {}
        
        for name, splitter in self.splitters.items():
            try:
                if name == 'token':
                    # Token分割器特殊处理
                    chunks = splitter.split_text(self.sample_doc)
                else:
                    chunks = splitter.split_text(self.sample_doc)
                
                results[name] = {
                    'chunk_count': len(chunks),
                    'avg_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                    'chunks': chunks[:3]  # 只取前3个用于展示
                }
                
                print(f"\n{name.upper()} 分割器:")
                print(f"  Chunks数量: {results[name]['chunk_count']}")
                print(f"  平均长度: {results[name]['avg_length']:.1f} 字符")
                
            except Exception as e:
                print(f"\n{name.upper()} 分割器错误: {e}")
                results[name] = None
        
        return results
    
    def demo_with_documents(self, file_path: str):
        """使用真实文档演示"""
        print(f"\n📂 加载并分割文件: {file_path}")
        print("=" * 60)
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return
        
        try:
            # 加载文档
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            if not documents:
                print("⚠️ 文档为空")
                return
            
            doc = documents[0]
            print(f"原始文档: {len(doc.page_content)} 字符")
            
            # 使用递归分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            
            # 分割文档
            chunks = text_splitter.split_text(doc.page_content)
            
            print(f"\n分割结果:")
            print(f"  Chunks数量: {len(chunks)}")
            
            # 显示统计信息
            self._show_chunk_statistics(chunks)
            
            # 显示前3个chunks
            print(f"\n前3个chunks预览:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nChunk #{i+1} ({len(chunk)} 字符):")
                print("-" * 40)
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                print(preview)
            
            return chunks
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return []
    
    def _show_chunk_statistics(self, chunks: List[str]):
        """显示chunks统计信息"""
        if not chunks:
            return
        
        lengths = [len(chunk) for chunk in chunks]
        
        print(f"\n📊 统计信息:")
        print(f"  总chunks数: {len(chunks)}")
        print(f"  平均长度: {sum(lengths)/len(lengths):.1f} 字符")
        print(f"  最小长度: {min(lengths)} 字符")
        print(f"  最大长度: {max(lengths)} 字符")
        print(f"  长度分布:")
        
        # 长度分布直方图
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')]
        hist = [0] * (len(bins) - 1)
        
        for length in lengths:
            for i in range(len(bins) - 1):
                if bins[i] <= length < bins[i+1]:
                    hist[i] += 1
                    break
        
        for i in range(len(hist)):
            if i == len(hist) - 1:
                print(f"    {bins[i]}+ 字符: {hist[i]} chunks")
            else:
                print(f"    {bins[i]}-{bins[i+1]-1} 字符: {hist[i]} chunks")
    
    def demo_advanced_techniques(self):
        """演示高级分割技巧"""
        print("\n🚀 高级分割技巧")
        print("=" * 60)
        
        # 1. 按句子分割
        print("\n1. 句子感知分割:")
        
        # 使用Spacy分割句子（需要安装spacy和中文模型）
        try:
            # 如果安装了spacy
            from langchain_text_splitters import SpacyTextSplitter
            
            # 注意：需要先安装spacy和中文模型
            # pip install spacy
            # python -m spacy download zh_core_web_sm
            
            spacy_splitter = SpacyTextSplitter(
                pipeline="zh_core_web_sm",
                chunk_size=300,
                chunk_overlap=50
            )
            
            # 使用示例文档的一部分
            sample_text = "RAG系统是现代AI应用的重要组成部分。它结合了检索和生成技术。可以有效提高回答的准确性。"
            sentences = spacy_splitter.split_text(sample_text)
            
            print(f"  原始文本: {sample_text}")
            print(f"  分割成 {len(sentences)} 个句子:")
            for i, sentence in enumerate(sentences):
                print(f"    {i+1}. {sentence}")
                
        except ImportError:
            print("  ⚠️  Spacy未安装，跳过句子分割演示")
            print("  安装命令: pip install spacy && python -m spacy download zh_core_web_sm")
        
        # 2. 语义分割策略
        print("\n2. 语义分割策略:")
        
        # 为不同内容类型设置不同chunk大小
        strategies = [
            ("技术文档", 800, 100),  # 需要更多上下文
            ("新闻文章", 500, 50),   # 中等长度
            ("对话记录", 300, 30),   # 较短
            ("代码文件", 1000, 150), # 保持代码块完整
        ]
        
        for doc_type, chunk_size, overlap in strategies:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            print(f"  {doc_type}: chunk_size={chunk_size}, overlap={overlap}")
        
        # 3. 动态chunk大小
        print("\n3. 动态chunk大小策略:")
        print("  - 根据内容密度调整")
        print("  - 在段落边界处停止")
        print("  - 避免在句子中间分割")
    
    
        
        


if __name__ == "__main__":
    demo = TextSplitterDemo()
    demo.demo_basic_splitting()
    demo.compare_splitters()
    # demo.demo_with_documents("data/sample.txt")
    # demo.demo_advanced_techniques()
    