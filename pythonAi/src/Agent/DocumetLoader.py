"""
统一的文档处理器
职责：处理所有文档相关的加载、清洗、分割操作
包括：知识库文档和用户上传文件
"""
import re
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import os
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """统一的文档处理器，所有文档操作都通过此类"""
    
    def __init__(self, rag_engine=None):
        """
        初始化文档处理器
        
        Args:
            rag_engine: RAG引擎实例（可选），用于获取embeddings等配置
        """
        self.rag_engine = rag_engine
        
        # 知识库文档的分割参数（精细分割）
        self.kb_chunk_size = 600
        self.kb_overlap = 60
        self.kb_md_chunk_size = 500  # Markdown二次分割大小
        self.kb_md_overlap = 50
        
        # 用户上传文件的分割参数（快速抽取）
        self.upload_chunk_size = 800
        self.upload_overlap = 120
        
        # 公共分隔符（按语义边界）
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        
        logger.info(f"✅ DocumentProcessor 初始化完成")
    
    # ========== 文本清洗 ==========
    
    def clean_text(self, text: str) -> str:
        """
        统一的文本清洗函数
        - 去除多余空行
        - 去除行首行尾空白
        - 过滤空行
        """
        if not text:
            return ""
        
        # 1. 将连续3个以上的换行替换为2个换行
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 2. 每行去除首尾空白
        lines = [line.strip() for line in text.split('\n')]
        
        # 3. 过滤掉完全空的行
        cleaned_lines = [line for line in lines if line]
        
        # 4. 重新组合（用单个换行连接）
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text
    
    # ========== 文件加载 ==========
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        从文件路径加载文档
        支持：.pdf, .txt, .md, .docx
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"正在加载文件: {file_path} (类型: {file_ext})")
        
        try:
            # 根据文件类型选择加载器
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_ext == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path, mode="single")
                docs = loader.load()
            else:  # .txt, .md 和其他文本文件
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            
            # 清洗每个文档的文本
            for doc in docs:
                doc.page_content = self.clean_text(doc.page_content)
                
                # 确保元数据中有source字段
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = os.path.basename(file_path)
            
            logger.info(f"  → 加载了 {len(docs)} 个原始文档片段")
            return docs
            
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            raise
    
    def load_file_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        从文本字符串创建文档（用于上传的文件）
        
        Args:
            text: 文本内容
            metadata: 元数据（可选）
            
        Returns:
            包含一个Document的列表
        """
        cleaned_text = self.clean_text(text)
        
        if metadata is None:
            metadata = {"source": "uploaded_file"}
        
        doc = Document(
            page_content=cleaned_text,
            metadata=metadata
        )
        
        return [doc]
    
    # ========== 文档分割 ==========
    
    def split_documents(self, docs: List[Document], mode: str = "knowledge_base") -> List[Document]:
        """
        分割文档（通用方法）
        
        Args:
            docs: 文档列表
            mode: 分割模式
                - "knowledge_base": 知识库模式（精细分割）
                - "upload": 上传文件模式（快速抽取）
                - "markdown": Markdown专用模式
                
        Returns:
            分割后的文档块列表
        """
        if not docs:
            return []
        
        # 根据模式选择分割参数
        if mode == "knowledge_base":
            chunk_size = self.kb_chunk_size
            chunk_overlap = self.kb_overlap
        elif mode == "upload":
            chunk_size = self.upload_chunk_size
            chunk_overlap = self.upload_overlap
        elif mode == "markdown":
            # Markdown模式走专门的处理
            return self._split_markdown_documents(docs)
        else:
            # 默认使用知识库模式
            chunk_size = self.kb_chunk_size
            chunk_overlap = self.kb_overlap
        
        # 创建分割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=False
        )
        
        # 执行分割
        splits = splitter.split_documents(docs)
        logger.info(f"  → 分割完成，得到 {len(splits)} 个文档块 (模式: {mode})")
        
        return splits
    
    def _split_markdown_documents(self, docs: List[Document]) -> List[Document]:
        """
        专门处理Markdown文档的分割
        策略：先按标题分割，再对长块二次分割
        """
        if not docs:
            return []
        
        # 合并所有文档内容
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # 按标题分割
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4")
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题在内容中
        )
        
        try:
            # 按标题分割
            splits_by_header = markdown_splitter.split_text(full_text)
            
            # 对过长的块进行二次分割
            final_splits = []
            md_text_splitter = MarkdownTextSplitter(
                chunk_size=self.kb_md_chunk_size,
                chunk_overlap=self.kb_md_overlap
            )
            
            for doc in splits_by_header:
                if len(doc.page_content) > self.kb_chunk_size * 1.2:  # 超过120%
                    # 二次分割
                    sub_splits = md_text_splitter.split_documents([doc])
                    final_splits.extend(sub_splits)
                else:
                    final_splits.append(doc)
            
            logger.info(f"  → Markdown分割完成，得到 {len(final_splits)} 个文档块")
            return final_splits
            
        except Exception as e:
            logger.error(f"Markdown分割失败，回退到普通分割: {e}")
            # 回退到普通分割
            return self.split_documents(docs, mode="knowledge_base")
    
    # ========== 便捷方法 ==========
    #开放一个知识库文档处理的接口， 后期添加管理员上传知识库文件的功能
    def process_for_knowledge_base(self, file_path: str) -> List[Document]:
        """
        处理知识库文件的完整流程（加载 + 清洗 + 分割）
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的文档块列表
        """
        # 1. 加载文件
        docs = self.load_file(file_path)
        
        # 2. 根据文件类型选择分割模式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.md':
            splits = self._split_markdown_documents(docs)
        else:
            splits = self.split_documents(docs, mode="knowledge_base")
        
        return splits
    
    def process_for_upload(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        处理上传文件的完整流程（从文本 + 清洗 + 分割）
        
        Args:
            text: 上传文件的文本内容
            metadata: 元数据
            
        Returns:
            分割后的文档块列表
        """
        # 1. 从文本创建文档
        docs = self.load_file_from_text(text, metadata)
        
        # 2. 使用上传模式分割
        splits = self.split_documents(docs, mode="upload")
        
        return splits