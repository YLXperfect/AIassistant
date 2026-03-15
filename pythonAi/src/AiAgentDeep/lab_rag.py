# lab_rag.py - 步骤1: 文档加载与分割
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

#文件路径  绝对路径
# DOCUMENT_PATH = "/Users/eval/Desktop/项目/项目/Python AI项目/Aiassistant/AIassistant/pythonAi/src/AiAgentDeep/ylx.txt"   
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建完整路径
DOCUMENT_PATH = os.path.join(current_dir, "ylx.txt")

def load_and_split_documents():
    """加载并分割文档"""
    try:
        # 1. 加载文档（尝试utf-8，失败则用gbk）
        try:
            loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
        except:
            loader = TextLoader(DOCUMENT_PATH, encoding="gbk")
        
        documents = loader.load()
        print(f"✅ 已加载文档，原始页数: {len(documents)}")
        print(f"   第一页预览: {documents[0].page_content[:200]}...\n")
        
        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"✅ 文档已分割为 {len(split_docs)} 个片段。")
        if split_docs:
            print(f"   片段示例 (1/{len(split_docs)}): {split_docs[0].page_content[:200]}...\n")
        return split_docs
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {DOCUMENT_PATH}")
        return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


#  - 步骤2: 向量化与存储
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain import zhipuai
def create_vector_store(split_docs):
    """将文档片段向量化并存储到ChromaDB"""
    print("\n=== 第二步：创建向量数据库 ===")
    
    # 1. 初始化智谱嵌入模型
    # 注意：模型名可能需要根据智谱最新文档调整
    try:
        embeddings = ZhipuAIEmbeddings(
            api_key=os.getenv("ZHIPUAI_API_KEY"),  # 复用你的对话模型密钥
            model="embedding-2"  # 或 "text_embedding"，请确认
        )
        print("✅ 嵌入模型已初始化")
    except Exception as e:
        print(f"❌ 嵌入模型初始化失败: {e}")
        print("   可能原因：1) API Key无权限 2) 模型名错误 3) 网络问题")
        return None
    
    # 2. 创建向量数据库（持久化到本地）
    try:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"  # 向量数据库本地目录
        )
        print("✅ 向量数据库已创建并保存至 ./chroma_db")
    except Exception as e:
        print(f"❌ 创建向量数据库失败: {e}")
        return None
    
    # 3. 测试检索功能
    test_query = "Eval"  # 使用你文档中的关键词
    print(f"\n🔍 测试检索: 查询 '{test_query}'")
    try:
        results = vectorstore.similarity_search(test_query, k=2)
        for i, doc in enumerate(results):
            print(f"  结果 {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")
    
    return vectorstore

# 修改主函数以包含第二步
if __name__ == "__main__":
    print("=== RAG第一步：文档加载与分割 ===")
    split_docs = load_and_split_documents()
    
    if split_docs:
        # 新增：执行第二步
        vectorstore = create_vector_store(split_docs)
