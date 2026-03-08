from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough ,RunnableLambda,RunnableBranch
from langchain_classic.retrievers import MultiQueryRetriever  # 可选多查询
import os
import re  # 导入正则表达式用于清洗

from langchain_community.document_loaders import PyPDFLoader, TextLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter,MarkdownHeaderTextSplitter




class RAGEngineLCEL:
    def __init__(self, persist_directory="./chroma_db",docs_path="./knowledge_base"): #向量库路径， 知识库文件夹路径
        self.persist_directory = persist_directory  # 路径
        self.docs_path = docs_path

        self.embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPUAI_API_KEY"))

        self.llm = ChatZhipuAI(model="glm-4.6", temperature=0.0, api_key=os.getenv("ZHIPUAI_API_KEY"),timeout = 90)

        #测试代码， 事先就已经创建好了知识库
        #self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

        # 文件列表（固定路径）
        # self.files_list = [
        #     "knowledge_base/jianli.pdf",
        #     "knowledge_base/star.md",
        #     "knowledge_base/rules.md"
        # ]
        #不再用硬编码， 采用扫描文件夹的方式添加文件
        self.files_list = self._scan_knowledge_base()


        # 检查数据库是否存在（用Chroma的get_collection检查）
        self.vectorstore = self._load_or_create_vectorstore()

        # RAG 链
        template = """根据以下上下文精准回答问题，只使用上下文信息，不要编造：
        上下文：
        {context}

        问题：{question}

        回答："""
        prompt = ChatPromptTemplate.from_template(template)
        # 高级检索器（MMR + 多查询）  MultiQueryRetriever 收到一个问题，用大模型把问题拆解成多个不同角度的相似问题，最后返回文档
        #Given a query, use an LLM to write a set of queries.
        #Retrieve docs for each query. Return the unique union of all retrieved docs
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
            ),
            # search_type="similarity_score_threshold",  # 使用分数阈值检索 ,测试知识库无问题答案，返回预设友好回复
            # search_kwargs={"k": 8, "score_threshold": 0.5 }
            # ), # 设置一个基础阈值，低于此的不返回

            llm=self.llm
        )
        

        # # 定义正常处理流程：prompt -> llm -> parser
        normal_chain = prompt | self.llm | StrOutputParser()

        # # 定义分支：如果输入字典中有 "__no_answer__" 键，直接返回预设消息 
        branch = RunnableBranch(
            (lambda x: isinstance(x, dict) and x.get("__no_answer__"),   #条件1
             lambda x: "抱歉，我在知识库中没有找到相关信息。你可以尝试换个问题。"), #分支1
            normal_chain  #   默认分支
        )
        
        self.rag_chain = (
            #第一个字典：准备输入，从输入中提取出 “question” 和检索到的 “context
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(self.check_and_mark)  # 新增检查步骤
            |branch
        )
        # '''{“context”: self.retriever, “question”: RunnablePassthrough()}：。它接收一个输入（即用户问题），然后并行执行两个操作：
        # retriever：接收问题，检索出相关文档，结果赋值给 context。
        # RunnablePassthrough()：简单地让原问题通过，赋值给 question。
        # 这一步的输出是一个字典：{“context”: “检索到的文本”, “question”: “原始问题”}。
        # | prompt：管道符将上一步的输出字典传递给提示模板 self.prompt。模板会按照定义，将 context 和 question 填入指定位置，生成完整的提示词字符串。
        # | llm：将提示词字符串传递给大语言模型。
        # | StrOutputParser()：将大模型的复杂响应对象解析为纯文本字符串
        # '''
    
    #增加回答的健壮性，如果知识库中没有返回该回答，防止模型幻觉
    def check_and_mark(self,inputs):
            # inputs 是一个字典，包含 'context' 和 'question'
            
        if not inputs.get('context') or len(inputs['context']) == 0:
                # 检索为空，返回一个带特殊键的字典
            print("答案未找到，进入预留友好返回")
            return {"__no_answer__": True, "question": inputs["question"]}
        else:
            return inputs  # 正常情况，原样返回

        # 检索type = similarity_score_threshold 的时候  测试是否能执行 if not docs_with_score:分支， 测试通过
            # docs_with_score = inputs.get("context", [])
            # if not docs_with_score:
            #     return {"__no_answer__": True, "question": inputs["question"]}
            # # 检查最高分是否低于我们认为的“相关”阈值
            # max_score = max(score for _, score in docs_with_score)  
            # if max_score < 0.75:  # 阈值需要根据你的数据和嵌入模型调整
            #     print(f"⚠️ 最高相关度分数 {max_score:.4f} 低于阈值，进入无答案分支")
            #     return {"__no_answer__": True, "question": inputs["question"]}
            # else:
            # # 正常情况，提取文档列表供后续步骤使用
            #     return {"context": [doc for doc, _ in docs_with_score], "question": inputs["question"]}
    def _scan_knowledge_base(self):
        """扫描knowledge_base文件夹，获取所有支持的文件"""
        supported_exts = ['.pdf', '.md', '.txt',]
        files = []
        if not os.path.exists(self.docs_path):
            print(f"⚠️ 知识库路径 {self.docs_path} 不存在，将创建空目录。")
            os.makedirs(self.docs_path)
            return files
        
        for file in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_exts):
                files.append(file_path)
        print(f"扫描到 {len(files)} 个知识库文件: {[os.path.basename(f) for f in files]}")
        return files

    def _clean_document_text(self, text):
        """清洗文档文本的辅助函数"""
        # 1. 去除多余的空行（将连续两个以上的换行替换为两个换行）
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # 2. 去除行首行尾的空白
        lines = [line.strip() for line in text.split('\n')]
        # 3. 过滤掉空行后重新组合（根据需求选择是否保留空行分隔）
        cleaned_lines = [line for line in lines if line]  # 完全去掉空行
        # 或者保留一个空行作为段落分隔：cleaned_text = '\n\n'.join(cleaned_lines)
        cleaned_text = '\n'.join(cleaned_lines)  # 简单连接
        return cleaned_text


    def _load_and_split_documents(self):
        """加载所有文档，并根据格式进行差异化分割和清洗"""
        all_docs = []  # 加载后的原始文档
        for file_path in self.files_list:
            print(f"正在处理: {file_path}")
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_path.endswith('.md'):
                # 对于Markdown文件，先用UnstructuredMarkdownLoader加载
                # loader = UnstructuredMarkdownLoader(file_path)
                # 使用 TextLoader 加载 Markdown 文件（无需 unstructured） 后续升级环境 安装Unstructured依赖包
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                #暂时不支持docx文件， 等安装Unstructured依赖包
                elif file_path.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                else:
                    print(f"⚠️ 跳过不支持的文件: {file_path}")
                    continue
                
                # **数据清洗步骤**：对每个文档的文本内容进行清洗
                for doc in docs:
                    original_text = doc.page_content
                    cleaned_text = self._clean_document_text(original_text)
                    doc.page_content = cleaned_text
                
                all_docs.extend(docs)
                print(f"  加载了 {len(docs)} 个文档片段")
            except Exception as e:
                print(f"❌ 处理文件 {file_path} 时出错: {e}，跳过此文件")
        
        # **差异化分割**
        all_splits = []
        
        # 按来源文件类型分组处理
        pdf_docs = [doc for doc in all_docs if doc.metadata.get('source', '').endswith('.pdf')]
        md_docs = [doc for doc in all_docs if doc.metadata.get('source', '').endswith('.md')]
        txt_docs = [doc for doc in all_docs if doc.metadata.get('source', '').endswith('.txt')]
        docx_docs = [doc for doc in all_docs if doc.metadata.get('source', '').endswith('.docx')]

        # 1. 处理 PDF 和 TXT（使用递归字符分割器，按自然边界）
        if pdf_docs or txt_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=60,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                keep_separator=False
            )
            if pdf_docs:
                pdf_splits = text_splitter.split_documents(pdf_docs)
                all_splits.extend(pdf_splits)
                print(f"📄 PDF 分割完成，得到 {len(pdf_splits)} 个块")
            if txt_docs:
                txt_splits = text_splitter.split_documents(txt_docs)
                all_splits.extend(txt_splits)
                print(f"📄 TXT 分割完成，得到 {len(txt_splits)} 个块")

        # 2. 处理 Markdown（先按标题分割，再对长块二次分割）
        if md_docs:
            headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            md_splits_by_header = []
            for doc in md_docs:
                # 注意：MarkdownHeaderTextSplitter 需要文本，返回的是 Document 列表
                splits = markdown_splitter.split_text(doc.page_content)
                for split in splits:
                    # 保留原始文件的元数据
                    split.metadata.update(doc.metadata)
                    md_splits_by_header.append(split)
            
            # 对过长的块进行二次分割
            final_md_splits = []
            md_text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
            for doc in md_splits_by_header:
                if len(doc.page_content) > 600:
                    sub_splits = md_text_splitter.split_documents([doc])
                    final_md_splits.extend(sub_splits)
                else:
                    final_md_splits.append(doc)
            all_splits.extend(final_md_splits)
            print(f"📑 Markdown 分割完成，得到 {len(final_md_splits)} 个块")

        # 3. 处理 DOCX（先用简单分割，后续可优化为按样式分割）
        if docx_docs:
            docx_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            docx_splits = docx_splitter.split_documents(docx_docs)
            all_splits.extend(docx_splits)
            print(f"📄 DOCX 分割完成，得到 {len(docx_splits)} 个块")

        print(f"✅ 总计生成 {len(all_splits)} 个文档块")
        return all_splits



    def _load_or_create_vectorstore(self):
        """加载或创建向量库：如果存在，直接加载；否则，加载文件并创建"""
        # 检查collection是否存在
        try:
            vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            if vectorstore._collection.count() > 0:
                print("✅ 加载现有向量库（已存在）。")
                return vectorstore
        except Exception as e:
            print(f"⚠️ 向量库不存在或错误：{e}，将重新创建。")

        print("🔄 创建新向量库...")
        if not self.files_list:
            print("⚠️ 知识库文件夹为空，将创建空的向量库。")
            # 创建空库并持久化
            vectorstore = Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            return vectorstore

        # 加载并分割文档
        all_splits = self._load_and_split_documents()

        # 创建向量库
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print("✅ 新向量库创建完成。")
        return vectorstore

        
    def query_document(self,question:str):
        return self.rag_chain.invoke(question)
    # def add_document(self,file_name:str):

    async def aquery_document(self, question: str):
        """异步版本：使用 ainvoke 非阻塞执行 RAG 链"""
        
        return await self.rag_chain.ainvoke(question)

    def debug_inspect_vectorstore(self, keyword: str, k: int = 20):
        """调试：直接查看向量库中与关键词相关的所有文档块"""
        # 1. 直接用相似度搜索获取原始块和分数
        docs_with_score = self.vectorstore.similarity_search_with_score(keyword, k=k)
    
        print(f"\n🔍【向量库诊断】关键词: '{keyword}'")
        print(f"检索到 {len(docs_with_score)} 个相关块\n")
    
        for i, (doc, score) in enumerate(docs_with_score):
            print(f"--- 块 {i+1} (相似度: {score:.4f}) ---")
            print(f"内容预览: {doc.page_content[:150].replace(chr(10), ' ')}")
            print(f"元数据: {doc.metadata}")
            print(f"完整内容长度: {len(doc.page_content)} 字符")
            print()
    
        return docs_with_score





if __name__ == "__main__":
    engine = RAGEngineLCEL()
    # 诊断所有包含“工作”的块
    # engine.debug_inspect_vectorstore(keyword="工作经历")
    # answer = engine.query_document("star法则是什么")
    # answer = engine.query_document("好的简历应该如何量化工作成果")
    answer = engine.query_document("jianli.pdf里面的工作经历有哪些")
    print(f"回答: {answer}")


    