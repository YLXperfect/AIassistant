from langchain_community.document_loaders import (
    TextLoader,          # 文本文件
    PyPDFLoader,         # PDF文件
    CSVLoader,           # CSV文件
    WebBaseLoader        # 网页
)

import os
import bs4

#加载网页要设置USER_AGENT
os.environ['USER_AGENT'] = 'MyLangChainRAG/1.0'

#RAG第一步， 文档加载
web_path = "https://nba.hupu.com/"
class documentLoad:

    #加载txt
    def query_document_txt(self,fileName):

        file_path = os.path.join(os.getcwd(), fileName)  #构建绝对路径
        if not os.path.exists(file_path):
            return f"文件 {fileName} 不存在。"
        
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()  #返回document对象， 不同文件类型，结构不一样
        return docs

#pdf
    def query_document_pdf(self,fileName):
        file_path = os.path.join(os.getcwd(), fileName)  #构建绝对路径
        if not os.path.exists(file_path):
            return f"文件 {fileName} 不存在。"
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs



#web
    def query_document_web(self,url):
        loader = WebBaseLoader(web_path=url)
        docs = loader.load()
        return print(f"{docs}")


        
if __name__ == "__main__":
    doc = documentLoad()
    # doc.query_document_txt('ylx.txt')
    '''
    [Document(metadata={'source': '/Users/eval/Desktop/项目/项目/Python AI项目/Aiassistant/AIassistant/pythonAi/ylx.txt'}, page_content='我是股神\n很爱学习\n爱炒股')]
    '''

    doc.query_document_pdf('jianli.pdf')
    # doc.query_document_web(web_path)

    



