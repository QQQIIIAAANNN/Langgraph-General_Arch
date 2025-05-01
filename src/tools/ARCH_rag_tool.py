from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import os

# 加載環境變量
load_dotenv()

# 初始化 OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 獲取向量資料庫的路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_2")

# @tool
# def ARCH_rag_tool(query: str, top_k: int = 5) -> str:
#     """
#     搜索 Chroma 向量資料庫，返回詳細文本內容(如果有的話包含圖片)及總結，但只返回總結，不包含原始的 context 與 query。
    
#     Args:
#         query (str): 查詢內容。
#         top_k (int): 返回的相關文檔數量，預設為 5。
        
#     Returns:
#         str: 僅包含總結結果的查詢結果。
#     """
#     try:
#         # 加載或初始化 Chroma 向量資料庫
#         vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)
        
#         # 在向量資料庫中執行相似性檢索
#         results = vectorstore.similarity_search(query, k=top_k)
#         if not results:
#             return "未找到與查詢相關的內容。"
        
#         # 聚合檢索到的內容（僅包括文本和表格，忽略圖片）
#         context_str = ""
#         for doc in results:
#             doc_type = doc.metadata.get('type', '').lower()
#             # 若 metadata 中有 original_content，優先使用；否則使用 page_content
#             content = doc.metadata.get('original_content', doc.page_content).strip()
            
#             if doc_type == 'image':
#                 continue  # 暫不處理圖片
#             elif doc_type == 'table':
#                 context_str += f"[表格] {content}\n"
#             else:
#                 context_str += f"[文本] {content}\n"
        
#         # 建立生成總結的模板，並指示只返回總結內容
#         answer_template = (
#             "請根據以下內容生成一個詳細的總結，僅返回總結內容，切勿包含原始的上下文或查詢資訊。\n"
#             "內容：{context}"
#             "查詢：{query}"
#         )
#         prompt_template = PromptTemplate.from_template(answer_template)
        
#         # 使用新版本的 RunnableSequence 調用方式
#         # 將 prompt_template 與 LLM 組合，類似於： prompt_template | llm
#         llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
#         chain = prompt_template | llm  # 組合為一個 RunnableSequence
        
#         # 執行鏈並僅返回總結結果
#         summary = chain.invoke({"context": context_str, "query": query})
        
#         return summary
        
#     except Exception as e:
#         return f"檢索或總結時發生錯誤：{str(e)}"


@tool
def ARCH_rag_tool(query: str, top_k: int = 5) -> str:
    """
    搜索 Chroma 向量資料庫，返回詳細文本內容(如果有的話包含圖片)及總結，不包含原始問題。
    
    Args:
        query (str): 查詢內容。
        top_k (int): 返回的相關文檔數量，預設為 5。
        
    Returns:
        str: 包含詳細文本內容和總結的查詢結果。
    """
    try:
        # 加載或初始化 Chroma 向量資料庫
        vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)
        
        # 在向量資料庫中執行相似性檢索
        results = vectorstore.similarity_search(query, k=top_k)
        
        if not results:
            return "未找到與查詢相關的內容。"
        
        # 聚合檢索到的內容（僅包括文本和表格，忽略圖片）
        context_str = ""
        for i, doc in enumerate(results):
            doc_type = doc.metadata.get('type', '').lower()
            # 若 metadata 中有 original_content，優先使用；否則使用 page_content
            content = doc.metadata.get('original_content', doc.page_content).strip()
            
            if doc_type == 'image':
                # 暫時不處理圖片
                continue
            elif doc_type == 'table':
                context_str += f"[表格] {content}\n"
            else:
                context_str += f"[文本] {content}\n"
        
        # 建立生成總結的模板，這裡不包含問題內容
        answer_template = """
        請根據以下內容提供一個對建築設計有用的完整詳細總結，列出所有關鍵點、關鍵數值資訊與重要信息：
        僅返回總結內容，切勿包含原始的上下文或查詢資訊。
        內容：{context}
        查詢：{query}
        """
        prompt_template = PromptTemplate.from_template(answer_template)
        
        answer_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1024),
            prompt=prompt_template
        )
        
        result = answer_chain.invoke({"context": context_str, "query": query})
        
        # 返回詳細文本內容和總結
        summary = result.get("text", "")
        return f"{summary}"
        
    except Exception as e:
        return f"檢索或總結時發生錯誤：{str(e)}"
