# from langchain_chroma import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os 

# load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# # vector_db_dir = "../knowledge/vector_db"
# current_dir = os.path.dirname(os.path.abspath(__file__))  # 當前檔案所在目錄
# vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_2")  # 拼接路徑
# os.makedirs(vector_db_dir, exist_ok=True)

# # 加載已存在的 Chroma 向量資料庫
# vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

# def query_vector_db(query: str, top_k: int = 5):
#     """
#     從向量資料庫中檢索與查詢相關的文檔。
#     Args:
#         query (str): 用戶的查詢。
#         top_k (int): 返回的相關文檔數量。
#     Returns:
#         list: 檢索到的相關上下文文檔。
#     """
#     results = vectorstore.similarity_search(query, k=top_k)
#     return [result.page_content for result in results]

# # 測試檢索
# query = "台南氣候資訊,台南微氣候資料,台南風向與噪音資料,台南日照小時與光照潛力,台南地区的太陽能潛力"
# retrieved_docs = query_vector_db(query)
# print("\n".join(retrieved_docs))


# # 測試檢索
import re
import io
import os
import uuid
import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_chroma import Chroma
from IPython.display import HTML, display, Image
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 設定向量資料庫路徑（請確認此路徑存在且有讀寫權限）
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_T")

# 建立嵌入模型與向量資料庫
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)

# 用來顯示 Base64 圖片的函數
def plt_img_base64(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        display(Image(data=img_data))
    except Exception as e:
        print("圖片解析失敗:", e)

# 改進的 Base64 判斷函數
def is_valid_base64(s):
    try:
        if not isinstance(s, str):
            return False
        decoded_data = base64.b64decode(s, validate=True)
        return len(decoded_data) > 0
    except Exception:
        return False

# 建立回答模板
answer_template = """
Answer the question based only on the following context, which can include text, images and tables:
{context}

Question: {question}
"""
prompt_template = PromptTemplate.from_template(answer_template)
answer_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1024),
    prompt=prompt_template
)

def answer(question, top_k=5):
    # 從向量資料庫中查詢相關文件
    relevant_docs = vectorstore.similarity_search(question, k=top_k)
    context = ""
    relevant_images = []  # 收集圖片的 Base64 資料
    
    for d in relevant_docs:
        doc_type = d.metadata.get('type', '').lower()
        content = d.metadata.get('original_content', d.page_content)

        if doc_type == 'text':
            context += "[text] " + content + "\n"
        elif doc_type == 'table':
            context += "[table] " + content + "\n"
        elif doc_type == 'image':
            context += "[image]\n"
            if is_valid_base64(content):
                relevant_images.append(content)
            else:
                print("⚠️ 無效的 Base64 圖片資料，已忽略")
        else:
            context += content + "\n"

    # 生成答案
    result = answer_chain.invoke({'context': context, 'question': question})
    
    return result, relevant_images

# 測試查詢
if __name__ == "__main__":
    query = "請說明曲面木構造提案時應該要達成的基本判斷標準，比如循環經濟、材料分割、製造效率、永續環保性、材料損耗率、減碳等等"
    result, images = answer(query, top_k=5)
    
    print("🔍 **回答:**\n", result)

    if images:
        print(f"\n📷 共找到 {len(images)} 張圖片，以下顯示：")
        for img in images:
            plt_img_base64(img)
    else:
        print("🚫 沒有找到圖片。")
