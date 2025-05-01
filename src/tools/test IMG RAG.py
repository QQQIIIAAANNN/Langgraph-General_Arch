#!/usr/bin/env python3
"""
Test Retrieval Script for Multi-modal RAG with Chroma

此腳本從先前建立並儲存的向量庫中讀取資料，
並使用 retriever 進行檢索測試。檢索結果只顯示圖片檔名，
避免直接輸出 base64 encoded 的圖片資料佔據整個視窗。

請先執行 embedding_script.py 以建立向量庫。
"""

import os
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_TM")
    
    if not os.path.isdir(vector_db_dir):
        raise FileNotFoundError("向量庫目錄不存在，請先執行 embedding_script.py 建立向量庫。")
    
    embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    vectorstore = Chroma(
        collection_name="mm_rag_clip_photos",
        embedding_function=embeddings,
        persist_directory=vector_db_dir
    )
    
    retriever = vectorstore.as_retriever()
    
    test_query = "最帥的外殼是甚麼?"  
    print("開始檢索，查詢內容：", test_query)
    results = retriever.get_relevant_documents(test_query)
    
    print("檢索結果：")
    for i, doc in enumerate(results, start=1):
        filename = doc.metadata.get("filename", "未知檔名")
        print(f"結果 {i}: 檔案名稱: {filename}")
        # 若需要同時顯示 base64 編碼內容，可取消下行註解
        # print("圖片 (base64 encoded):", doc.page_content)
        print("-" * 30)

if __name__ == "__main__":
    main()
