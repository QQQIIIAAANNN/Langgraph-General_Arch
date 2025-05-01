import os
import shutil
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# 設定向量資料庫路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_TM")
image_source_dir = os.path.join(current_dir, "../../knowledge/shell_images")  # 原始圖片目錄
cache_dir = os.path.join(current_dir, "../../knowledge/selected_shell_cache")  # 複製目標目錄

# 確保緩存目錄存在
os.makedirs(cache_dir, exist_ok=True)

# 初始化 OpenCLIP Embeddings
embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

# 加載 Chroma 向量資料庫
vectorstore = Chroma(
    collection_name="mm_rag_clip_photos",
    embedding_function=embeddings,
    persist_directory=vector_db_dir
)

# RAG 檢索
def search_top_shell_images(query, top_k=5):
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents(query, k=top_k)
    
    selected_images = {}
    for i, doc in enumerate(results):
        filename = doc.metadata.get("filename", f"未知檔名_{i+1}")
        selected_images[filename] = filename

    return selected_images

# 複製前 5 名圖片到 `selected_shell_cache`
def copy_top_images(image_dict, top_n=3):
    copied_files = []
    for i, filename in enumerate(list(image_dict.keys())[:top_n]):
        source_path = os.path.join(image_source_dir, filename)
        target_path = os.path.join(cache_dir, filename)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_files.append(target_path)

    return copied_files

# 定義工具
@tool
def IMG_rag_tool(design_goal: str) -> dict:
    """
    根據設計目標、需求、類型與偏好，檢索 5 張最佳外殼方案圖，並複製前 5 名到 `selected_shell_cache`。
    
    Args:
        design_goal: 設計目標
    
    Returns:
        dict: 包含前 5 張圖片的檢索結果（filename 作為 key，metadata 作為 value）
    """
    # 生成查詢語句
    query = f"{design_goal}"

    # 檢索前 5 名圖片
    selected_images = search_top_shell_images(query, top_k=5)

    # 複製前 3 名到 `selected_shell_cache`
    copied_files = copy_top_images(selected_images, top_n=5)

    print(f"✅ RAG 檢索完成，已選擇 {len(selected_images)} 張圖片")
    print(f"📌 已複製 {len(copied_files)} 張圖片到 selected_shell_cache")

    return selected_images
