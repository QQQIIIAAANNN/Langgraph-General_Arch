import os
from PIL import Image
from dotenv import load_dotenv
from langchain.tools import tool
from google.generativeai import GenerativeModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 加載環境變量
load_dotenv()

# 設定 API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 確保 API Key 存在
if not GEMINI_API_KEY:
    raise ValueError("❌ 缺少 GEMINI_API_KEY，請在 .env 文件中設置！")

# 設置向量資料庫（使用 text-embedding-3-small）
current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

# Gemini 處理圖片
def analyze_image_with_gemini(image_path, prompt):
    """
    使用 Gemini 分析圖片內容，獲取建築外立面資訊。
    """
    try:
        model = GenerativeModel("gemini-1.5-flash")
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])

        return response.text.strip() if response and hasattr(response, "text") else "Gemini 未返回結果"
    except Exception as e:
        print(f"❌ Gemini 分析圖片時發生錯誤: {e}")
        return None

# RAG 搜索相關資料
def search_rag_data(query):
    """
    在向量資料庫中查找與建築外立面相關的資料。
    """
    try:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in docs]) if docs else "❌ 未找到相關資料"
    except Exception as e:
        print(f"❌ RAG 搜索錯誤: {e}")
        return "❌ RAG 搜索失敗"

# LangChain 工具：生成建築設計相關的資訊（不包含 GPT 處理）
@tool
def prompt_generation(fused_image_path: str) -> dict:
    """
    1. 使用 Gemini 解析圖片外立面材質、細節、形式。
    2. 使用 RAG 查找與外立面設計相關的物環資料。
    
    Args:
        fused_image_path (str): 建築方案圖片 (e.g., "./output/RenderImg.png")
    
    Returns:
        dict: 包含 "image_description" (圖片分析結果) 和 "rag_data" (向量資料庫搜尋結果)
    """
    # 圖片分析 Prompt
    image_prompt = "辨識當前方案中的建築外立面材質、細節、形式。"
    image_description = analyze_image_with_gemini(fused_image_path, image_prompt)
    rag_data = search_rag_data(image_description)

    return {
        "image_description": image_description,
        "rag_data": rag_data
    }