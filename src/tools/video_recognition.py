import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.tools import tool

# 加載環境變量
load_dotenv()

# 獲取 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 初始化 Gemini 模型
model = genai.GenerativeModel('gemini-2.0-flash')

@tool
def video_recognition(video_paths, prompt: str) -> str:
    """
    使用 Gemini API 分析影片檔案，支援單個和多個檔案，以及 mp4 和 gif 格式。
    主要目的是要辨識生成3D模型的結果，需要3D模型生成的影片(使用路徑及檔名)。

    Args:
        video_paths (list 或 str): 影片檔案路徑，可以是單個檔案路徑字串，也可以是檔案路徑列表。
        prompt (str): 用於引導模型分析的提示語。

    Returns:
        str: 模型分析結果。
    """
    # 根據輸入類型轉換成 list 格式
    if isinstance(video_paths, dict):
        paths = list(video_paths.values())
    elif isinstance(video_paths, list):
        paths = []
        for item in video_paths:
            if isinstance(item, str) and ',' in item:
                paths.extend([p.strip() for p in item.split(',') if p.strip()])
            else:
                paths.append(item)
    elif isinstance(video_paths, str):
        paths = [video_paths]
    else:
        raise ValueError("不支持的 video_paths 類型，請傳入 list、dict 或 str。")

    results = []
    for video_path in paths:
        try:
            # 判斷檔案類型
            if video_path.lower().endswith(".mp4"):
                mime_type = "video/mp4"
            else:
                return f"不支援的影片格式：{video_path}"

            # 開啟影片檔案
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()

            # 準備請求內容
            contents = [
                {
                    "mime_type": mime_type,
                    "data": video_data
                },
                prompt,
            ]

            # 發送請求並獲取回應
            response = model.generate_content(contents)
            response.resolve()  # 確保回應已完全解析
            results.append(response.text)

        except Exception as e:
            return f"分析影片時發生錯誤：{e}"

    return "\n\n".join(results)