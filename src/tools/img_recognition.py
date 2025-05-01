import os
from PIL import Image
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
from langchain.tools import tool

# 加載環境變量
load_dotenv()

# 獲取 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 確保 API Key 設置正確
if not GEMINI_API_KEY:
    raise ValueError("❌ 缺少 GEMINI_API_KEY，請在 .env 文件中設置！")


def process_images_with_gemini(image_paths, prompt):
    """
    使用 Gemini 模型處理多張圖片並生成文字描述。

    :param image_paths: 一個包含圖片路徑的列表(可以使用複數圖片路徑)
    :param prompt: 分析目標的提示文字
    :return: Gemini 生成的文字描述
    """
    try:
        # 初始化 Gemini 模型
        model = GenerativeModel("gemini-2.0-flash")

        # 打開所有圖片
        images = [Image.open(path) for path in image_paths if os.path.exists(path)]

        if not images:
            return "❌ 無法找到任何有效的圖片，請檢查圖片路徑。"

        # 使用 Gemini 進行內容生成
        response = model.generate_content([prompt] + images)

        # 提取並返回內容
        return response.text.strip() if response and hasattr(response, "text") else "Gemini 未返回任何內容，請檢查輸入。"

    except Exception as e:
        print(f"❌ Gemini 處理過程中發生錯誤: {e}")
        return None


@tool
def img_recognition(image_paths, prompt: str) -> str:
    """
    使用 Gemini 模型識別圖片，支援多種格式的圖片路徑輸入，可同時處理多張圖片。
    
    Args:
        image_paths: 圖片路徑的集合，支援以下格式：
                     - list，例如: ["/path/to/image1.png", "/path/to/image2.png"]
                     - dict，例如: {"fused_image": "/path/to/fused_image.png", "future_image": "/path/to/future_image.png"}
                     - 單一 str，例如: "/path/to/image.png"
        prompt (str): 用於引導 Gemini 生成描述的提示詞。

    Returns:
        str: Gemini 生成的文字描述，或錯誤提示。
    """
    # 根據輸入類型轉換成 list 格式
    if isinstance(image_paths, dict):
        paths = list(image_paths.values())
    elif isinstance(image_paths, list):
        paths = []
        for item in image_paths:
            # 如果元素為字串且包含逗號，則拆分為多個圖片路徑
            if isinstance(item, str) and ',' in item:
                paths.extend([p.strip() for p in item.split(',') if p.strip()])
            else:
                paths.append(item)
    elif isinstance(image_paths, str):
        paths = [image_paths]
    else:
        raise ValueError("不支持的 image_paths 類型，請傳入 list、dict 或 str。")
    
    return process_images_with_gemini(paths, prompt) or "❌ 無法識別圖片內容，請檢查輸入。"




