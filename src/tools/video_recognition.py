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
def video_recognition(params) -> str:
    """
    使用 Gemini API 分析影片檔案，支援單個和多個檔案，以及 mp4 和 gif 格式。
    主要目的是要辨識生成3D模型的結果，需要3D模型生成的影片(使用路徑及檔名)。

    Args:
        params (dict): 包含以下參數：
            - video_paths (list 或 str): 影片檔案路徑，可以是單個檔案路徑字串，也可以是檔案路徑列表。
            - prompt (str): 用於引導模型分析的提示語。

    Returns:
        str: 模型分析結果。
    """
    # 提取參數
    if isinstance(params, dict):
        video_paths = params.get("video_paths", [])
        prompt = params.get("prompt", "請分析這個影片")
    else:
        # 兼容直接傳入兩個參數的舊方式
        video_paths = params
        prompt = "請分析這個影片"
    
    # 確保 video_paths 是列表格式
    if isinstance(video_paths, dict):
        paths = list(video_paths.values())
    elif isinstance(video_paths, list):
        paths = []
        for item in video_paths:
            if isinstance(item, str):
                if ',' in item:  # 處理逗號分隔的路徑字串
                    paths.extend([p.strip() for p in item.split(',') if p.strip()])
                else:
                    paths.append(item)
            else:
                paths.append(item)
    elif isinstance(video_paths, str):
        if ',' in video_paths:  # 處理逗號分隔的多個路徑
            paths = [p.strip() for p in video_paths.split(',') if p.strip()]
        else:
            paths = [video_paths]
    else:
        raise ValueError("不支持的 video_paths 類型，請傳入 list、dict 或 str。")

    # 確保有有效的路徑
    if not paths:
        return "沒有提供有效的影片路徑"
    
    # 處理每個影片並收集結果
    results = []
    for video_path in paths:
        try:
            # 檢查路徑是否為字串
            if not isinstance(video_path, str):
                results.append(f"跳過非字串路徑: {video_path}")
                continue
                
            # 檢查檔案是否存在
            if not os.path.exists(video_path):
                results.append(f"檔案不存在: {video_path}")
                continue
                
            # 判斷檔案類型
            if video_path.lower().endswith(".mp4"):
                mime_type = "video/mp4"
            elif video_path.lower().endswith(".mov"):
                mime_type = "video/mov"
            elif video_path.lower().endswith(".avi"):
                mime_type = "video/avi"
            elif video_path.lower().endswith(".webm"):
                mime_type = "video/webm"
            elif video_path.lower().endswith(".mp3"):
                mime_type = "audio/mp3"  # 也支援音訊檔案
            else:
                results.append(f"不支援的影片格式：{video_path}")
                continue

            # 開啟影片檔案
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()

            # 將路徑資訊加入提示中，使模型知道正在分析哪個檔案
            file_specific_prompt = f"分析影片檔案 '{os.path.basename(video_path)}'：\n{prompt}"

            # 準備請求內容
            contents = [
                {
                    "mime_type": mime_type,
                    "data": video_data
                },
                file_specific_prompt,
            ]

            # 發送請求並獲取回應
            response = model.generate_content(contents)
            response.resolve()  # 確保回應已完全解析
            
            # 將檔案名稱添加到結果開頭，以便於區分多個影片的結果
            file_result = f"===== 影片分析: {os.path.basename(video_path)} =====\n{response.text}"
            results.append(file_result)

        except Exception as e:
            results.append(f"分析影片 '{video_path}' 時發生錯誤：{e}")

    # 合併所有結果
    all_results = "\n\n" + "="*50 + "\n\n".join(results)
    
    # 加上整體評估結果
    if all([r for r in results if "整體評估：通過" in r]):
        return f"整體評估：通過 (所有影片均通過)\n{all_results}"
    else:
        return f"整體評估：失敗 (至少一個影片未通過或發生錯誤)\n{all_results}"