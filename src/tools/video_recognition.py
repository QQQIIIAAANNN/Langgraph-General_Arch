import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.tools import tool
import mimetypes

# 加載環境變量
load_dotenv()

# 獲取 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("錯誤：找不到 GEMINI_API_KEY 環境變數。")
    genai_configured = False
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_configured = True
    except Exception as config_e:
        print(f"錯誤：配置 Gemini API 時出錯: {config_e}")
        genai_configured = False

# 僅在 genai 配置成功後才初始化模型
model = None
if genai_configured:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini 模型初始化成功。")
    except Exception as model_init_e:
        print(f"錯誤：初始化 Gemini 模型時出錯: {model_init_e}")
        model = None
else:
    print("警告：由於 API Key 配置失敗，未初始化 Gemini 模型。")

@tool
def video_recognition(video_path: str, prompt: str) -> str:
    """
    使用 Gemini API 分析單個影片檔案。
    主要目的是要辨識生成3D模型的結果，需要3D模型生成的影片(使用路徑及檔名)。

    Args:
        video_path (str): 單個影片檔案的路徑。
        prompt (str): 用於引導模型分析的提示語。

    Returns:
        str: 模型分析結果或錯誤訊息。
    """
    if model is None:
        return "錯誤：Gemini 模型未成功初始化，無法分析影片。"

    if not isinstance(video_path, str) or not os.path.exists(video_path):
         return f"錯誤：提供的影片路徑無效或不存在 -> '{video_path}'"

    try:
        mime_type, _ = mimetypes.guess_type(video_path)
        supported_mime_types = [
            "video/mp4", "video/mpeg", "video/mov", "video/avi",
            "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"
        ]
        if not mime_type or mime_type not in supported_mime_types:
            return f"錯誤：不支援的影片格式或無法判斷 MIME 類型 ({mime_type}) -> '{os.path.basename(video_path)}'"

        print(f"  [video_recognition tool] 正在上傳檔案: {os.path.basename(video_path)} (MIME: {mime_type})")
        try:
            video_file_api = genai.upload_file(path=video_path)
            print(f"  [video_recognition tool] 檔案上傳成功: {video_file_api.name}")

            while video_file_api.state.name == "PROCESSING":
                print("  [video_recognition tool] 等待檔案處理完成...")
                import time
                time.sleep(10)
                video_file_api = genai.get_file(video_file_api.name)

            if video_file_api.state.name == "FAILED":
                return f"錯誤：檔案 API 處理影片失敗 -> '{os.path.basename(video_path)}'"
            if video_file_api.state.name != "ACTIVE":
                return f"錯誤：檔案 API 處理後影片狀態未知 ({video_file_api.state.name}) -> '{os.path.basename(video_path)}'"

            print(f"  [video_recognition tool] 檔案處理完成 (狀態: {video_file_api.state.name})，準備發送請求...")

        except Exception as upload_e:
             print(f"  [video_recognition tool] 檔案上傳或狀態檢查出錯: {upload_e}")
             return f"錯誤：使用 File API 上傳或檢查影片時出錯 -> '{os.path.basename(video_path)}': {upload_e}"

        contents = [
            video_file_api,
            prompt,
        ]

        print(f"  [video_recognition tool] 正在調用 generate_content...")
        response = model.generate_content(contents)

        try:
             print(f"  [video_recognition tool] 正在刪除已處理的 File API 檔案: {video_file_api.name}")
             genai.delete_file(video_file_api.name)
        except Exception as delete_e:
             print(f"  [video_recognition tool] 警告：刪除 File API 檔案時出錯: {delete_e}")

        analysis_text = response.text if hasattr(response, 'text') else f"錯誤：模型未返回有效的文本回應 (類型: {type(response)})"
        print(f"  [video_recognition tool] 分析完成: {analysis_text[:100]}...")
        return analysis_text

    except Exception as e:
        import traceback
        print(f"分析影片 '{os.path.basename(video_path)}' 時發生未預期錯誤：")
        traceback.print_exc()
        return f"錯誤：分析影片 '{os.path.basename(video_path)}' 時發生未預期錯誤: {e}"