"""Tool for generating and editing images using Gemini."""
from langchain.tools import tool
from google import genai
from google.genai import types
import os
import base64
import uuid
from PIL import Image
import io
import mimetypes
from typing import List, Optional, Dict, Any, Union
import traceback # Import traceback

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API key using the client style
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("需要設定 GEMINI_API_KEY 環境變數")
# Initialize the client globally or ensure it's initialized before use
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("Gemini Client 初始化成功。")
except Exception as e:
    print(f"Gemini Client 初始化失敗: {e}")
    # Depending on your application structure, you might want to exit or raise
    raise RuntimeError(f"無法初始化 Gemini Client: {e}")

# --- Constants ---
OUTPUT_IMAGE_CACHE_DIR = "./output/cache/render_cache"
os.makedirs(OUTPUT_IMAGE_CACHE_DIR, exist_ok=True)

# --- Helper Function (extracted from the class) ---
def _process_image_input(image_input: Any) -> Optional[Dict[str, Any]]:
    """
    Processes various image input types into the required format for the Gemini API.
    Supported types: file path (str), PIL.Image, bytes, base64 string.

    Returns:
        A dictionary like {"mime_type": ..., "data": ...} or None if processing fails.
    """
    image_bytes = None
    mime_type = None

    try:
        if isinstance(image_input, str):
            # Check if it's a base64 string first
            try:
                # Basic check for base64 pattern (might need refinement)
                if image_input.startswith('data:image'):
                    # data:image/png;base64,iVBORw0KGgo...
                    header, encoded = image_input.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    image_bytes = base64.b64decode(encoded)
                elif len(image_input) % 4 == 0 and '=' not in image_input[-3:] and base64.b64decode(image_input, validate=True):
                     # More robust check for plain base64 (allows for padding)
                     image_bytes = base64.b64decode(image_input)
                     # Attempt to guess mime type from decoded bytes (less reliable)
                     try:
                         # Try to load into PIL to guess format
                         img = Image.open(io.BytesIO(image_bytes))
                         img_format = img.format or "PNG" # Default if format unknown
                         mime_type = Image.MIME.get(img_format.upper()) or f"image/{img_format.lower()}"
                     except Exception:
                         mime_type = "image/png" # Fallback
                         print("Warning: Assuming image/png for raw base64 input after failed format detection.")
                else:
                     # Assume it's a file path
                     if os.path.exists(image_input):
                         mime_type, _ = mimetypes.guess_type(image_input)
                         if not mime_type or not mime_type.startswith("image/"):
                             print(f"Warning: Could not determine image MIME type for path {image_input}. Skipping.")
                             return None
                         with open(image_input, "rb") as f:
                             image_bytes = f.read()
                     else:
                         print(f"Error: Image path does not exist: {image_input}")
                         return None
            except (ValueError, TypeError, Exception):
                 # If base64 decode fails or other error, assume it's a path
                 if os.path.exists(image_input):
                     mime_type, _ = mimetypes.guess_type(image_input)
                     if not mime_type or not mime_type.startswith("image/"):
                         print(f"Warning: Could not determine image MIME type for path {image_input}. Skipping.")
                         return None
                     with open(image_input, "rb") as f:
                         image_bytes = f.read()
                 else:
                     print(f"Error: Image path does not exist: {image_input}")
                     return None
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            img_format = image_input.format if image_input.format else "PNG" # Default to PNG
            # Ensure format is supported for saving
            save_format = img_format.upper()
            if save_format not in Image.SAVE:
                print(f"Warning: Original format {save_format} not directly savable, converting to PNG.")
                save_format = "PNG"
            image_input.save(buffer, format=save_format)
            image_bytes = buffer.getvalue()
            mime_type = Image.MIME.get(save_format)
            if not mime_type: mime_type = f"image/{save_format.lower()}" # Fallback MIME type
        elif isinstance(image_input, bytes):
            image_bytes = image_input
            # Attempt to guess mime type from bytes
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img_format = img.format or "PNG"
                mime_type = Image.MIME.get(img_format.upper()) or f"image/{img_format.lower()}"
            except Exception:
                # Cannot reliably determine mime type from bytes alone, default assumption
                mime_type = "image/png" # Default assumption for raw bytes
                print("Warning: Assuming image/png for raw bytes input after failed format detection.")
        else:
            print(f"Error: Unsupported image input type: {type(image_input)}")
            return None

        if image_bytes and mime_type:
            return {"mime_type": mime_type, "data": image_bytes}
        else:
            print("Error: Failed to extract image bytes or determine MIME type.")
            return None

    except Exception as e:
        print(f"Error processing image input: {e}")
        return None

# --- Tool Function ---
@tool("gemini_image_generation")
def generate_gemini_image(prompt: str, image_inputs: Optional[List[Any]] = None, i: int = 1, model_name: str = "gemini-2.0-flash-exp-image-generation") -> Dict[str, Any]:
    """
    使用指定的 Gemini 模型，根據文字提示和可選的圖片輸入來生成或編輯圖片。
    使用 client.models.generate_content 方式調用。

    處理多種情境：
    - 文字生成圖片：僅提供 `prompt`。
    - 圖片編輯 / 文字與圖片生成圖片：提供 `prompt` 和 `image_inputs`。
    - 多模態輸入：`image_inputs` 中可以處理檔案路徑、PIL 圖片、位元組或 base64 字串。

    **注意：** 模型有時可能只輸出文字或停止生成。
    請嘗試在提示中明確要求圖片輸出（例如，"生成一張..."或"將圖片更新為..."）。

    Args:
        prompt (str): 圖片生成/編輯的文字描述或指令。
        image_inputs (Optional[List[Any]]): 圖片輸入的列表。
        model_name (str): 要使用的 Gemini 模型名稱。
        i (int): 要生成的圖片數量 (預設為 1)。

    Returns:
        Dict[str, Any]: 包含生成結果的字典。
    """
    global client # Access the globally initialized client
    if client is None:
         return {"error": "Gemini Client 未成功初始化。"}

    print(f"Gemini Image Gen Tool (Client Mode): Received prompt='{prompt[:50]}...', {len(image_inputs) if image_inputs else 0} image inputs.")
    
    # 準備 contents 列表 - 首先加入文本提示
    if image_inputs:
        # 圖生圖／圖片編輯
        # 使用者提供的 prompt 作為編輯指令
        text_prompt = f"根據以下提示「{prompt}」，並參考所附的圖片，修改並生成 {i} 張圖片。"
    else:
        # 純文字生成圖片
        text_prompt = f"請生成 {i} 張圖片，內容為：「{prompt}」。"
    # --- END MODIFIED ---
    
    contents = [text_prompt]  # 開始只有文字提示
    
    # 處理圖片輸入
    if image_inputs:
        for img_input in image_inputs:
            try:
                # 處理圖片輸入並獲取字節和MIME類型
                if isinstance(img_input, str) and os.path.exists(img_input):
                    # 如果是文件路徑，讀取文件並獲取MIME類型
                    mime_type, _ = mimetypes.guess_type(img_input)
                    if not mime_type or not mime_type.startswith("image/"):
                        mime_type = "image/png"  # 默認MIME類型
                    with open(img_input, "rb") as f:
                        img_bytes = f.read()
                    print(f"成功載入圖片: {img_input}")
                    
                    # 創建Part物件並添加到contents
                    img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                    contents.append(img_part)
                else:
                    print(f"Warning: 跳過無效的圖片輸入: {img_input}")
            except Exception as e:
                print(f"處理圖片輸入時發生錯誤: {e}")
                continue
    
    print(f"Gemini Image Gen Tool: 處理了 {len(contents)-1} 張圖片。最終 contents 長度: {len(contents)}")
    
    try:
        # 使用 client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                # 使用大寫格式，與官方範例一致
                response_modalities=['TEXT', 'IMAGE']
             )
        )

        print("--- Gemini API Response (Client Mode) ---")

        text_result = ""
        generated_files_info = []

        # 處理回應
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_result += part.text + "\n"
                    elif hasattr(part, "inline_data") and part.inline_data:
                        # 處理內嵌圖片數據
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type
                        if not mime_type.startswith("image/"): continue
                        extension = mime_type.split('/')[-1] if '/' in mime_type else 'png'
                        if extension == 'jpeg': extension = 'jpg'
                        filename = f"gemini_gen_{uuid.uuid4()}.{extension}"
                        output_path = os.path.abspath(os.path.join(OUTPUT_IMAGE_CACHE_DIR, filename))
                        try:
                            with open(output_path, "wb") as image_file:
                                image_file.write(image_data)
                            print(f"圖片已保存至: {output_path}")
                            generated_files_info.append({"filename": filename, "path": output_path, "type": mime_type})
                        except Exception as save_e:
                            print(f"保存生成的圖片 {filename} 時發生錯誤: {save_e}")

        if not generated_files_info and "generate" in prompt.lower() and ("image" in prompt.lower() or "圖片" in prompt.lower()):
             print("Warning: Gemini 圖片生成工具沒有生成任何圖片檔案，儘管提示要求了圖片。")

        return {
            "text_response": text_result.strip(),
            "generated_files": generated_files_info
        }

    except Exception as e:
        print(f"使用 Gemini 圖片生成工具 (Client Mode) 時發生嚴重錯誤: {e}")
        traceback.print_exc()
        error_details = str(e)
        try:
             response_obj = None
             if hasattr(e, 'response'):
                 response_obj = e.response
             elif len(e.args) > 0 and hasattr(e.args[0], 'prompt_feedback'):
                 response_obj = e.args[0]
             if response_obj and hasattr(response_obj, 'prompt_feedback'):
                 error_details += f"\nPrompt Feedback: {response_obj.prompt_feedback}"
        except Exception:
             pass

        return {"error": f"使用 Gemini 圖片生成工具時發生錯誤: {error_details}"}

