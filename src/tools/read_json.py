import json
import re

def read_json_file(file_path: str) -> dict:
    """
    讀取混合文本文件（包含 JSON 和其他文本）內容。
    
    Args:
        file_path (str): 文件的完整路徑。
        
    Returns:
        dict: 提取的內容，包括 JSON 數據和其他文本。如果文件無效或為空則返回錯誤消息。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {"error": "文件為空"}

        # 提取 JSON 部分（使用正則表達式）
        json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        json_data = None
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return {"error": f"JSON 解碼失敗: {e}"}

        # 提取非 JSON 文本部分
        non_json_content = re.sub(r"```json\n.*?\n```", "", content, flags=re.DOTALL).strip()

        # 返回提取的內容
        return {
            "success": True,
            "json_data": json_data if json_data else {},
            "non_json_content": non_json_content
        }

    except FileNotFoundError:
        return {"error": f"文件未找到: {file_path}"}
    except Exception as e:
        return {"error": f"未知錯誤: {e}"}
