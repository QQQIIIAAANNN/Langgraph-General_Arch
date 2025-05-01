import websocket
import uuid
import json
import urllib.request
import urllib.parse
import requests
import os
import time
import random
import shutil
from langchain.tools import tool
import uuid # 確保導入 uuid

# ComfyUI WebSocket 伺服器設置
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
# 目標輸出目錄（請確保此目錄存在或程序能創建）
OUTPUT_MODEL_CACHE_DIR = "./output/model_cache"

@tool
def generate_3D(image_path: str) -> dict:
    """
    需要開啟comfyUI。使用 ComfyUI，透過 WebSocket API 獲取輸出結果，
    從中提取生成的模型 (.glb) 與影片 (.mp4)。
    若 output_files 為空，則使用 fallback 邏輯從 outputs/3d 資料夾中讀取，
    模型檔案預期命名為 3d.glb，影片檔案預期命名為 3d_preview.mp4，
    最後將檔案寫入 OUTPUT_MODEL_CACHE_DIR，並使用 UUID 重新命名。

    Args:
        image_path (str): 輸入圖片路徑

    Returns:
        dict: 包含模型和影片檔名的字典，例如 {"model": "model_result_uuid.glb", "video": "video_result_uuid.mp4"}
              如果某個檔案生成失敗，對應的值會是 None。
    """
    def queue_prompt(prompt):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    
    def get_file(filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
            return response.read()
    
    def get_history(prompt_id):
        with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def get_outputs(ws, prompt):
        prompt_id = queue_prompt(prompt)['prompt_id']
        output_files = {}
        try:
            while True:
                out = ws.recv()
                # 若接收到二進位資料則跳過
                if isinstance(out, bytes):
                    continue
                message = json.loads(out)
                if (message.get('type') == 'executing' and 
                    message.get('data', {}).get('prompt_id') == prompt_id):
                    # 當 node 為 None 代表執行完成
                    if message.get('data', {}).get('node') is None:
                        break
        except websocket.WebSocketConnectionClosedException:
            print("WebSocket 連線已關閉，跳出等待。")
        
        # 透過歷史記錄取得輸出結果
        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'files' in node_output:
                for file_info in node_output['files']:
                    file_name = file_info['filename']
                    file_data = get_file(file_name, file_info['subfolder'], file_info['type'])
                    output_files[file_name] = file_data
            elif 'images' in node_output:
                for file_info in node_output['images']:
                    file_name = file_info['filename']
                    file_data = get_file(file_name, file_info['subfolder'], file_info['type'])
                    output_files[file_name] = file_data
        return output_files

    def upload_file(file, subfolder="", overwrite=False):
        try:
            body = {"image": file}
            data = {}
            if overwrite:
                data["overwrite"] = "true"
            if subfolder:
                data["subfolder"] = subfolder
            resp = requests.post(f"http://{server_address}/upload/image", files=body, data=data)
            if resp.status_code == 200:
                data = resp.json()
                path = data["name"]
                if "subfolder" in data and data["subfolder"]:
                    path = data["subfolder"] + "/" + path
            else:
                print(f"{resp.status_code} - {resp.reason}")
                path = ""
        except Exception as error:
            print(error)
            path = ""
        return path

    # 載入 workflow JSON 文件
    json_file_path = os.path.abspath("./src/tools/comfyUI/IF_TRELLIS_WF_single.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        workflow_jsondata = f.read()
    prompt = json.loads(workflow_jsondata)
    
    # 上傳輸入圖片，觸發渲染流程
    shell_filename = os.path.basename(image_path)
    # Ensure the file exists before opening
    if not os.path.exists(image_path):
        return {"error": f"Input image path does not exist: {image_path}"}
    with open(image_path, "rb") as f:
        comfyui_path_image = upload_file(f, "", True)
        # Check if upload was successful
        if not comfyui_path_image:
             return {"error": "Failed to upload image to ComfyUI."}
    
    # 更新 workflow 中圖片輸入的節點（假設節點 "67" 為圖片輸入）
    # 使用上傳後返回的路徑/名稱
    prompt["67"]["inputs"]["image"] = comfyui_path_image
    # 更新隨機種子（如有需要）
    prompt["84"]["inputs"]["seed"] = random.randint(1, 1000000000)
    
    # 建立 WebSocket 連線並獲取輸出結果
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    print("WebSocket 連接成功，開始生成模型和影片")
    output_files = get_outputs(ws, prompt)
    try:
        if ws.sock and ws.sock.connected:
            ws.close()
    except Exception as e:
        print("關閉 WebSocket 時發生錯誤:", e)
    
    # 若 output_files 為空，則改用 fallback 方法從 outputs/3d 資料夾讀取
    if not output_files:
        fallback_folder = r"D:\ComfyUI3D_windows_portable\ComfyUI\output\3d"
        print("使用 fallback 機制讀取", fallback_folder)
        output_files = {}
        model_path = os.path.join(fallback_folder, "3d.glb")
        video_path = os.path.join(fallback_folder, "3d_preview.mp4")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                output_files["3d.glb"] = f.read()
        else:
            print("找不到 fallback 模型檔案:", model_path)
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                output_files["3d_preview.mp4"] = f.read()
        else:
            print("找不到 fallback 影片檔案:", video_path)
    
    # 確保目標輸出目錄存在
    os.makedirs(OUTPUT_MODEL_CACHE_DIR, exist_ok=True)
    
    model_filename = None
    video_filename = None
    file_uuid = uuid.uuid4() # 使用 UUID 命名

    for file_name, file_data in output_files.items():
        # 依據檔案名稱判斷類型並寫入
        if file_name.lower().startswith("3d") and file_name.lower().endswith(".glb") and model_filename is None:
            model_filename = f"model_result_{file_uuid}.glb"
            model_saved_path = os.path.join(OUTPUT_MODEL_CACHE_DIR, model_filename)
            with open(model_saved_path, 'wb') as f:
                f.write(file_data)
            print(f"模型已保存至: {model_saved_path}")
        elif file_name.lower().startswith("3d_preview") and file_name.lower().endswith(".mp4") and video_filename is None:
            video_filename = f"video_result_{file_uuid}.mp4"
            video_saved_path = os.path.join(OUTPUT_MODEL_CACHE_DIR, video_filename)
            with open(video_saved_path, 'wb') as f:
                f.write(file_data)
            print(f"影片已保存至: {video_saved_path}")
    
    return {"model": model_filename, "video": video_filename}
