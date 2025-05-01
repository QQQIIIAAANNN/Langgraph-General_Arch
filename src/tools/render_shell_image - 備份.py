import websocket
import uuid
import json
import urllib.request
import urllib.parse
import requests
import os
import random
import glob
import shutil
from PIL import Image
import io
from langchain.tools import tool


# ComfyUI WebSocket 伺服器設置
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

OUTPUT_SHELL_CACHE_DIR = "./output/render_cache"


@tool
def render_shell_image(current_round: int, two_d: str, selected_shell_images: dict) -> str:
    """
    使用 ComfyUI WebSocket API 渲染圖片，根據 workflow 和指定的 base_map 及 shell_types 生成。
    
    Args:
        current_round (int): 目前輪次，決定要選擇的外殼圖。
        two_d (str): 方案圖片。
        selected_shell_images (dict): RAGImageSelection 任務返回的外殼圖片字典。
    
    Returns:
        str: 渲染後圖片的保存路徑（多個路徑以逗號分隔）。
    """

    # 函數: 發送 workflow 請求
    def queue_prompt(prompt):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    # 函數: 接收 WebSocket 圖片數據
    def get_image(filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
            return response.read()

    def get_history(prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
            return json.loads(response.read())

    def get_images(ws, prompt):
        prompt_id = queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
                # bytesIO = BytesIO(out[8:])
                # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
                continue #previews are binary data

        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

        return output_images

    def upload_file(file, subfolder="", overwrite=False):
        try:
            # Wrap file in formdata so it includes filename
            body = {"image": file}
            data = {}
            
            if overwrite:
                data["overwrite"] = "true"
    
            if subfolder:
                data["subfolder"] = subfolder

            resp = requests.post(f"http://{server_address}/upload/image", files=body,data=data)
            
            if resp.status_code == 200:
                data = resp.json()
                # Add the file to the dropdown list and update the widget value
                path = data["name"]
                if "subfolder" in data:
                    if data["subfolder"] != "":
                        path = data["subfolder"] + "/" + path
                
            else:
                print(f"{resp.status_code} - {resp.reason}")
        except Exception as error:
            print(error)
        return path


    # 加載 workflow JSON 文件
    json_file_path = os.path.abspath("./src/tools/comfyUI/case_study_workflow.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        workflow_jsondata = f.read()

    prompt = json.loads(workflow_jsondata)

    # 定義 base_map 和 shell_file 的路徑&提取檔案名稱
    base_map_path = os.path.abspath(os.path.join("./input/2D", two_d))

    shell_image_list = list(selected_shell_images.keys())  # 取得所有圖片名稱
    if not shell_image_list:
        raise ValueError("❌ RAGImageSelection 未返回任何圖片，無法渲染！")
    selected_shell_filename = shell_image_list[min(current_round, len(shell_image_list) - 1)] 
    shell_file_path = os.path.abspath(os.path.join("./knowledge/selected_shell_cache", selected_shell_filename))

    if not os.path.exists(base_map_path) or not os.path.exists(shell_file_path):
        raise FileNotFoundError("未找到 base_map 或 shell_file 文件，請檢查文件路徑！")

    base_map_filename = os.path.basename(base_map_path)
    shell_file_filename = os.path.basename(shell_file_path)

    with open(base_map_path, "rb") as f:
        comfyui_path_image = upload_file(f, "", True)

    with open(shell_file_path, "rb") as f:
        comfyui_path_image = upload_file(f, "", True)

    # 配置 workflow 中的節點
    prompt["62"]["inputs"]["image"] = base_map_filename  # Site Image Node
    prompt["74"]["inputs"]["image"] = shell_file_filename  # Target Image Node

    # # 隨機種子
    # prompt["230"]["inputs"]["seed"] = random.randint(1, 1000000000)

    # 連接 WebSocket 並獲取圖片
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    print("WebSocket 連接成功")
    images = get_images(ws, prompt)
    ws.close()

    # 保存圖片
    # output_dir = os.path.abspath( "./output/case_study")
    os.makedirs(OUTPUT_SHELL_CACHE_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_SHELL_CACHE_DIR, f"shell_result_{current_round}.png")
    output_filename = os.path.basename(output_path)

    if images:
        for node_id in images:
            for idx, image_data in enumerate(images[node_id]):
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path)
                print(f"圖片保存至: {output_path}")
                return output_filename

    # # 將生成的圖片複製到 OUTPUT_SHELL_CACHE_DIR 並重命名
    # if os.path.exists(OUTPUT_SHELL_CACHE_DIR):
    #     shutil.rmtree(OUTPUT_SHELL_CACHE_DIR)  # 清空目錄
    # os.makedirs(OUTPUT_SHELL_CACHE_DIR, exist_ok=True)

    # target_path = os.path.join(OUTPUT_SHELL_CACHE_DIR, "shell_result.png")
    # shutil.copy(output_path, target_path)
    # print(f"成功複製 {output_path} 到 {target_path}")

    raise RuntimeError("未生成任何圖片，請檢查輸入數據或工具配置！")
