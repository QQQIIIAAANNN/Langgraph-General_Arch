import websocket
import uuid
import json
import urllib.request
import urllib.parse
import requests
import os
import random
import glob
from PIL import Image
import io
from langchain.tools import tool
import uuid # 確保導入 uuid


# ComfyUI WebSocket 配置
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

OUTPUT_DIR = "./output"
# OUTPUT_CASE_DIR = "./output/case_study"
OUTPUT_SHELL_CACHE_DIR = "./output/cache"


@tool
def simulate_future_image(outer_prompt: str, render_image: str) -> str:
    """
    需要開啟comfyUI。
    使用 ComfyUI WebSocket API，根據 `outer_prompt` 和 `render_image` 生成未來情境模擬圖。
    檔案名稱將包含唯一 ID。
    Args:
        outer_prompt (str): LLM 生成的建築情境 Prompt
        render_image (str): 來自 ImageRenderTask 的已渲染圖片檔名 (相對於 render_cache)。
    Returns:
        str: 渲染後的圖片檔名。如果失敗則返回錯誤訊息。
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
    json_file_path = os.path.abspath("./src/tools/comfyUI/scenario_simulation_workflow.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        workflow_jsondata = f.read()

    prompt = json.loads(workflow_jsondata)
    
    # 確定輸入圖片路徑
    shell_result_img = os.path.abspath(os.path.join(OUTPUT_SHELL_CACHE_DIR, render_image))
    if not os.path.exists(shell_result_img):
        # 返回錯誤訊息而不是拋出異常，讓 ToolAgent 處理
        return f"Error: 未找到渲染圖片: {shell_result_img}"

    try:
        with open(shell_result_img, "rb") as f:
            comfyui_path_image = upload_file(f, "", True)
            if not comfyui_path_image:
                 return "Error: Failed to upload render_image to ComfyUI."
    except Exception as e:
        return f"Error opening or uploading render_image: {e}"


    prompt["177"]["inputs"]["text"] = outer_prompt  # Set Positive Prompt
    prompt["185"]["inputs"]["image"] = comfyui_path_image # Input Image

    # # 設置 seed 為 -1（隨機種子）
    # prompt["166"]["inputs"]["seed"] = random.randint(1, 1000000000)

    # 連接 WebSocket
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = get_images(ws, prompt)
    ws.close()

    # 保存渲染後的圖片
    os.makedirs(OUTPUT_SHELL_CACHE_DIR, exist_ok=True)
    file_uuid = uuid.uuid4()
    # 使用 UUID 命名
    output_filename = f"future_render_{file_uuid}.png"
    output_path = os.path.join(OUTPUT_SHELL_CACHE_DIR, output_filename)

    if images:
        # 保存第一個生成的圖片
        for node_id in images:
            if images[node_id]: # 確保節點有圖片輸出
                image_data = images[node_id][0]
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path)
                print(f"模擬完成，圖片保存至: {output_path}")
                return output_filename # 返回檔名
            else:
                # 如果遍歷完所有節點都沒有找到圖片
                return "Error: 未生成任何圖片數據，請檢查 ComfyUI workflow 或節點輸出。"

    # 如果 images 為空
    return "Error: 未生成任何圖片，請檢查 ComfyUI 連線或 workflow 配置！"


