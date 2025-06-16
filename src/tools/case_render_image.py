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
def case_render_image(outer_prompt: str, i: int, strength:str) -> str:
    """
    需要開啟comfyUI。使用 ComfyUI WebSocket API 渲染圖片，每次循環生成一張圖片，總共生成 i 張，
    並在每次生成前以隨機數更新隨機種子。檔案名稱將包含唯一 ID。

    Args:
        outer_prompt (str): LLM 生成的木構造建築設計 Prompt
        i:生成次數，1-4
        strength: 木構造建築的強度，0.0-0.8，數字越大，網格木構造建築的強度越高。

    Returns:
        str: 生成圖片的檔名，以逗號分隔，例如：
             shell_result_uuid1_1.png,shell_result_uuid1_2.png,...
    """

    def queue_prompt(prompt):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

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
                        break  # 執行完成
            else:
                # 這裡可以處理二進位資料（例如 latent previews），目前略過
                continue

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

    # 載入 workflow JSON 文件
    json_file_path = os.path.abspath("./src/tools/comfyUI/case_render_workflow.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        workflow_jsondata = f.read()

    prompt = json.loads(workflow_jsondata)
    
    os.makedirs(OUTPUT_SHELL_CACHE_DIR, exist_ok=True)
    
    # 根據傳入的 i 進行循環生成，i 預設為 1
    saved_filenames = []
    base_uuid = uuid.uuid4() # 為這一批生成創建一個基礎 UUID
    for iteration in range(1, i + 1):
        prompt["87"]["inputs"]["value"] = outer_prompt
        prompt["69"]["inputs"]["noise_seed"] = random.randint(1, 1000000000)
        # prompt["80"]["inputs"]["strength_model"] = strength if strength else 0
        prompt["80"]["inputs"]["strength_model"] = 0

        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        print(f"WebSocket 連接成功，開始第 {iteration} 次生成")
        
        single_images = get_images(ws, prompt)
        ws.close()
        
        # 從返回結果中取出第一張圖片
        for node_id in single_images:
            if single_images[node_id]:
                image_data = single_images[node_id][0]
                # 使用 UUID 和 iteration 命名
                output_filename = f"shell_result_{base_uuid}_{iteration}.png"
                output_path = os.path.join(OUTPUT_SHELL_CACHE_DIR, output_filename)
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path)
                print(f"圖片保存至: {output_path}")
                saved_filenames.append(output_filename) # 只保存檔名
                break

    # 返回以逗號分隔的檔名字串
    return ",".join(saved_filenames)


