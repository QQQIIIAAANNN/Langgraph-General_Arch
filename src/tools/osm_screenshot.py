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
from typing import Dict, Any

# --- Constants ---
CANVASFLARE_API_KEY = "e3f2251094c7198c4bff1d2b045c0545" # 您的 Canvasflare API 金鑰
CANVASFLARE_ENDPOINT = f"https://api.canvasflare.com/render?api_key={CANVASFLARE_API_KEY}"
OUTPUT_DIR = r"D:\MA system\LangGraph\output\cache\search_cache" # 使用原始字串以避免反斜線問題
DEFAULT_ZOOM = 15
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_DELAY_MS = 1000 # 增加延遲以確保地圖完全載入

# --- Helper Function to create HTML ---
def create_map_html(lat: float, lon: float, zoom: int) -> str:
    """Generates HTML content for displaying an OpenStreetMap using Leaflet."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>OSM Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ height: {DEFAULT_HEIGHT}px; width: {DEFAULT_WIDTH}px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{lat}, {lon}], {zoom});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);
        L.marker([{lat}, {lon}]).addTo(map); // 添加標記
    </script>
</body>
</html>
"""

# --- Helper Function to create CSS ---
def create_map_css() -> str:
    """Generates basic CSS for the map container."""
    # 基本 CSS 已經包含在 HTML 的 <style> 標籤中，但 API 可能需要此參數
    return f"#map {{ height: {DEFAULT_HEIGHT}px; width: {DEFAULT_WIDTH}px; }}"

# --- Langchain Tool ---
@tool
def get_osm_screenshot(latitude: float, longitude: float, zoom: int = DEFAULT_ZOOM) -> str:
    """
    Generates a screenshot of an OpenStreetMap centered at the given latitude and longitude.

    Args:
        latitude: The latitude of the center point.
        longitude: The longitude of the center point.
        zoom: The zoom level (default: 15).

    Returns:
        The local file path of the downloaded screenshot, or an error message.
    """
    print(f"--- Executing get_osm_screenshot ---")
    print(f"Input - Latitude: {latitude}, Longitude: {longitude}, Zoom: {zoom}")

    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    html_content = create_map_html(latitude, longitude, zoom)
    css_content = create_map_css()

    payload: Dict[str, Any] = {
        "html": html_content,
        "css": css_content,
        "viewport_width": DEFAULT_WIDTH,
        "viewport_height": DEFAULT_HEIGHT,
        "ms_delay": DEFAULT_DELAY_MS, # 等待地圖圖塊加載
        "device_scale": 1 # 可以調整以獲得更高解析度，例如 2
    }
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        print(f"Sending request to Canvasflare API: {CANVASFLARE_ENDPOINT}")
        response = requests.post(CANVASFLARE_ENDPOINT, json=payload, headers=headers, timeout=60) # 增加超時
        response.raise_for_status() # 檢查 HTTP 錯誤狀態碼

        response_data = response.json()
        print(f"Canvasflare API Response: {response_data}")

        if 'imageUrl' not in response_data:
            error_msg = f"Error: 'imageUrl' not found in Canvasflare API response: {response_data}"
            print(error_msg)
            return error_msg

        image_url = response_data['imageUrl']
        print(f"Image URL received: {image_url}")

        # 下載圖片
        image_filename = f"osm_screenshot_{uuid.uuid4()}.png"
        local_image_path = os.path.join(OUTPUT_DIR, image_filename)

        print(f"Downloading image to: {local_image_path}")
        if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
            error_msg = f"Error: Invalid image URL received: {image_url}"
            print(error_msg)
            return error_msg

        img_response = requests.get(image_url, stream=True, timeout=60)
        img_response.raise_for_status()

        with open(local_image_path, 'wb') as f:
            shutil.copyfileobj(img_response.raw, f)

        print(f"Image successfully downloaded: {local_image_path}")
        return local_image_path

    except requests.exceptions.RequestException as e:
        error_msg = f"Error during Canvasflare API request or image download: {e}"
        print(error_msg)
        # 如果有 response，嘗試打印更多錯誤信息
        if e.response is not None:
            try:
                print(f"API Error Response: {e.response.text}")
            except Exception:
                pass # 忽略打印錯誤時的錯誤
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        return error_msg

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # 範例：台北 101
    test_lat = 25.033964
    test_lon = 121.564468
    # 使用 invoke 方法並傳遞字典作為輸入
    result = get_osm_screenshot.invoke({
        "latitude": test_lat,
        "longitude": test_lon,
        "zoom": 17
    })
    print(f"\nTool execution result: {result}")

    # 範例：倫敦眼 (同樣更新)
    # test_lat_london = 51.5033
    # test_lon_london = -0.1195
    # result_london = get_osm_screenshot.invoke({
    #     "latitude": test_lat_london,
    #     "longitude": test_lon_london
    #     # zoom 會使用預設值 15
    # })
    # print(f"\nTool execution result (London): {result_london}")
