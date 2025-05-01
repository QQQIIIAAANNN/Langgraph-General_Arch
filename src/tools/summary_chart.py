import os
import json
import base64

# 定義輸出目錄
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_image_to_base64(image_path):
    """將圖片轉換為 Base64 格式"""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"


def load_json_content(json_path):
    """讀取 JSON 文件並返回其內容"""
    if not os.path.exists(json_path):
        return f"Error: JSON file {json_path} does not exist."
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return f"Error: Unable to parse JSON file {json_path}."


def summary_chart(final_summary_json: str, output_html_path: str) -> str:
    """
    從 final_summary.json 數據生成流程圖，並動態嵌入每個 JSON 文件的詳細內容，導出為 HTML 文件。

    :param final_summary_json: 包含 final_summary 數據的 JSON 文件路徑
    :param output_html_path: 將流程圖輸出為 HTML 文件的路徑
    :return: 成功生成流程圖的 HTML 文件路徑
    """
    # 確認輸入文件是否存在
    if not os.path.exists(final_summary_json):
        return f"Error: Final summary JSON file {final_summary_json} does not exist."

    # 加載數據
    try:
        with open(final_summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    except json.JSONDecodeError:
        return f"Error: Unable to parse JSON file: {final_summary_json}"

    if not summary_data:
        return "Error: No valid data found in the JSON file."

    # 找出最高平均分數的方案
    best_summary = max(summary_data, key=lambda x: x.get("Average_score", 0))
    best_round = best_summary["current_round"]
    best_shell = best_summary["selected_shells"]
    best_score = best_summary["Average_score"]

    # 創建節點內容
    flow_chart_nodes = []  # 儲存所有方案的 HTML 節點
    for data in summary_data:
        is_best = data["current_round"] == best_round  # 是否為推薦方案

        # 將圖片轉換為 Base64 格式
        base_map_base64 = encode_image_to_base64(data["base_map"])
        shell_result_img_base64 = encode_image_to_base64(data["shell_result_img"])
        shell_future_img_base64 = encode_image_to_base64(data["shell_future_img"])

        # 動態讀取詳細 JSON 文件的內容
        analysis_details = "N/A"
        shell_result_details = "N/A"
        feedback_details = "N/A"

        # 讀取分析數據 JSON 文件
        if data.get("analysis_data") and os.path.exists(data["analysis_data"]):
            analysis_content = load_json_content(data["analysis_data"])
            if isinstance(analysis_content, dict):
                analysis_details = json.dumps(analysis_content, indent=4, ensure_ascii=False)

        # 讀取外殼結果 JSON 文件
        if data.get("shell_result") and os.path.exists(data["shell_result"]):
            shell_result_content = load_json_content(data["shell_result"])
            if isinstance(shell_result_content, dict):
                shell_result_details = json.dumps(shell_result_content, indent=4, ensure_ascii=False)

        # 讀取反饋結果 JSON 文件
        if data.get("feedback_result") and os.path.exists(data["feedback_result"]):
            feedback_content = load_json_content(data["feedback_result"])
            if isinstance(feedback_content, dict):
                feedback_details = json.dumps(feedback_content, indent=4, ensure_ascii=False)

        # 動態生成節點，嵌入 JSON 的內容
        node = f"""
        <div style="border: {'2px solid green' if is_best else '2px solid #ccc'};
                    padding: 20px; margin: 10px; border-radius: 5px; 
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
                    background-color: {'#e6ffe6' if is_best else '#f9f9f9'};">
            <h4 style="color: {'green' if is_best else 'black'};">
                Round {data['current_round']} - {data['selected_shells']}
                {'(Recommended)' if is_best else ''}
            </h4>
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                {f'<a href="{data["base_map"]}" target="_blank"><img src="{base_map_base64}" style="height: 150px; margin: 5px;"></a>' if base_map_base64 else ''}
                {f'<a href="{data["shell_result_img"]}" target="_blank"><img src="{shell_result_img_base64}" style="height: 150px; margin: 5px;"></a>' if shell_result_img_base64 else ''}
                {f'<a href="{data["shell_future_img"]}" target="_blank"><img src="{shell_future_img_base64}" style="height: 150px; margin: 5px;"></a>' if shell_future_img_base64 else ''}
            </div>
            <button onclick="toggleDetails('analysis-{data['current_round']}')" 
                    style="background-color: #007bff; color: white; border: none; padding: 10px; cursor: pointer;">
                Show Analysis Data
            </button>
            <div id="analysis-{data['current_round']}" style="display: none; margin-top: 10px; max-height: 200px; overflow-y: scroll; background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc;">
                <pre>{analysis_details}</pre>
            </div>
            <button onclick="toggleDetails('shell-result-{data['current_round']}')" 
                    style="background-color: #007bff; color: white; border: none; padding: 10px; cursor: pointer;">
                Show Shell Result
            </button>
            <div id="shell-result-{data['current_round']}" style="display: none; margin-top: 10px; max-height: 200px; overflow-y: scroll; background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc;">
                <pre>{shell_result_details}</pre>
            </div>
            <button onclick="toggleDetails('feedback-{data['current_round']}')" 
                    style="background-color: #007bff; color: white; border: none; padding: 10px; cursor: pointer;">
                Show Feedback Details
            </button>
            <div id="feedback-{data['current_round']}" style="display: none; margin-top: 10px; max-height: 200px; overflow-y: scroll; background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc;">
                <pre>{feedback_details}</pre>
            </div>
        </div>
        """
        flow_chart_nodes.append(node)

    # 添加推薦節點
    recommendation_node = f"""
    <div style="border: 2px solid #000; padding: 20px; margin: 10px; 
                border-radius: 5px; background-color: #e6ffe6;">
        <h3>Final Recommendation</h3>
        <p style="font-weight: bold; color: green;">
            The best shell design is: Round {best_round} - {best_shell} 
            with an Average Score of {best_score:.2f}
        </p>
    </div>
    """

    # 構建 HTML 頁面
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Final Summary Chart</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
        </style>
        <script>
            function toggleDetails(id) {{
                const element = document.getElementById(id);
                if (element.style.display === "none") {{
                    element.style.display = "block";
                }} else {{
                    element.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Final Summary Flow Chart</h1>
            {''.join(flow_chart_nodes)}  <!-- 確保每個節點都正確拼接 -->
            {recommendation_node}
        </div>
    </body>
    </html>
    """

    # 將內容寫入文件
    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        return f"Error: Unable to save HTML file. {str(e)}"

    return f"Flow chart successfully generated: {output_html_path}"
