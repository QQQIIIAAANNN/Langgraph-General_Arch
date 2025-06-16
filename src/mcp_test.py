# src/mcp_test.py
import os
import asyncio
import json
import traceback
import sys
import uuid
import base64
import time
import re
from collections import defaultdict

# --- Add PIL for image loading (Keep for initial message construction if needed elsewhere) ---
try:
    from PIL import Image as PILImage
except ImportError:
    print("錯誤：需要 Pillow 庫來處理圖像。請執行 pip install Pillow")
    PILImage = None # Indicate PIL is not available

if sys.platform.startswith("win"):
    # 強制使用 ProactorEventLoop，以支援 subprocess
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List, Union # Added Union
from contextlib import asynccontextmanager
import requests
import atexit
import platform

# --- 移除日誌記錄 ---
print("日誌記錄已移除，將使用 print 輸出。")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
# from mcp.server.fastmcp import Image as MCPImage # 不再直接需要 MCPImage

load_dotenv()

# --- LLM Setup (修改：移除 convert_system_message_to_human) ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("錯誤：找不到 GEMINI_API_KEY 環境變數。")
        exit(1)
    else:
        agent_llm = ChatGoogleGenerativeAI(  #gemini-2.5-pro-exp-03-25  "gemini-2.5-pro-preview-03-25"
            model="gemini-2.5-flash-preview-04-17", #gemini-2.0-flash "gemini-2.5-flash-preview-04-17" 
            temperature=0.1,
            google_api_key=api_key
        )
        print(f"Agent LLM ({agent_llm.model}) 初始化成功 (convert_system_message_to_human=False/Default)。")
    utility_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    print("Utility LLM (OpenAI for Router) 初始化成功。")
except Exception as e:
    print(f"ERROR: 無法初始化 LLM。錯誤: {e}")
    traceback.print_exc()
    exit(1)

# --- MCP Server Configurations (修改 OSM 設定) ---
MCP_CONFIGS = {
    "rhino": {
        "command": "Z:\\miniconda3\\envs\\rhino_mcp\\python.exe",
        "args": ["-m","rhino_mcp.server"],
        "transport": "stdio",
    },
    "revit": {
        "command": "node",
        "args": ["D:\\MA system\\LangGraph\\src\\mcp\\revit-mcp\\build\\index.js"],
        "transport": "stdio",
    },
    "pinterest": {
        "command": "node", # Assuming node runs the JS file
        "args": ["D:\\MA system\\LangGraph\\src\\mcp\\pinterest-mcp-server\\dist\\pinterest-mcp-server.js"],
        "transport": "stdio", # Assuming stdio, adjust if needed
    },
    # --- 修改 OSM MCP 配置，使用 osm-mcp-server 命令 ---
    "osm": {
      "command": "osm-mcp-server", # 使用 osm-mcp-server 命令
      "args": [],  # 不需要額外參數
      "transport": "stdio",
    }
}

# --- 全局變數 (移除 _mcp_clients_initialized) ---
_loaded_mcp_tools: Dict[str, List[BaseTool]] = {}
_mcp_clients: Dict[str, MultiServerMCPClient] = {}
_mcp_init_lock = asyncio.Lock()

# =============================================================================
# 定義狀態 (State)
# =============================================================================
class MCPAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    initial_request: str
    initial_image_path: Optional[str]
    target_mcp: Optional[str]
    task_complete: bool = False
    # --- 用於存儲截圖/下載結果的字段 ---
    saved_image_path: Optional[str] # Stores the path returned by Rhino/Pinterest/OSM
    saved_image_data_uri: Optional[str] # Stores the generated data URI
    # --- <<< 新增：連續文本響應計數器 >>> ---
    consecutive_llm_text_responses: int = 0 # Track consecutive non-tool/non-completion AI messages
    # --- MODIFIED: Add screenshot counter for Rhino ---
    rhino_screenshot_counter: int = 0 
    # --- END MODIFICATION ---
    last_executed_node: Optional[str] = None # 記錄最後執行的節點名稱
    # --- 新增: 存儲CSV報告路徑 ---
    saved_csv_path: Optional[str] = None

# =============================================================================
# 本地工具定義 (Local Tools)
# =============================================================================
@tool
def create_planned_data_summary_csv(data_rows: List[Dict[str, Union[str, float]]], total_area: float, bcr: Optional[float], far: Optional[float], filename: str = "planned_project_summary.csv") -> str:
    """
    根據「規劃好」的設計數據生成CSV摘要文件。
    此工具不與Rhino互動；它只記錄計畫中提供的數據。
    在規劃階段結束時使用此工具，以創建設計意圖的摘要。

    Args:
        data_rows: 一個字典列表，每個字典代表一個空間。必須包含 'name' (str), 'area' (float), 'percentage' (float) 和 'floor' (str, 例如 "Floor 1") 鍵。
        total_area: 規劃的總樓地板面積 (float)。
        bcr: 規劃的建蔽率 (float, 百分比)。如果無則為空。
        far: 規劃的容積率 (float)。如果無則為空。
        filename: 輸出的CSV文件名。預設為 "planned_project_summary.csv"。

    Returns:
        一個確認成功和保存文件路徑的字串，以 [CSV_FILE_PATH]: 為前綴。
    """
    import csv
    import time
    from collections import defaultdict
    output_dir = r"D:\MA system\LangGraph\output\space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base, ext = os.path.splitext(filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}_{timestamp}{ext}"
    file_path = os.path.join(output_dir, new_filename)
    headers = ["Space Name", "Area (sqm)", "Percentage (%)"]

    spaces_by_floor = defaultdict(list)
    for row in data_rows:
        floor = row.get('floor', 'Unassigned')
        spaces_by_floor[floor].append(row)
    
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            
            writer.writerow(["Project Summary (Based on Plan)"])
            writer.writerow(["Total Planned Floor Area (sqm)", round(total_area, 2)])
            writer.writerow(["Planned Building Coverage Ratio (%)", bcr if bcr is not None else "N/A"])
            writer.writerow(["Planned Floor Area Ratio", far if far is not None else "N/A"])
            writer.writerow([])
            
            writer.writerow(["Planned Space Details"])

            sorted_floors = sorted(spaces_by_floor.keys())
            for floor in sorted_floors:
                writer.writerow([])
                writer.writerow([f"--- {floor} ---"])
                writer.writerow(headers)
                for row in spaces_by_floor[floor]:
                    writer.writerow([
                        row.get('name', 'N/A'), 
                        round(row.get('area', 0.0), 2),
                        round(row.get('percentage', 0.0), 2)
                    ])
                
        return f"[CSV_FILE_PATH]:{file_path}"
    except Exception as e:
        return f"[ERROR] Failed to create planned summary table: {str(e)}"

# --- 新增: 本地工具列表 ---
LOCAL_TOOLS = [create_planned_data_summary_csv]

# =============================================================================
# 工具管理 (使用 print 替換 logging)
# =============================================================================
async def initialize_single_mcp(mcp_name: str) -> tuple[Optional[MultiServerMCPClient], List[BaseTool]]:
    """初始化單個 MCP 連接並獲取其工具 (使用 print)。"""
    print(f"--- [Lazy Init] 正在初始化 {mcp_name} MCP 連接 ---")
    config_item = MCP_CONFIGS.get(mcp_name)
    if not config_item:
        print(f"  !!! [Lazy Init] 錯誤: 在 MCP_CONFIGS 中找不到 {mcp_name} 的配置。")
        return None, []

    client = None
    tools = []
    try:
        # --- 命令和路徑檢查 (使用 print) ---
        command_path = config_item['command']
        # 檢查命令是否存在 (對 'python' 這類通用命令可能不適用)
        if command_path != "python" and not os.path.exists(command_path) and command_path != sys.executable:
            print(f"  !!! [Lazy Init] 警告: 命令路徑 '{command_path}' 不存在。")
        # 檢查 args 中的文件路徑 (如果有的話)
        for arg in config_item.get('args', []):
             # Check if it looks like a file path and doesn't exist
             if ('/' in arg or '\\' in arg) and not os.path.exists(arg):
                  print(f"  !!! [Lazy Init] 警告: 參數中的路徑 '{arg}' 不存在。")

        # print(f"  - [Lazy Init] 使用配置: {config_item}")
        print(f"  - [Lazy Init] 正在初始化 {mcp_name} Client...")
        try:
            single_server_config = {mcp_name: config_item}
            client = MultiServerMCPClient(single_server_config)
            print(f"  - [Lazy Init] {mcp_name} Client 初始化完成。")
        except Exception as client_init_e:
            print(f"  !!! [Lazy Init] 客戶端初始化錯誤: {client_init_e}")
            traceback.print_exc()
            return None, []

        # --- 連接和獲取工具 (使用 print) ---
        try:
            print(f"  - [Lazy Init] 正在啟動 {mcp_name} Client 連接 (__aenter__)...")
            await client.__aenter__()
            print(f"  - [Lazy Init] {mcp_name} Client 連接成功。")

            print(f"  - [Lazy Init] [開始] 正在從 {mcp_name} 獲取工具 (get_tools)...")
            try:
                tools = client.get_tools()
                print(f"  - [Lazy Init] [完成] 從 {mcp_name} 獲取工具完成。數量: {len(tools)}")
                if not tools:
                    print(f"  !!! [Lazy Init] 警告: {mcp_name} 返回了空的工具列表 !!!")
                else:
                    # --- 打印工具信息 (可選，保持開啟以供調試) ---
                    print(f"  --- 可用工具列表 ({mcp_name}) ---")
                    for i, tool in enumerate(tools):
                        tool_info = f"    工具 {i+1}: Name='{tool.name}'"
                        if hasattr(tool, 'description') and tool.description:
                             tool_info += f", Desc='{tool.description[:60]}...'"
                        print(tool_info)
                    print(f"  --- 工具列表結束 ({mcp_name}) ---")
            except Exception as tools_e:
                print(f"  !!! [Lazy Init] 獲取工具錯誤: {tools_e}")
                traceback.print_exc()
                tools = []
        except Exception as enter_e:
            print(f"  !!! [Lazy Init] 客戶端連接或獲取工具錯誤: {enter_e}")
            traceback.print_exc()
            if client:
                try:
                    print(f"  -- [Cleanup Attempt] 嘗試關閉失敗的 {mcp_name} client...")
                    await client.__aexit__(type(enter_e), enter_e, enter_e.__traceback__)
                    print(f"  -- [Cleanup Attempt] 關閉 {mcp_name} client 完成。")
                except Exception as exit_e:
                    print(f"  -- [Cleanup Attempt] 關閉 {mcp_name} client 時也發生錯誤: {exit_e}")
                    traceback.print_exc()
            client = None
            tools = []
        print(f"--- [Lazy Init] {mcp_name.capitalize()} 初始化流程完成 ---")
    except Exception as inner_e:
        print(f"!!!!! [Lazy Init] 錯誤: 在處理 {mcp_name} MCP 時發生外部錯誤 !!!!!")
        traceback.print_exc()
        client = None
        tools = []
    return client, tools

# --- shutdown_mcp_clients (使用 print) ---
async def shutdown_mcp_clients(clients_to_shutdown: Dict[str, MultiServerMCPClient]):
    print("\n--- [Cleanup] 正在關閉 MCP Client 連接 ---")
    if not clients_to_shutdown:
        print("  沒有需要關閉的客戶端。")
        return
    for name, client in clients_to_shutdown.items():
        try:
            print(f"  - 正在關閉 {name} Client (__aexit__)...")
            await client.__aexit__(None, None, None)
            print(f"  - {name} Client 已關閉")
        except Exception as close_e:
            print(f"錯誤: 關閉 {name} Client 時發生錯誤: {close_e}")
            traceback.print_exc()
    print("--- [Cleanup] 所有 MCP Client 已嘗試關閉 ---")

# --- _sync_cleanup (使用 print, 移除 _mcp_clients_initialized 檢查) ---
def _sync_cleanup():
    global _mcp_clients
    # 只檢查 _mcp_clients 是否有內容
    if _mcp_clients:
        print("--- [atexit] 檢測到需要清理 MCP 客戶端 ---")
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                loop.create_task(shutdown_mcp_clients(_mcp_clients))
                print("--- [atexit] 清理任務已創建 (循環運行中) ---")
            else:
                loop.run_until_complete(shutdown_mcp_clients(_mcp_clients))
                print("--- [atexit] 清理任務已同步執行 ---")
        except Exception as cleanup_err:
            print(f"--- [atexit] 執行異步清理時出錯: {cleanup_err} ---")
            traceback.print_exc()
        finally:
            _mcp_clients = {}
    else:
         print("--- [atexit] 無需清理 MCP 客戶端 ---")

atexit.register(_sync_cleanup)

# --- get_mcp_tools (使用 print) ---
async def get_mcp_tools(mcp_name: str) -> List[BaseTool]:
    global _loaded_mcp_tools, _mcp_clients
    if mcp_name in _loaded_mcp_tools:
        # print(f"--- [Lazy Load] 使用已緩存的 {mcp_name} MCP 工具 ---")
        return _loaded_mcp_tools[mcp_name]

    async with _mcp_init_lock:
        if mcp_name in _loaded_mcp_tools:
            # print(f"--- [Lazy Load] 使用已緩存的 {mcp_name} MCP 工具 (after lock) ---")
            return _loaded_mcp_tools[mcp_name]

        print(f"--- [Lazy Load] 觸發 {mcp_name} MCP 工具初始化 ---")
        client, tools = await initialize_single_mcp(mcp_name)

        _loaded_mcp_tools[mcp_name] = tools
        if client:
             _mcp_clients[mcp_name] = client

        print(f"--- [Lazy Load] {mcp_name} MCP 工具初始化完成並緩存 (找到 {len(tools)} 個工具) ---")
        return tools

# =============================================================================
# 提示詞定義 (修改 AGENT_EXECUTION_PROMPT 加入最終截圖指令)
# =============================================================================
# --- 通用 Rhino/Revit 執行提示 ---
RHINO_AGENT_EXECUTION_PROMPT = SystemMessage(content="""你是一個嚴格按計劃執行任務的助手，專門為 CAD/BIM 環境生成指令。消息歷史中包含了用戶請求和一個分階段目標的計劃。
**你的核心任務：根據計劃，執行且僅執行下一個未完成的步驟。嚴禁重複已完成的步驟。**

**定位下一步驟的演算法 (必須嚴格遵守):**
1.  **檢查歷史紀錄:** 查看最近的幾條消息。你的主要依據是最後一條 `ToolMessage`。
2.  **匹配上次動作:** 將 `ToolMessage` 的結果與 `[目標階段計劃]:` 中的步驟進行比對，找出它對應的是計劃中的第幾個步驟。
3.  **確定下一步:** 緊接在上一步之後的那個步驟，就是你現在需要執行的**唯一目標**。例如，如果上一步是計劃的第 1 步，你現在就必須執行第 2 步。
4.  **初始情況:** 如果歷史紀錄中沒有 `ToolMessage` (代表這是計劃生成後的第一次執行)，則從計劃的第 1 步開始。
5.  **錯誤處理:** 如果 `ToolMessage` 指出上一步驟執行失敗，你的任務是分析錯誤原因，並嘗試**修正並重新執行同一個步驟**。
                                             
**執行規則:**                                                                       
1.  **要調用工具來執行動作，請必須生成 `tool_calls` 在首位的 AIMessage 以請求該工具調用**。**不要僅用文字描述你要調用哪個工具，而是實際生成工具調用指令。** 一次只生成一個工具調用請求。
2.  嚴格禁止使用 f-string 格式化字串。請使用 `.format()` 或 `%` 進行字串插值。(此為 IronPython 2.7 環境限制)
3.  **仔細參考工具描述或 Mcp 文檔確認函數用法與參數正確性，必須實際生成結構化的工具呼叫指令。**
4.  **多方案管理 (重要):**
    * 當生成多個方案時，**每個方案必須完全獨立**，視為單獨的任務序列處理
    * **方案隔離原則:**
        * **每個方案必須有自己的頂層圖層**，使用 `rs.AddLayer("方案A_描述")` 創建
        * **切換方案前必須隱藏前一方案的圖層**，使用 `rs.LayerVisible("前一方案名", False)`
        * **所有物件必須正確配置到其所屬方案的圖層**，使用 `rs.CurrentLayer("方案X_描述::子圖層")`
        * **完成每個方案後必須截圖**，再開始下一個方案
    * **避免方案間的量體重疊**，可考慮在不同方案間使用座標偏移
5.  **量體生成策略:**
    * **空間操作優先使用布林運算**：使用 `rs.BooleanUnion()`、`rs.BooleanDifference()`、`rs.BooleanIntersection()` 創造複雜形態
    * **善用幾何變換**：使用旋轉、縮放、移動等操作調整物件姿態，創造更豐富的空間層次
    * **避免無效量體**：不要創建過小、位置不合理或對空間表達無貢獻的量體
    * **注意 IronPython 2.7 語法限制**：Rhino 8使用IronPython 2.7，禁止使用Python 3特有語法   
6.  **曲面造型策略:**
        *   **曲面創建類別：**
            *   **掃掠 (Sweep):**
                *   `rs.AddSweep1(rail_curve_id, shape_curve_ids)`: 將剖面曲線列表 `shape_curve_ids` 沿單一軌道 `rail_curve_id` 掃掠成曲面。注意剖面曲線的方向和順序。
                *   `rs.AddSweep2(rail_curve_ids, shape_curve_ids)`: 將剖面曲線列表 `shape_curve_ids` 沿兩個軌道列表 `rail_curve_ids` 掃掠成曲面。注意剖面曲線的方向、順序及與軌道的接觸。
            *   **放樣 (Loft):**
                *   `rs.AddLoftSrf(curve_ids, start_pt=None, end_pt=None, type=0, style=0, simplify=0, closed=False)`: 在有序的曲線列表 `curve_ids` 之間創建放樣曲面。注意曲線方向和接縫點。可指定類型、樣式等。
            *   **網格曲面 (Network Surface):**
                *   `rs.AddNetworkSrf(curve_ids)`: 從一組相交的曲線網絡 `curve_ids` 創建曲面。所有 U 方向曲線必須與所有 V 方向曲線相交。
            *   **平面曲面 (Planar Surface):**
                *   `rs.AddPlanarSrf(curve_ids)`: 從一個或多個封閉的*平面*曲線列表 `curve_ids` 創建平面曲面。曲線必須共面且封閉。
        *   **實體創建類別：**
            *   **擠出 (Extrusion):**
                *   `rs.ExtrudeCurve(curve_id, path_curve_id)`: 將輪廓線 `curve_id` 沿路徑曲線 `path_curve_id` 擠出成曲面。
                *   `rs.ExtrudeCurveStraight(curve_id, start_point, end_point)` 或 `rs.ExtrudeCurveStraight(curve_id, direction_vector)`: 將曲線 `curve_id` 沿直線擠出指定距離和方向。
                *   `rs.ExtrudeCurveTapered(curve_id, distance, direction, base_point, angle)`: 將曲線 `curve_id` 沿 `direction` 方向擠出 `distance` 距離，同時以 `base_point` 為基準、按 `angle` 角度進行錐化。
                *   `rs.ExtrudeSurface(surface_id, path_curve_id, cap=True/False)`: 將曲面 `surface_id` 沿路徑曲線 `path_curve_id` 擠出成實體或開放形狀，可選是否封口 (`cap`)。
7.  **Rhino 圖層管理 (重要):** 當生成 Rhino 代碼時：
        *   如果當前階段目標**明確要求**在特定圖層上操作，**必須**在相關操作（如創建物件）**之前**包含 `rs.CurrentLayer('目標圖層名稱')` 指令。
        *   如果目標涉及控制圖層可見性（例如，準備截圖），**必須**包含 `rs.LayerVisible('圖層名', True/False)` 指令。
    *   **截圖前的圖層準備：在調用 `capture_focused_view` 進行截圖之前，必須確保只有與當前截圖目標直接相關的圖層是可見的。所有其他不相關的圖層，特別是那些可能遮擋目標視圖的圖層（例如，其他樓層、其他設計方案的頂層圖層、輔助線圖層等），都應使用 `rs.LayerVisible('圖層名', False)` 進行隱藏。 使用透視/兩點透視截圖時須確保相關圖層都有開啟**
8. **最終步驟 (Rhino/Revit):**
    *   對於 Rhino/Revit 任務，每當完成一個方案或一個樓層就**必須**要調用 `capture_focused_view` 工具來截取畫面。截圖時如果設定相機位置，確保(`target_position`)位於方案的中心點。
    *   **僅當消息歷史清楚地表明計劃中的最後階段目標已成功執行**，你才能生成文本回復：`全部任務已完成` 以結束整個任務。
9. 如果當前階段目標不需要工具即可完成（例如，僅需總結信息），請生成說明性的自然語言回應。
10.若遇工具錯誤，分析錯誤原因 (尤其是代碼執行錯誤)，**嘗試修正你的工具調用參數或生成的代碼**，然後再次請求工具調用。如果無法修正，請報告問題。
11.規劃數據摘要報告 (空間規劃任務的必要首步):僅當**任務是關於**空間佈局規劃** (例如，量體配置等)，你**必須在第一個步驟**執行生成摘要報告。
                                             
**常規執行：對於計劃中的任何步驟，不要用自然語言解釋你要做什麼，直接生成包含 Tool Calls 結構的工具調用。**
**關鍵指令：不要用自然語言解釋你要做什麼，直接根據你用上述演算法定位到的下一步驟，生成包含 Tool Calls 結構的工具調用。**
**絕對指令：不要延續[目標階段計劃]生成 "任務完成" 或將任務完成當作一個步驟。當前一個訊息是[目標階段計劃]時直接進行工具調用，不要包含描述性文本！**
                                             
**可用工具清單:**
你能夠使用以下工具來完成計劃中的步驟。你必須使用這些工具，並嚴格按照其參數要求來生成工具調用。
{tool_descriptions}""")

# --- Pinterest 執行提示 ---
PINTEREST_AGENT_EXECUTION_PROMPT = SystemMessage(content="""你是一個 Pinterest 圖片搜索助手。
你的任務是：
1.  分析用戶請求和計劃（如果有的話）。
2.  如果計劃指示調用 `pinterest_search_and_download` 工具，請立即生成該工具調用。
3.  工具參數應包含從用戶請求中提取的 `keyword` (搜索關鍵詞) 和可選的 `limit` (下載數量)。
4.  **最終步驟：** 在工具成功執行並返回圖片路徑後，你的最終回應應該是：「圖片搜索和下載完成」。以結束任務。
請直接生成工具調用或最終完成訊息。""")

# --- OSM 執行提示 ---
OSM_AGENT_EXECUTION_PROMPT = SystemMessage(content="""你是一個 OpenStreetMap 地圖助手。
你的任務是：
1.  分析用戶請求和計劃（如果有的話）。
2.  如果計劃指示調用 `geocode_and_screenshot` 工具，請立即生成該工具調用。
3.  **地址/座標處理 (geocode_and_screenshot):**
        *   **檢查使用者輸入**：查看初始請求或當前目標是否包含明確的**經緯度座標**（例如 "2X.XXX 1XX.XXX" 或類似格式）。
        *   **如果找到座標**：直接將**座標字串 "緯度,經度"** (例如 "2X.XXX 1XX.XXX") 作為 `address` 參數的值傳遞給 `geocode_and_screenshot` 工具。**不要**嘗試將座標轉換成地址。
        *   **如果只找到地址**：請**嘗試將其簡化**，例如 "號碼, 街道名稱, 城市, 國家" 再傳遞給 `address` 參數。如果持續地理編碼失敗，可以嘗試進一步簡化。
4.  **最終步驟：** 在工具成功執行並返回截圖路徑後，你的最終回應應該是：「地圖截圖已完成」。
請直接生成工具調用或最終完成訊息。""")


# --- Router Prompt (MODIFIED) ---
ROUTER_PROMPT = """你是一個智能路由代理。根據使用者的**初始請求文本**，判斷應將任務分配給哪個專業領域的代理。
目前可用的代理有：
- 'revit': 主要處理與 Revit 建築資訊模型相關的請求。
- 'rhino': 主要處理與 Rhino 3D 模型相關的請求。
- 'pinterest': 主要處理與 Pinterest 圖片搜索和下載相關的請求。
- 'osm': 主要處理與 OpenStreetMap 地圖相關的請求。

分析以下**初始使用者請求文本**，並決定最適合處理此請求的代理。生成模型的任務以rhino為主，除非特別指定用revit。
你的回應必須是 'revit', 'rhino', 'pinterest' 或 'osm'。請只回應目標代理的名稱。

初始使用者請求文本：
"{user_request_text}"
"""

PLAN_PREFIX = "[目標階段計劃]:\n"

# --- Fallback Agent Prompt ---
FALLBACK_PROMPT = SystemMessage(content="""你是一個補救與驗證助手。主要助手可能已完成其步驟、卡住了，或聲稱任務已完成。
    你的任務是：
    1.  仔細分析消息歷史，特別是 `[目標階段計劃]:` 和最近幾條主要助手的回應。
    2.  **分析主要助手狀態**：
        *   如果主要助手的最後一條回應**不是工具調用**，而是描述性文本（例如 "正在執行階段 X..." 或類似的對話），這通常表示主要助手**卡住了**或者未能按預期生成工具調用。
    3.  **驗證完成狀態 (如果主要助手聲稱完成或歷史表明可能已到最後階段)**：
        *   查看 `[目標階段計劃]:`，識別出計劃中的**最後一個階段目標**。檢查最近的消息歷史，請獨立判斷這個**最後的階段目標是否已經成功執行完畢**。
    4.  **確定下一步**：
        *   如果根據上述驗證，計劃中的**最後一個階段目標確實已成功執行**，請**只輸出**文本消息：`[FALLBACK_CONFIRMED_COMPLETION]`。
        *   如果主要助手**卡住了**（如第 2 點所述），或者任務**未完成** (例如，最後的計劃步驟未完成，或者還有更早的計劃步驟未完成且你可以識別出來)，並且你可以根據計劃和歷史確定下一個**應該執行的階段目標**，請**生成執行該目標所需的 `tool_calls`**。直接輸出包含工具調用的 AIMessage。**優先嘗試從計劃中找到下一個應該執行的步驟並為其生成工具調用。**
        *   如果任務**未完成**，且你無法根據現有信息確定下一步、無法恢復流程（例如，無法識別計劃的最後一步，或無法判斷其是否完成，或無法為卡住的助手找到解決方案），請**只輸出**文本消息：`[FALLBACK_CANNOT_RECOVER]`。

   **關鍵：不要重複主要助手剛剛完成的步驟。專注於未完成的目標或驗證最終狀態。如果主要助手明顯卡在某個描述性文本而未生成工具調用，你的首要任務是根據計劃推斷並生成正確的工具調用。**
   
   消息歷史:
   {relevant_history}
   """)

# =============================================================================
# 輔助函數：執行工具 (修改以處理 Pinterest 下載路徑)
# =============================================================================
async def execute_tools(agent_action: AIMessage, selected_tools: List[BaseTool]) -> List[ToolMessage]:
    """執行 AI Message 中的工具調用，處理 capture_focused_view 和 pinterest download 返回，並確保 ToolMessage content 非空字串。"""
    tool_messages = []
    if not agent_action.tool_calls:
        return tool_messages
    name_to_tool_map = {tool.name: tool for tool in selected_tools}
    print(f"    準備執行 {len(agent_action.tool_calls)} 個工具調用...")
    for tool_call in agent_action.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"      >> 調用工具: {tool_name} (ID: {tool_call_id})")

        tool_to_use = name_to_tool_map.get(tool_name)
        if not tool_to_use:
            error_msg = f"錯誤：找不到名為 '{tool_name}' 的工具。"
            print(f"      !! {error_msg}")
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name))
            continue

        observation_str = f"[未成功執行工具 {tool_name}]"
        final_content = "UNEXPECTED_TOOL_EXECUTION_FAILURE"
        observation = None

        try:
            # --- 參數處理 (保持不變) ---
            if not isinstance(tool_args, dict):
                 try:
                     tool_args_dict = json.loads(str(tool_args)) if isinstance(tool_args, str) and str(tool_args).strip().startswith('{') else {"input": tool_args}
                 except json.JSONDecodeError:
                     tool_args_dict = {"input": tool_args}
            else:
                 tool_args_dict = tool_args

            # --- 調用工具 (ainvoke) ---
            print(f"        調用 {tool_name}.ainvoke...")
            observation = await tool_to_use.ainvoke(tool_args_dict, config=None)
            print(f"        {tool_name}.ainvoke 調用完成。觀察值類型: {type(observation).__name__}")

            # --- 轉換 observation 為字串 ---

            # --- 處理 capture_viewport 返回 (保持不變) ---
            if tool_name == "capture_focused_view" and isinstance(observation, str):
                if observation.startswith("[Error]"):
                    final_content = f"[Error: Viewport Capture Failed]: {observation}"
                    print(f"      !! 工具 '{tool_name}' 返回錯誤信息: {observation}")
                else:
                    final_content = f"[IMAGE_FILE_PATH]:{observation}"
                    print(f"      << 工具 '{tool_name}' 返回文件路徑字符串: {observation}")

            # --- 處理 pinterest_search_and_download 返回 (MODIFIED to return JSON list of paths) ---
            elif tool_name == "pinterest_search_and_download" and isinstance(observation, list):
                 print(f"      << 工具 '{tool_name}' 返回列表。正在解析下載路徑...")
                 try:
                     print(f"         DEBUG: Raw observation list received:\n{json.dumps(observation, indent=2, ensure_ascii=False)}")
                 except Exception as json_e:
                     print(f"         DEBUG: Could not JSON dump observation: {json_e}")
                     print(f"         DEBUG: Raw observation list (repr): {repr(observation)}")

                 download_paths = []
                 full_text_output = []
                 expected_prefix = "保存位置: " # 使用簡體中文前綴

                 for text_item in observation:
                     if isinstance(text_item, str):
                         full_text_output.append(text_item)
                         if text_item.startswith(expected_prefix):
                             path = text_item.split(expected_prefix, 1)[1].strip()
                             if path and os.path.exists(path): # <<< ADDED: Check if path exists >>>
                                 print(f"         提取到有效路徑: {path}")
                                 download_paths.append(path)
                             elif path:
                                 print(f"         警告: 從 '{text_item}' 提取的路徑不存在: {path}")
                             else:
                                 print(f"         警告: 從 '{text_item}' 提取的路徑為空。")
                     else:
                         print(f"         警告: 觀察列表中的項目不是預期的字串: {type(text_item)} - {repr(text_item)}")
                         full_text_output.append(str(text_item))

                 print(f"         找到 {len(download_paths)} 個有效下載路徑。")
                 if download_paths:
                     # <<< MODIFIED: Return JSON list of paths >>>
                     try:
                         final_content = json.dumps({"downloaded_paths": download_paths})
                         print(f"         返回 JSON 列表: {final_content}")
                     except Exception as json_e:
                         print(f"         !! JSON 序列化下載路徑列表時出錯: {json_e}")
                         final_content = "[Error serializing download paths]"
                     # <<< END MODIFIED >>>
                 else:
                     failure_mentioned = any("失败" in t or "failed" in t.lower() for t in full_text_output)
                     if failure_mentioned:
                         final_content = "\n".join(full_text_output) if full_text_output else "Pinterest tool ran with download errors, no valid paths reported."
                     else:
                         final_content = "\n".join(full_text_output) if full_text_output else "Pinterest tool ran but no valid download paths found."
                     print(f"         未找到有效下載路徑，返回文本輸出: {final_content[:100]}...")

            # --- 處理 bytes (保持不變) ---
            elif isinstance(observation, bytes):
                try:
                    observation_str = observation.decode('utf-8', errors='replace')
                    print(f"      << 工具 '{tool_name}' 返回 bytes，已解碼。")
                except Exception as decode_err:
                    observation_str = f"[Error Decoding Bytes: {decode_err}]"
                    print(f"      !! 工具 '{tool_name}' 返回 bytes，解碼失敗: {decode_err}")
                final_content = observation_str if observation_str else "DECODED_EMPTY_STRING"

            # --- 處理 dict/list (排除 capture_viewport 和 pinterest) ---
            elif isinstance(observation, (dict, list)):
                if isinstance(observation, list) and not observation:
                     error_msg = f"工具 '{tool_name}' 的 ainvoke 返回了空列表 `[]`。這可能表示 langchain-mcp-adapters 在處理工具響應時內部出錯，或者工具本身未按預期返回 (檢查工具實現)。"
                     print(f"      !! {error_msg}")
                     final_content = "ADAPTER_RETURNED_EMPTY_LIST"
                else:
                    try:
                        observation_str = json.dumps(observation, ensure_ascii=False, indent=2)
                        print(f"      << 工具 '{tool_name}' 返回普通 dict/list，已序列化為 JSON 字串。")
                    except TypeError as json_err:
                        observation_str = f"[Error JSON Serializing Result: {json_err}] 回退到 str(): {str(observation)}"
                        print(f"      !! 工具 '{tool_name}' 返回 dict/list，JSON 序列化失敗: {json_err}。回退到 str()")
                    except Exception as ser_err:
                        observation_str = f"[Error Serializing Result: {ser_err}]"
                        print(f"      !! 工具 '{tool_name}' 返回 dict/list，序列化時發生未知錯誤: {ser_err}")
                    final_content = observation_str

            # --- 處理其他類型 (保持不變) ---
            else:
                try:
                    temp_str = str(observation)
                    if temp_str == "[]":
                         print(f"      !! 工具 '{tool_name}' 返回值 string 化後為 '[]'，可能表示錯誤或空列表。原始類型: {type(observation).__name__}")
                         observation_str = "TOOL_RETURNED_EMPTY_LIST_STR"
                    elif temp_str == "":
                        observation_str = "EMPTY_TOOL_RESULT"
                        print(f"      << 工具 '{tool_name}' 返回空字串，已替換為佔位符。")
                    elif observation is None:
                        observation_str = "NONE_TOOL_RESULT"
                        print(f"      << 工具 '{tool_name}' 返回 None，已替換為佔位符。")
                    else:
                        observation_str = temp_str
                        print(f"      << 工具 '{tool_name}' 返回其他類型 ({type(observation).__name__})，已使用 str() 轉換。")
                except Exception as str_conv_err:
                     observation_str = f"[Error Converting Result to String: {str_conv_err}]"
                     print(f"      !! 工具 '{tool_name}' 返回其他類型，str() 轉換失敗: {str_conv_err}")
                final_content = observation_str

            # 最終防線 (保持不變)
            if not final_content:
                final_content = "FINAL_CONTENT_EMPTY"
                print(f"      !! 警告：最終 final_content 為空，使用最終佔位符。")

            tool_messages.append(ToolMessage(content=final_content, tool_call_id=tool_call_id, name=tool_name))

        except Exception as tool_exec_e:
            error_msg = f"錯誤：執行或處理工具 '{tool_name}' 時失敗: {tool_exec_e}"
            print(f"      !! {error_msg}")
            print(f"         調用時參數: {tool_args_dict}")
            if observation is not None:
                print(f"         ainvoke 返回的觀察值 (類型 {type(observation).__name__}): {repr(observation)[:500]}")
            traceback.print_exc()
            tool_messages.append(ToolMessage(content=str(error_msg), tool_call_id=tool_call_id, name=tool_name))

    return tool_messages


# =============================================================================
# 核心函數：調用 LLM 執行計劃步驟 (添加詳細打印)
# =============================================================================
async def call_llm_with_tools(
    messages: List[BaseMessage],
    selected_tools: List[BaseTool],
    execution_prompt: SystemMessage # <<< 新增參數
) -> AIMessage:
    """
    調用 agent_llm (Gemini) 根據消息歷史（含計劃）和可用工具來執行下一步。
    輸入消息應已包含多模態內容。
    """
    print(f"  >> 調用 Agent LLM ({agent_llm.model}) 執行下一步 (使用提示: {execution_prompt.content[:50]}...)...")
    try:
        # --- 使用輔助函數獲取 Gemini 兼容的工具定義 ---
        print("     正在準備 Gemini 兼容的工具定義列表...")
        gemini_compatible_tools = _prepare_gemini_compatible_tools(selected_tools)
        print(f"     獲取了 {len(gemini_compatible_tools)} 個 Gemini 兼容的工具定義。")

        # --- 綁定工具到 agent_llm ---
        print("     正在將 MCP 工具 (含手動定義) 綁定到 LLM...")
        llm_with_tools = agent_llm.bind_tools(gemini_compatible_tools) # Use corrected tools
        print("     MCP 工具綁定完成。")

        # --- 配置 Runnable 移除回調 ---
        print("     正在配置 LLM runnable 以移除回調 (with_config)...")
        llm_configured_no_callbacks = llm_with_tools.with_config({"callbacks": None})
        print("     LLM runnable 配置完成 (callbacks=None)。")

        # --- 準備調用消息 ---
        # messages 列表應已包含正確格式化的多模態內容 (如果有的話)
        current_call_messages = [execution_prompt] + messages # <<< 修改：使用傳入的 execution_prompt
        print(f"     LLM 輸入消息數 (含執行提示): {len(current_call_messages)}")

        # --- 添加詳細打印 (檢查多模態消息格式) ---
        print("-" * 40)
        print(">>> DEBUG: Messages Sent to LLM.ainvoke:")
        for i, msg in enumerate(current_call_messages):
            print(f"  Message {i} ({type(msg).__name__}):")
            try:
                # 使用更安全的方式獲取和打印內容
                if isinstance(msg.content, str):
                    content_repr = repr(msg.content)
                elif isinstance(msg.content, list):
                     # 對列表內容進行部分表示，避免過長
                     content_repr = "[" + ", ".join(repr(item)[:100] + ('...' if len(repr(item)) > 100 else '') for item in msg.content) + "]"
                else:
                     content_repr = repr(msg.content)
                print(f"    Content: {content_repr[:1000]}{'...' if len(content_repr) > 1000 else ''}")
            except Exception as repr_err:
                print(f"    Content: [Error representing content: {repr_err}]")

            if isinstance(msg, AIMessage) and msg.tool_calls:
                try:
                    tool_calls_repr = repr(msg.tool_calls)
                    print(f"    Tool Calls: {tool_calls_repr[:500]}{'...' if len(tool_calls_repr) > 500 else ''}")
                except Exception as repr_err:
                    print(f"    Tool Calls: [Error representing tool_calls: {repr_err}]")
            elif isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id'):
                 print(f"    Tool Call ID: {msg.tool_call_id}")
        print("-" * 40)
        # --- 結束詳細打印 ---

        # --- 執行 LLM 調用 (使用配置後的 Runnable，無 config 參數) ---
        print("     正在調用配置後的 LLM.ainvoke...")
        response = await llm_configured_no_callbacks.ainvoke(current_call_messages) # 直接傳遞消息列表
        print(f"  << LLM 調用完成。")
        if isinstance(response, AIMessage) and response.tool_calls:
             print(f"     LLM 請求調用 {len(response.tool_calls)} 個工具。")
        elif isinstance(response, AIMessage):
             print(f"     LLM 返回內容: {response.content[:150]}...")
             if "任務已完成" in response.content.lower():
                 print("     偵測到 '任務已完成'。")
        else:
             print(f"     LLM 返回非預期類型: {type(response).__name__}")

        return response

    except Exception as e:
        print(f"!! 執行 LLM 調用 (call_llm_with_tools) 時發生錯誤: {e}")
        traceback.print_exc()
        # ... (錯誤處理保持不變) ...
        error_content = f"執行 LLM 決策時發生錯誤: {e}"
        str_e = str(e)
        if isinstance(e, ValueError) and "Unexpected message with type" in str_e:
             error_content = f"內部錯誤：調用 LLM 時消息順序或類型不匹配。錯誤: {e}"
        elif "Function and/or coroutine must be provided" in str_e or "bind_tools" in str_e.lower():
             error_content = f"內部錯誤：綁定或調用工具時出錯。檢查工具定義或LLM兼容性。錯誤: {e}"
        elif "InvalidArgument: 400" in str_e:
             reason = "未知原因"
             if "missing field" in str_e:
                 reason = f"工具 Schema 無效 (即使手動修正後，仍可能存在問題或影響其他工具)"
             elif "function declaration" in str_e:
                  reason = f"工具函數聲明格式錯誤"
             elif "contents" in str_e: # 檢查是否是內容格式錯誤
                 reason = f"消息內容格式錯誤，可能多模態輸入未被正確處理"
             error_content = f"內部錯誤：傳遞給 Gemini 的數據無效 ({reason})。錯誤: {e}"
        else:
             # 保留通用錯誤處理
             pass # error_content 已在 try 塊外定義

        return AIMessage(content=error_content)


# --- NEW HELPER FUNCTION for preparing Gemini-compatible tools ---
def _prepare_gemini_compatible_tools(mcp_tools: List[BaseTool]) -> List[Union[BaseTool, Dict]]:
    """
    為 Gemini LLM 準備工具列表，手動修正特定工具的 schema。
    """
    print("     [Helper] 準備 Gemini 兼容的工具定義列表...")
    tools_for_binding = []
    if not mcp_tools:
        print("     [Helper] 警告: 傳入的 mcp_tools 列表為空。")
        return []

    for tool in mcp_tools:
        if not tool or not hasattr(tool, 'name'):
            print(f"     [Helper] 警告: 工具列表中發現無效工具對象: {tool}")
            continue

        if tool.name == "get_scene_objects_with_metadata":
            print(f"     [Helper] 為 '{tool.name}' 創建手動 Gemini FunctionDeclaration...")
            manual_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "filters": {
                            "type": "OBJECT",
                            "description": "篩選條件，例如 {'layer': 'Default', 'name': 'Cube*'}",
                            "nullable": True,
                        },
                        "metadata_fields": {
                            "type": "ARRAY",
                            "description": "要返回的元數據欄位列表，例如 ['name', 'layer', 'short_id']",
                            "nullable": True,
                            "items": { "type": "STRING" }
                        }
                    },
                }
            }
            tools_for_binding.append(manual_declaration)
        elif tool.name == "zoom_to_target" or tool.name == "capture_focused_view":
            print(f"     [Helper] 為含 bounding_box 參數的工具 '{tool.name}' 創建手動 Gemini FunctionDeclaration...")
            properties = {
                "view": { "type": "STRING", "description": "視圖名稱或ID", "nullable": True }
            }
            if tool.name == "zoom_to_target":
                properties.update({
                    "object_ids": { "type": "ARRAY", "description": "要縮放到的對象ID列表", "nullable": True, "items": {"type": "STRING"} },
                    "all_views": { "type": "BOOLEAN", "description": "是否應用於所有視圖", "nullable": True }
                })
            elif tool.name == "capture_focused_view":
                properties.update({
                    "projection_type": { "type": "STRING", "description": "投影類型: 'parallel', 'perspective', 'two_point'", "nullable": True },
                    "lens_angle": { "type": "NUMBER", "description": "透視或兩點投影的鏡頭角度", "nullable": True },
                    "camera_position": { "type": "ARRAY", "description": "相機位置的 [x, y, z] 坐標", "nullable": True, "items": {"type": "NUMBER"} },
                    "target_position": { "type": "ARRAY", "description": "目標點的 [x, y, z] 坐標", "nullable": True, "items": {"type": "NUMBER"} },
                    "layer": { "type": "STRING", "description": "用於篩選顯示註釋的圖層名稱", "nullable": True },
                    "show_annotations": { "type": "BOOLEAN", "description": "是否顯示物件註釋", "nullable": True },
                    "max_size": { "type": "INTEGER", "description": "截圖的最大尺寸", "nullable": True }
                })
            
            # Correct bounding_box for Gemini: items need a type for the inner array's elements
            properties["bounding_box"] = {
                "type": "ARRAY",
                "description": "邊界框的8個角點坐標 [[x,y,z], [x,y,z], ...]",
                "nullable": True,
                "items": { # This 'items' describes the outer array (list of points)
                    "type": "ARRAY", # Each item is an array (a point)
                    "items": { # This 'items' describes the inner array (coordinates of a point)
                        "type": "NUMBER" # Each coordinate is a NUMBER
                    }
                }
            }
            manual_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": { "type": "OBJECT", "properties": properties }
            }
            tools_for_binding.append(manual_declaration)
        else:
            tools_for_binding.append(tool) # Add other tools as they are
            # print(f"     [Helper] 保留標準 MCP BaseTool 對象: {tool.name}")
    
    if not tools_for_binding and mcp_tools: # If all tools were invalid or some other issue
        print("     [Helper] 警告: 工具準備後列表為空，但原始列表非空。可能所有工具都無法處理。")
    elif not tools_for_binding and not mcp_tools:
        pass # Expected if input was empty
    else:
        print(f"     [Helper] 完成 Gemini 兼容工具準備，共 {len(tools_for_binding)} 個。")
    return tools_for_binding
# --- END NEW HELPER FUNCTION ---

# =============================================================================
# 圖節點 (Graph Nodes)
# =============================================================================

RPM_DELAY = 6.5 # 比 6 秒稍長一點，留點餘裕

# --- Router Node (MODIFIED to handle pinterest) ---
async def route_mcp_target(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """使用 utility_llm 判斷用戶初始請求文本應路由到哪個 MCP (revit, rhino, pinterest)。"""
    print("--- 執行 MCP 路由節點 ---")

    # --- NEW: Check if target_mcp is already set in the state ---
    pre_set_target_mcp = state.get("target_mcp")
    valid_mcp_targets = ["revit", "rhino", "pinterest", "osm"]
    if pre_set_target_mcp and pre_set_target_mcp in valid_mcp_targets:
        print(f"  檢測到已預設 target_mcp: '{pre_set_target_mcp}'。直接使用此目標，跳過 LLM 路由。")
        return {"target_mcp": pre_set_target_mcp, "last_executed_node": "router_skipped_due_to_preset"}
    # --- END NEW ---

    initial_request_text = state.get('initial_request', '')
    if not initial_request_text:
        print("錯誤：狀態中未找到 'initial_request' 且 target_mcp 未預設。默認為 rhino。")
        # {{ edit_1 }}
        return {"target_mcp": "rhino", "last_executed_node": "router_defaulted_rhino_no_request"}
        # {{ end_edit_1 }}

    print(f"  根據初始請求文本路由: '{initial_request_text[:150]}...'")
    prompt = ROUTER_PROMPT.format(user_request_text=initial_request_text)
    try:
        response = await utility_llm.ainvoke([SystemMessage(content=prompt)], config=config)
        route_decision = response.content.strip().lower()
        print(f"  LLM 路由決定: {route_decision}")
        if route_decision in valid_mcp_targets: # Use the list here
            # {{ edit_2 }}
            return {"target_mcp": route_decision, "last_executed_node": "router_llm_decision"}
            # {{ end_edit_2 }}
        else:
            print(f"  警告: LLM 路由器的回應無法識別 ('{route_decision}')。預設為 rhino。")
            # {{ edit_3 }}
            return {"target_mcp": "rhino", "last_executed_node": "router_defaulted_rhino_unknown_llm_response"}
            # {{ end_edit_3 }}
    except Exception as e:
        print(f"  路由 LLM 呼叫失敗: {e}")
        traceback.print_exc()
        # {{ edit_4 }}
        return {"target_mcp": "rhino", "last_executed_node": "router_defaulted_rhino_llm_exception"}
        # {{ end_edit_4 }}


# <<< 新增：訊息剪枝輔助函式 >>>
MAX_RECENT_INTERACTIONS_DEFAULT = 18
MAX_RECENT_INTERACTIONS_FORCING = 23

def _prune_messages_for_llm(full_messages: List[BaseMessage], max_recent_interactions: int = MAX_RECENT_INTERACTIONS_DEFAULT) -> List[BaseMessage]:
    if not full_messages:
        return []

    initial_human_message = None
    plan_ai_message = None

    # 找到初始的 HumanMessage (通常是列表中的第一個)
    if full_messages and isinstance(full_messages[0], HumanMessage): # Added check for full_messages not empty
        initial_human_message = full_messages[0]

    # 找到最新的計劃 AIMessage
    PLAN_PREFIX = "[目標階段計劃]:\n"
    for msg in reversed(full_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX):
            plan_ai_message = msg
            break

    pruned_list = []
    added_message_ids = set() # 使用物件 id 來避免重複添加完全相同的訊息實例

    # 1. 添加初始 HumanMessage (如果存在)
    if initial_human_message:
        pruned_list.append(initial_human_message)
        added_message_ids.add(id(initial_human_message))

    # 2. 添加計劃 AIMessage (如果存在且與 initial_human_message 不同)
    if plan_ai_message and id(plan_ai_message) not in added_message_ids:
        # 確保計劃訊息不是列表中的第一個 HumanMessage (雖然不太可能，但以防萬一)
        if not (initial_human_message and id(plan_ai_message) == id(initial_human_message)):
            pruned_list.append(plan_ai_message)
            added_message_ids.add(id(plan_ai_message))

    # 3. 確定近期互動的候選訊息 (排除已添加的 initial_human_message 和 plan_ai_message)
    recent_interaction_candidates = []
    for msg in full_messages:
        if id(msg) not in added_message_ids:
            recent_interaction_candidates.append(msg)
    
    # 選取最後 N 條作為實際的近期互動訊息
    actual_recent_interactions = recent_interaction_candidates[-max_recent_interactions:]

    # 4. 將近期互動訊息添加到剪枝後的列表
    #    將 initial_human_message 和 plan_ai_message 放在前面，然後是 recent_interactions
    #    這裡的邏輯是重新構建 pruned_list，而不是在現有的 pruned_list 後追加
    final_pruned_list = []
    temp_added_ids = set()

    if initial_human_message:
        final_pruned_list.append(initial_human_message)
        temp_added_ids.add(id(initial_human_message))

    if plan_ai_message and id(plan_ai_message) not in temp_added_ids:
        final_pruned_list.append(plan_ai_message)
        temp_added_ids.add(id(plan_ai_message))
    
    for msg in actual_recent_interactions:
        if id(msg) not in temp_added_ids: # 避免再次添加 plan 或 initial human message 如果它們恰好在尾部
            final_pruned_list.append(msg)
            # temp_added_ids.add(id(msg)) # 不需要，因為是從尾部取的

    # --- 日誌記錄剪枝後的訊息 (可選，用於調試) ---
    # print(f"    原始訊息數量: {len(full_messages)}, 剪枝後訊息數量: {len(final_pruned_list)}")
    # pruned_message_summary = []
    # for i, m_obj in enumerate(final_pruned_list):
    #     m_content_str = ""
    #     if isinstance(m_obj.content, str):
    #         m_content_str = m_obj.content[:30].replace("\n", " ") + "..."
    #     elif isinstance(m_obj.content, list) and m_obj.content:
    #         first_item_content = m_obj.content[0]
    #         if isinstance(first_item_content, dict) and first_item_content.get("type") == "text":
    #             m_content_str = first_item_content.get("text", "")[:30] + "..."
    #         else:
    #             m_content_str = str(first_item_content)[:30] + "..."
    #     elif m_obj.content is None:
    #         m_content_str = "[None Content]"
    #     else:
    #         m_content_str = f"[{type(m_obj.content).__name__} Content]"
    #     pruned_message_summary.append(f"      {i}: {type(m_obj).__name__} - '{m_content_str}'")
    # print("    剪枝後訊息預覽:\n" + "\n".join(pruned_message_summary))
    # --- 結束日誌記錄 ---

    return final_pruned_list
# <<< 結束：訊息剪枝輔助函式 >>>

# =============================================================================
# Agent Nodes (修改：處理 Pinterest ToolMessage，返回最終結果)
# =============================================================================
async def agent_node_logic(state: MCPAgentState, config: RunnableConfig, mcp_name: str) -> Dict:
    """通用 Agent 節點邏輯：處理特定工具消息，規劃，或執行下一步。"""
    print(f"--- 執行 {mcp_name.upper()} Agent 節點 ---")
    current_messages = list(state['messages'])
    last_message = current_messages[-1] if current_messages else None
    current_consecutive_responses = state.get("consecutive_llm_text_responses", 0)
    # Ensure rhino_screenshot_counter is present in the state, default to 0 if not
    current_rhino_screenshot_counter = state.get("rhino_screenshot_counter", 0)

    # --- 處理 capture_viewport, OSM, Pinterest 的 ToolMessage 返回 ---
    IMAGE_PATH_PREFIX = "[IMAGE_FILE_PATH]:"
    OSM_IMAGE_PATH_PREFIX = "[OSM_IMAGE_PATH]:" # Assuming OSM tool returns this prefix
    CSV_PATH_PREFIX = "[CSV_FILE_PATH]:"

    if isinstance(last_message, ToolMessage):
        # Handle Local CSV Creation Tool
        if last_message.name == "create_planned_data_summary_csv":
            if last_message.content.startswith(CSV_PATH_PREFIX):
                csv_path = last_message.content[len(CSV_PATH_PREFIX):]
                print(f"  檢測到計劃數據CSV報告已生成於: {csv_path}")
                # This tool is called after planning is done. The next step is to start executing the modeling.
                # Returning a message here allows the agent to acknowledge and proceed.
                return {
                    "messages": [AIMessage(content=f"計劃總結報告已在規劃階段完成，並保存於 {csv_path}。現在開始執行模型建構。")],
                    "saved_csv_path": csv_path,
                    "task_complete": False, # Modeling is not yet done
                    "consecutive_llm_text_responses": 0,
                    "last_executed_node": f"{mcp_name}_agent"
                }

        # Handle Rhino/Revit Screenshot Path
        if last_message.name == "capture_focused_view" and isinstance(last_message.content, str):
            if last_message.content.startswith(IMAGE_PATH_PREFIX):
                print("  檢測到 capture_viewport 工具返回的文件路徑。") 
                uuid_image_path = last_message.content[len(IMAGE_PATH_PREFIX):]
                print(f"    原始文件路徑 (UUID based): {uuid_image_path}")
                
                new_image_path_for_state = uuid_image_path # Default to original if rename fails
                data_uri_for_state = None
                # {{ edit_2 }}
                # --- MODIFIED: Renaming logic for Rhino screenshots ---
                if mcp_name == "rhino":
                    current_rhino_screenshot_counter += 1 # Increment counter from state
                    
                    # Sanitize initial_request for use in filename (take first 20 chars, replace spaces, keep alphanum and underscore)
                    req_str_part = state.get('initial_request', 'RhinoTask')
                    sanitized_req_prefix = "".join(filter(lambda x: x.isalnum() or x == '_', req_str_part.replace(" ", "_")[:20]))
                    
                    original_extension = os.path.splitext(uuid_image_path)[1]
                    new_filename = f"{sanitized_req_prefix}_Shot-{current_rhino_screenshot_counter}{original_extension}"
                    
                    try:
                        if os.path.exists(uuid_image_path):
                            new_renamed_path = os.path.join(os.path.dirname(uuid_image_path), new_filename)
                            os.rename(uuid_image_path, new_renamed_path)
                            new_image_path_for_state = new_renamed_path # Use renamed path
                            print(f"    文件已重命名為: {new_renamed_path}")
                        else:
                            print(f"  !! 錯誤：capture_viewport 返回的原始文件路徑不存在: {uuid_image_path}。無法重命名。")
                            # new_image_path_for_state remains uuid_image_path, which is problematic if it doesn't exist.
                            # Consider how to handle this error - perhaps return an error message.
                            # For now, it will proceed and likely fail to generate URI / be found later.
                    except Exception as rename_err:
                        print(f"  !! 重命名文件 '{uuid_image_path}' 至 '{new_filename}' 時出錯: {rename_err}")
                        # new_image_path_for_state remains uuid_image_path
                # --- END MODIFICATION ---
                # {{ end_edit_2 }}

                try:
                    if not os.path.exists(new_image_path_for_state):
                        print(f"  !! 錯誤：處理後的圖像文件路徑不存在: {new_image_path_for_state}")
                        # {{ edit_3 }}
                        return { 
                              "messages": [AIMessage(content=f"截圖文件未找到: {new_image_path_for_state}。")],
                              "saved_image_path": None, "saved_image_data_uri": None,
                              "task_complete": False, 
                              "consecutive_llm_text_responses": 0,
                              "rhino_screenshot_counter": current_rhino_screenshot_counter # Return updated counter
                              # {{ end_edit_3 }}
                          }
                    with open(new_image_path_for_state, "rb") as f: image_bytes = f.read()
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    mime_type = "image/png" 
                    ext = os.path.splitext(new_image_path_for_state)[1].lower()
                    if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                    data_uri_for_state = f"data:{mime_type};base64,{base64_data}"
                    # {{ edit_4 }}
                    return {
                         "messages": [AIMessage(content=f"已成功截取畫面並保存至 {new_image_path_for_state}。")],
                         "saved_image_path": new_image_path_for_state, 
                         "saved_image_data_uri": data_uri_for_state,
                         "task_complete": False, 
                         "consecutive_llm_text_responses": 0,
                         "rhino_screenshot_counter": current_rhino_screenshot_counter # Return updated counter
                         # {{ end_edit_4 }}
                    }
                except Exception as img_proc_err:
                    print(f"  !! 處理截圖文件 '{new_image_path_for_state}' 或編碼時出錯: {img_proc_err}")
                    # {{ edit_5 }}
                    return { 
                         "messages": [AIMessage(content=f"處理截圖文件 '{new_image_path_for_state}' 時失敗: {img_proc_err}。")],
                         "task_complete": False, 
                         "consecutive_llm_text_responses": 0,
                         "rhino_screenshot_counter": current_rhino_screenshot_counter # Return updated counter
                         # {{ end_edit_5 }}
                     }
            elif last_message.content.startswith("[Error: Viewport Capture Failed]:"): 
                error_msg = last_message.content 
                print(f"  檢測到 capture_viewport 工具返回錯誤: {error_msg}")
                # {{ edit_6 }}
                return {"messages": [AIMessage(content=f"任務因截圖錯誤而中止: {error_msg}")], "task_complete": True, "consecutive_llm_text_responses": 0, "rhino_screenshot_counter": current_rhino_screenshot_counter} 
                # {{ end_edit_6 }}

        # Handle OSM Screenshot Path
        elif last_message.name == "geocode_and_screenshot" and isinstance(last_message.content, str) and last_message.content.startswith(OSM_IMAGE_PATH_PREFIX): 
            print("  檢測到 geocode_and_screenshot 工具返回的文件路徑。") 
            image_path = last_message.content[len(OSM_IMAGE_PATH_PREFIX):]
            print(f"    OSM 文件路徑: {image_path}")
            try:
                # ... (OSM Image processing logic: check exists, read, encode, create data URI) ...
                if not os.path.exists(image_path):
                    print(f"  !! 錯誤：收到的 OSM 圖像文件路徑不存在: {image_path}") # Corrected Indentation
                    return {"messages": [AIMessage(content=f"地圖處理完畢，但截圖文件未找到: {image_path}")], "task_complete": True, "consecutive_llm_text_responses": 0} # OSM task is likely done
                with open(image_path, "rb") as f: image_bytes = f.read() # Corrected Indentation
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                # ... (mime type detection) ...
                mime_type = "image/png" # Default or detect # Corrected Indentation
                data_uri = f"data:{mime_type};base64,{base64_data}"
                return {
                    "messages": [AIMessage(content=f"地圖截圖已完成。\n截圖已保存至 {image_path}。")],
                    "saved_image_path": image_path, "saved_image_data_uri": data_uri,
                    "task_complete": True, # OSM task usually ends here
                    "consecutive_llm_text_responses": 0
                }
            except Exception as img_proc_err:
                print(f"  !! 處理 OSM 截圖文件 '{image_path}' 或編碼時出錯: {img_proc_err}")
                return {"messages": [AIMessage(content=f"地圖截圖已完成，但處理文件 '{image_path}' 時失敗: {img_proc_err}")], "task_complete": True, "consecutive_llm_text_responses": 0} # Corrected Indentation

        # Handle Pinterest Download Paths
        elif last_message.name == "pinterest_search_and_download": # Corrected Indentation
            print("  檢測到 pinterest_search_and_download 工具返回。") # Corrected Indentation
            content = last_message.content
            saved_paths_list = None
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "downloaded_paths" in data and isinstance(data["downloaded_paths"], list):
                    saved_paths_list = data["downloaded_paths"]
                    print(f"    成功解析到 {len(saved_paths_list)} 個下載路徑。")
                else:
                     print(f"    ToolMessage content is JSON but missing 'downloaded_paths' list: {content[:200]}...")
            except json.JSONDecodeError:
                print(f"    ToolMessage content is not JSON (likely text output or error): {content[:200]}...")
            except Exception as e:
                print(f"    解析 Pinterest ToolMessage content 時出錯: {e}")

            if saved_paths_list:
                 last_path = saved_paths_list[-1] if saved_paths_list else None
                 data_uri = None
                 if last_path:
                     try:
                        # ... (Image processing for last Pinterest image) ...
                         with open(last_path, "rb") as f: image_bytes = f.read()
                         base64_data = base64.b64encode(image_bytes).decode('utf-8')
                         # ... (mime type detection) ...
                         mime_type = "image/png" # Default or detect
                         data_uri = f"data:{mime_type};base64,{base64_data}"
                     except Exception as img_proc_err:
                         print(f"    !! 處理最後一個 Pinterest 文件 '{last_path}' 或編碼時出錯: {img_proc_err}")

                 return {
                     "messages": [AIMessage(content=f"Pinterest 圖片搜索和下載完成，共找到 {len(saved_paths_list)} 個有效文件。")],
                     "saved_image_path": last_path, # Keep last for reference
                     "saved_image_data_uri": data_uri, # Keep last for reference
                     "saved_image_paths": saved_paths_list, # Store the full list
                     "task_complete": True, # Assume Pinterest is final step
                     "consecutive_llm_text_responses": 0
                 }
            else:
                 print("    Pinterest 工具未返回有效路徑列表，任務可能未成功或未找到圖片。")
                 # Return text message, reset counter, task likely complete based on Pinterest prompt
                 return {"messages": [AIMessage(content=f"Pinterest 任務處理完成，但未找到或處理下載路徑。工具輸出: {content[:200]}...")], "task_complete": True, "consecutive_llm_text_responses": 0} # Corrected Indentation
            # Add more elif blocks here if other tools return specific results needing processing

    # --- 如果不是處理特定工具返回，則執行正常規劃/執行邏輯 ---
    try:
        # ... (Planning/Execution logic starts here) ...
        initial_image_path = state.get('initial_image_path')
        has_input_image = initial_image_path and os.path.exists(initial_image_path)
        if has_input_image: print(f"  檢測到初始圖片輸入: {initial_image_path}")
        else: print("  未檢測到有效初始圖片輸入。")

        if not current_messages or not isinstance(current_messages[0], HumanMessage):
             print("!! 錯誤：狀態 'messages' 為空或第一個消息不是 HumanMessage。")
             return {"messages": [AIMessage(content="內部錯誤：缺少有效的初始用戶請求消息。")]}
        initial_user_message_obj = current_messages[0]
        initial_user_text = ""
        if isinstance(initial_user_message_obj.content, str): initial_user_text = initial_user_message_obj.content
        elif isinstance(initial_user_message_obj.content, list):
            for item in initial_user_message_obj.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    initial_user_text = item.get("text", ""); break
        if not initial_user_text:
            print("!! 錯誤：無法從初始 HumanMessage 提取文本內容。")
            return {"messages": [AIMessage(content="內部錯誤：無法解析初始用戶請求文本。")]}
        print(f"  使用初始文本 '{initial_user_text[:100]}...' 作為基礎。")

        # PLAN_PREFIX = "[目標階段計劃]:\n" # <<< 移除此處的局部定義 >>>
        plan_exists = any(
            isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)
            for msg in current_messages
        )

        # ========================
        # === PLANNING PHASE ===
        # ========================
        if not plan_exists:
            print(f"  檢測到無計劃，進入規劃階段...")
            # --- 獲取工具用於規劃提示 ---
            mcp_tools = await get_mcp_tools(mcp_name)
            print(f"  獲取了 {len(mcp_tools)} 個 {mcp_name} MCP 工具 (用於規劃提示)。")
            if not mcp_tools: print(f"  警告：未找到 {mcp_name} 工具！")

            # --- 新增: 將本地工具加入列表 ---
            all_available_tools = mcp_tools + LOCAL_TOOLS
            print(f"  提供給規劃師的工具總數: {len(all_available_tools)} (MCP: {len(mcp_tools)}, Local: {len(LOCAL_TOOLS)})")

            # --- 選擇規劃提示 ---
            active_planning_prompt_content = ""
            if mcp_name in ["rhino", "revit"]:
                active_planning_prompt_content = """你是一位優秀的任務規劃助理，專門為 CAD/BIM 任務制定計劃。
            基於使用者提供的文字請求、可選的圖像以及下方列出的可用工具，生成一個清晰的、**分階段目標**的計劃。

            **重要要求：**
            1.  **量化與具體化:** 對於幾何操作 (Rhino/Revit)，每個階段目標**必須**包含盡可能多的**具體數值、尺寸、座標、角度、數量、距離、方向、或清晰的空間關係描述**。
            2.  **邏輯順序:** 確保階段目標按邏輯順序排列，後續步驟依賴於先前步驟的結果。
            3.  **基地與座標系統意識 (Rhino - 極度重要):**
                *   **確立基準方位:** 在進行任何與基地佈局相關的規劃時，**第一步必須是確立一個清晰的座標系統和方向基準**。明確定義「北」方與其他「東、西、南」對應的向量（例如，世界座標的Y軸正方向 `(0, 1, 0)`），並在後續所有步驟中嚴格遵守此基準。
                *   **邊界意識:** 如果任務提供了基地邊界，**必須**將處理基地邊界作為優先步驟。
                    *   a. 規劃創建或識別代表基地邊界的曲線。
                    *   b. 在規劃放置任何建築量體、道路或景觀元素之前，**必須**先驗證其預計位置**完全位於**已定義的基地邊界內部。可以規劃獲取基地邊界的 bounding box 作為快速檢查。
            4.  **空間佈局規劃 (Rhino):**
                    *   當任務涉及空間配置或多個量體的佈局時，計劃應明確描述這些量體之間的**拓撲關係** (如相鄰、共享面、包含) 和**相對位置** (如A在B的上方，C在D的西側並偏移X單位)。
                    *   **空間單元化原則：原則上，每一個獨立的功能空間（例如客廳、單獨的臥室、廚房、衛生間等）都應該規劃為一個獨立的幾何量體。避免使用單一量體代表多個不同的功能空間。為每個規劃生成的獨立空間量體或重要動線元素指定一個有意義的臨時名稱或標識符，並在後續的建模步驟中通過 Rhino 的 `add_object_metadata()` 功能將此名稱賦予對應的 Rhino 物件。**
                    *   **圖層規劃 - 初始設定：** 在開始任何建模或創建新的方案/基礎圖層 (如 "方案A", "Floor_1") 之前，**必須**規劃一個步驟：首先獲取當前場景中的所有圖層列表，然後將所有已存在的**頂層圖層**及其子圖層設置為不可見。這樣可以確保在一個乾淨的環境中開始新的設計工作。之後再創建並設置當前工作所需的圖層。
                    *   **圖層規劃 - 動線表達與分層 (Rhino):**
                        *   對於**水平動線**（例如走廊、通道），如果需要視覺化，建議規劃使用非常薄的板狀量體來示意其路徑和寬度。這些水平動線元素**必須**規劃到其所服務的樓層圖層下的**子圖層**中，例如：`Floor_1::Corridors_F1` 或 `Floor_Ground::Horizontal_Circulation`。
                        *   對於**垂直動線**（例如樓梯、坡道、電梯井），則應規劃使用合適的3D量體來表達其佔據的空間和形態。這些垂直動線元素通常規劃到一個獨立的頂層圖層下，例如 `Circulation::Vertical_Core` 或 `Stairs_Elevators`。
                        *   所有動線元素也必須根據其服務的樓層或連接關係，正確地規劃到相應的圖層下。
                    *   在進行複雜的空間佈局規劃時，可以先(以文字描述的形式)構思一個2D平面上的關係草圖，標註出各個獨立空間量體和動線的大致位置、尺寸和鄰接關係，然後再將此2D關係轉化為3D建模步驟的規劃。
                *   規劃時需仔細考慮並確保最終生成的**量體數量、各個空間量體的具體位置和尺寸**符合設計意圖和空間邏輯。 **對於每個創建的空間，必須使用 `rs.AddTextDot("空間名稱", (x,y,z))` 在其量體中心附近標示空間名稱。絕對禁止使用 `rs.AddText()` 或 `rs.SetUserText()`。**
            5.  **多方案與多樓層處理 (Rhino):**
                *   如果用戶請求中明確要求"多方案"或"不同選項"，**必須**將每個方案視為一個**獨立的、完整的任務序列**來規劃。
                *   為每個方案指定一個清晰的名稱或標識符 (例如 "方案A_現代風格", "方案B_傳統風格")，並在整個方案的規劃和執行階段中使用此標識。
                *   計劃應清晰地標示每個方案的開始和結束。
                *   **對於包含多個樓層的設計方案，在完成每一樓層的主要建模內容後，應規劃一次詳細的截圖步驟。多方案規劃時每一方案完成後也同樣。 (參考下方截圖規劃詳細流程)。**
                *   對於多樓層可以規劃同時展示所有樓層的截圖總覽，但對於多方案不用。
            6.  **造型與形態規劃 (Rhino):**
                *   當任務目標涉及'造型方案'、'形態生成'或對現有量體進行'外觀設計'時，規劃階段應積極考慮如何利用布林運算 (如加法、減法、交集) 和幾何變換 (如扭轉、彎曲、陣列、縮放、旋轉) 等高級建模技巧來達成獨特且具有空間感的「虛、實」幾何形態。
                *   **如要創造更具特殊性、流動性或有機感的造型，應考慮並規劃使用多種曲面生成與編輯技巧。規劃時應考慮工具的輸入要求：**
                    *   **曲面應用技巧：** 優先規劃從曲線或曲面創建實體或有厚度的曲面，不要只是開放曲面。應用上盡量不要混雜保持造型純粹性。
                    *   **曲面創建類別：**
                        *   **掃掠 (Sweep):**
                            *   `rs.AddSweep1(rail_curve_id, shape_curve_ids)`: 將剖面曲線列表 `shape_curve_ids` 沿單一軌道 `rail_curve_id` 掃掠成曲面。注意剖面曲線的方向和順序。
                            *   `rs.AddSweep2(rail_curve_ids, shape_curve_ids)`: 將剖面曲線列表 `shape_curve_ids` 沿兩個軌道列表 `rail_curve_ids` 掃掠成曲面。注意剖面曲線的方向、順序及與軌道的接觸。
                        *   **放樣 (Loft):**
                            *   `rs.AddLoftSrf(curve_ids, start_pt=None, end_pt=None, type=0, style=0, simplify=0, closed=False)`: 在有序的曲線列表 `curve_ids` 之間創建放樣曲面。注意曲線方向和接縫點。可指定類型、樣式等。
                        *   **網格曲面 (Network Surface):**
                            *   `rs.AddNetworkSrf(curve_ids)`: 從一組相交的曲線網絡 `curve_ids` 創建曲面。所有 U 方向曲線必須與所有 V 方向曲線相交。
                        *   **平面曲面 (Planar Surface):**
                            *   `rs.AddPlanarSrf(curve_ids)`: 從一個或多個封閉的*平面*曲線列表 `curve_ids` 創建平面曲面。曲線必須共面且封閉。
                    *   **實體創建類別：**
                        *   **擠出 (Extrusion):**
                            *   `rs.ExtrudeCurve(curve_id, path_curve_id)`: 將輪廓線 `curve_id` 沿路徑曲線 `path_curve_id` 擠出成曲面。
                            *   `rs.ExtrudeCurveStraight(curve_id, start_point, end_point)` 或 `rs.ExtrudeCurveStraight(curve_id, direction_vector)`: 將曲線 `curve_id` 沿直線擠出指定距離和方向。
                            *   `rs.ExtrudeCurveTapered(curve_id, distance, direction, base_point, angle)`: 將曲線 `curve_id` 沿 `direction` 方向擠出 `distance` 距離，同時以 `base_point` 為基準、按 `angle` 角度進行錐化。
                            *   `rs.ExtrudeSurface(surface_id, path_curve_id, cap=True/False)`: 將曲面 `surface_id` 沿路徑曲線 `path_curve_id` 擠出成實體或開放形狀，可選是否封口 (`cap`)。
                *   在計劃中明確指出預計在哪些步驟使用這些技巧，以及預期達成的形態效果和所需的輸入物件。造型上應具有特殊的美學價值並符合設計概念。
            7.  **圖像參考規劃 (若有提供圖像):**
                *   在生成具體的建模計劃之前，**必須**先進行詳細的"圖像分析與解讀"階段。
                *   規劃時應基於：觀察到的主要建築體塊組成和它們之間的**空間布局關係**（例如，穿插、並列、堆疊）；估計主要部分之間的精確長、寬、高比例關係；主次要量體的位置關係；主要的立面特徵（重點是整體形態）；柱子及其他特殊形式。
                *   **必須**將上述圖像分析得出的觀察結果，轉化為後續 Rhino 建模步驟中的具體參數和操作指導。**需特別注意絕對座標上的位置關係；方體的高度及角度關係；長短邊的方向關係，以構成符合圖片目標的建築塊體。**
                *   **如果任務是參考圖片進行空間佈局(或量體配置)規劃，要在主要建築塊體的關係下發展詳細量體及空間配置。不需要建立精確立面等細部特徵。**
            8.  **截圖規劃詳細流程 (Rhino/Revit):**
                *   **截圖策略：** 規劃應分為兩個主要截圖階段，以確保成果的完整展示：
                    1.  **整體視圖階段：** 在所有主要建模步驟完成後，首先規劃生成一到兩個能夠展示整體設計的**透視 (`perspective`) 或兩點透視 (`two_point`)** 視圖。在執行此階段的截圖時，**必須確保所有與設計方案相關的圖層（例如所有樓層、外部造型、基地等）都是可見的**，以呈現完整的模型。
                    2.  **分層平面圖階段：** 如果是量體或平面規劃任務，在整體視圖截圖完成後，再針對**每一個樓層**規劃生成單獨的、**俯視的平行投影 (`parallel`)** 視圖。在執行此階段的截圖時，**必須只顯示當前正在截圖的樓層圖層**，並隱藏所有其他不相關的樓層圖層，以確保平面圖的清晰性。
                *   **每張截圖的詳細步驟：**
                    a.  **設定視圖與投影：** 明確指定投影模式 (`perspective`, `two_point`, `parallel`) 並設定適當的鏡頭角度。對於俯視平面圖，必須使用 `parallel` 模式。**對於目的為渲染的模型，適用於`two_point`並配合相機採用人視角(`z=2`,`lens_angle=10`)；展示其他類型的模型，適用於`perspective`。**
                    b.  **管理圖層可見性 (關鍵)：** 規劃獲取當前場景中所有圖層的列表後，根據上述的「整體視圖」或「分層平面圖」策略，精確地規劃顯示或隱藏哪些圖層。
                    c.  **設定相機 (可選但建議)：** 對於透視圖，規劃設定相機位置 (`camera_position`) 和目標點 (`target_position`)，以獲得最佳視角。例如，人視角高度約為 `z=2`,`lens_angle=10`，鳥瞰視角可以更高。**不論何時，必須確保相機旋轉設定為0，切勿讓其變為-90度。**
                    d.  **執行截圖：** 規劃調用 `capture_focused_view` 工具。 建議設定相機後還要使用zoom功能鎖定目標。 
            9.  **目標狀態:** 計劃應側重於**每個階段要達成的目標狀態**，說明該階段完成後場景應有的變化。
                *   **最後一個計劃應包含"全部任務已完成"時的相關行動，引導實際執行時的處理。**
            10.  **規劃數據摘要報告 (空間規劃任務的必要首步):**
                *   **僅當**任務是關於**空間佈局規劃** (例如，量體配置等)，你**必須**將生成摘要報告作為計劃的**第一個步驟**。
                *   **此步驟基於你即將制定的後續建模步驟，先行總結和報告規劃的量化數據。如果是要求分析已有的方案，則應該要先分析再進行數據摘要整理。**
                *   **規劃的第一步應如下：**
                    1.  **預先匯總:** 在腦中構思好所有建模步驟後，審查你計劃要創建的所有空間（如客廳、臥室等）的名稱、**所屬樓層**和具體尺寸/面積。
                    2.  **計算匯總數據:** 基於這些規劃數值，計算出總面積、每個空間的面積佔比，以及建蔽率(BCR)和容積率(FAR)（如果適用）。
                    3.  **規劃首個工具調用:** 將匯總好的數據（`data_rows` - 其中每個空間字典需包含 `name`, `area`, `percentage` **和 `floor`**，`total_area`, `bcr`, `far`）作為參數，將對 `create_planned_data_summary_csv` 工具的調用規劃為整個計劃的**第 1 步**。
                    4.  **後續步驟:** 在此報告步驟之後，再依次列出所有實際的 Rhino 模型建構步驟。

            **rhino提醒:目前單位是M(公尺)。**這個計劃應側重於**每個階段要達成的目標狀態並包含細節**，而不是具體的工具使用細節。將任務分解成符合邏輯順序及細節的多個階段目標。直接輸出這個階段性目標計劃，不要额外的開場白或解釋。
            可用工具如下 ({mcp_name}):
            {tool_descriptions}"""
            elif mcp_name == "pinterest":
                # Define Pinterest planning prompt content here or use a global variable
                active_planning_prompt_content = f"""用戶請求使用 Pinterest 進行圖片搜索。
                可用工具 ({mcp_name}):
                - pinterest_search_and_download: {{"description": "Searches Pinterest for images based on a keyword and downloads them. Args: keyword (str), limit (int, optional)."}}
                請制定一個單一步驟計劃來使用 pinterest_search_and_download 工具，目標是根據用戶請求搜索並下載圖片。
                計劃的最終步驟應明確指出調用 `pinterest_search_and_download`。"""
            elif mcp_name == "osm":
                 # Define OSM planning prompt content here or use a global variable
                 active_planning_prompt_content = f"""用戶請求使用 OpenStreetMap 生成地圖截圖。
                 可用工具 ({mcp_name}):
                 - geocode_and_screenshot: {{"description": "Geocodes an address or uses coordinates to take a screenshot from OpenStreetMap. Args: address (str: address or 'lat,lon')."}}
                 請制定一個單一步驟計劃來使用 geocode_and_screenshot 工具，目標是根據用戶請求生成地圖截圖。
                 計劃的最終步驟應明確指出調用 `geocode_and_screenshot`。"""
            else: # Fallback
                tool_descriptions_for_fallback_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
                active_planning_prompt_content = f"請為使用 {mcp_name} 的任務制定計劃。可用工具：\n{tool_descriptions_for_fallback_str}"

            # --- 格式化規劃提示 (Only for Rhino/Revit as others have descriptions embedded or generated above) ---
            planning_system_content_final = active_planning_prompt_content
            if mcp_name in ["rhino", "revit"]:
                tool_descriptions_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_available_tools])
                planning_system_content_final = active_planning_prompt_content.format(
                    mcp_name=mcp_name,
                    tool_descriptions=tool_descriptions_for_prompt
                )
            # Note: No formatting needed for Pinterest/OSM/Fallback as their prompts are already complete strings

            planning_system_message = SystemMessage(content=planning_system_content_final)
            print(f"    為 {mcp_name} 構造了規劃 SystemMessage")

            # --- 構造規劃 HumanMessage ---
            planning_human_content = [{"type": "text", "text": initial_user_text}]
            if has_input_image:
                try:
                    with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    # Determine mime type properly if possible, default to png
                    mime_type="image/png"
                    file_ext = os.path.splitext(initial_image_path)[1].lower()
                    if file_ext in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
                    elif file_ext == '.gif': mime_type = 'image/gif'
                    elif file_ext == '.webp': mime_type = 'image/webp'

                    planning_human_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                    })
                    print("    已將初始圖片添加到規劃 HumanMessage 中。")
                except Exception as img_read_err:
                    print(f"    !! 無法讀取或編碼初始圖片: {img_read_err}")
                    # Fallback to text only if image fails
                    planning_human_content = [{"type": "text", "text": initial_user_text}]

            # Ensure content is always a list for multi-modal models
            if not isinstance(planning_human_content, list):
                 planning_human_content = [{"type": "text", "text": str(planning_human_content)}] # Should not happen with above logic, but safe fallback

            planning_human_message_user_input = HumanMessage(content=planning_human_content)

            # --- 調用 LLM 進行規劃 ---
            print(f"     正在調用 LLM ({agent_llm.model}) 進行規劃...")
            plan_message = None
            try:
                # Use the main agent LLM for planning
                planning_llm_no_callbacks = agent_llm.with_config({"callbacks": None})
                planning_response = await planning_llm_no_callbacks.ainvoke(
                    [planning_system_message, planning_human_message_user_input]
                )

                if isinstance(planning_response, AIMessage) and planning_response.content:
                    # Prepend the prefix to identify it as a plan
                    plan_content = PLAN_PREFIX + planning_response.content.strip()
                    plan_message = AIMessage(content=plan_content)
                    print(f"  生成階段目標計劃:\n------\n{plan_content[:500]}...\n------")
                else:
                    # Handle cases where planning LLM failed or returned unexpected format
                    error_msg = "LLM 未能生成有效計劃。"
                    if isinstance(planning_response, AIMessage) and not planning_response.content:
                         error_msg += " (回應內容為空)"
                    elif not isinstance(planning_response, AIMessage):
                         error_msg += f" (返回類型為 {type(planning_response).__name__})"
                    print(f"  !! {error_msg}")
                    plan_message = AIMessage(content=f"無法為您的請求制定計劃。({error_msg})") # Provide some error info

            except Exception as planning_err:
                 error_msg = f"調用規劃 LLM 時發生錯誤: {planning_err}"
                 print(f"  !! {error_msg}")
                 traceback.print_exc()
                 plan_message = AIMessage(content=error_msg) # Return the error message
            finally:
                print(f"     規劃 LLM 調用結束，等待 {RPM_DELAY} 秒...")
                await asyncio.sleep(RPM_DELAY)
                print("     等待結束。")

            # --- *** 規劃完成後直接返回，觸發 should_continue *** ---
            # Return the plan message (or error message if planning failed)
            # Reset counter as this node completed its current task (planning)
            return {"messages": [plan_message] if plan_message else [], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent"}

        # ==========================
        # === EXECUTION PHASE ===
        # ==========================
        else:
            print(f"  檢測到已有計劃，進入執行階段...")
            # --- 獲取 MCP 工具 ---
            mcp_tools = await get_mcp_tools(mcp_name)
            print(f"  獲取了 {len(mcp_tools)} 個 {mcp_name} MCP 工具 (用於執行)。")
            if not mcp_tools: print(f"  警告：執行階段未找到 {mcp_name} 工具！")

            # --- 組合所有可用工具 ---
            all_tools_for_execution = mcp_tools + LOCAL_TOOLS

            # --- 選擇執行提示 ---
            active_execution_prompt_template = None # Use template now
            if mcp_name in ["rhino", "revit"]:
                # Use the globally defined RHINO_AGENT_EXECUTION_PROMPT
                active_execution_prompt_template = RHINO_AGENT_EXECUTION_PROMPT
            elif mcp_name == "pinterest":
                 # Use the globally defined PINTEREST_AGENT_EXECUTION_PROMPT
                 active_execution_prompt_template = PINTEREST_AGENT_EXECUTION_PROMPT # No formatting needed
            elif mcp_name == "osm":
                 # Use the globally defined OSM_AGENT_EXECUTION_PROMPT
                 active_execution_prompt_template = OSM_AGENT_EXECUTION_PROMPT # No formatting needed
            else: # Fallback
                print(f"  警告：執行階段找不到為 {mcp_name} 定義的特定執行提示，將使用 Rhino/Revit 後備提示。")
                active_execution_prompt_template = RHINO_AGENT_EXECUTION_PROMPT

            if not active_execution_prompt_template:
                 # Safety check
                 print(f"  !! 嚴重錯誤：未能為 {mcp_name} 確定有效的執行提示！")
                 return {"messages": [AIMessage(content=f"內部錯誤：無法為 {mcp_name} 加載執行指令。")], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent_error"}

            # --- NEW: Format execution prompt with tools for relevant agents ---
            active_execution_prompt = None
            if "{tool_descriptions}" in active_execution_prompt_template.content:
                tool_descriptions_for_exec = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools_for_execution])
                active_execution_prompt = SystemMessage(
                    content=active_execution_prompt_template.content.format(tool_descriptions=tool_descriptions_for_exec)
                )
            else:
                # For prompts that don't need tool formatting (like Pinterest/OSM)
                active_execution_prompt = active_execution_prompt_template
            # --- END NEW ---

            # --- 判斷是否為計劃生成後首次執行 ---
            is_first_execution_after_plan = False
            # 如果 plan_exists (我們在執行分支) 且最後一條消息是有效的計劃消息,
            # 這意味著我們剛從規劃階段過渡到執行階段的第一步。
            if plan_exists and isinstance(last_message, AIMessage) and \
               isinstance(last_message.content, str) and \
               last_message.content.strip().startswith(PLAN_PREFIX):
                
                # 再次確認這不是一個包含 PLAN_PREFIX 的錯誤消息
                is_actual_plan_msg = "無法為您的請求制定計劃" not in last_message.content and \
                                     "調用規劃 LLM 時發生錯誤" not in last_message.content
                if is_actual_plan_msg:
                    is_first_execution_after_plan = True
                    print("    檢測到這是計劃生成後的第一個執行調用 (最後一條消息是有效的計劃)。")

            # --- 準備執行階段的消息 ---
            messages_for_execution = current_messages
            # Ensure the first HumanMessage includes the image if provided and not already multi-modal
            if has_input_image and isinstance(messages_for_execution[0], HumanMessage) and not isinstance(messages_for_execution[0].content, list):
                # ... (修正 HumanMessage 以包含圖片的邏輯不變) ...
                 print("   修正執行階段的初始 HumanMessage 以包含圖片...")
                 try:
                     # Re-read image and create multi-modal content
                     with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                     mime_type="image/png" # Re-detect or use default
                     file_ext = os.path.splitext(initial_image_path)[1].lower()
                     if file_ext in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
                     elif file_ext == '.gif': mime_type = 'image/gif'
                     elif file_ext == '.webp': mime_type = 'image/webp'

                     initial_human_content = [
                         {"type": "text", "text": initial_user_text}, # Use the extracted text
                         {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}
                     ]
                     messages_for_execution[0] = HumanMessage(content=initial_human_content)
                 except Exception as img_read_err:
                     print(f"   !! 無法讀取或編碼初始圖片用於執行階段: {img_read_err}")
                     # Proceed with text-only if image fails during execution prep


            # --- 調用 LLM 執行下一步 ---
            execution_response = None
            try:
                # --- PRUNE MESSAGES ---
                pruned_messages_for_llm = messages_for_execution # 預設不剪枝

                if mcp_name == "rhino":
                    max_interactions_for_rhino_pruning = MAX_RECENT_INTERACTIONS_DEFAULT
                    if is_first_execution_after_plan:
                        max_interactions_for_rhino_pruning = 2
                        print(f"    為 Rhino 首次執行調用，設定 max_interactions_for_pruning={max_interactions_for_rhino_pruning} (保留初始請求、計劃和少量近期互動)。")
                    else:
                        print(f"    為 Rhino 非首次執行調用，使用預設歷史記錄交互數量: {max_interactions_for_rhino_pruning}")
                    
                    print(f"  Rhino: 準備執行 LLM 調用，原始待處理消息數: {len(messages_for_execution)}")
                    pruned_messages_for_llm = _prune_messages_for_llm(messages_for_execution, max_interactions_for_rhino_pruning)
                else: # 對於 revit, pinterest, osm 等其他 MCP
                    print(f"  {mcp_name.upper()}: 不執行訊息剪枝。原始待處理消息數: {len(messages_for_execution)}")
                    # pruned_messages_for_llm 已設為 messages_for_execution (不剪枝)

                print(f"  剪枝後/處理後傳遞給 LLM 的消息數: {len(pruned_messages_for_llm)}")
                
                execution_response = await call_llm_with_tools(pruned_messages_for_llm, all_tools_for_execution, active_execution_prompt)

            finally:
                print(f"     執行 LLM 調用結束，等待 {RPM_DELAY} 秒...")
                await asyncio.sleep(RPM_DELAY)
                print("     等待結束。")

            # --- 更新連續空響應計數器 ---
            new_consecutive_responses = 0 # Reset by default
            if isinstance(execution_response, AIMessage):
                has_tool_calls = hasattr(execution_response, 'tool_calls') and execution_response.tool_calls
                has_content = execution_response.content is not None and execution_response.content.strip() != ""
                if has_tool_calls:
                    new_consecutive_responses = 0 # Corrected Indentation
                    print(f"  LLM 返回 {len(execution_response.tool_calls)} 個工具調用，重置連續文本響應計數器為 0。")
                elif has_content:
                    # Includes error messages, completion messages, etc.
                    new_consecutive_responses = 0 # Corrected Indentation
                    print(f"  LLM 返回帶有內容的文本消息 ('{execution_response.content[:50]}...')，重置連續文本響應計數器為 0。")
                else: # No tool calls, no content (empty string or None)
                    new_consecutive_responses = current_consecutive_responses + 1 # Corrected Indentation
                    print(f"  LLM 返回空內容且無工具調用，遞增連續文本響應計數器為 {new_consecutive_responses}。")
            else: # Not an AIMessage (e.g., internal error in call_llm_with_tools returned something else)
                new_consecutive_responses = 0
                print(f"  最終返回非 AIMessage 類型 ({type(execution_response).__name__})，重置連續文本響應計數器為 0。") # Corrected Indentation

            # --- 檢查計數器閾值 ---
            task_complete_due_to_counter = False
            messages_to_return = [] # Initialize list for messages to add to state this turn
            if new_consecutive_responses >= 3:
                print(f"  已連續收到 {new_consecutive_responses} 次無效響應，將標記任務完成。") # Corrected Indentation
                task_complete_due_to_counter = True
                error_msg = f"[系統錯誤：連續 {new_consecutive_responses} 次未能生成有效工具調用或完成消息，任務強制終止。]" # Corrected Indentation
                # Append the problematic response if it exists and isn't the error message itself
                if execution_response and (not isinstance(execution_response, AIMessage) or execution_response.content != error_msg): # Corrected Indentation
                    messages_to_return.append(execution_response) # Corrected Indentation
                messages_to_return.append(AIMessage(content=error_msg)) # Add the termination message
            elif execution_response: # If counter not exceeded, add the valid response from LLM # Corrected Indentation
                messages_to_return.append(execution_response) # Corrected Indentation

            # --- 返回執行結果 ---
            return_dict = {
                "messages": messages_to_return,
                "consecutive_llm_text_responses": new_consecutive_responses,
                "last_executed_node": f"{mcp_name}_agent", # 更新執行的節點名
                "rhino_screenshot_counter": current_rhino_screenshot_counter # Pass back updated counter
            }
            if task_complete_due_to_counter:
                return_dict["task_complete"] = True # Mark task complete if counter triggered

            return return_dict

    except Exception as e:
        print(f"!! 執行 {mcp_name.upper()} Agent 節點時發生外部錯誤: {e}")
        traceback.print_exc()
        # Return error message and reset counter
        # {{ edit_2 }}
        return {"messages": [AIMessage(content=f"執行 {mcp_name} Agent 時發生外部錯誤: {e}")], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent_error", "rhino_screenshot_counter": current_rhino_screenshot_counter}
        # {{ end_edit_2 }}

# --- 具體的 Agent Nodes (添加 OSM) ---
async def call_revit_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    return await agent_node_logic(state, config, "revit")

async def call_rhino_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    return await agent_node_logic(state, config, "rhino")

# --- 新增 Pinterest Agent Node ---
async def call_pinterest_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    return await agent_node_logic(state, config, "pinterest")

# --- 新增 OSM Agent Node ---
async def call_osm_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    return await agent_node_logic(state, config, "osm")

# --- Tool Executor Node (保持不變) ---
async def agent_tool_executor(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """執行 Agent 請求的工具調用 (使用 print)。"""
    print("--- 執行 Agent 工具節點 ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("  最後消息沒有工具調用，跳過。")
        # {{ edit_1 }}
        return {"last_executed_node": "agent_tool_executor_skipped"}
        # {{ end_edit_1 }}

    target_mcp = state.get("target_mcp")
    if not target_mcp:
         error_msg = "錯誤：狀態中缺少 'target_mcp'，無法執行工具。"
         print(f"  !! {error_msg}")
         error_tool_messages = [ ToolMessage(content=error_msg, tool_call_id=tc.get("id"), name=tc.get("name", "unknown_tool")) for tc in last_message.tool_calls ]
         # {{ edit_2 }}
         return {"messages": error_tool_messages, "last_executed_node": "agent_tool_executor_error"}
         # {{ end_edit_2 }}

    print(f"  目標 MCP: {target_mcp}")
    try:
        mcp_tools = await get_mcp_tools(target_mcp)
        all_tools_for_execution = mcp_tools + LOCAL_TOOLS
        print(f"  使用 {len(all_tools_for_execution)} 個總工具 ({target_mcp} MCP: {len(mcp_tools)}, Local: {len(LOCAL_TOOLS)})。")
        tool_messages = await execute_tools(last_message, all_tools_for_execution) # 移除 state 參數
        print(f"  工具執行完成，返回 {len(tool_messages)} 個 ToolMessage。")
        # {{ edit_3 }}
        return {"messages": tool_messages, "last_executed_node": "agent_tool_executor"}
        # {{ end_edit_3 }}
    except Exception as e:
        print(f"!! 執行 Agent 工具節點時發生錯誤: {e}")
        traceback.print_exc()
        error_msg = f"執行工具時出錯: {e}"
        error_tool_messages = [ ToolMessage(content=error_msg, tool_call_id=tc.get("id"), name=tc.get("name", "unknown_tool")) for tc in last_message.tool_calls ]
        # {{ edit_4 }}
        return {"messages": error_tool_messages, "last_executed_node": "agent_tool_executor_error"}
        # {{ end_edit_4 }}

# --- Fallback Agent Node ---
async def call_fallback_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """調用補救 LLM 嘗試恢復流程。"""
    print("--- 執行 Fallback Agent 節點 ---")
    current_messages = state['messages']
    target_mcp = state.get("target_mcp") # Needed to get tools for binding

    if not target_mcp:
         print("  !! Fallback Agent 錯誤：狀態中缺少 'target_mcp'。")
         return {"messages": [AIMessage(content="[FALLBACK_ERROR] Missing target MCP in state.")]}

    # 提取相關歷史記錄用於提示
    plan_message = next((msg for msg in reversed(current_messages) if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)), None)
    plan_content_for_prompt = ""
    if plan_message and isinstance(plan_message.content, str):
        # MODIFIED: Use the full plan content for the prompt
        plan_content_for_prompt = plan_message.content.strip()
        print(f"  提取到完整計劃內容 (用於 Fallback Prompt): {plan_content_for_prompt[:500]}...")

    # 只取最近幾條消息 + 計劃 (計劃已單獨處理，這裡只取非計劃的近期消息)
    # MODIFIED: _prune_messages_for_llm now only gets recent *non-plan* messages if plan is found
    # Or, we can build the history string more explicitly. Let's build it explicitly for clarity.
    
    history_items = []
    # Add the initial human message if available (usually the first)
    if current_messages and isinstance(current_messages[0], HumanMessage):
        first_human_msg = current_messages[0]
        # Summarize the initial human message if it's the one with the image content list
        if isinstance(first_human_msg.content, list):
            text_part = ""
            for item in first_human_msg.content:
                 if isinstance(item, dict) and item.get("type") == "text":
                      text_part = item.get("text", "")
                      break
            history_items.append(f"初始用戶請求 (HumanMessage): {text_part[:300]}...") # Summarize initial request text
        else:
             history_items.append(f"初始用戶請求 (HumanMessage): {str(first_human_msg.content)[:300]}...") # Summarize initial request string

    # Add the plan message's full content (already extracted above)
    if plan_content_for_prompt:
         history_items.append(f"\n---\n完整目標階段計劃 (AIMessage):\n{plan_content_for_prompt}\n---")


    # Add recent messages (excluding the initial human message and the plan message if they are at the end)
    # Let's grab the last N messages, but skip the first if it's the initial human, and skip the last if it's the plan message itself.
    messages_for_recent_history = current_messages[1:] # Skip the first message assuming it's the initial Human
    if messages_for_recent_history and plan_message and id(messages_for_recent_history[-1]) == id(plan_message):
         messages_for_recent_history = messages_for_recent_history[:-1] # Skip the plan message if it's the last

    # Get the last few relevant messages (e.g., last 5-7 interactions)
    max_recent = 7 # Limit recent history to avoid overwhelming the LLM
    recent_messages_to_summarize = messages_for_recent_history[-max_recent:]


    for msg in recent_messages_to_summarize:
        msg_summary = f"{type(msg).__name__}: "
        if isinstance(msg.content, str):
            msg_summary += f"{msg.content[:500]}..." if len(msg.content) > 500 else msg.content
        elif isinstance(msg.content, list):
            # Summarize list content (e.g., tool message with file path)
            summary_parts = []
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    summary_parts.append(item.get("text", "")[:100] + "...")
                elif isinstance(item, str): # Handle ToolMessage content which might be JSON string or simple string
                    summary_parts.append(item[:100] + "...")
                else:
                    summary_parts.append(f"[{type(item).__name__} content]")
            msg_summary += " | ".join(summary_parts)
        elif hasattr(msg, 'tool_calls') and msg.tool_calls: # Check for tool_calls attribute
            # Summarize tool calls
            tool_call_summaries = []
            for tc in msg.tool_calls:
                 tool_call_summaries.append(f"ToolCall(name={tc.get('name', 'N/A')}, args={str(tc.get('args', {}))[:100]}...)")
            msg_summary += f"ToolCalls: {'; '.join(tool_call_summaries)}"

        history_items.append(msg_summary)

    # Join the history items into a single string for the prompt
    relevant_history_str = "\n".join(history_items)


    prompt_content = FALLBACK_PROMPT.content.format(relevant_history=relevant_history_str)
    fallback_system_message = SystemMessage(content=prompt_content)
    print(f"  Fallback Agent Prompt (Partial Preview):\n{prompt_content[:1000]}...") # Print a longer preview

    original_fallback_response = None
    fallback_response_to_return = None
    mcp_tools_raw = [] # Define outside try for access in parsing block

    try:
        # 獲取工具以供綁定（補救 LLM 也需要知道可用工具）
        mcp_tools_raw = await get_mcp_tools(target_mcp) # Assign to mcp_tools_raw
        if not mcp_tools_raw:
             print(f"  !! Fallback Agent 警告：未找到 {target_mcp} 工具！")
        
        # --- 使用輔助函數準備 Gemini 兼容的工具 ---
        gemini_compatible_fallback_tools = _prepare_gemini_compatible_tools(mcp_tools_raw)

        # 使用 agent_llm (Gemini)
        fallback_llm = agent_llm 
        llm_with_tools = fallback_llm.bind_tools(gemini_compatible_fallback_tools) # Bind corrected tools
        llm_configured = llm_with_tools.with_config({"callbacks": None})

        messages_for_llm_invoke = [fallback_system_message]
        # Add a neutral HumanMessage to ensure the 'contents' field is not empty
        # when the SystemMessage is potentially moved to 'system_instruction' by LangChain.
        # This message also serves as a conversational turn for the LLM to respond to.
        # Using "." is a common minimal prompt to trigger a response based on system instructions.
        messages_for_llm_invoke.append(HumanMessage(content="."))

        # original_fallback_response = await llm_configured.ainvoke([fallback_system_message]) # OLD
        original_fallback_response = await llm_configured.ainvoke(messages_for_llm_invoke) # NEW
        print(f"  Fallback Agent 原始響應: {original_fallback_response}")
        
        fallback_response_to_return = original_fallback_response # Default


        # --- Reinstated: Process fallback_response to extract tool_calls from content if necessary ---
        if isinstance(original_fallback_response, AIMessage) and \
           isinstance(original_fallback_response.content, str) and \
           not original_fallback_response.tool_calls: 
            
            content_str = original_fallback_response.content.strip()
            is_potential_json_tool_call = False
            if (content_str.startswith('{') and content_str.endswith('}')):
                 is_potential_json_tool_call = True
            elif content_str.startswith('```json'):
                 match = re.match(r'^```json\s*(\{.*?\})\s*```$', content_str, re.DOTALL | re.IGNORECASE)
                 if match:
                     content_str = match.group(1).strip()
                     is_potential_json_tool_call = True
                 else:
                     if "tool_calls" in content_str and ("recipient_name" in content_str or "name" in content_str) : # Added "name"
                          cleaned_md_json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', original_fallback_response.content.strip(), flags=re.IGNORECASE)
                          if cleaned_md_json_str.strip().startswith('{'):
                              content_str = cleaned_md_json_str.strip()
                              is_potential_json_tool_call = True
            
            if is_potential_json_tool_call:
                try:
                    parsed_json = json.loads(content_str)
                    if isinstance(parsed_json, dict) and "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
                        processed_tool_calls = []
                        for tc_orig in parsed_json["tool_calls"]:
                            if isinstance(tc_orig, dict):
                                tc = tc_orig.copy() 
                                tool_name_to_set = None
                                tool_args_to_set = tc.get("parameters", tc.get("args", {}))
                                raw_name = tc.get("recipient_name", tc.get("name"))

                                if raw_name:
                                    func_name_part = raw_name
                                    if raw_name.startswith("functions."):
                                        func_name_part = raw_name.split("functions.", 1)[1]
                                    
                                    found_tool_match = False
                                    # Use mcp_tools_raw which contains the original BaseTool objects
                                    for t_obj in mcp_tools_raw: 
                                        if t_obj.name == func_name_part: 
                                            tool_name_to_set = t_obj.name
                                            found_tool_match = True
                                            break
                                        if t_obj.name.endswith(f"_{func_name_part}"): 
                                            tool_name_to_set = t_obj.name
                                            found_tool_match = True
                                            break
                                    if not found_tool_match:
                                         print(f"  Fallback Agent: Could not reliably map name '{raw_name}' to a known tool. Using '{func_name_part}'.")
                                         tool_name_to_set = func_name_part
                                else:
                                    print(f"  Fallback Agent: Tool call missing 'recipient_name' or 'name': {tc_orig}")
                                    continue 
                                
                                new_tc_entry = {
                                    "name": tool_name_to_set,
                                    "args": tool_args_to_set,
                                    "id": tc.get("id", str(uuid.uuid4()))
                                }
                                processed_tool_calls.append(new_tc_entry)
                            
                        if processed_tool_calls:
                             placeholder_content = "[Fallback agent initiated tool call via content parsing.]"
                             fallback_response_to_return = AIMessage(
                                 content=placeholder_content, 
                                 tool_calls=processed_tool_calls,
                                 id=original_fallback_response.id if original_fallback_response else str(uuid.uuid4()), 
                                 additional_kwargs=original_fallback_response.additional_kwargs if original_fallback_response else {},
                                 response_metadata=original_fallback_response.response_metadata if original_fallback_response else {},
                                 # tool_call_chunks should be fine as None/default if not streaming
                             )
                             print(f"  Fallback Agent: Reconstructed AIMessage with tool_calls attribute: {fallback_response_to_return.tool_calls} and content: '{placeholder_content}'")
                        else:
                            print("  Fallback Agent: Parsed JSON from content, but 'tool_calls' list was empty or malformed after processing.")
                    # else:
                        # print(f"  Fallback Agent: Content was JSON, but not in expected tool_calls format. Parsed: {json.dumps(parsed_json, indent=2)}")


                except json.JSONDecodeError:
                    print(f"  Fallback Agent: Content looked like JSON for tool call but failed to parse: {content_str[:200]}...")
                except Exception as e_proc:
                    print(f"  Fallback Agent: Error processing content for tool_calls: {e_proc} on content {content_str[:200]}")
        # --- END Reinstated Parsing ---


    except Exception as e:
        print(f"!! Fallback Agent 調用 LLM 或解析時發生錯誤: {e}") # Modified error message
        traceback.print_exc()
        # Ensure fallback_response_to_return is an AIMessage
        if not isinstance(fallback_response_to_return, AIMessage):
            fallback_response_to_return = AIMessage(content=f"[FALLBACK_LLM_ERROR_OR_PARSING] {e}")
        else: # If it was already an AIMessage (e.g. from LLM and parsing failed later), append error
            fallback_response_to_return.content += f" [Error during post-processing: {e}]"

    finally:
        # 短暫等待，避免速率限制
        await asyncio.sleep(RPM_DELAY / 2) # Shorter delay for fallback?
        print("     Fallback Agent 等待結束。")

    return {"messages": [fallback_response_to_return] if fallback_response_to_return else [], "last_executed_node": "fallback_agent"}

# =============================================================================
# Conditional Edge Logic (修改 should_continue 處理 task_complete)
# =============================================================================
def should_continue(state: MCPAgentState) -> str:
    """確定是否繼續處理請求、調用工具、調用補救或結束。"""
    print("--- 判斷是否繼續 ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None
    target_mcp = state.get("target_mcp", "unknown")
    last_node = state.get("last_executed_node")

    # --- 優先檢查 task_complete 標誌 (通常由 agent_node_logic 中的工具結果或連續錯誤觸發) ---
    if state.get("task_complete"):
        print(f"  檢測到 task_complete 標誌 (可能來自工具或連續錯誤) -> end")
        return END

    if not last_message:
        print("  消息列表為空 -> end")
        return END

    # --- 檢查 AI 是否請求工具調用 (來自任何 Agent，包括 Fallback) ---
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"  AI請求工具 ({len(last_message.tool_calls)}個從 {last_node}) -> agent_tool_executor")
        return "agent_tool_executor" 

    # --- 處理計劃生成 (通常由 primary agent 在沒有計劃時觸發) ---
    if isinstance(last_message, AIMessage) and isinstance(last_message.content, str) and last_message.content.strip().startswith(PLAN_PREFIX):
        is_actual_plan = "無法為您的請求制定計劃" not in last_message.content and "調用規劃 LLM 時發生錯誤" not in last_message.content
        if is_actual_plan:
            if last_node and (last_node.endswith("_agent") or last_node.endswith("_planner")):
                 print(f"  最後消息是新生成的計劃 (來自 {last_node}) -> 返回 {target_mcp}_agent 執行第一步")
                 if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
                     return f"{target_mcp}_agent"
                 else:
                     print(f"  警告: 無效的 target_mcp ('{target_mcp}') 在計劃生成後。無法返回 Agent -> end")
                     return END 
            else: 
                 print(f"  !! 錯誤: 計劃意外來自非 Agent 節點 ({last_node}) -> end (異常)")
                 return END
        else: 
             print(f"  最後消息是計劃生成錯誤 ('{last_message.content[:50]}...') -> end") 
             return END

    # --- 檢查是否為工具執行結果 (ToolMessage) ---
    if isinstance(last_message, ToolMessage):
        print(f"  最後消息是 ToolMessage (來自工具 '{last_message.name}') -> 返回 {target_mcp}_agent 處理結果")
        if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
            return f"{target_mcp}_agent" 
        else:
            print(f"  警告: 無效的 target_mcp ('{target_mcp}')，無法返回 Agent 處理工具結果 -> end")
            return END

    # --- 處理 AIMessage (非計劃，且沒有 tool_calls) ---
    if isinstance(last_message, AIMessage):
        # 確保 content_str 是實際的字串，如果 content 為 None，則預設為空字串以便安全處理
        raw_content = last_message.content
        content_str = str(raw_content).lower() if raw_content is not None else ""

        # --- 處理 Fallback Agent 的輸出 (沒有 tool_calls attribute) ---
        if last_node == "fallback_agent":
            fallback_end_keywords = [
                "[fallback_cannot_recover]", "[fallback_error]", "[fallback_llm_error]",
                "[fallback_llm_error_or_parsing]", "[fallback_confirmed_completion]",
            ]
            if any(keyword in content_str for keyword in fallback_end_keywords):
                if "[fallback_confirmed_completion]" in content_str:
                    print(f"  檢測到 Fallback Agent 確認任務成功完成 ('{content_str[:50]}...') -> end")
                else:
                    print(f"  檢測到 Fallback Agent 明確的失敗/無法恢復消息 ('{content_str[:50]}...') -> end")
                return END
            else:
                print(f"  !! 錯誤: Fallback Agent ({last_node}) 輸出非工具/非明確結束信號的 AIMessage ('{content_str[:50]}...') -> end (異常)")
                return END

        # --- 處理來自 主要 Agent / Planner 的 AIMessage ---
        if last_node and (last_node.endswith("_agent") or last_node.endswith("_planner")):
            # 1. 檢查主要 Agent/Planner 的完成關鍵字
            primary_agent_completion_keywords = [ "全部任務已完成", "圖片搜索和下載完成", "地圖截圖已完成", ]
            if any(keyword in content_str for keyword in primary_agent_completion_keywords):
                print(f"  檢測到主要 Agent/Planner ({last_node}) 的完成消息 ('{content_str[:50]}...'). 路由到 fallback_agent 進行驗證。")
                return "fallback_agent"

            # 2. 檢查主要 Agent/Planner 的內容是否為空
            #    (沒有 tool_calls 的情況已在最前面處理)
            if not content_str.strip(): # 如果內容為空或僅包含空白字符
                print(f"  來自主要 Agent/Planner ({last_node}) 的 AIMessage 內容為空或僅空白。路由到 fallback_agent。")
                return "fallback_agent"
            
            # 3. 如果內容非空且不是完成關鍵字，則是主要 Agent/Planner 的中間步驟文本。
            #    路由回主要 Agent 繼續其自身邏輯。
            print(f"  來自主要 Agent/Planner ({last_node}) 的中間文本 AIMessage ('{content_str[:50]}...'). 路由回 {target_mcp}_agent。")
            if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
                return f"{target_mcp}_agent"
            else: 
                print(f"  警告: 無效的 target_mcp ('{target_mcp}') for primary agent AIMessage. 路由到 fallback_agent。")
                return "fallback_agent"

        # --- 處理來自 agent_tool_executor 的 AIMessage ---
        # (這通常是在 agent_node_logic 處理 ToolMessage 後生成的文本消息，
        #  例如 "screenshot saved at X", "Pinterest download complete", "OSM map ready")
        if last_node == "agent_tool_executor":
            # 這類消息是資訊性的。主要 Agent 需要看到它們才能繼續執行計劃。
            # 如果這裡的消息為空，也應該路由到 fallback。
            if not content_str.strip(): # 如果內容為空或僅包含空白字符
                print(f"  來自 agent_tool_executor 的 AIMessage 內容為空或僅空白。路由到 fallback_agent。")
                return "fallback_agent"

            print(f"  來自 agent_tool_executor 的 AIMessage (工具結果處理後的信息) ('{content_str[:50]}...'). 路由回 {target_mcp}_agent。")
            if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
                return f"{target_mcp}_agent"
            else:
                print(f"  警告: 無效的 target_mcp ('{target_mcp}') for agent_tool_executor AIMessage. 路由到 fallback_agent。")
                return "fallback_agent"

        # --- 其他 AIMessage 的捕獲 ---
        # (例如，來自未知節點，或以上邏輯未能覆蓋的情況)
        print(f"  來自節點 '{last_node}' 的無法分類的 AIMessage (無工具、非計劃) ('{content_str[:50]}...'). 路由到 fallback_agent。")
        return "fallback_agent"

    # --- 其他意外情況 ---
    elif isinstance(last_message, HumanMessage):
        print("  在流程中意外出現 HumanMessage (非初始請求) -> end (異常)")
        return END
    else:
        print(f"  未知的最後消息類型 ({type(last_message).__name__}) 或無法處理的狀態 -> end")
        return END

# =============================================================================
# 建立和編譯 LangGraph (添加 OSM 節點和邊)
# =============================================================================
workflow = StateGraph(MCPAgentState)
workflow.add_node("router", route_mcp_target)
workflow.add_node("revit_agent", call_revit_agent)
workflow.add_node("rhino_agent", call_rhino_agent)
workflow.add_node("pinterest_agent", call_pinterest_agent)
workflow.add_node("osm_agent", call_osm_agent)
workflow.add_node("agent_tool_executor", agent_tool_executor)
# --- 新增 Fallback Node ---
workflow.add_node("fallback_agent", call_fallback_agent)

workflow.set_entry_point("router")

# --- Router Edges (保持不變) ---
workflow.add_conditional_edges(
    "router",
    lambda x: x.get("target_mcp"),
    {
        "revit": "revit_agent",
        "rhino": "rhino_agent",
        "pinterest": "pinterest_agent",
        "osm": "osm_agent"
        # No default to END here, router should always pick one.
        # If router fails, it defaults to "revit" internally or could be made to go to END.
    }
)

# --- Primary Agent Edges ---
# 由於 should_continue 的邏輯已修改，主要 agent 不再直接連接到 END。
# 它們會請求工具 (agent_tool_executor)，處理工具結果後返回自身，或者如果它們卡住/聲稱完成，
# should_continue 會將它們路由到 fallback_agent。
workflow.add_conditional_edges(
    "revit_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "revit_agent": "revit_agent", # For loop after tool execution if more steps
        "fallback_agent": "fallback_agent", # If stuck or claims completion
        END: END # Only if should_continue returns END for critical errors (e.g. no plan, no message)
    }
)
workflow.add_conditional_edges(
    "rhino_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "rhino_agent": "rhino_agent",
        "fallback_agent": "fallback_agent",
        END: END
    }
)
workflow.add_conditional_edges(
    "pinterest_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "pinterest_agent": "pinterest_agent",
        "fallback_agent": "fallback_agent",
        END: END
    }
)
workflow.add_conditional_edges(
    "osm_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "osm_agent": "osm_agent",
        "fallback_agent": "fallback_agent",
        END: END
    }
)

# --- Fallback Agent Edges ---
workflow.add_conditional_edges(
    "fallback_agent",
    should_continue, # Reuse the same logic
    {
        "agent_tool_executor": "agent_tool_executor", # Fallback succeeded in generating tool call
        "revit_agent": "revit_agent",
        "rhino_agent": "rhino_agent",
        "pinterest_agent": "pinterest_agent",
        "osm_agent": "osm_agent",
        # For now, this setup relies on FALLBACK_PROMPT guiding it to either tool_call or [FALLBACK_CANNOT_RECOVER]
        "fallback_agent": "fallback_agent", # Allows fallback to re-evaluate if it produces text instead of tools/end
        END: END # If should_continue detects explicit fallback failure or other critical errors
    }
)


# --- Tool Executor Edges ---
# After tools are executed, should_continue will route to the correct primary agent
# (revit_agent, rhino_agent, etc.) based on the target_mcp in the state,
# or to fallback_agent if the primary agent then gets stuck.
workflow.add_conditional_edges(
   "agent_tool_executor",
   should_continue, # should_continue correctly routes ToolMessages back to the target_mcp_agent
   {
       "revit_agent": "revit_agent",
       "rhino_agent": "rhino_agent",
       "pinterest_agent": "pinterest_agent",
       "osm_agent": "osm_agent",
       "fallback_agent": "fallback_agent", # This path is less likely if ToolMessage logic in should_continue is robust
                                        # as ToolMessages should go to primary agents.
                                        # However, if a primary agent immediately yields to fallback after a tool, this covers it.
       END: END # If should_continue determines an end condition after tool execution (e.g. task_complete set by tool)
   }
)

graph = workflow.compile()
# --- 修改 Graph Name ---
graph.name = "Router_AgentPlanning_MCP_Agent_V23_FallbackLastStepVerify" # Update version
print(f"LangGraph 編譯完成: {graph.name}")






