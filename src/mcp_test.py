# src/mcp_test.py
import os
import asyncio
import json
import traceback
import sys
import uuid
import base64
import time

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
from langchain_core.tools import BaseTool
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
        "command": "Z:\\miniconda3\\envs\\rhino_mcp\\python",
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
你的任務是：
1.  分析計劃和執行歷史(對於計畫不可跳過，應思考如何通過組合工具和操作來做到這個目標狀態)。
2.  **嚴格遵循提供的計劃。** 你的首要任務是識別並執行計劃中的**第一個具體動作/階段目標**。通常最近的工具執行結果 (`ToolMessage`) 和AI回應表明達成預期成果，代表更前面的階段已經完成只是訊息被省略。請直接下一個未完成目標。
3.  決定達成該目標所需的**第一個具體動作**。
4.  **如果需要調用工具來執行此動作，請必須生成 `tool_calls` 在首位的 AIMessage 以請求該工具調用**。**不要僅用文字描述你要調用哪個工具，而是實際生成工具調用指令。** 一次只生成一個工具調用請求。
5.  嚴格禁止使用 f-string 格式化字串。請使用 `.format()` 或 `%` 進行字串插值。(此為 IronPython 2.7 環境限制)
6.  **仔細參考工具描述或 Mcp 文檔確認函數用法與參數正確性，必須實際生成結構化的工具呼叫指令。**
7.  **多方案管理 (重要):**
    * 當生成多個方案時，**每個方案必須完全獨立**，視為單獨的任務序列處理
    * **方案隔離原則:**
        * **每個方案必須有自己的頂層圖層**，使用 `rs.AddLayer("方案A_描述")` 創建
        * **切換方案前必須隱藏前一方案的圖層**，使用 `rs.LayerVisible("前一方案名", False)`
        * **所有物件必須正確配置到其所屬方案的圖層**，使用 `rs.CurrentLayer("方案X_描述::子圖層")`
        * **完成每個方案後必須截圖**，再開始下一個方案
    * **避免方案間的量體重疊**，可考慮在不同方案間使用座標偏移
8.  **量體生成策略:**
    * **空間操作優先使用布林運算**：使用 `rs.BooleanUnion()`、`rs.BooleanDifference()`、`rs.BooleanIntersection()` 創造複雜形態
    * **善用幾何變換**：使用旋轉、縮放、移動等操作調整物件姿態，創造更豐富的空間層次
    * **避免無效量體**：不要創建過小、位置不合理或對空間表達無貢獻的量體
    * **注意 IronPython 2.7 語法限制**：Rhino 8使用IronPython 2.7，禁止使用Python 3特有語法   
9.  **Rhino 圖層管理 (重要):** 當生成 Rhino 代碼時：
        *   如果當前階段目標**明確要求**在特定圖層上操作，**必須**在相關操作（如創建物件）**之前**包含 `rs.CurrentLayer('目標圖層名稱')` 指令。
        *   如果目標涉及控制圖層可見性（例如，準備截圖），**必須**包含 `rs.LayerVisible('圖層名', True/False)` 指令。
        *   **截圖前的圖層準備：在調用 `capture_focused_view` 進行截圖之前，必須確保只有與當前截圖目標直接相關的圖層是可見的。所有其他不相關的圖層，特別是那些可能遮擋目標視圖的圖層（例如，其他樓層、其他設計方案的頂層圖層、輔助線圖層等），都應使用 `rs.LayerVisible('圖層名', False)` 進行隱藏。**
10. **最終步驟 (Rhino/Revit):**
    *   對於 Rhino/Revit 任務，每當完成一個方案或一個樓層就**必須**要調用 `capture_focused_view` 工具來截取畫面。**僅當消息歷史清楚地表明計劃中的最後階段目標已成功執行**，你才能生成文本回復：`全部任務已完成` 以結束整個任務。
11. 如果當前階段目標不需要工具即可完成（例如，僅需總結信息），請生成說明性的自然語言回應。
12. 若遇工具錯誤，分析錯誤原因 (尤其是代碼執行錯誤)，**嘗試修正你的工具調用參數或生成的代碼**，然後再次請求工具調用。如果無法修正，請報告問題。

**常規執行：對於計劃中的任何後續步驟，如果該步驟需要與 Rhino/Revit 環境互動，你的回應也必須是工具調用。不要用自然語言解釋你要做什麼，直接生成工具調用。**
**關鍵指令：只要下一步是工具操作，你的回應中**必須**包含 Tool Calls 結構。直到錯誤或是處理完最終任務後，才可生成純文字的完成訊息。**""")

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

分析以下**初始使用者請求文本**，並決定最適合處理此請求的代理。
你的回應必須是 'revit', 'rhino', 'pinterest' 或 'osm'。請只回應目標代理的名稱。

初始使用者請求文本：
"{user_request_text}"
"""

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
        # --- 手動構造工具定義列表 (僅處理 MCP 工具) ---
        print("     準備 MCP 工具定義列表，手動修正特殊參數...")
        tools_for_binding = []
        for tool in selected_tools:
            if tool.name == "get_scene_objects_with_metadata":
                # 保留原有的手動宣告
                print(f"     正在為 '{tool.name}' 創建手動 Gemini FunctionDeclaration...")
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
                print(f"     手動定義已創建: {tool.name}")
            elif tool.name == "zoom_to_target" or tool.name == "capture_focused_view":
                print(f"     正在為含bounding_box參數的工具 '{tool.name}' 創建手動 Gemini FunctionDeclaration...")
                # 構建共用的基本屬性
                properties = {
                    "view": {
                        "type": "STRING",
                        "description": "視圖名稱或ID",
                        "nullable": True
                    }
                }
                
                # 根據工具名稱添加特定屬性
                if tool.name == "zoom_to_target":
                    properties.update({
                        "object_ids": {
                            "type": "ARRAY",
                            "description": "要縮放到的對象ID列表",
                            "nullable": True,
                            "items": {"type": "STRING"}
                        },
                        "all_views": {
                            "type": "BOOLEAN",
                            "description": "是否應用於所有視圖",
                            "nullable": True
                        }
                    })
                elif tool.name == "capture_focused_view":
                    properties.update({
                        "projection_type": {
                            "type": "STRING",
                            "description": "投影類型: 'parallel', 'perspective', 'two_point'",
                            "nullable": True
                        },
                        "lens_angle": {
                            "type": "NUMBER",
                            "description": "透視或兩點投影的鏡頭角度",
                            "nullable": True
                        },
                        # --- 新增相機參數定義 ---
                        "camera_position": {
                            "type": "ARRAY",
                            "description": "相機位置的 [x, y, z] 坐標",
                            "nullable": True,
                             "items": {"type": "NUMBER"}
                        },
                        "target_position": {
                             "type": "ARRAY",
                             "description": "目標點的 [x, y, z] 坐標",
                             "nullable": True,
                             "items": {"type": "NUMBER"}
                         },
                         # --- 結束新增 ---
                        "layer": {
                            "type": "STRING",
                            "description": "用於篩選顯示註釋的圖層名稱",
                            "nullable": True
                        },
                        "show_annotations": {
                            "type": "BOOLEAN",
                            "description": "是否顯示物件註釋",
                            "nullable": True
                        },
                        "max_size": {
                            "type": "INTEGER",
                            "description": "截圖的最大尺寸",
                            "nullable": True
                        }
                    })
                
                # 為兩個工具都添加正確的bounding_box結構
                properties["bounding_box"] = {
                    "type": "ARRAY",
                    "description": "邊界框的8個角點坐標 [[x,y,z], [x,y,z], ...]",
                    "nullable": True,
                    "items": {
                        "type": "ARRAY",
                        "items": {
                            "type": "NUMBER"
                        }
                    }
                }
                
                # 創建完整的手動宣告
                manual_declaration = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "OBJECT",
                        "properties": properties
                    }
                }
                
                tools_for_binding.append(manual_declaration)
                print(f"     手動定義已創建: {tool.name}")
            else:
                tools_for_binding.append(tool)
                print(f"     保留標準 MCP BaseTool 對象: {tool.name}")

        # --- 綁定工具到 agent_llm ---
        print("     正在將 MCP 工具 (含手動定義) 綁定到 LLM...")
        llm_with_tools = agent_llm.bind_tools(tools_for_binding)
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


# =============================================================================
# 圖節點 (Graph Nodes)
# =============================================================================

RPM_DELAY = 6.5 # 比 6 秒稍長一點，留點餘裕

# --- Router Node (MODIFIED to handle pinterest) ---
async def route_mcp_target(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """使用 utility_llm 判斷用戶初始請求文本應路由到哪個 MCP (revit, rhino, pinterest)。"""
    print("--- 執行 MCP 路由節點 ---")
    initial_request_text = state.get('initial_request', '')
    if not initial_request_text:
        print("錯誤：狀態中未找到 'initial_request'。默認為 revit。")
        return {"target_mcp": "revit"}

    print(f"  根據初始請求文本路由: '{initial_request_text[:150]}...'")
    prompt = ROUTER_PROMPT.format(user_request_text=initial_request_text)
    try:
        response = await utility_llm.ainvoke([SystemMessage(content=prompt)], config=config)
        route_decision = response.content.strip().lower()
        print(f"  LLM 路由決定: {route_decision}")
        if route_decision in ["revit", "rhino", "pinterest", "osm"]:
            return {"target_mcp": route_decision}
        else:
            print(f"  警告: LLM 路由器的回應無法識別 ('{route_decision}')。預設為 revit。")
            return {"target_mcp": "revit"}
    except Exception as e:
        print(f"  路由 LLM 呼叫失敗: {e}")
        traceback.print_exc()
        return {"target_mcp": "revit"}


# <<< 新增：訊息剪枝輔助函式 >>>
MAX_RECENT_INTERACTIONS_DEFAULT = 5
MAX_RECENT_INTERACTIONS_FORCING = 10

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
    """通用 Agent 節點邏輯：處理特定工具消息 (截圖, Pinterest 下載)，規劃，或執行下一步。"""
    print(f"--- 執行 {mcp_name.upper()} Agent 節點 ---")
    current_messages = list(state['messages'])
    last_message = current_messages[-1] if current_messages else None
    # --- <<< 新增：獲取當前計數器 >>> ---
    current_consecutive_responses = state.get("consecutive_llm_text_responses", 0)
    # new_consecutive_responses = 0 # 預設重置 # <<< 由後續邏輯決定是否重置或遞增
    
    # --- 處理 capture_viewport 返回的文件路徑 (Rhino/Revit) ---
    IMAGE_PATH_PREFIX = "[IMAGE_FILE_PATH]:"
    if isinstance(last_message, ToolMessage) and last_message.name == "capture_focused_view" and last_message.content.startswith(IMAGE_PATH_PREFIX):
        print("  檢測到 capture_viewport 工具返回的文件路徑。")
        image_path = last_message.content[len(IMAGE_PATH_PREFIX):]
        print(f"    文件路徑: {image_path}")
        try:
            if not os.path.exists(image_path):
                 print(f"  !! 錯誤：收到的圖像文件路徑不存在: {image_path}")
                 # 返回錯誤訊息，但不結束任務
                 return {
                     "messages": [AIMessage(content=f"截圖文件未找到: {image_path}。請檢查 Rhino 端保存路徑。")],
                     "saved_image_path": None, # 清除路徑
                     "saved_image_data_uri": None, # 清除URI
                     "task_complete": False, # 不再將截圖視為任務完成
                     "consecutive_llm_text_responses": 0
                 }

            with open(image_path, "rb") as f: image_bytes = f.read()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png"
            if file_extension == ".jpeg" or file_extension == ".jpg": mime_type = "image/jpeg"
            data_uri = f"data:{mime_type};base64,{base64_data}"
            print(f"    推斷 MIME 類型: {mime_type}")

            # 返回帶有圖片資訊的 AIMessage，但不標記 task_complete
            # LLM 會根據這個消息和計劃決定下一步
            return {
                "messages": [AIMessage(content=f"已成功截取畫面並保存至 {image_path}。請繼續執行計劃的後續步驟。")],
                "saved_image_path": image_path,
                "saved_image_data_uri": data_uri,
                "task_complete": False, # 重要的修改：不再將截圖視為任務完成
                "consecutive_llm_text_responses": 0 # 重置計數器
            }
        except Exception as img_proc_err:
            print(f"  !! 處理截圖文件 '{image_path}' 或編碼時出錯: {img_proc_err}")
            traceback.print_exc()
            # 返回錯誤訊息，但不結束任務
            return {
                "messages": [AIMessage(content=f"處理截圖文件 '{image_path}' 時失敗: {img_proc_err}。請檢查錯誤並重試。")],
                "task_complete": False, # 不再將截圖視為任務完成
                "consecutive_llm_text_responses": 0
            }

    # --- 處理 OSM 返回的文件路徑 (新增) ---
    OSM_IMAGE_PATH_PREFIX = "[OSM_IMAGE_PATH]:"
    if isinstance(last_message, ToolMessage) and last_message.name == "geocode_and_screenshot" and last_message.content.startswith(OSM_IMAGE_PATH_PREFIX):
        print("  檢測到 geocode_and_screenshot 工具返回的文件路徑。")
        image_path = last_message.content[len(OSM_IMAGE_PATH_PREFIX):]
        print(f"    OSM 文件路徑: {image_path}")
        try:
            if not os.path.exists(image_path):
                 print(f"  !! 錯誤：收到的 OSM 圖像文件路徑不存在: {image_path}")
                 return {"messages": [AIMessage(content=f"地圖處理完畢，但截圖文件未找到: {image_path}")], "task_complete": True, "consecutive_llm_text_responses": 0}

            with open(image_path, "rb") as f: image_bytes = f.read()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png"
            if file_extension == ".jpeg" or file_extension == ".jpg": mime_type = "image/jpeg"
            data_uri = f"data:{mime_type};base64,{base64_data}"
            print(f"    推斷 MIME 類型: {mime_type}")

            return {
                "messages": [AIMessage(content=f"地圖截圖已完成。\n截圖已保存至 {image_path}。")],
                "saved_image_path": image_path,
                "saved_image_data_uri": data_uri,
                "task_complete": True,
                "consecutive_llm_text_responses": 0 # 重置計數器
            }
        except Exception as img_proc_err:
            print(f"  !! 處理 OSM 截圖文件 '{image_path}' 或編碼時出錯: {img_proc_err}")
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"地圖截圖已完成，但處理文件 '{image_path}' 時失敗: {img_proc_err}")], "task_complete": True, "consecutive_llm_text_responses": 0} # 重置計數器

    # --- 處理 capture_viewport 返回的錯誤消息 ---
    elif isinstance(last_message, ToolMessage) and last_message.name == "capture_focused_view" and last_message.content.startswith("[Error: Viewport Capture Failed]:"):
         error_msg = last_message.content
         print(f"  檢測到 capture_viewport 工具返回錯誤: {error_msg}")
         return {"messages": [AIMessage(content=f"任務因截圖錯誤而終止: {error_msg}")], "task_complete": True, "consecutive_llm_text_responses": 0} # 重置計數器

    # --- 處理 pinterest_search_and_download 返回的文件路徑列表 (MODIFIED) ---
    if isinstance(last_message, ToolMessage) and last_message.name == "pinterest_search_and_download":
        print("  檢測到 pinterest_search_and_download 工具返回。")
        content = last_message.content
        saved_paths_list = None
        try:
            # Try to parse the content as JSON which should contain the list
            data = json.loads(content)
            if isinstance(data, dict) and "downloaded_paths" in data and isinstance(data["downloaded_paths"], list):
                saved_paths_list = data["downloaded_paths"]
                print(f"    成功解析到 {len(saved_paths_list)} 個下載路徑。")
            else:
                 print(f"    ToolMessage content is JSON but missing 'downloaded_paths' list: {content[:200]}...")
        except json.JSONDecodeError:
            # If it's not JSON, it might be an error message or plain text
            print(f"    ToolMessage content is not JSON (likely text output or error): {content[:200]}...")
        except Exception as e:
            print(f"    解析 Pinterest ToolMessage content 時出錯: {e}")

        if saved_paths_list:
             # --- MODIFIED: Store the list and mark task complete ---
             print(f"    將下載的路徑列表存儲到狀態中。")
             # Store the list of paths. Assume Pinterest is the final step.
             # We still store the last path in the single fields for potential compatibility
             # or quick access, but the primary source is the list.
             last_path = saved_paths_list[-1] if saved_paths_list else None
             data_uri = None
             if last_path:
                 try:
                     with open(last_path, "rb") as f: image_bytes = f.read()
                     base64_data = base64.b64encode(image_bytes).decode('utf-8')
                     # ... (mime type detection) ...
                     mime_type = "image/png" # Default or detect
                     file_extension = os.path.splitext(last_path)[1].lower()
                     if file_extension == ".jpeg" or file_extension == ".jpg": mime_type = "image/jpeg"
                     elif file_extension == ".gif": mime_type = "image/gif"
                     elif file_extension == ".webp": mime_type = "image/webp"
                     data_uri = f"data:{mime_type};base64,{base64_data}"
                 except Exception as img_proc_err:
                     print(f"    !! 處理最後一個 Pinterest 文件 '{last_path}' 或編碼時出錯: {img_proc_err}")

             return {
                 "messages": [AIMessage(content=f"Pinterest 圖片搜索和下載完成，共找到 {len(saved_paths_list)} 個有效文件。")],
                 "saved_image_path": last_path, # Keep last for reference
                 "saved_image_data_uri": data_uri, # Keep last for reference
                 "saved_image_paths": saved_paths_list, # Store the full list <<< NEW FIELD >>>
                 "task_complete": True, # Assume Pinterest is final step
                 "consecutive_llm_text_responses": 0 # 重置計數器
             }
        else:
             # If parsing failed or no paths found, just pass the message content along
             print("    Pinterest 工具未返回有效路徑列表，任務可能未成功或未找到圖片。")
             # Let should_continue decide the next step based on the text message
             return {"messages": [AIMessage(content=f"Pinterest 任務處理完成，但未找到或處理下載路徑。工具輸出: {content[:200]}...")], "consecutive_llm_text_responses": 0}

    # --- 如果不是處理特定工具返回，則執行正常規劃/執行邏輯 ---
    try:
        # ... (獲取初始消息、圖片路徑的邏輯保持不變) ...
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


        # 獲取 MCP 工具
        mcp_tools = await get_mcp_tools(mcp_name)
        print(f"  獲取了 {len(mcp_tools)} 個 {mcp_name} MCP 工具。")
        if not mcp_tools: print(f"  警告：未找到 {mcp_name} 工具！")
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])

        # --- 為 Rhino/Revit 定義規劃提示模板 ---
        RHINO_PLANNING_PROMPT_TEMPLATE = """你是一位優秀的任務規劃助理，專門為 CAD/BIM 任務制定計劃。
            基於使用者提供的文字請求、可選的圖像以及下方列出的可用工具，生成一個清晰的、**分階段目標**的計劃。

            **重要要求：**
            1.  **量化與具體化:** 對於幾何操作 (Rhino/Revit)，每個階段目標**必須**包含盡可能多的**具體數值、尺寸、座標、角度、數量、距離、方向、或清晰的空間關係描述**。
            2.  **邏輯順序:** 確保階段目標按邏輯順序排列，後續步驟依賴於先前步驟的結果。
            3.  **空間佈局規劃 (Rhino):**
                *   當任務涉及空間配置或多個量體的佈局時，計劃應明確描述這些量體之間的**拓撲關係** (如相鄰、共享面、包含) 和**相對位置** (如A在B的上方，C在D的西側並偏移X單位)。
                *   **空間單元化原則：原則上，每一個獨立的功能空間（例如客廳、單獨的臥室、廚房、衛生間等）都應該規劃為一個獨立的幾何量體。避免使用單一量體代表多個不同的功能空間。為每個規劃生成的獨立空間量體或重要動線元素指定一個有意義的臨時名稱或標識符，並在後續的建模步驟中通過 Rhino 的 `add_object_metadata()` 功能將此名稱賦予對應的 Rhino 物件。**
                *   **圖層規劃 - 初始設定：** 在開始任何建模或創建新的方案/基礎圖層 (如 "方案A", "Floor_1") 之前，**必須**規劃一個步驟：首先獲取當前場景中的所有圖層列表，然後將所有已存在的**頂層圖層**及其子圖層設置為不可見。這樣可以確保在一個乾淨的環境中開始新的設計工作。之後再創建並設置當前工作所需的圖層。
                *   **圖層規劃 - 動線表達與分層 (Rhino):**
                    *   對於**水平動線**（例如走廊、通道），如果需要視覺化，建議規劃使用 Rhino 中的線條 (`rs.AddLine()`) 或非常薄的板狀量體來示意其路徑和寬度。這些水平動線元素**必須**規劃到其所服務的樓層圖層下的**子圖層**中，例如：`Floor_1::Corridors_F1` 或 `Floor_Ground::Horizontal_Circulation`。
                    *   對於**垂直動線**（例如樓梯、坡道、電梯井），則應規劃使用合適的3D量體來表達其佔據的空間和形態。這些垂直動線元素通常規劃到一個獨立的頂層圖層下，例如 `Circulation::Vertical_Core` 或 `Stairs_Elevators`。
                    *   所有動線元素也必須根據其服務的樓層或連接關係，正確地規劃到相應的圖層下。
                *   在進行複雜的空間佈局規劃時，可以先(以文字描述的形式)構思一個2D平面上的關係草圖，標註出各個獨立空間量體和動線的大致位置、尺寸和鄰接關係，然後再將此2D關係轉化為3D建模步驟的規劃。
                *   規劃時需仔細考慮並確保最終生成的**量體數量、各個空間量體的具體位置和尺寸**符合設計意圖和空間邏輯。
            4.  **多方案與多樓層處理 (Rhino):**
                *   如果用戶請求中明確要求"多方案"或"不同選項"，**必須**將每個方案視為一個**獨立的、完整的任務序列**來規劃。
                *   為每個方案指定一個清晰的名稱或標識符 (例如 "方案A_現代風格", "方案B_傳統風格")，並在整個方案的規劃和執行階段中使用此標識。
                *   計劃應清晰地標示每個方案的開始和結束。
                *   **對於包含多個樓層的設計方案，在完成每一樓層的主要建模內容後，應規劃一次詳細的截圖步驟 (參考下方截圖規劃詳細流程)。**
            5.  **造型與形態探索 (Rhino):**
                *   當任務目標涉及'造型探索'、'形態生成'或對現有量體進行'外觀設計'時，規劃階段應積極考慮如何利用布林運算 (如加法、減法、交集) 和幾何變換 (如扭轉、彎曲、陣列、縮放、旋轉) 等高級建模技巧來達成獨特或複雜的「虛、實」幾何形態。在計劃中明確指出預計在哪些步驟使用這些技巧。
                *   造型探索不需要規劃同時展示所有方案的截圖總覽。
            6.  **圖像參考規劃 (若有提供圖像):**
                *   在生成具體的建模計劃之前，**必須**先進行一個詳細的"圖像分析與解讀"階段。
                *   此階段的輸出(文字描述)應包含：觀察到的主要建築體塊組成和它們之間的**空間布局關係**（例如，穿插、並列、堆疊）；估計的整體及主要部分的長、寬、高比例關係；主要的立面特徵（重點是整體形態而非細節雕刻）；可識別的屋頂形式；以及整體呈現的建築風格。
                *   **必須**將上述圖像分析階段得出的觀察結果，轉化為後續 Rhino 建模步驟中的具體參數和操作指導。
            7.  **截圖規劃詳細流程 (Rhino/Revit):**
                *   每當計劃需要截圖時 (例如完成一個樓層、一個設計方案，或應用戶明確要求)，**必須**規劃以下完整步驟：
                    1.  **設定視圖投影模式：** 明確指定是平行投影 (`parallel`)、透視投影 (`perspective`) 還是兩點透視 (`two_point`)。
                    2.  **(可選)調整相機：** 如果需要特定視角（非標準頂視、前視等），規劃設定相機位置 (`camera_position`)、目標點 (`target_position`) 和/或透鏡角度 (`lens_angle`)。
                    3.  **管理圖層可見性 (關鍵步驟)：**
                        a.  規劃獲取當前場景中所有圖層的列表。
                        b.  識別出當前截圖目標**直接相關**的圖層（例如，如果要截取 "Floor_1" 的俯視圖，則相關圖層是 "Floor_1" 及其所有子圖層，如 `Floor_1::Walls`, `Floor_1::Corridors_F1` 等）。
                        c.  規劃遍歷所有圖層，將所有**不屬於**上述直接相關圖層集合的**其他頂層圖層**（及其所有子圖層，通常通過隱藏其頂層母圖層實現）設置為**不可見**。
                        d.  確保所有與當前截圖目標**直接相關**的圖層均設置為**可見**，所有當前可見的目標物件都完整顯示在視圖。
                    4.  **執行截圖：** 規劃調用 `capture_focused_view` 工具。此工具本身具備縮放視圖到目標的功能。
                    **在截圖時，必須確保當前樓層的視圖不被遮擋，尤其是俯視圖。**
                    **對於空間佈局規劃任務，適用於平行投影；適用於對於渲染用建模，適用於兩點透視並配合相機採用人視角；對於造型探索皆適用但需確保造型能夠完美呈現。**
            8.  **目標狀態:** 計劃應側重於**每個階段要達成的目標狀態**，說明該階段完成後場景應有的變化。
                *   **不要在計劃中包含"任務已完成"或"回覆使用者"等描述，這些會在實際執行時由系統自動處理**。

            **rhino提醒: 目前單位是M(公尺)。對於量體配置方案建議使用parallel或perspective從上方俯視；對於渲染用建模建議使用two point perspective從人眼視角截圖**
            這個計劃應側重於**每個階段要達成的目標狀態並包含細節**，而不是具體的工具使用細節。將任務分解成符合邏輯順序及細節的多個階段目標。
            直接輸出這個階段性目標計劃，不要额外的開場白或解釋。

            可用工具如下 ({mcp_name}):
            {tool_descriptions}"""

        # --- 根據 mcp_name 選擇規劃和執行提示 ---
        active_planning_prompt_content = ""
        active_execution_prompt = None

        if mcp_name in ["rhino", "revit"]:
            # 直接使用定義好的模板字串
            active_planning_prompt_content = RHINO_PLANNING_PROMPT_TEMPLATE
            active_execution_prompt = RHINO_AGENT_EXECUTION_PROMPT
        elif mcp_name == "pinterest":
            active_planning_prompt_content = f"""用戶請求使用 Pinterest 進行圖片搜索。
            可用工具 ({mcp_name}):
            - pinterest_search_and_download: {{"description": "Searches Pinterest for images based on a keyword and downloads them. Args: keyword (str), limit (int, optional)."}}
            請制定一個單一步驟計劃來使用 pinterest_search_and_download 工具，目標是根據用戶請求搜索並下載圖片。
            計劃的最終步驟應明確指出調用 `pinterest_search_and_download`。"""
            active_execution_prompt = PINTEREST_AGENT_EXECUTION_PROMPT
        elif mcp_name == "osm":
            active_planning_prompt_content = f"""用戶請求使用 OpenStreetMap 生成地圖截圖。
            可用工具 ({mcp_name}):
            - geocode_and_screenshot: {{"description": "Geocodes an address or uses coordinates to take a screenshot from OpenStreetMap. Args: address (str: address or 'lat,lon')."}}
            請制定一個單一步驟計劃來使用 geocode_and_screenshot 工具，目標是根據用戶請求生成地圖截圖。
            計劃的最終步驟應明確指出調用 `geocode_and_screenshot`。"""
            active_execution_prompt = OSM_AGENT_EXECUTION_PROMPT
        else: # 其他未知 MCP 或未來擴展
            print(f"警告：找不到為 {mcp_name} MCP 定義的特定規劃提示。將使用通用提示。")
            # 可以設定一個非常通用的後備規劃提示，或直接跳過規劃，依賴執行提示
            # 為了安全，如果 active_planning_prompt_content 未被賦值，後續的 format 可能會出錯。
            # 但此處的後備邏輯也會嘗試 format，所以需要確保它有占位符。
            tool_descriptions_for_fallback = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
            active_planning_prompt_content = f"請為使用 {{mcp_name}} 的任務制定計劃。可用工具：\n{{tool_descriptions}}" # 使用雙大括號來轉義，以便後續 .format
            active_execution_prompt = RHINO_AGENT_EXECUTION_PROMPT # 後備執行提示


        # 檢查是否需要規劃
        PLAN_PREFIX = "[目標階段計劃]:\n"
        plan_exists = any(
            isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)
            for msg in current_messages
        )

        messages_to_return = []

        if not plan_exists:
            print(f"  需要為 {mcp_name} 生成執行計劃...")
            
            planning_system_content_final = active_planning_prompt_content # 預設使用已構建好的特定提示
            if mcp_name in ["rhino", "revit"]: # 只有 Rhino/Revit 需要格式化 tool_descriptions
                tool_descriptions_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
                planning_system_content_final = active_planning_prompt_content.format(
                    mcp_name=mcp_name, 
                    tool_descriptions=tool_descriptions_for_prompt
                )
            elif mcp_name not in ["pinterest", "osm"]: # 對於通用後備情況
                 tool_descriptions_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
                 planning_system_content_final = active_planning_prompt_content.format( # active_planning_prompt_content 應該是帶占位符的模板
                     mcp_name=mcp_name,
                     tool_descriptions=tool_descriptions_for_prompt
                 )
            # 對於 Pinterest 和 OSM，active_planning_prompt_content 已經是 f-string 渲染後的結果（mcp_name已填充，工具描述硬編碼）
            # 無需再次格式化。
            
            planning_system_message = SystemMessage(content=planning_system_content_final)
            print(f"    為 {mcp_name} 構造了規劃 SystemMessage")

            # --- 構造規劃 HumanMessage (保持不變) ---
            planning_human_content = [{"type": "text", "text": initial_user_text}]
            if has_input_image:
                try:
                    with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    mime_type="image/png" # Assume png, adjust if needed
                    planning_human_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                    })
                    print("    已將初始圖片添加到規劃 HumanMessage 中。")
                except Exception as img_read_err:
                    print(f"    !! 無法讀取或編碼初始圖片: {img_read_err}")
                    planning_human_content = initial_user_text

            if not isinstance(planning_human_content, list):
                 planning_human_content = [{"type": "text", "text": str(planning_human_content)}]
            planning_human_message_user_input = HumanMessage(content=planning_human_content)

            print(f"     正在調用 LLM ({agent_llm.model}) 進行規劃...")
            plan_message = None
            try:
                planning_llm_no_callbacks = agent_llm.with_config({"callbacks": None})
                planning_response = await planning_llm_no_callbacks.ainvoke(
                    [planning_system_message, planning_human_message_user_input]
                )
                if isinstance(planning_response, AIMessage) and planning_response.content:
                    plan_content = PLAN_PREFIX + planning_response.content.strip()
                    plan_message = AIMessage(content=plan_content)
                    print(f"  生成階段目標計劃:\n------\n{plan_content}\n------")
                else:
                    print("  !! LLM 未能生成有效計劃。")
                    plan_message = AIMessage(content="抱歉，無法為您的請求制定計劃。")
            except Exception as planning_err:
                 print(f"  !! 調用 LLM 進行規劃時發生錯誤: {planning_err}")
                 traceback.print_exc()
                 error_message = f"調用規劃 LLM 時出錯: {planning_err}"
                 plan_message = AIMessage(content=error_message)
            finally:
                print(f"     規劃 LLM 調用結束，等待 {RPM_DELAY} 秒...")
                await asyncio.sleep(RPM_DELAY)
                print("     等待結束。")

            if plan_message: messages_to_return.append(plan_message)
        else:
             print("  檢測到已有計劃，跳過規劃步驟。")

        # 調用 call_llm_with_tools 執行下一步
        messages_for_execution = current_messages + messages_to_return
        if has_input_image and isinstance(messages_for_execution[0], HumanMessage) and not isinstance(messages_for_execution[0].content, list):
             # ... (修正 HumanMessage 包含圖片的邏輯保持不變) ...
             print("   修正執行階段的初始 HumanMessage 以包含圖片...")
             try:
                 with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                 img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                 mime_type="image/png" # Assume png
                 initial_human_content = [
                     {"type": "text", "text": initial_user_text},
                     {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}
                 ]
                 messages_for_execution[0] = HumanMessage(content=initial_human_content)
             except Exception as img_read_err:
                 print(f"   !! 無法讀取或編碼初始圖片用於執行階段: {img_read_err}")


        execution_response = None
        plan_was_generated_this_run = bool(messages_to_return) # 檢查本次運行是否生成了計劃

        try:
            # --- PRUNE MESSAGES before main execution call ---
            print(f"  準備執行 LLM 調用，原始待處理消息數: {len(messages_for_execution)}")
            pruned_messages_for_llm = _prune_messages_for_llm(messages_for_execution, MAX_RECENT_INTERACTIONS_DEFAULT)
            print(f"  剪枝後傳遞給 LLM 的消息數: {len(pruned_messages_for_llm)}")
            # 首次嘗試根據歷史+（可能有的）新計劃執行
            execution_response = await call_llm_with_tools(pruned_messages_for_llm, mcp_tools, active_execution_prompt) # <<< 傳遞 active_execution_prompt

            # --- <<< REVISED FORCING LOGIC >>> ---
            # 檢查是否：
            # 1. 計劃是在 *本次* 運行中生成的
            # 2. 響應是 AIMessage
            # 3. 響應 *沒有* 包含工具調用
            if plan_was_generated_this_run and isinstance(execution_response, AIMessage) and not execution_response.tool_calls:
                print("  計劃階段剛結束，但首次執行響應未包含工具調用。嘗試強制生成工具調用...")
                
                # 強制提示本身不需要特定於MCP，因為它的目標是產生任何工具調用
                forcing_system_prompt_content = f"""你必須根據剛才生成的計劃執行第一個階段目標。
                你之前的回應意圖執行一個動作，但未能正確生成工具調用。
                現在，請立即為這個**當前應該執行的動作**生成對應的工具調用指令 (`tool_calls`)。
                不要添加任何解釋、確認或對話性文本。直接輸出包含 `tool_calls` 的 AIMessage。
                第一個階段目標通常是檢查環境或獲取信息，例如調用 '{mcp_tools[0].name if mcp_tools else "get_layers"}'。"""
                # 注意：這裡的 forcing_system_prompt 是一個新的 SystemMessage，它將與 active_execution_prompt 不同
                # forcing_system_prompt_for_call = SystemMessage(content=forcing_system_prompt_content)


                try:
                    # `pruned_messages_for_llm` 是導致 execution_response 的訊息歷史
                    # `execution_response` 是有問題的回應
                    # `forcing_system_prompt` (新的) 是新的指示
                    # 在強制調用時，我們應該使用原始導致問題的 active_execution_prompt，再加上 forcing_system_prompt_content 形成上下文
                    
                    # messages_for_forcing_unpruned = pruned_messages_for_llm + [execution_response, forcing_system_prompt_for_call] # 原本的
                    
                    # 新的組合方式：使用原 execution prompt + 有問題的訊息 + 強制訊息
                    # 確保 active_execution_prompt 是有效的
                    if not active_execution_prompt: active_execution_prompt = RHINO_AGENT_EXECUTION_PROMPT

                    # Forcing messages should include:
                    # 1. The original human request
                    # 2. The plan
                    # 3. The problematic AI response (execution_response)
                    # 4. The new forcing system prompt
                    # We construct this based on `pruned_messages_for_llm` which contains 1 & 2, then add 3 & 4.
                    # The `call_llm_with_tools` will prepend its own system prompt (which would be active_execution_prompt again if we didn't modify for forcing)
                    # So for forcing, we need to construct the messages carefully.
                    # Let call_llm_with_tools prepend the *forcing_system_prompt_content*
                    
                    forcing_llm_execution_prompt = SystemMessage(content=forcing_system_prompt_content)
                    
                    # The messages to pass to call_llm_with_tools for forcing should be the history that led to the bad response
                    # `pruned_messages_for_llm` IS that history, minus the system prompt.
                    # Then call_llm_with_tools will add the new `forcing_llm_execution_prompt`.
                    messages_for_forcing_context = pruned_messages_for_llm + [execution_response]


                    # --- PRUNE MESSAGES for forcing call ---
                    print(f"    準備強制 LLM 調用，原始待處理消息數: {len(messages_for_forcing_context)}") # messages_for_forcing_unpruned
                    pruned_messages_for_forcing = _prune_messages_for_llm(messages_for_forcing_context, MAX_RECENT_INTERACTIONS_FORCING)
                    print(f"    剪枝後傳遞給強制 LLM 的消息數: {len(pruned_messages_for_forcing)}")
                    
                    print(f"    調用 LLM 強制生成工具調用 (基於 {len(pruned_messages_for_forcing)} 條剪枝後消息)...")
                    # 使用 forcing_llm_execution_prompt 進行強制調用
                    forced_response = await call_llm_with_tools(pruned_messages_for_forcing, mcp_tools, forcing_llm_execution_prompt)
                    
                    # 檢查強制調用是否 *成功* 生成了工具調用
                    if isinstance(forced_response, AIMessage) and forced_response.tool_calls:
                        print("    成功強制生成工具調用。替換原始響應。")
                        # 用強制生成的響應替換原始有問題的響應
                        execution_response = forced_response
                    else:
                        # 記錄失敗但暫時保留原始有問題的響應
                        # (後續的計數器可能會捕獲這個問題)
                        print("    警告：強制生成工具調用失敗或未產生工具調用。保留原始響應。")
                        if isinstance(forced_response, AIMessage):
                            print(f"      強制調用返回內容: '{forced_response.content[:100]}...'")
                        else:
                            print(f"      強制調用返回類型: {type(forced_response).__name__}")

                except Exception as force_err:
                    print(f"    強制生成工具調用時發生錯誤: {force_err}")
                    traceback.print_exc()
                    # 如果強制失敗，保留原始有問題的響應

            # --- <<< END REVISED FORCING LOGIC >>> ---

            # --- 基於 *最終* 的 execution_response 更新計數器 (依照使用者建議修改) ---
            new_consecutive_responses = 0 # 預設重置

            if isinstance(execution_response, AIMessage):
                has_tool_calls = hasattr(execution_response, 'tool_calls') and execution_response.tool_calls
                has_content = execution_response.content is not None and execution_response.content.strip() != ""

                if has_tool_calls:
                    # 如果有工具調用，總是重置計數器
                    new_consecutive_responses = 0
                    print(f"  LLM 返回 {len(execution_response.tool_calls)} 個工具調用，重置連續文本響應計數器為 0。")
                elif has_content:
                    # 如果沒有工具調用，但有內容 (包括錯誤信息、完成信息等)，也重置計數器
                    new_consecutive_responses = 0
                    print(f"  LLM 返回帶有內容的文本消息 ('{execution_response.content[:50]}...')，重置連續文本響應計數器為 0。")
                else: # (既沒有工具調用，也沒有內容，即 content is None or content is empty string)
                    new_consecutive_responses = current_consecutive_responses + 1
                    print(f"  LLM 返回空內容且無工具調用，遞增連續文本響應計數器為 {new_consecutive_responses}。")
            else:
                # 如果 execution_response 不是 AIMessage (例如發生錯誤或類型非預期)
                # 這種情況通常意味著流程中斷或出現更嚴重的問題，重置計數器是合理的
                new_consecutive_responses = 0
                print(f"  最終返回非 AIMessage 類型 ({type(execution_response).__name__})，重置連續文本響應計數器為 0。")
            
            # --- 修改：提前檢查計數器閾值，確保在所有場景下都執行 ---
            task_complete_due_to_counter = False
            if new_consecutive_responses >= 3:
                print(f"  已連續收到 {new_consecutive_responses} 次無效響應，將標記任務完成。")
                task_complete_due_to_counter = True
                # 添加一個明確的終止消息
                error_msg = f"[系統錯誤：連續 {new_consecutive_responses} 次未能生成有效工具調用或完成消息，任務強制終止。]"
                # 確保 messages_to_return 包含這個錯誤消息
                # 如果 execution_response 存在且不是這個錯誤消息，先添加它
                if execution_response and (not isinstance(execution_response, AIMessage) or execution_response.content != error_msg):
                     messages_to_return.append(execution_response)
                messages_to_return.append(AIMessage(content=error_msg))
            
            # ... 等待 asyncio.sleep 的部分 ...
            
            # 明確的任務完成返回
            if task_complete_due_to_counter:
                print(f"  由於連續無效回應達到上限，返回 task_complete=True")
                return {
                    "messages": messages_to_return, # messages_to_return 此時應已包含終止消息
                    "consecutive_llm_text_responses": 0, # 重置計數器
                    "task_complete": True # 明確設置任務完成
                }
            
            # ... 處理非計數器觸發場景的部分 ...
            
        finally:
             print(f"     執行 LLM 調用結束，等待 {RPM_DELAY} 秒...")
             await asyncio.sleep(RPM_DELAY)
             print("     等待結束。")

        # 除非計數器已觸發完成，否則添加最終響應
        if execution_response and not task_complete_due_to_counter:
            messages_to_return.append(execution_response)

        # 返回新生成的消息、更新後的計數器和 task_complete 狀態
        return_dict = {
            "messages": messages_to_return,
            "consecutive_llm_text_responses": new_consecutive_responses # 返回更新後的計數
        }
        # 如果由計數器觸發，添加 task_complete 標誌
        if task_complete_due_to_counter:
            return_dict["task_complete"] = True

        return return_dict

    except Exception as e:
        print(f"!! 執行 {mcp_name.upper()} Agent 節點時發生外部錯誤: {e}")
        traceback.print_exc()
        # 出錯時也重置計數器
        return {"messages": [AIMessage(content=f"執行 {mcp_name} Agent 時發生外部錯誤: {e}")], "consecutive_llm_text_responses": 0}

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
        return {}

    target_mcp = state.get("target_mcp")
    if not target_mcp:
         error_msg = "錯誤：狀態中缺少 'target_mcp'，無法執行工具。"
         print(f"  !! {error_msg}")
         error_tool_messages = [ ToolMessage(content=error_msg, tool_call_id=tc.get("id"), name=tc.get("name", "unknown_tool")) for tc in last_message.tool_calls ]
         return {"messages": error_tool_messages}

    print(f"  目標 MCP: {target_mcp}")
    try:
        selected_tools = await get_mcp_tools(target_mcp)
        print(f"  使用 {len(selected_tools)} 個 {target_mcp} 工具。")
        tool_messages = await execute_tools(last_message, selected_tools) # 移除 state 參數
        print(f"  工具執行完成，返回 {len(tool_messages)} 個 ToolMessage。")
        return {"messages": tool_messages}
    except Exception as e:
        print(f"!! 執行 Agent 工具節點時發生錯誤: {e}")
        traceback.print_exc()
        error_msg = f"執行工具時出錯: {e}"
        error_tool_messages = [ ToolMessage(content=error_msg, tool_call_id=tc.get("id"), name=tc.get("name", "unknown_tool")) for tc in last_message.tool_calls ]
        return {"messages": error_tool_messages}

# =============================================================================
# Conditional Edge Logic (修改 should_continue 處理 task_complete)
# =============================================================================
def should_continue(state: MCPAgentState) -> str:
    """確定是否繼續處理請求、調用工具或結束。"""
    print("--- 判斷是否繼續 ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None
    target_mcp = state.get("target_mcp", "unknown")
    consecutive_responses = state.get("consecutive_llm_text_responses", 0) # 獲取狀態中的計數

    # --- 優先檢查 task_complete 標誌 ---
    # 這個標誌現在可能由 agent_node_logic 因計數器達到上限而設置
    if state.get("task_complete"):
        print(f"  檢測到 task_complete 標誌 (可能來自計數器或工具處理) -> end")
        return END

    if not last_message:
        print("  消息列表為空 -> end")
        return END

    # --- 根據最後一條消息的類型和內容判斷 ---
    if isinstance(last_message, AIMessage):
        print("  最後消息是 AIMessage")
        content_str = ""
        
        # 檢查是否為嚴格的計劃消息 (只檢查前綴)
        is_plan_message = False
        if isinstance(last_message.content, str):
            content_str = ' '.join(last_message.content.lower().split())
            # 嚴格檢查前綴，並且確保不是系統錯誤消息
            is_plan_message = (last_message.content.strip().startswith("[目標階段計劃]:") and 
                               not content_str.startswith("[系統錯誤")) 
            print(f"  消息是計劃消息 (檢查前綴): {is_plan_message}")

        # 1. 檢查是否有工具調用請求 (最高優先級)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"  AI請求工具 ({len(last_message.tool_calls)}個) -> agent_tool_executor")
            return "agent_tool_executor"
        
        # 2. 如果是嚴格的計劃消息 (帶前綴)，則繼續執行
        if is_plan_message:
            print(f"  這是嚴格的計劃消息，繼續執行 -> {target_mcp}_agent")
            return f"{target_mcp}_agent"
        
        # --- 後續的完成/錯誤/計數器檢查邏輯 ---
        # 3. 檢查正常消息中的完成關鍵字 (這裡 is_plan_message 必為 False)
        completion_keywords = [
            "全部任務已完成",
            "最終畫面已截取",
            "任務因截圖錯誤而終止",
            "地圖截圖已完成",
            "圖片搜索和下載完成",
            "[系統錯誤", # 將系統錯誤也視為完成條件
        ]
        # 確保只在非計劃消息中檢查完成關鍵字
        is_explicit_completion = not is_plan_message and any(keyword in content_str for keyword in completion_keywords)

        if is_explicit_completion:
            print(f"  檢測到明確的完成/系統錯誤消息 ('{last_message.content[:50]}...') -> end")
            return END
        # 移除冗餘的錯誤檢查，因為已包含在 completion_keywords 中
        # elif "執行 llm 決策時發生錯誤" in content_str or "內部錯誤：" in content_str:
        #     print(f"  AI 返回決策錯誤消息 ('{last_message.content[:50]}...') -> end")
        #     return END

        # 4. 如果不是工具調用、不是明確完成/錯誤消息，則檢查計數器
        else:
            # 即使內容為空字串 ''，也算作一次無效響應，計數器會增加 (這部分邏輯移到 agent_node_logic 中處理)
            # 這裡只檢查計數器的結果
            if consecutive_responses >= 3:
                # 這個分支理論上不應該再被觸發，因為 agent_node_logic 會提前設置 task_complete
                # 但保留作為最後防線
                content_preview = ""
                if last_message and isinstance(last_message, AIMessage):
                    content_preview = f"(內容預覽: '{last_message.content[:20]}...')" if last_message.content else "(空消息)"
                
                print(f"  [should_continue Failsafe] AI未請求工具也未宣告明確完成，已連續 {consecutive_responses} 次 {content_preview} -> end (達到上限)")
                return END
            else:
                # 計數未達上限，返回 Agent 重新決策
                print(f"  AI未請求工具也未宣告明確完成 (連續 {consecutive_responses} 次)，返回 agent 重新決策...")
                if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
                    return f"{target_mcp}_agent" # 返回到 agent 節點
                else:
                    # 這個情況理論上不應該發生
                    print(f"  警告: 無效的 target_mcp ('{target_mcp}')，無法返回 Agent -> end")
                    return END

    elif isinstance(last_message, ToolMessage):
        # 工具執行完成後，總是應該回到 Agent 來處理結果
        print(f"  最後消息是 ToolMessage (來自工具 '{last_message.name}') -> 返回 {target_mcp}_agent 處理結果")
        if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
            return f"{target_mcp}_agent"
        else:
            print(f"  警告: 無效的 target_mcp ('{target_mcp}')，無法返回 Agent -> end")
            return END

    elif isinstance(last_message, HumanMessage):
         if len(messages) == 1:
              print("  只有初始 HumanMessage -> 邏輯應由 Router 處理 (返回 END 避免死循環)")
              return END
         else:
              print("  在流程中意外出現 HumanMessage -> end (異常)")
              return END
    else:
        print(f"  未知的最後消息類型 ({type(last_message).__name__}) -> end")
        return END

# =============================================================================
# 建立和編譯 LangGraph (添加 OSM 節點和邊)
# =============================================================================
workflow = StateGraph(MCPAgentState)
workflow.add_node("router", route_mcp_target)
workflow.add_node("revit_agent", call_revit_agent)
workflow.add_node("rhino_agent", call_rhino_agent)
# --- 新增 Pinterest Node ---
workflow.add_node("pinterest_agent", call_pinterest_agent)
# --- 新增 OSM Node ---
workflow.add_node("osm_agent", call_osm_agent)
workflow.add_node("agent_tool_executor", agent_tool_executor)

workflow.set_entry_point("router")

# --- 修改 Router Edges ---
workflow.add_conditional_edges(
    "router",
    lambda x: x.get("target_mcp"),
    {
        "revit": "revit_agent",
        "rhino": "rhino_agent",
        "pinterest": "pinterest_agent",
        "osm": "osm_agent"
    }
)

# --- 新增 Edges for OSM Agent ---
workflow.add_conditional_edges(
    "revit_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "revit_agent": "revit_agent", # <<< 新增：允許返回自身重新決策
        END: END
    }
)
workflow.add_conditional_edges(
    "rhino_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "rhino_agent": "rhino_agent", # <<< 新增：允許返回自身重新決策
        END: END
    }
)
# Add edges for the new pinterest agent
workflow.add_conditional_edges(
    "pinterest_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "pinterest_agent": "pinterest_agent", # <<< 新增：允許返回自身重新決策
        END: END # Pinterest agent might also end directly or request tools
    }
)

# Add edges for the new osm agent
workflow.add_conditional_edges(
    "osm_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "osm_agent": "osm_agent", # <<< 新增：允許返回自身重新決策
        END: END # OSM agent might also end directly or request tools
    }
)
# --- END MODIFIED ---

# --- 修改 Tool Executor Edges ---
workflow.add_conditional_edges(
   "agent_tool_executor",
   should_continue, # should_continue now returns the correct agent name or END
   {
       "revit_agent": "revit_agent",
       "rhino_agent": "rhino_agent",
       "pinterest_agent": "pinterest_agent",
       "osm_agent": "osm_agent",
       END: END
   }
)

graph = workflow.compile()
# --- 修改 Graph Name ---
graph.name = "Router_AgentPlanning_MCP_Agent_V18_OSM"
print(f"LangGraph 編譯完成: {graph.name}")


