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
        "command": "C:\\Users\\User\\miniconda3\\envs\\rhino_mcp\\python",
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
AGENT_EXECUTION_PROMPT = SystemMessage(content="""你是一個嚴格按計劃執行任務的助手，專門為 CAD/BIM 環境生成指令或處理地理/圖像請求。消息歷史中包含了用戶請求和一個分階段目標的計劃。
你的任務是：
1.  分析計劃和執行歷史。
2.  識別出計劃中**第一個尚未完成的階段目標**。
3.  決定達成該目標所需的**第一個具體動作**。
4.  **如果需要調用工具來執行此動作，請必須生成 `tool_calls` 在首位的 AIMessage 以請求該工具調用**。**不要僅僅用文字描述你要調用哪個工具，而是實際生成工具調用指令。** 一次只生成一個工具調用請求。
5.  嚴格禁止使用 f-string 格式化字串。請使用 `.format()` 或 `%` 進行字串插值。
6.  **仔細參考工具描述或 Mcp 文檔確認函數用法與參數正確性，必須實際生成結構化的工具呼叫指令。**
    *   **Rhino 圖層管理 (重要):** 當生成 Rhino 代碼 (`execute_rhino_code` 或 `RhinoMCPCoordinator` 的 `user_request`) 時：
        *   如果當前階段目標**明確要求**在特定圖層上操作，**必須**在相關操作（如創建物件）**之前**包含 `rs.CurrentLayer('目標圖層名稱')` 指令。
        *   如果目標涉及控制圖層可見性（例如，準備截圖），**必須**包含 `rs.LayerVisible('圖層名', True/False)` 指令。
        *   **可用函數參考:**
            *   `rs.CurrentLayer('LayerName')`: 設置當前活動圖層。
            *   `rs.LayerVisible('LayerName', True/False)`: 設置圖層可見性。
            *   `rs.IsLayer('LayerName')`: 檢查圖層是否存在。
            *   `rs.AddLayer('LayerName', color=(R,G,B))`: 添加新圖層（可選顏色）。
    *   **地址/座標處理 (OSM - geocode_and_screenshot):**
        *   **檢查使用者輸入**：查看初始請求或當前目標是否包含明確的**經緯度座標**（例如 "2X.XXX 1XX.XXX" 或類似格式）。
        *   **如果找到座標**：直接將**座標字串 "緯度,經度"** (例如 "2X.XXX 1XX.XXX") 作為 `address` 參數的值傳遞給 `geocode_and_screenshot` 工具。**不要**嘗試將座標轉換成地址。
        *   **如果只找到地址**：請**嘗試將其簡化**，例如 "號碼, 街道名稱, 城市, 國家" 再傳遞給 `address` 參數。如果持續地理編碼失敗，可以嘗試進一步簡化。
7.  **最終步驟：**
    *   對於 Rhino/Revit 任務，當所有用戶請求的目標達成後，計劃的最後一步**應該**是調用 `capture_focused_view` 工具來截取最終畫面。
    *   對於 Pinterest 任務，計劃的最後一步應該是調用 `pinterest_search_and_download` (如果尚未完成) 或生成總結訊息。
    *   對於 OSM，計劃主要使用 (`geocode_and_screenshot`)。
    請生成對應的 `tool_calls` 或總結訊息。
8.  如果當前階段目標不需要工具即可完成（例如，僅需總結信息），請生成說明性的自然語言回應。
9.  若遇工具錯誤，分析錯誤原因 (尤其是代碼執行錯誤)，**嘗試修正你的工具調用參數或生成的代碼**，然後再次請求工具調用。如果無法修正，請報告問題。

**關鍵指令：只要下一步是工具操作，你的回應中**必須**包含 Tool Calls 結構。直到錯誤或是處理完最終任務後，才可生成純文字的完成訊息。**""")

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
    selected_tools: List[BaseTool]
) -> AIMessage:
    """
    調用 agent_llm (Gemini) 根據消息歷史（含計劃）和可用工具來執行下一步。
    輸入消息應已包含多模態內容。
    """
    print(f"  >> 調用 Agent LLM ({agent_llm.model}) 執行下一步...")
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
        current_call_messages = [AGENT_EXECUTION_PROMPT] + messages
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

RPM_DELAY = 6 # 比 6 秒稍長一點，留點餘裕

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

# =============================================================================
# Agent Nodes (修改：處理 Pinterest ToolMessage，返回最終結果)
# =============================================================================
async def agent_node_logic(state: MCPAgentState, config: RunnableConfig, mcp_name: str) -> Dict:
    """通用 Agent 節點邏輯：處理特定工具消息 (截圖, Pinterest 下載)，規劃，或執行下一步。"""
    print(f"--- 執行 {mcp_name.upper()} Agent 節點 ---")
    current_messages = list(state['messages'])
    last_message = current_messages[-1] if current_messages else None

    # --- 處理 capture_viewport 返回的文件路徑 (Rhino/Revit) ---
    IMAGE_PATH_PREFIX = "[IMAGE_FILE_PATH]:"
    if isinstance(last_message, ToolMessage) and last_message.name == "capture_focused_view" and last_message.content.startswith(IMAGE_PATH_PREFIX):
        print("  檢測到 capture_viewport 工具返回的文件路徑。")
        image_path = last_message.content[len(IMAGE_PATH_PREFIX):]
        print(f"    文件路徑: {image_path}")
        try:
            if not os.path.exists(image_path):
                 print(f"  !! 錯誤：收到的圖像文件路徑不存在: {image_path}")
                 return {"messages": [AIMessage(content=f"任務處理完畢，但最終截圖文件未找到: {image_path}")], "task_complete": True}

            with open(image_path, "rb") as f: image_bytes = f.read()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png"
            if file_extension == ".jpeg" or file_extension == ".jpg": mime_type = "image/jpeg"
            data_uri = f"data:{mime_type};base64,{base64_data}"
            print(f"    推斷 MIME 類型: {mime_type}")

            return {
                "messages": [AIMessage(content=f"任務已完成。\n最終畫面已截取並保存至 {image_path}。")], # 簡化消息，URI 在狀態中
                "saved_image_path": image_path,
                "saved_image_data_uri": data_uri,
                "task_complete": True # 標記任務完成
            }
        except Exception as img_proc_err:
            print(f"  !! 處理截圖文件 '{image_path}' 或編碼時出錯: {img_proc_err}")
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"任務已完成，但處理截圖文件 '{image_path}' 時失敗: {img_proc_err}")], "task_complete": True}

    # --- 處理 OSM 返回的文件路徑 (新增) ---
    OSM_IMAGE_PATH_PREFIX = "[OSM_IMAGE_PATH]:"
    if isinstance(last_message, ToolMessage) and last_message.name == "geocode_and_screenshot" and last_message.content.startswith(OSM_IMAGE_PATH_PREFIX):
        print("  檢測到 geocode_and_screenshot 工具返回的文件路徑。")
        image_path = last_message.content[len(OSM_IMAGE_PATH_PREFIX):]
        print(f"    OSM 文件路徑: {image_path}")
        try:
            if not os.path.exists(image_path):
                 print(f"  !! 錯誤：收到的 OSM 圖像文件路徑不存在: {image_path}")
                 return {"messages": [AIMessage(content=f"地圖處理完畢，但截圖文件未找到: {image_path}")], "task_complete": True}

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
                "task_complete": True
            }
        except Exception as img_proc_err:
            print(f"  !! 處理 OSM 截圖文件 '{image_path}' 或編碼時出錯: {img_proc_err}")
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"地圖截圖已完成，但處理文件 '{image_path}' 時失敗: {img_proc_err}")], "task_complete": True}

    # --- 處理 capture_viewport 返回的錯誤消息 ---
    elif isinstance(last_message, ToolMessage) and last_message.name == "capture_focused_view" and last_message.content.startswith("[Error: Viewport Capture Failed]:"):
         error_msg = last_message.content
         print(f"  檢測到 capture_viewport 工具返回錯誤: {error_msg}")
         return {"messages": [AIMessage(content=f"任務因截圖錯誤而終止: {error_msg}")], "task_complete": True}

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
                 "task_complete": True # Assume Pinterest is final step
             }
        else:
             # If parsing failed or no paths found, just pass the message content along
             print("    Pinterest 工具未返回有效路徑列表，任務可能未成功或未找到圖片。")
             # Let should_continue decide the next step based on the text message
             return {"messages": [AIMessage(content=f"Pinterest 任務處理完成，但未找到或處理下載路徑。工具輸出: {content[:200]}...")]}

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

        # 檢查是否需要規劃
        PLAN_PREFIX = "[目標階段計劃]:\n"
        plan_exists = any(
            isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)
            for msg in current_messages
        )

        messages_to_return = []

        if not plan_exists:
            print("  需要生成執行計劃...")
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
            # --- 調整 planning prompt slightly for Pinterest ---
            planning_system_content = f"""你是一位優秀的任務規劃助理，專門為 CAD/BIM、圖像搜索或地理空間任務制定計劃。
            基於使用者提供的文字請求、可選的圖像以及下方列出的可用工具，生成一個清晰的、**分階段目標**的計劃。

            **重要要求：**
            1.  **量化與具體化:** 對於幾何操作 (Rhino/Revit)，每個階段目標**必須**包含盡可能多的**具體數值、尺寸、座標、角度、數量、距離、方向、或清晰的空間關係描述**。
            2.  **邏輯順序:** 確保階段目標按邏輯順序排列，後續步驟依賴於先前步驟的結果。
            3.  **最終步驟:**
                *   對於 Rhino，計劃的最後一步**必須**是調用 `capture_focused_view` 工具來截取最終畫面。請確保相機位置在水平面之上。
                *   對於 Pinterest，計劃通常是搜索 (`pinterest_search_and_download`)。
                *   對於 OSM，計劃通常是地理編碼加截圖 (`geocode_and_screenshot`)。
            4.  **目標狀態:** 計劃應側重於**每個階段要達成的目標狀態**，說明該階段完成後場景應有的變化。

            **rhino提醒: 目前單位是M(公尺)。對於量體配置方案建議使用parallel或perspective從上方俯視；對於渲染用建模建議使用two point perspective從人眼視角截圖**
            這個計劃應側重於**每個階段要達成的目標狀態**，而不是具體的工具使用細節。將任務分解成符合邏輯順序的多個階段目標。
            直接輸出這個階段性目標計劃，不要額外的開場白或解釋。

            可用工具如下 ({mcp_name}):
            {tool_descriptions}"""
            planning_system_message = SystemMessage(content=planning_system_content)
            print("    構造了規劃 SystemMessage")

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
        try:
            execution_response = await call_llm_with_tools(messages_for_execution, mcp_tools)
        finally:
             print(f"     執行 LLM 調用結束，等待 {RPM_DELAY} 秒...")
             await asyncio.sleep(RPM_DELAY)
             print("     等待結束。")

        if execution_response: messages_to_return.append(execution_response)

        # 返回新生成的消息 (不標記 task_complete，由 should_continue 決定)
        return {"messages": messages_to_return}

    except Exception as e:
        print(f"!! 執行 {mcp_name.upper()} Agent 節點時發生外部錯誤: {e}")
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"執行 {mcp_name} Agent 時發生外部錯誤: {e}")]}

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

    # --- 優先檢查 task_complete 標誌 ---
    if state.get("task_complete"):
        print("  檢測到 task_complete 標誌 -> end")
        return END

    if not last_message:
        print("  消息列表為空 -> end")
        return END

    # --- 根據最後一條消息的類型和內容判斷 ---
    if isinstance(last_message, AIMessage):
        print("  最後消息是 AIMessage")
        content_str = ""
        if isinstance(last_message.content, str):
            content_str = last_message.content.lower()

        # 檢查是否有工具調用請求
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"  AI請求工具 ({len(last_message.tool_calls)}個) -> agent_tool_executor")
            return "agent_tool_executor"
        # 檢查是否是表示完成的消息 (即使沒有 task_complete 標誌也結束)
        elif "任務已完成" in content_str or "最終畫面已截取" in content_str or "任務因截圖錯誤而終止" in content_str:
             print("  檢測到最終完成或錯誤消息 -> end")
             return END
        # 檢查是否是特定的、應導致終止的 LLM 錯誤消息
        elif "執行 LLM 決策時發生錯誤" in content_str or "內部錯誤：" in content_str:
             print("  AI 返回決策錯誤 -> end")
             return END
        # 預設行為：如果 AI 既未完成也未請求工具調用，則結束 (避免死循環)
        else:
             print(f"  AI未請求工具也未宣告完成 (content: '{content_str[:100]}...') -> end")
             return "agent_tool_executor"

    elif isinstance(last_message, ToolMessage):
        # 工具執行完成後，總是應該回到 Agent 來處理結果
        print(f"  最後消息是 ToolMessage (來自工具 '{last_message.name}') -> 返回 {target_mcp}_agent 處理結果")
        # --- 修改：Route back to specific agent ---
        if target_mcp in ["revit", "rhino", "pinterest", "osm"]:
            return f"{target_mcp}_agent"
        else:
            print(f"  警告: 無效的 target_mcp ('{target_mcp}')，無法返回 Agent -> end")
            return END
        # --- END MODIFIED ---

    elif isinstance(last_message, HumanMessage):
         # ... (HumanMessage 處理邏輯保持不變) ...
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
        END: END
    }
)
workflow.add_conditional_edges(
    "rhino_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        END: END
    }
)
# Add edges for the new pinterest agent
workflow.add_conditional_edges(
    "pinterest_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        END: END # Pinterest agent might also end directly or request tools
    }
)

# Add edges for the new pinterest agent
workflow.add_conditional_edges(
    "osm_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        END: END # Pinterest agent might also end directly or request tools
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


