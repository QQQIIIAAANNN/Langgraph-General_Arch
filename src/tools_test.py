import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import TypedDict, Annotated, List, Dict, Sequence, Literal, Optional
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import traceback
from contextlib import asynccontextmanager

# 載入 .env 設定
load_dotenv()

# --- LLM for Routing ---
# Make sure OPENAI_API_KEY is set in your .env file
try:
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except ImportError:
    print("ERROR: langchain-openai is not installed. Please install it: pip install langchain-openai")
    exit(1)
except Exception as e:
    print(f"ERROR: Could not initialize ChatOpenAI. Check OPENAI_API_KEY. Error: {e}")
    exit(1)


# =============================================================================
# 導入工具
# =============================================================================
# 從工具檔案導入 @tool 裝飾的函數
from src.tools.gemini_image_generation_tool import generate_gemini_image as image_tool
from src.tools.gemini_search_tool import perform_grounded_search as search_tool

# =============================================================================
# 定義狀態 (State)
# =============================================================================
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 新增一個欄位來儲存路由決策 (可選，但有助於追蹤)
    route_decision: Optional[str] = None

# =============================================================================
# 定義圖節點 (Graph Nodes)
# =============================================================================

def execute_image_generation(state: GraphState) -> Dict[str, Sequence[BaseMessage]]:
    """
    執行 Gemini 圖片生成工具。
    返回包含更新後 messages 的字典。
    """
    print("--- 執行圖片生成節點 ---")
    last_message = state['messages'][-1] # Get the last message, assuming it's the user prompt for the tool
    if not isinstance(last_message, HumanMessage):
         print("警告: 圖片生成節點預期最後一條訊息是 HumanMessage，但收到了", type(last_message))
         # Attempt to use the *first* message as a fallback, similar to previous logic
         if state['messages'] and isinstance(state['messages'][0], HumanMessage):
             user_message_content = state['messages'][0].content
             print("使用第一條訊息作為 Prompt (Fallback)。")
         else:
             error_msg = "錯誤：找不到有效的用戶 Prompt 進行圖片生成。"
             print(error_msg)
             # 返回包含錯誤訊息的狀態更新
             return {"messages": [AIMessage(content=error_msg)]} # 返回字典
    else:
        user_message_content = last_message.content

    print(f"圖片生成 Prompt: {user_message_content}")
    tool_input = {"prompt": user_message_content}
    new_messages: List[BaseMessage] = [] # 用於儲存此節點要添加的新訊息
    try:
        # Call the image generation tool function directly
        result = image_tool(tool_input)
        print(f"圖片生成工具回傳結果: {result}")
        # Format the result before adding it as a message
        if result.get("error"):
            response_content = f"圖片生成失敗: {result['error']}"
        elif not result.get("generated_files"):
            response_content = f"模型回覆 (未生成圖片): {result.get('text_response', '[無文字回覆]')}"
        else:
            files_info = "\n".join([f"- {f['filename']} (儲存於: {f['path']})" for f in result["generated_files"]])
            response_content = f"成功生成圖片:\n{files_info}\n\n模型文字回覆:\n{result.get('text_response', '[無文字回覆]')}"

        new_messages.append(AIMessage(content=response_content))
    except Exception as e:
        print(f"呼叫圖片生成工具時發生錯誤: {e}")
        traceback.print_exc()
        new_messages.append(AIMessage(content=f"圖片生成節點內部錯誤: {e}"))

    # 返回包含新訊息的字典以更新狀態
    return {"messages": new_messages}

def execute_search(state: GraphState) -> Dict[str, Sequence[BaseMessage]]:
    """
    執行 Gemini Grounded Search 工具。
    返回包含更新後 messages 的字典。
    """
    print("--- 執行搜尋節點 ---")
    last_message = state['messages'][-1] # Get the last message, assuming it's the user query
    if not isinstance(last_message, HumanMessage):
        print("警告: 搜尋節點預期最後一條訊息是 HumanMessage，但收到了", type(last_message))
        if state['messages'] and isinstance(state['messages'][0], HumanMessage):
             user_query = state['messages'][0].content
             print("使用第一條訊息作為查詢 (Fallback)。")
        else:
             error_msg = "錯誤：找不到有效的用戶查詢進行搜尋。"
             print(error_msg)
             # 返回包含錯誤訊息的字典
             return {"messages": [AIMessage(content=error_msg)]} # 返回字典
    else:
        user_query = last_message.content

    print(f"搜尋查詢: {user_query}")
    tool_input = {"query": user_query}
    new_messages: List[BaseMessage] = [] # 用於儲存此節點要添加的新訊息
    try:
        # Call the search tool function directly
        result = search_tool(tool_input)
        print(f"搜尋工具回傳結果: {result}")

        # Format the result
        if result.get("error"):
            response_content = f"搜尋失敗: {result['error']}"
        else:
            response_content = f"搜尋結果:\n{result.get('text_content', '[無文字回覆]')}"
            if result.get("grounding_sources"):
                sources = "\n".join([f"- [{s['title']}]({s['uri']})" for s in result["grounding_sources"]])
                response_content += f"\n\n參考來源:\n{sources}"
            if result.get("search_suggestions"):
                suggestions = "\n".join([f"- {s}" for s in result["search_suggestions"]])
                response_content += f"\n\n相關搜尋建議:\n{suggestions}"

        new_messages.append(AIMessage(content=response_content))
    except Exception as e:
        print(f"呼叫搜尋工具時發生錯誤: {e}")
        traceback.print_exc()
        new_messages.append(AIMessage(content=f"搜尋節點內部錯誤: {e}"))

    # 返回包含新訊息的字典以更新狀態
    return {"messages": new_messages}

# --- Router Node ---
ROUTING_PROMPT = SystemMessage(
    content="""你是一個路由代理。根據使用者的請求，判斷應該使用哪個工具。
使用者請求可能是要求生成圖片，或是進行資訊搜尋/回答問題。

回應必須是以下其中一個：
'image_generation' - 如果使用者主要想生成或創建一張圖片。
'search' - 如果使用者主要想搜尋資訊、查詢事實、詢問問題或需要最新資訊。

請只回應 'image_generation' 或 'search'。不要包含任何其他文字。"""
)

# 修改 route_query 的返回類型為字典
def route_query(state: GraphState) -> Dict[str, str]:
    """
    使用 LLM (gpt-4o-mini) 判斷用戶查詢意圖。
    返回包含路由決策的字典，例如 {"route_decision": "image_generation"}。
    """
    print("--- 執行路由節點 ---")
    messages = state['messages']
    if not messages or not isinstance(messages[-1], HumanMessage):
        print("錯誤：路由節點無法找到有效的用戶訊息。")
        # 返回一個表示錯誤的字典
        return {"route_decision": "__error__"}

    user_query = messages[-1].content
    print(f"待路由的查詢: {user_query}")

    try:
        routing_messages = [ROUTING_PROMPT, HumanMessage(content=user_query)]
        response = router_llm.invoke(routing_messages)
        route = response.content.strip().lower()
        print(f"LLM 路由決定: {route}")

        if "image_generation" in route:
            decision = "image_generation"
        elif "search" in route:
            decision = "search"
        else:
            print(f"警告: LLM 路由器的回應無法識別 ('{route}')。預設為搜尋。")
            decision = "search" # Default route

        # 返回包含路由決策的字典
        return {"route_decision": decision}
    except Exception as e:
        print(f"路由 LLM 呼叫失敗: {e}")
        traceback.print_exc()
        return {"route_decision": "__error__"}

# =============================================================================
# 建立 LangGraph
# =============================================================================
workflow = StateGraph(GraphState)

# 新增節點
workflow.add_node("classify_intent", route_query)
workflow.add_node("image_generation", execute_image_generation)
workflow.add_node("search_execution", execute_search)

# 設定圖的流程
workflow.set_entry_point("classify_intent")

# 設定條件邊
# 修改 lambda 函數以從 route_query 返回的字典中提取決策
workflow.add_conditional_edges(
    "classify_intent",
    lambda x: x.get("route_decision", "__error__"), # 從字典中獲取 'route_decision'
    {
        "image_generation": "image_generation",
        "search": "search_execution",
        "__error__": END
    }
)

# 從工作節點到結束
workflow.add_edge("image_generation", END)
workflow.add_edge("search_execution", END)

# 編譯 LangGraph
graph = workflow.compile()
graph.name = "Gemini_Image_Search_Router_Graph"
print("LangGraph 編譯完成: Gemini_Image_Search_Router_Graph")

# =============================================================================
# 執行測試 (範例 - 需要取消註解並運行)
# =============================================================================
if __name__ == "__main__":
    import asyncio
    import traceback # 確保導入 traceback

    async def run_graph():
        while True:
            user_prompt = input("\n請輸入您的請求 (圖片生成 或 搜尋查詢)，或輸入 'exit' 離開: ")
            if user_prompt.lower() == 'exit':
                break

            print(f"\n使用者輸入: {user_prompt}")
            inputs = {"messages": [HumanMessage(content=user_prompt)]}
            print("--- 開始執行 Graph ---")
            try: # 添加 try...except 以捕捉執行期間的錯誤
                async for output in graph.astream(inputs):
                    for key, value in output.items():
                        print(f"節點 '{key}' 的輸出:")
                        # 檢查 value 是否是字典並且包含 'messages'
                        if isinstance(value, dict) and 'messages' in value:
                             # 只打印新添加的消息內容，避免打印整個狀態
                             new_messages = value['messages']
                             for msg in new_messages:
                                  print(f"  - {type(msg).__name__}: {msg.content[:200]}...") # 打印消息類型和部分內容
                        elif isinstance(value, dict) and 'route_decision' in value:
                             print(f"  - 路由決策: {value['route_decision']}")
                        else:
                             print(f"  - {value}") # 打印原始值（如果不是預期的格式）
                        print("-" * 30)
            except Exception as graph_error:
                 print("\n--- Graph 執行時發生錯誤 ---")
                 traceback.print_exc() # 打印詳細的錯誤追蹤

            print("=" * 50)

    try:
        # asyncio.run(run_graph()) # 在某些環境可能導致循環錯誤
        # 使用 get_event_loop().run_until_complete 通常更穩定
        asyncio.get_event_loop().run_until_complete(run_graph())
    except RuntimeError as e:
         if "Cannot run the event loop while another loop is running" in str(e):
              print("發現現有事件循環。嘗試在新循環中運行...")
              loop = asyncio.new_event_loop()
              asyncio.set_event_loop(loop)
              loop.run_until_complete(run_graph())
         else:
              print(f"運行時錯誤: {e}")
              traceback.print_exc() # 打印錯誤追蹤
    except Exception as main_error: # 捕捉其他可能的錯誤
        print(f"運行 run_graph 時發生未預期的錯誤: {main_error}")
        traceback.print_exc() # 打印錯誤追蹤


    print("\nLangGraph 測試結束。")
