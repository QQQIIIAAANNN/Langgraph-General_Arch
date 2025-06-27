import os
import re
import ast
import json
import base64
import time # 新增導入
from dotenv import load_dotenv
from langgraph.graph.state import StateGraph, START, END
from typing import List
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from src.memory import get_long_term_store, get_short_term_memory
from src.tools.img_recognition import img_recognition
from src.tools.prompt_generation import prompt_generation
from src.tools.case_render_image import case_render_image
from src.tools.simulate_future_image import simulate_future_image
from src.tools.ARCH_rag_tool import ARCH_rag_tool
from src.tools.video_recognition import video_recognition
from src.tools.generate_3D import generate_3D
from src.tools.gemini_search_tool import perform_grounded_search
from src.tools.gemini_image_generation_tool import generate_gemini_image
from src.T_config import GraphOverallConfig # LLM_INSTANCE 和 PROMPTS_INSTANCE 仍然可以導入，但節點內主要用 config

# 載入 .env 設定
load_dotenv()

# 全局的 LLM_INSTANCE 和 PROMPTS_INSTANCE 主要作為後備或用於非節點上下文
# 節點內部應使用從 config 傳入的實例
# default_llm = LLM_INSTANCE # 可以保留，但節點內不直接用
# default_prompts = PROMPTS_INSTANCE # 可以保留，但節點內不直接用


# =============================================================================
# 輔助函數，用於確保配置是 GraphOverallConfig 的實例
# =============================================================================
def ensure_graph_overall_config(config_input: any) -> GraphOverallConfig: # 接受更通用的類型
    if isinstance(config_input, GraphOverallConfig):
        print("DEBUG ensure_graph_overall_config: Input is already GraphOverallConfig instance.")
        return config_input

    actual_config_dict = None
    print(f"DEBUG ensure_graph_overall_config: Received config_input type: {type(config_input)}, value: {repr(config_input)}")

    if hasattr(config_input, 'get') and callable(getattr(config_input, 'get')): # 檢查是否像字典一樣操作
        # 標準的 LangGraph RunnableConfig 將配置放在 'configurable' 鍵下
        # 我們也處理直接傳遞普通字典的情況
        if 'configurable' in config_input and isinstance(config_input['configurable'], dict):
            print("DEBUG ensure_graph_overall_config: Detected RunnableConfig-like structure, using config_input['configurable']")
            actual_config_dict = config_input['configurable']
        elif all(isinstance(k, str) for k in config_input.keys()): # 粗略檢查是否為普通字典
            print("DEBUG ensure_graph_overall_config: Assuming config_input is the plain config dictionary.")
            actual_config_dict = dict(config_input) # 確保是普通字典
        else:
            print(f"DEBUG ensure_graph_overall_config: config_input is dict-like but not recognized structure: {repr(config_input)}")
            # 如果不是期望的結構，但仍然是 dict-like，嘗試直接使用它
            # 這可能在某些情況下有效，但在其他情況下可能導致 Pydantic 錯誤
            actual_config_dict = dict(config_input)


    if actual_config_dict is None:
        raise TypeError(f"Could not extract a valid configuration dictionary from input type {type(config_input)}. Value: {repr(config_input)}")

    print(f"DEBUG ensure_graph_overall_config: Final dictionary for Pydantic instantiation: {repr(actual_config_dict)}")
    print(f"DEBUG ensure_graph_overall_config: Value of 'run_site_analysis' in final dict for Pydantic: {actual_config_dict.get('run_site_analysis')}")
    
    try:
        # 在實例化前打印 GraphOverallConfig 中 run_site_analysis_raw_value 字段的預設值和別名
        field_info_raw = GraphOverallConfig.model_fields.get("run_site_analysis_raw_value")
        if field_info_raw:
            print(f"DEBUG ensure_graph_overall_config: Model field 'run_site_analysis_raw_value' - default: {field_info_raw.default}, alias: {field_info_raw.alias}")
        else:
            print("DEBUG ensure_graph_overall_config: Could not get field_info for 'run_site_analysis_raw_value'")

        instance = GraphOverallConfig(**actual_config_dict)
        
        # 打印實例化後的值
        print(f"DEBUG ensure_graph_overall_config: Instance created. instance.run_site_analysis_raw_value = {getattr(instance, 'run_site_analysis_raw_value', 'N/A')}, instance.run_site_analysis = {instance.run_site_analysis}")
        return instance
    except Exception as e:
        print(f"DEBUG ensure_graph_overall_config: Error during GraphOverallConfig instantiation: {e}")
        print(f"DEBUG ensure_graph_overall_config: Failing dictionary was: {repr(actual_config_dict)}")
        raise e


# =============================================================================
# 建立 LLM 實例（不再設定 system_message 屬性）
# =============================================================================
# 這部分綁定工具的 LLM 實例，如果工具調用也需要配置化，則需要更複雜的處理
# 目前假設工具綁定的 LLM 可以使用預設配置的 LLM
# 或者，這些綁定可以在 invoke 時動態創建，基於傳入的 config.llm_config
# 為簡化，暫時保留全局 llm 的用法進行工具綁定

# 使用 T_config 中的預設配置來初始化一個 LLM 實例，主要用於工具綁定
_temp_default_config_for_tools = GraphOverallConfig()
_tool_binding_llm = _temp_default_config_for_tools.llm_config.get_llm()

llm_with_img = _tool_binding_llm.bind_tools([img_recognition])
llm_with_ARCHrag = _tool_binding_llm.bind_tools([ARCH_rag_tool])
llm_with_prompt_gen = _tool_binding_llm.bind_tools([prompt_generation])
llm_with_gen2 = _tool_binding_llm.bind_tools([simulate_future_image])


# =============================================================================
# 定義全局狀態 (GlobalState)
# =============================================================================
class GlobalState(TypedDict, total=False):
    設計目標x設計需求x方案偏好 : Annotated[Sequence[BaseMessage], add_messages]
    design_summary: str
    analysis_img: str
    site_analysis: str
    design_advice: list
    outer_prompt: list
    case_image: list
    future_image: list
    perspective_3D: list
    model_3D: list
    GATE1: str
    GATE2: str 
    GATE_REASON1: str
    GATE_REASON2: str
    current_round: int
    evaluation_count: int
    evaluation_result: list
    evaluation_status: str
    final_evaluation: str

def custom_add_messages(existing: list, new: list) -> list:
    # 確保 new 為列表
    if not isinstance(new, list):
        new = [new]
    processed_new = []
    for msg in new:
        # 如果訊息具有 role 屬性且預設為 "human"，則修改為 "system"
        if hasattr(msg, "role"):
            if msg.role == "human":
                msg.role = "system"
        processed_new.append(msg)
    return existing + processed_new

# =============================================================================
# 各任務定義
# =============================================================================

# 問題代理任務：提示用戶輸入設計目標、設計需求、方案類型、方案偏好 OK
class QuestionTask:
    def __init__(self, state: GlobalState):
        self.state = state
    
    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state
        
        active_config = ensure_graph_overall_config(config)

        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language

        user_input = self.state["設計目標x設計需求x方案偏好"][0].content
        print("✅ 用戶的設計需求已記錄：", user_input)

        # 生成用於檢索的查詢，現在會更通用，不僅限於 ARCH_rag_tool
        # 原來的 keyword_prompt_content 可以繼續使用或微調
        keyword_prompt_content = active_config.question_task_keyword_prompt_template.format(
            user_input=user_input,
            llm_output_language=active_language
        )
        keywords_msg = current_llm.invoke([SystemMessage(content=keyword_prompt_content)])
        # keywords_text 現在作為一個通用的搜索查詢
        search_query_text = keywords_msg.content.strip()
        print("生成的通用搜索查詢：", search_query_text)

        # 1. 使用 ARCH_rag_tool
        arch_rag_results = ""
        try:
            arch_rag_msg_content = ARCH_rag_tool.invoke(search_query_text) # 或者使用更精確的查詢
            if isinstance(arch_rag_msg_content, str):
                arch_rag_results = arch_rag_msg_content
            print("ARCH_rag_tool 檢索結果：", arch_rag_results)
        except Exception as e:
            print(f"⚠️ ARCH_rag_tool 調用失敗: {e}")
            arch_rag_results = "ARCH RAG 工具檢索失敗。"

        # 2. 使用 perform_grounded_search
        grounded_search_results_text = ""
        # grounded_search_files = [] # 如果需要處理圖片等文件
        try:
            # 假設 perform_grounded_search 返回一個字典，包含 text_content 和 images
            # 我們主要關心 text_content
            # 查詢的 prompt 可以與 ARCH_rag_tool 的查詢相同，或者針對性調整
            # 這裡我們使用相同的 search_query_text
            # 提示：perform_grounded_search 的查詢可以更自然語言化
            # 例如："查詢關於 {木構造} {數位製造} 的 {pavilion 設計案例} 和 {構造細節}"
            # 這裡的 search_query_text 已經是 LLM 生成的關鍵詞，可能需要包裝一下
            
            # 建立一個更適合 perform_grounded_search 的查詢
            # 可以直接用用戶輸入，或者結合 LLM 生成的關鍵詞
            grounded_search_query = (
                f"針對用戶的建築設計需求 '{user_input}'，"
                f"尋找相關的案例、構造細節、製造工法（特別是木構造、數位製造、傳統製造工法、木結構）、"
                f"減碳與循環永續性策略、技術細節及研究理論。"
                f"生成的關鍵詞參考：{search_query_text}"
            )
            print(f"Grounded Search 查詢: {grounded_search_query}")

            search_tool_output = perform_grounded_search({"query": grounded_search_query})
            
            if isinstance(search_tool_output, dict):
                grounded_search_results_text = search_tool_output.get("text_content", "")
                # 如果需要處理圖片:
                # returned_images = search_tool_output.get("images", [])
                # for img_info in returned_images:
                #     # 處理圖片邏輯...
                #     pass
            elif isinstance(search_tool_output, str): # 向下兼容，如果工具直接返回字符串
                grounded_search_results_text = search_tool_output
            
            print("perform_grounded_search 檢索結果 (文本部分)：", grounded_search_results_text)
        except Exception as e:
            print(f"⚠️ perform_grounded_search 調用失敗: {e}")
            grounded_search_results_text = "Grounded Search 工具檢索失敗。"

        # 合併兩個 RAG 工具的結果
        # 可以簡單拼接，或者讓 LLM 稍後在總結時自行判斷重要性
        combined_rag_info = f"傳統知識庫檢索結果:\n{arch_rag_results}\n\n網路與文獻搜索結果:\n{grounded_search_results_text}"
        
        # 更新 summary prompt 以包含合併後的 RAG 信息
        summary_input_content = active_config.question_task_summary_prompt_template.format(
            user_input=user_input,
            rag_msg=combined_rag_info, # 使用合併後的資訊
            llm_output_language=active_language
        )
        summary_msg = current_llm.invoke([SystemMessage(content=summary_input_content)])
        self.state["design_summary"] = f"用戶需求:\n{user_input}\n\n設計目標初步總結與分析 (基於檢索資訊):\n{summary_msg.content}"
        print("✅ 設計目標總結完成！")
        print(self.state["design_summary"])
        return {          
            "設計目標x設計需求x方案偏好": self.state["設計目標x設計需求x方案偏好"], 
            "design_summary": self.state["design_summary"]
            }

# 場地分析任務：讀取 JSON 並分析基地資訊，同時透過 LLM 呼叫進一步解讀資料 OK
class SiteAnalysisTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        print("DEBUG: SiteAnalysisTask.run CALLED")
        if state is not None:
            self.state = state
        
        active_config = ensure_graph_overall_config(config)
        print(f"DEBUG SiteAnalysisTask: Processed active_config.run_site_analysis: {active_config.run_site_analysis}")

        if not active_config.run_site_analysis:
            skip_message = "用戶配置要求跳過基地分析 (run_site_analysis=False)。"
            print(f"ℹ️ {skip_message}")
            self.state["site_analysis"] = skip_message
            self.state["analysis_img"] = "無圖片 (基地分析已跳過)"
            return {
                "site_analysis": self.state["site_analysis"],
                "analysis_img": self.state["analysis_img"],
            }

        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language
        
        # --- 步驟 1: 分析 user input 的內容獲取經緯度和地點 ---
        user_design_input = ""
        if self.state.get("設計目標x設計需求x方案偏好") and isinstance(self.state["設計目標x設計需求x方案偏好"], Sequence) and len(self.state["設計目標x設計需求x方案偏好"]) > 0:
            user_design_input = self.state["設計目標x設計需求x方案偏好"][0].content
            print(f"ℹ️ SiteAnalysisTask [Step 1]: 取得用戶設計需求內容: '{user_design_input[:100]}...'")
        else:
            error_message = "⚠️ SiteAnalysisTask [Step 1]: 未能在狀態中找到有效的用戶設計需求輸入。無法提取基地資訊。"
            print(error_message)
            self.state["site_analysis"] = error_message
            self.state["analysis_img"] = "無圖片 (缺少用戶設計需求)"
            return {"site_analysis": self.state["site_analysis"], "analysis_img": self.state["analysis_img"]}

        region = "未知"
        geo_location = "未知"
        try:
            extract_prompt_content = active_config.site_analysis_extract_site_info_prompt_template.format(
                user_design_input=user_design_input,
                llm_output_language=active_language
            )
            print(f"DEBUG SiteAnalysisTask [Step 1]: Prompt for extracting site info: {extract_prompt_content}")
            site_info_msg = current_llm.invoke([SystemMessage(content=extract_prompt_content)])
            site_info_json_str = site_info_msg.content.strip()
            print(f"ℹ️ SiteAnalysisTask [Step 1]: LLM 提取的基地資訊 (原始字串): {site_info_json_str}")
            
            if site_info_json_str.startswith("```json"):
                site_info_json_str = site_info_json_str[7:]
            if site_info_json_str.endswith("```"):
                site_info_json_str = site_info_json_str[:-3]
            site_info_json_str = site_info_json_str.strip()

            parsed_site_info = json.loads(site_info_json_str)
            region = parsed_site_info.get("region", "未知")
            geo_location = parsed_site_info.get("geo_location", "未知")
            print(f"✅ SiteAnalysisTask [Step 1]: 解析後的 Region: {region}, GeoLocation: {geo_location}")

        except AttributeError as e:
            # 捕獲 active_config.site_analysis_extract_site_info_prompt_template 不存在的錯誤
            error_message = f"⚠️ SiteAnalysisTask [Step 1]: 提取基地資訊時發生 AttributeError (可能 Prompt 模板未在 T_config.py 中正確定義或加載): {e}"
            print(error_message)
            self.state["site_analysis"] = error_message
            self.state["analysis_img"] = "無圖片 (配置錯誤)"
            return {"site_analysis": self.state["site_analysis"], "analysis_img": self.state["analysis_img"]}
        except json.JSONDecodeError as e:
            print(f"⚠️ SiteAnalysisTask [Step 1]: 解析LLM提取的基地資訊JSON失敗: {e}. 原始字串: '{site_info_json_str}'")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 1]: 提取基地資訊時發生其他錯誤: {e}")

        if region == "未知" and geo_location == "未知":
            print("⚠️ SiteAnalysisTask [Step 1]: 未能從用戶輸入中明確提取有效的地區或地理位置資訊。後續分析可能受影響，但將繼續嘗試。")

        # --- 步驟 2: 使用以上資訊及圖片 "D:/MA system/LangGraph/input/2D/map.png" 進行圖片辨識分析 ---
        # 注意：檔名已從 base_map.png 改為 map.png
        base_map_path_str = "./input/2D/map.png" 
        
        print(f"DEBUG SiteAnalysisTask [Step 2]: Checking for image file at: '{base_map_path_str}'")
        if not os.path.exists(base_map_path_str):
            error_message = f"❌ SiteAnalysisTask [Step 2]: 圖片文件未找到於 '{base_map_path_str}'。無法進行圖片辨識。"
            print(error_message)
            self.state["site_analysis"] = self.state.get("site_analysis", "") + " " + error_message # 附加錯誤
            self.state["analysis_img"] = "缺少圖片文件 (map.png)，無法進行辨識。"
            # 即使圖片缺失，我們仍然可以嘗試基於文本的RAG和分析，所以不一定立即返回
            # 但如果圖片是核心，則應該返回
            # 為了符合流程，如果圖片辨識是必要的，這裡應該返回
            return {"site_analysis": self.state["site_analysis"], "analysis_img": self.state["analysis_img"]}
        
        print(f"✅ SiteAnalysisTask [Step 2]: 圖片文件 '{base_map_path_str}' 已找到。")
        
        initial_img_analysis_content = "圖片辨識失敗或未執行。" # 預設值
        try:
            img_rec_prompt_content = active_config.site_analysis_img_recognition_prompt_template.format(
                region=region, 
                geo_location=geo_location, 
                llm_output_language=active_language
            )
            print(f"DEBUG SiteAnalysisTask [Step 2]: Prompt for image recognition: {img_rec_prompt_content}")
            initial_img_analysis_content = img_recognition.invoke({ 
                "image_paths": base_map_path_str, # 使用絕對路徑
                "prompt": img_rec_prompt_content,
            })
            self.state["analysis_img"] = initial_img_analysis_content 
            print(f"✅ SiteAnalysisTask [Step 2]: 初步圖片辨識結果: '{str(initial_img_analysis_content)[:200]}...'")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 2]: 執行圖片辨識時發生錯誤: {e}")
            self.state["analysis_img"] = f"圖片辨識失敗: {e}"
            # 根據需求決定是否在此處返回

        # --- 步驟 3: 根據辨識結果生成關鍵字 ---
        site_rag_keywords = "未知關鍵字" # 預設值
        try:
            rag_keywords_gen_prompt = active_config.site_analysis_rag_keywords_prompt_template.format(
                region=region,
                geo_location=geo_location,
                initial_img_analysis_summary=str(initial_img_analysis_content), #確保是字符串 
                llm_output_language=active_language
            )
            print(f"DEBUG SiteAnalysisTask [Step 3]: Prompt for RAG keyword generation: {rag_keywords_gen_prompt}")
            keywords_msg = current_llm.invoke([SystemMessage(content=rag_keywords_gen_prompt)])
            site_rag_keywords = keywords_msg.content.strip()
            print(f"✅ SiteAnalysisTask [Step 3]: 生成的基地分析RAG關鍵字: {site_rag_keywords}")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 3]: 生成RAG關鍵字時發生錯誤: {e}")

        # --- 步驟 4: 進行ARCH_rag_tool以及perform_grounded_search ---
        arch_rag_site_results = "ARCH RAG 工具檢索無結果或失敗。"
        try:
            print(f"DEBUG SiteAnalysisTask [Step 4]: Invoking ARCH_rag_tool with keywords: {site_rag_keywords}")
            arch_rag_msg_content = ARCH_rag_tool.invoke(site_rag_keywords)
            if isinstance(arch_rag_msg_content, str):
                arch_rag_site_results = arch_rag_msg_content
            print(f"✅ SiteAnalysisTask [Step 4]: ARCH_rag_tool 基地資訊檢索結果: '{arch_rag_site_results[:200]}...'")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 4]: ARCH_rag_tool 調用失敗: {e}")

        grounded_search_site_results_text = "Grounded Search 工具檢索無結果或失敗。"
        try:
            grounded_search_site_query = (
                f"查詢關於地點 '{region}' (經緯度: {geo_location}) 的詳細背景資料，"
                f"包括都市計畫規範、建築法規、氣候資料、日照、人文歷史、水文地質、周邊環境等。"
                f"參考關鍵字：{site_rag_keywords}"
            )
            print(f"DEBUG SiteAnalysisTask [Step 4]: Invoking perform_grounded_search with query: {grounded_search_site_query}")
            search_tool_output = perform_grounded_search({"query": grounded_search_site_query})
            if isinstance(search_tool_output, dict):
                grounded_search_site_results_text = search_tool_output.get("text_content", "")
            elif isinstance(search_tool_output, str):
                grounded_search_site_results_text = search_tool_output
            print(f"✅ SiteAnalysisTask [Step 4]: perform_grounded_search 檢索結果 (文本部分): '{grounded_search_site_results_text[:200]}...'")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 4]: perform_grounded_search 調用失敗: {e}")

        combined_site_rag_info = (
            f"內部知識庫檢索 (ARCH_rag_tool):\n{arch_rag_site_results}\n\n"
            f"網路與文獻搜索 (perform_grounded_search):\n{grounded_search_site_results_text}"
        )

        # --- 步驟 5: 執行最終的 LLM 分析，整合圖片辨識和所有 RAG 資訊 ---
        final_analysis_result_content = "最終基地分析報告生成失敗。"
        try:
            llm_analysis_prompt_content = active_config.site_analysis_llm_prompt_template.format(
                region=region,
                geo_location=geo_location,
                analysis_img=str(initial_img_analysis_content), # 確保是字符串
                rag_supplementary_info=combined_site_rag_info,
                llm_output_language=active_language
            )
            print(f"DEBUG SiteAnalysisTask [Step 5]: Prompt for final LLM analysis: {llm_analysis_prompt_content}")
            analysis_result_msg = current_llm.invoke([SystemMessage(content=llm_analysis_prompt_content)])
            final_analysis_result_content = analysis_result_msg.content
            self.state["site_analysis"] = final_analysis_result_content
            print(f"✅ SiteAnalysisTask [Step 5]: 最終的、整合了RAG的分析報告: '{final_analysis_result_content[:200]}...'")
        except Exception as e:
            print(f"⚠️ SiteAnalysisTask [Step 5]: 生成最終分析報告時發生錯誤: {e}")
            self.state["site_analysis"] = f"最終基地分析報告生成失敗: {e}"

        print("✅ SiteAnalysisTask.run COMPLETED")
        return {
            "analysis_img": self.state["analysis_img"], # 保持為初步圖片辨識結果
            "site_analysis": self.state["site_analysis"], # 更新為最終分析報告
        }

# 設計方案任務 OK
class RAGdesignThinking:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state        
        
        active_config = ensure_graph_overall_config(config)
        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language

        design_goal_summary = self.state.get("design_summary", "無設計目標總結") # 從 QuestionTask 獲取的總結
        analysis_result = self.state.get("site_analysis", "無基地分析結果")
        current_round = self.state.get("current_round", 0)
        improvement = self.state.get("GATE_REASON1", "") 

        # 新的 "keywords_prompt_content" - 現在是引導設計方向，而不是生成 RAG 關鍵詞
        # 它會參考 design_goal_summary 和 analysis_result
        # rag_design_thinking_keywords_prompt_template 在 T_config.py 中也需要更新
        design_directions_prompt_content = active_config.rag_design_thinking_keywords_prompt_template.format(
            design_goal_summary=design_goal_summary, # 更新變數名
            analysis_result=analysis_result,
            llm_output_language=active_language
        )
        response_design_directions_msg = current_llm.invoke([SystemMessage(content=design_directions_prompt_content)])
        # design_directions_text 現在是設計方向的文本描述
        design_directions_text = response_design_directions_msg.content.strip()
        print("生成的設計方向指引：", design_directions_text)

        # 不再調用 ARCH_rag_tool
        # RAG_msg_content 現在可以直接使用 design_directions_text 或其他相關內容
        # 如果 complete_scheme_prompt_template 仍然需要 rag_msg 變數，
        # 我們可以將 design_directions_text 作為 rag_msg 傳入，
        # 或者修改 complete_scheme_prompt_template 以接受 design_directions
        
        # 假設 complete_scheme_prompt_template 中的 rag_msg 現在代表更廣泛的參考資料或設計指引
        # 我們將 design_directions_text 用於此處，或者您可以選擇其他更有意義的內容
        # T_config.py 中的 rag_design_thinking_complete_scheme_prompt_template 的描述也應該更新
        complete_scheme_prompt_content = active_config.rag_design_thinking_complete_scheme_prompt_template.format(
            design_goal_summary=design_goal_summary, # 更新變數名
            improvement=improvement,
            analysis_result=analysis_result,
            # rag_msg=RAG_msg_content, # 原來的 RAG 結果
            design_directions=design_directions_text, # 新的設計方向指引
            llm_output_language=active_language
        )
        complete_response_msg = current_llm.invoke([SystemMessage(content=complete_scheme_prompt_content)])
        complete_scheme_content = complete_response_msg.content

        new_scheme_entry = {"round": int(current_round), "proposal": str(complete_scheme_content)}
        existing_advice = self.state.get("design_advice", [])
        if not isinstance(existing_advice, list): 
            existing_advice = []
        updated_advice = custom_add_messages(existing_advice, [new_scheme_entry])
        self.state["design_advice"] = updated_advice

        print("✅ 設計建議已完成！")
        print(f"最終設計建議：{self.state['design_advice']}")
        return {"design_advice": self.state["design_advice"]}

# GATE 檢查方案（請回答：有/沒有） OK
class GateCheck1:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state
        
        active_config = ensure_graph_overall_config(config)
        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language

        design_advice_raw = self.state.get("design_advice", [])
        design_summary = self.state.get("design_summary", "無目標")

        def ensure_dict(item):
            if isinstance(item, dict):
                return item
            if isinstance(item, str):
                try:
                    # Try to parse if it's a JSON string representing a dict
                    parsed_item = json.loads(item)
                    if isinstance(parsed_item, dict):
                        return parsed_item
                except json.JSONDecodeError:
                    # If it's not a JSON string, and we expect dicts, this might be an issue.
                    # For now, we'll assume items are either dicts or JSON strings of dicts.
                    print(f"⚠️ 無法解析項目為字典: {item}")
                    return None # Or handle as per logic, e.g., wrap in a default dict structure
            print(f"⚠️ 項目不是字典或JSON字串: {item}")
            return None

        design_advice_list = []
        if isinstance(design_advice_raw, list):
            for item in design_advice_raw:
                d = ensure_dict(item)
                if d is not None:
                    design_advice_list.append(d)
        else:
             # Handle case where design_advice_raw is not a list (e.g. initial empty string)
            print(f"⚠️ design_advice_raw 不是列表: {design_advice_raw}")

        # 使用新條件：沒有 "state" 鍵的對象作為 current proposals
        current_proposals = [
            advice for advice in design_advice_list if isinstance(advice, dict) and "state" not in advice
        ]
        historical_proposals = [
            advice for advice in design_advice_list if isinstance(advice, dict) and "state" in advice
        ]

        if not current_proposals:
            print("⚠️ 當前無符合條件的設計建議方案（未找到不含 state 鍵的對象）。")
            self.state["GATE1"] = "沒有"
            self.state["GATE_REASON1"] = "當前無符合條件的設計建議方案（未找到不含 state 鍵的對象）"
            # Ensure design_advice remains a list
            if not isinstance(self.state.get("design_advice"), list):
                self.state["design_advice"] = []
            return {"GATE1": self.state["GATE1"], "GATE_REASON1": self.state["GATE_REASON1"], "design_advice": self.state["design_advice"]}

        # 格式化輸出以供 prompt 使用
        formatted_current = json.dumps(current_proposals, ensure_ascii=False, indent=2)
        formatted_previous = json.dumps(historical_proposals, ensure_ascii=False, indent=2)

        gate1_prompt_content = active_config.gate_check1_prompt_template.format(
            design_summary=design_summary,
            formatted_current=formatted_current,
            formatted_previous=formatted_previous,
            llm_output_language=active_language
        )

        llm_response_msg = current_llm.invoke([SystemMessage(content=gate1_prompt_content)])
        response_content = llm_response_msg.content
        response_lines = [line.strip() for line in response_content.splitlines() if line.strip()]
        
        evaluation_result = "沒有" # Default
        reason = "LLM 回覆格式不符或為空" # Default

        if not response_lines:
            print("⚠️ LLM 回覆為空，請檢查提示格式。")
        else:
            evaluation_result = response_lines[0]
            reason = response_lines[1] if len(response_lines) > 1 else "空"

        # 判斷評估結果，決定 state 鍵值
        state_value = True if evaluation_result == "有" else False

        # 為每個當前方案字典新增 state 鍵
        for advice_dict_item in current_proposals: # Renamed to avoid conflict
            if isinstance(advice_dict_item, dict): # Ensure it's a dict before assigning
                 advice_dict_item["state"] = state_value

        # 覆蓋 design_advice: 將 historical_proposals 與更新後的 current_proposals 合併
        # existing_advice was already processed into design_advice_list.
        # We need to reconstruct design_advice based on historical and updated current.
        updated_design_advice_list = historical_proposals + current_proposals
        self.state["design_advice"] = updated_design_advice_list

        # 最後 self.state["GATE1"] 僅返回評判結果（"有" 或 "没有"）
        self.state["GATE1"] = evaluation_result
        self.state["GATE_REASON1"] = reason

        print(f"【GateCheck】已收到評審結果：{evaluation_result}，原因：{reason}")
        return {"GATE1": self.state["GATE1"], "GATE_REASON1": self.state["GATE_REASON1"], "design_advice": self.state["design_advice"]}
        

# 外殼 Prompt 生成：呼叫 LLM（使用 prompt 生成工具）根據基地資訊與融合圖生成設計 prompt OK
class OuterShellPromptTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state

        active_config = ensure_graph_overall_config(config)
        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language

        current_round = self.state.get("current_round", 0)
        design_advice_list_raw = self.state.get("design_advice", [])
        
        # 整合來自 GateCheck1 和 GateCheck2 的改進建議
        improvement_from_gate1 = self.state.get("GATE_REASON1", "")
        improvement_from_gate2 = self.state.get("GATE_REASON2", "")

        improvement_texts = []
        if improvement_from_gate1:
            improvement_texts.append(f"對文字設計方案的改進建議: {improvement_from_gate1}")
        if improvement_from_gate2:
            improvement_texts.append(f"對上一批生成圖像的改進建議: {improvement_from_gate2}")

        improvement = "\n".join(improvement_texts) if improvement_texts else "無"
        
        design_advice_list = []
        if isinstance(design_advice_list_raw, list):
            design_advice_list = [item for item in design_advice_list_raw if isinstance(item, dict)]


        # 過濾出當前輪次且 state 為 True 的設計方案（必須是字典格式）
        valid_advices = [
            advice for advice in design_advice_list
            if advice.get("round") == current_round and advice.get("state") == True
        ]
        
        advice_text = "無目標"
        if valid_advices:
            selected_advice = valid_advices[0]
            advice_text = selected_advice.get("proposal", "無目標")
        else:
            print(f"⚠️ OuterShellPromptTask: 未找到輪次 {current_round} 且 state 為 True 的有效設計建議。")


        gpt_prompt_content = active_config.outer_shell_gpt_prompt_template.format(
            advice_text=advice_text,
            improvement=improvement,
            llm_output_language=active_language
        )

        gpt_output_msg = current_llm.invoke([SystemMessage(content=gpt_prompt_content)])
        final_prompt_text = gpt_output_msg.content if hasattr(gpt_output_msg, "content") else "❌ GPT 生成失敗"

        lora_guidance_prompt_content = active_config.outer_shell_lora_prompt_template.format(
            final_prompt=final_prompt_text,
            llm_output_language=active_language
        )

        gpt_output2_msg = current_llm.invoke([SystemMessage(content=lora_guidance_prompt_content)])
        lora_value_str = gpt_output2_msg.content.strip() if hasattr(gpt_output2_msg, "content") else "0.5"
        
        try:
            # Attempt to convert to float, ensure it's a number
            float(lora_value_str)
        except ValueError:
            print(f"⚠️ LoRA權重生成非數字 '{lora_value_str}', 使用預設值 0.5")
            lora_value_str = "0.5"


        new_prompt_entry = {"round":int(current_round),"prompt": str(final_prompt_text),"lora":str(lora_value_str)}
        
        existing_prompts_list = self.state.get("outer_prompt", [])
        if not isinstance(existing_prompts_list, list):
            existing_prompts_list = []
        
        updated_prompts = existing_prompts_list + [new_prompt_entry]
        self.state["outer_prompt"] = updated_prompts


        print("✅ 生成外殼 Prompt 完成！")
        print(f"📌 外殼 Prompt: {final_prompt_text}, LoRA: {lora_value_str}")
        return {"outer_prompt": self.state["outer_prompt"]}

# 方案情境生成：呼叫 LLM（使用圖片生成工具）根據外殼 prompt 與融合圖生成未來情境圖 OK
class CaseScenarioGenerationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state

        active_config = ensure_graph_overall_config(config)
        current_round = self.state.get("current_round", 0)
        outer_prompt_list_raw = self.state.get("outer_prompt", [])
        
        # 從配置中讀取要生成的圖片數量
        num_images_to_generate = active_config.case_scenario_image_count

        outer_prompt_list = []
        if isinstance(outer_prompt_list_raw, list):
            outer_prompt_list = [item for item in outer_prompt_list_raw if isinstance(item, dict)]

        # 篩選出當前輪次且不含 "state" 鍵的字典 (i.e., the latest one for this round)
        current_round_prompts = [
            item for item in outer_prompt_list 
            if item.get("round") == current_round and "state" not in item
        ]

        prompt_to_use = ""
        lora_to_use = "0.5" # Default

        if current_round_prompts:
            latest_prompt_entry = current_round_prompts[-1] # Get the last one for the current round
            prompt_to_use = latest_prompt_entry.get("prompt", "")
            lora_to_use = latest_prompt_entry.get("lora", "0.5")
        else:
            print(f"⚠️ CaseScenarioGenerationTask: 未找到輪次 {current_round} 的外殼 prompt。")
            if outer_prompt_list:
                latest_prompt_entry = outer_prompt_list[-1]
                prompt_to_use = latest_prompt_entry.get("prompt", "")
                lora_to_use = latest_prompt_entry.get("lora", "0.5")
                print(f"↪️  使用最後一個可用的 prompt: {prompt_to_use}")


        if not prompt_to_use:
            print("❌ CaseScenarioGenerationTask: 無可用 prompt 生成圖片。")
            if not isinstance(self.state.get("case_image"), list):
                self.state["case_image"] = []
            # 返回一個表示失敗的條目或保持為空列表
            self.state["case_image"] = custom_add_messages(
                self.state.get("case_image", []), 
                [{"round": current_round, "id_in_round": 1, "filename": "無Prompt生成失敗", "image_url": "未生成", "path": "無", "description": "無可用 prompt 生成圖片。"}]
            )
            return {"case_image": self.state["case_image"]}

        generated_image_infos = []
        # 與 GateCheck2 和 UnifiedImageGenerationTask 統一快取目錄
        render_cache_dir = os.path.join(os.getcwd(), "output", "render_cache")
        os.makedirs(render_cache_dir, exist_ok=True)

        # 調用一次 case_render_image 工具，讓它根據 num_images_to_generate 生成所有圖片
        all_generated_filenames_str = case_render_image.invoke({
            "current_round": current_round,
            "outer_prompt": prompt_to_use,
            "i": num_images_to_generate, # 生成總數
            "strength": lora_to_use
        })

        generated_filenames_list = []
        if all_generated_filenames_str and isinstance(all_generated_filenames_str, str):
            # 工具返回逗號分隔的檔名
            generated_filenames_list = [fn.strip() for fn in all_generated_filenames_str.split(',') if fn.strip()]
        
        if not generated_filenames_list:
            print(f"⚠️ 圖片生成工具未返回任何有效文件名 (工具返回: {all_generated_filenames_str})。")
            generated_image_infos.append({
                "round": current_round,
                "id_in_round": 1, # 標記一個錯誤條目
                "filename": "工具未返回文件名",
                "image_url": "未生成",
                "path": "無",
                "description": f"圖片生成工具未返回任何有效文件名 (工具返回: {all_generated_filenames_str})。"
            })
        else:
            print(f"🖼️ 工具返回了 {len(generated_filenames_list)} 個文件名: {generated_filenames_list}")
            for idx, generated_filename in enumerate(generated_filenames_list):
                image_url = "未生成"
                path_for_state = "處理失敗或文件未找到"
                description = f"Round {current_round}, Image {idx+1}/{num_images_to_generate}."
                # 確保文件名是有效的字符串且不是錯誤標記
                if generated_filename and isinstance(generated_filename, str) and \
                   generated_filename not in ["生成失敗", "文件未找到", "工具未返回文件名"]: # 添加更多可能的錯誤標記
                    
                    image_path_in_cache = os.path.join(render_cache_dir, generated_filename)
                    if os.path.exists(image_path_in_cache):
                        try:
                            with open(image_path_in_cache, "rb") as image_file:
                                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                            image_url = f"data:image/png;base64,{encoded_image}" # 假設總是PNG，或從文件名推斷
                            path_for_state = image_path_in_cache
                            description += f" Successfully processed file '{generated_filename}'."
                            print(f"✅ 成功處理圖片: {generated_filename}")
                        except Exception as e:
                            print(f"⚠️ 無法讀取或編碼圖片文件 {generated_filename}: {e}")
                            image_url = "讀取或編碼失敗"
                            path_for_state = image_path_in_cache # 路徑存在但處理失敗
                            description += f" Failed to read or encode file '{generated_filename}': {e}."
                    else:
                        print(f"⚠️ 工具聲稱生成了圖片 '{generated_filename}' 但在路徑 '{image_path_in_cache}' 未找到。")
                        image_url = "文件於快取未找到"
                        path_for_state = image_path_in_cache # 記錄下嘗試的路徑
                        description += f" File '{generated_filename}' not found at expected path."
                else:
                    print(f"⚠️ 工具返回了無效或錯誤標記的檔名: '{generated_filename}'")
                    image_url = "生成失敗（工具報告）"
                    description += f" Tool returned an invalid or error filename: '{generated_filename}'."

                generated_image_infos.append({
                    "round": current_round,
                    "id_in_round": idx + 1, # 使用列表索引+1作為輪次內ID，與工具內部迭代對應
                    "filename": generated_filename,
                    "image_url": image_url,
                    "path": path_for_state,
                    "description": description
                })

        existing_images_list = self.state.get("case_image", [])
        if not isinstance(existing_images_list, list):
            existing_images_list = []
        
        updated_images_list = existing_images_list + generated_image_infos # 直接添加列表
        self.state["case_image"] = updated_images_list

        print(f"✅ 方案情境圖處理完成，共處理 {len(generated_image_infos)} 條圖片資訊。")
        print(f"詳細圖片資訊: {generated_image_infos}")
        return {"case_image": self.state["case_image"]}    

# class UnifiedImageGenerationTask:
#     def __init__(self, state: GlobalState):
#         self.state = state

#     def run(self, state: GlobalState, config: GraphOverallConfig | dict):
#         if state is not None:
#             self.state = state

#         active_config = ensure_graph_overall_config(config)
#         current_llm = active_config.llm_config.get_llm() # 用於格式化 prompt
#         active_language = active_config.llm_output_language
        
#         current_round = self.state.get("current_round", 0)
#         design_advice_list_raw = self.state.get("design_advice", [])
#         improvement = self.state.get("GATE_REASON1", "") 
#         num_tool_calls = active_config.case_scenario_image_count 

#         design_advice_list = []
#         if isinstance(design_advice_list_raw, list):
#             design_advice_list = [item for item in design_advice_list_raw if isinstance(item, dict)]

#         valid_advices = [
#             advice for advice in design_advice_list
#             if advice.get("round") == current_round and advice.get("state") == True
#         ]
        
#         advice_text = "A creative timber pavilion." 
#         if valid_advices:
#             selected_advice = valid_advices[0] 
#             advice_text = selected_advice.get("proposal", advice_text)
#         else:
#             print(f"⚠️ UnifiedImageGenerationTask: 未找到輪次 {current_round} 且 state 為 True 的有效設計建議。將使用預設提案。")

#         image_gen_prompt_text_template = active_config.outer_shell_gpt_prompt_template.format(
#             advice_text=advice_text,
#             improvement=improvement,
#             llm_output_language=active_language 
#         )
        
#         final_image_prompt_msg = current_llm.invoke([SystemMessage(content=image_gen_prompt_text_template)])
#         final_image_prompt = final_image_prompt_msg.content.strip() if hasattr(final_image_prompt_msg, "content") else "Error generating image prompt."
        
#         if "Error generating image prompt" in final_image_prompt:
#              print(f"❌ UnifiedImageGenerationTask: LLM 生成圖像 Prompt 失敗。")
#         else:
#             print(f"✅ UnifiedImageGenerationTask: 生成的最終圖像 Prompt (用於所有調用): '{final_image_prompt[:200]}...'")

#         existing_outer_prompts = self.state.get("outer_prompt", [])
#         if not isinstance(existing_outer_prompts, list):
#             existing_outer_prompts = []
        
#         prompts_from_other_rounds = [
#             p for p in existing_outer_prompts if isinstance(p, dict) and p.get("round") != current_round
#         ]
#         new_prompt_entry = {"round": current_round, "prompt": final_image_prompt}
#         self.state["outer_prompt"] = prompts_from_other_rounds + [new_prompt_entry]
#         print(f"ℹ️ UnifiedImageGenerationTask: outer_prompt 已更新，當前輪次 {current_round} 的 prompt: '{final_image_prompt[:100]}...'")

#         generated_image_infos = []
        
#         base_render_cache_dir = os.path.join(os.getcwd(), "output", "cache", "render_cache")
#         os.makedirs(base_render_cache_dir, exist_ok=True) 


#         if "Error generating image prompt" in final_image_prompt or not final_image_prompt:
#             print(f"❌ UnifiedImageGenerationTask: 因 Prompt 生成失敗，跳過所有圖像生成調用。")
#             existing_case_images = self.state.get("case_image", [])
#             if not isinstance(existing_case_images, list): existing_case_images = []
#             error_entry = {
#                 "round": current_round,
#                 "id_in_round": 1, 
#                 "filename": "Prompt生成失敗", # Basename
#                 "image_url": "未生成",
#                 "description": "LLM failed to generate a valid image prompt.",
#                 "path": "無" 
#             }
#             images_from_other_rounds_img = [img for img in existing_case_images if isinstance(img, dict) and img.get("round") != current_round]
#             self.state["case_image"] = images_from_other_rounds_img + [error_entry]
#             return {
#                 "case_image": self.state["case_image"],
#                 "outer_prompt": self.state["outer_prompt"]
#             }

#         for call_idx in range(num_tool_calls):
#             print(f"ℹ️ UnifiedImageGenerationTask: 開始第 {call_idx + 1}/{num_tool_calls} 次圖像生成調用...")
            
#             image_path_to_store = f"處理錯誤_{call_idx + 1}.png" 
#             image_url = "未生成"
#             full_description = ""
#             tool_error_desc = None
#             current_call_tool_text_response = None
#             file_type_for_url = "image/png" 

#             try:
#                 tool_output = generate_gemini_image.invoke({
#                     "prompt": final_image_prompt, 
#                     "image_inputs": [], 
#                     "i": 1
#                 })

#                 current_call_tool_generated_files = []
#                 current_call_tool_image_bytes_list = [] 
                
#                 if isinstance(tool_output, dict):
#                     current_call_tool_generated_files = tool_output.get("generated_files", [])
#                     current_call_tool_image_bytes_list = tool_output.get("image_bytes", []) 
#                     current_call_tool_text_response = tool_output.get("text_response")
#                     tool_error_desc = tool_output.get("error")

#                     if current_call_tool_text_response:
#                         print(f"  ↪️ 調用 {call_idx + 1}: 工具文字回饋: '{current_call_tool_text_response[:100]}...'")

#                     if tool_error_desc:
#                         print(f"  ⚠️ 調用 {call_idx + 1}: 圖像生成工具報告錯誤: {tool_error_desc}")
#                         image_path_to_store = f"工具錯誤_調用{call_idx + 1}.png" # This will become a basename later
#                     elif not current_call_tool_generated_files:
#                         print(f"  ⚠️ 調用 {call_idx + 1}: 工具未返回任何文件資訊。 Files: {current_call_tool_generated_files}")
#                         tool_error_desc = "Tool did not return any file information."
#                         image_path_to_store = f"無文件資訊_調用{call_idx + 1}.png" # Basename
#                     else:
#                         file_info = current_call_tool_generated_files[0]
#                         file_type_for_url = file_info.get("file_type", "image/png") 
                        
#                         path_from_file_info = file_info.get("path")
#                         filename_from_file_info = file_info.get("filename") 

#                         resolved_image_path = None
#                         if isinstance(path_from_file_info, str) and os.path.isabs(path_from_file_info):
#                             resolved_image_path = path_from_file_info
#                             print(f"  DEBUG UnifiedImageGenerationTask: 使用工具提供的絕對路徑 'path': '{resolved_image_path}'")
#                         elif isinstance(filename_from_file_info, str):
#                             resolved_image_path = os.path.join(base_render_cache_dir, os.path.basename(filename_from_file_info))
#                             print(f"  DEBUG UnifiedImageGenerationTask: 從 'filename' ('{filename_from_file_info}') 和 cache_dir 構造路徑: '{resolved_image_path}'")
#                         else:
#                             tool_error_desc = "Tool returned invalid 'path' or 'filename' in file_info."
#                             image_path_to_store = f"路徑無效_調用{call_idx + 1}.png" # Basename
#                             print(f"  ⚠️ 調用 {call_idx + 1}: 工具返回的文件資訊中 'path' 和 'filename' 均無效。 Path: {path_from_file_info}, Filename: {filename_from_file_info}")
                        
#                         if resolved_image_path:
#                             image_path_to_store = resolved_image_path 
#                             print(f"  DEBUG UnifiedImageGenerationTask: 解析得到的待檢查路徑: '{image_path_to_store}' (類型: {type(image_path_to_store)}) for call {call_idx + 1}")

#                             img_bytes_data_for_url = None
#                             if current_call_tool_image_bytes_list and isinstance(current_call_tool_image_bytes_list[0].get("data"), bytes):
#                                 img_bytes_data_for_url = current_call_tool_image_bytes_list[0].get("data")
#                                 print(f"  ℹ️ 調用 {call_idx + 1}: 工具直接返回了圖片字節數據。")
                            
#                             if not os.path.exists(image_path_to_store):
#                                  print(f"  ⚠️ 調用 {call_idx + 1}: 解析後的圖片路徑 '{image_path_to_store}' 文件不存在。")
#                                  tool_error_desc = tool_error_desc or f"Resolved image file does not exist: {os.path.basename(image_path_to_store)}"
#                             else:
#                                 if not img_bytes_data_for_url:
#                                     print(f"  ℹ️ 調用 {call_idx + 1}: 文件 '{os.path.basename(image_path_to_store)}' 存在，但工具未直接返回字節。嘗試從文件讀取以生成URL...")
#                                     try:
#                                         with open(image_path_to_store, "rb") as f_read:
#                                             img_bytes_data_for_url = f_read.read()
#                                         print(f"    ✅ 成功從文件讀取字節數據: {os.path.basename(image_path_to_store)}")
#                                     except Exception as e_read_file:
#                                         print(f"    ⚠️ 從文件讀取字節數據失敗: {os.path.basename(image_path_to_store)}, Error: {e_read_file}")
#                                         tool_error_desc = tool_error_desc or f"Failed to read file bytes: {e_read_file}"
                                
#                                 if img_bytes_data_for_url:
#                                     try:
#                                         encoded_image = base64.b64encode(img_bytes_data_for_url).decode('utf-8')
#                                         image_url = f"data:{file_type_for_url};base64,{encoded_image}" 
#                                         print(f"  ✅ 調用 {call_idx + 1}: 成功處理圖片並生成URL: {os.path.basename(image_path_to_store)}")
#                                     except Exception as e_encode:
#                                         print(f"  ⚠️ 調用 {call_idx + 1}: 無法編碼圖片數據 for {os.path.basename(image_path_to_store)}: {e_encode}")
#                                         image_url = "編碼失敗"
#                                         tool_error_desc = tool_error_desc or f"Encoding failed: {e_encode}"
#                                 elif not tool_error_desc : 
#                                      image_url = "讀取字節失敗"
#                 else: 
#                     print(f"  ⚠️ 調用 {call_idx + 1}: 工具返回了意外的輸出格式: {type(tool_output)}")
#                     tool_error_desc = "Unexpected tool output format."
#                     image_path_to_store = f"格式錯誤_調用{call_idx + 1}.png" # Basename

#             except Exception as e_invoke:
#                 print(f"  💥 調用 {call_idx + 1} 期間調用工具時發生意外錯誤: {e_invoke}")
#                 tool_error_desc = f"Exception during tool call: {e_invoke}"
#                 image_path_to_store = f"調用異常_{call_idx+1}.png" # Basename
            
#             base_description = (
#                 f"Agent: UnifiedImageGeneration; Round: {current_round}; "
#                 f"CallNum: {call_idx + 1}/{num_tool_calls}; " 
#                 f"Prompt: {final_image_prompt[:50]}..."
#             )
#             full_description = base_description
#             if current_call_tool_text_response: 
#                 full_description += f" | ToolTextResponse: {current_call_tool_text_response}"
#             if tool_error_desc:
#                 full_description += f" | Error: {tool_error_desc}"
#                 if "處理錯誤" in image_path_to_store and not os.path.isabs(image_path_to_store): 
#                     image_path_to_store = f"具體錯誤_{call_idx + 1}_{tool_error_desc[:20].replace(' ','_')}.png" # Basename

#             # 準備存儲到 state 的數據
#             # final_path_for_state 應該是絕對路徑或標準化的錯誤標記
#             # final_filename_for_state 應該是 basename 或標準化的錯誤標記
            
#             final_path_for_state = "路徑錯誤或生成失敗" # Default error path
#             final_filename_for_state = f"處理錯誤_{call_idx + 1}.png" # Default error filename (basename)

#             if os.path.isabs(image_path_to_store): # 如果 image_path_to_store 已經是絕對路徑
#                 if os.path.exists(image_path_to_store):
#                     final_path_for_state = image_path_to_store
#                     final_filename_for_state = os.path.basename(image_path_to_store)
#                 else: # 絕對路徑但文件不存在
#                     final_path_for_state = image_path_to_store # 存儲嘗試的路徑
#                     final_filename_for_state = os.path.basename(image_path_to_store) + "_文件不存在"
#                     # image_url 應已是 "未生成" 或錯誤狀態
#             elif not any(err_tag in image_path_to_store for err_tag in ["錯誤", "無效", "異常"]):
#                 # 如果 image_path_to_store 不是絕對路徑且不是已知錯誤標記 (例如，它是從工具返回的 basename)
#                 potential_abs_path = os.path.join(base_render_cache_dir, os.path.basename(image_path_to_store))
#                 if os.path.exists(potential_abs_path):
#                     final_path_for_state = potential_abs_path
#                     final_filename_for_state = os.path.basename(potential_abs_path)
#                 else:
#                     final_path_for_state = potential_abs_path # 存儲嘗試的路徑
#                     final_filename_for_state = os.path.basename(image_path_to_store) + "_文件不存在"
#             else: # image_path_to_store 本身就是一個錯誤標記 (例如 "工具錯誤_...")
#                 final_filename_for_state = image_path_to_store # 使用這個錯誤標記作為檔名
#                 # final_path_for_state 保持為 "路徑錯誤或生成失敗"

#             generated_image_infos.append({
#                 "round": current_round,
#                 "id_in_round": call_idx + 1, 
#                 "filename": final_filename_for_state, # 存儲 basename 或錯誤標記
#                 "image_url": image_url,
#                 "description": full_description,
#                 "path": final_path_for_state # 存儲絕對路徑或標準化錯誤標記
#             })
            
#             if call_idx < num_tool_calls - 1:
#                 print(f"  ℹ️ 調用 {call_idx + 1} 完成後延遲 5 秒...") 
#                 time.sleep(5)

#         existing_images_list = self.state.get("case_image", [])
#         if not isinstance(existing_images_list, list):
#             existing_images_list = []
        
#         images_from_other_rounds_img_final = [img for img in existing_images_list if isinstance(img, dict) and img.get("round") != current_round]
#         updated_images_list = images_from_other_rounds_img_final + generated_image_infos
#         self.state["case_image"] = updated_images_list

#         print(f"✅ UnifiedImageGenerationTask: 所有圖像生成調用完成，共處理 {len(generated_image_infos)} 條圖片資訊。")
#         if generated_image_infos:
#             for idx, info in enumerate(generated_image_infos):
#                  # 在最終日誌中，filename 應該只顯示檔名部分
#                  display_filename = info.get('filename', '未知檔名')
#                  if isinstance(display_filename, str) and os.path.isabs(display_filename) and not any(err_tag in display_filename for err_tag in ["錯誤", "無效", "異常", "失敗"]):
#                      display_filename = os.path.basename(display_filename)

#                  print(f"  詳細圖片資訊 ({idx+1}): Filename='{display_filename}', Path='{info.get('path')}', URL: {'有內容' if info.get('image_url') and info.get('image_url').startswith('data:image') else info.get('image_url', '未定義')}")
        
#         return {
#             "case_image": self.state["case_image"],
#             "outer_prompt": self.state["outer_prompt"] 
#         }

# GATE 檢查方案（請回答：有/沒有） OK
class GateCheck2:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state

        active_config = ensure_graph_overall_config(config)
        current_llm = active_config.llm_config.get_llm() # 雖然 img_recognition 主要使用，但 prompt 模板可能需要語言
        active_language = active_config.llm_output_language

        current_round = self.state.get("current_round", 0) 
        case_images_raw = self.state.get("case_image", [])
        design_advice_list_raw = self.state.get("design_advice", [])
        
        case_images_list = []
        if isinstance(case_images_raw, list):
            case_images_list = [item for item in case_images_raw if isinstance(item, dict)]

        design_advice_list = []
        if isinstance(design_advice_list_raw, list):
            design_advice_list = [item for item in design_advice_list_raw if isinstance(item, dict)]


        # 過濾出當前輪次且 state 為 True 的設計方案（必須是字典格式）
        valid_advices = [
            advice for advice in design_advice_list
            if advice.get("round") == current_round and advice.get("state") == True
        ]
        
        advice_text = "無目標"
        if valid_advices:
            selected_advice = valid_advices[0]
            advice_text = selected_advice.get("proposal", "無目標")
        else:
            print(f"⚠️ GateCheck2: 未找到輪次 {current_round} 且 state 為 True 的有效設計建議。")


        # Filter images for the current round from self.state["case_image"]
        current_round_image_infos = [
            img_info for img_info in case_images_list
            if img_info.get("round") == current_round and 
               isinstance(img_info.get("filename"), str) and # Ensure filename is a string
               img_info.get("filename") not in ["未生成", "文件未找到", "生成失敗", "工具報告錯誤", "工具未返回文件名", "Prompt生成失敗", "無效文件名"] # 更多可能的錯誤標記
        ]

        if not current_round_image_infos:
            print(f"⚠️ GateCheck2: 當前輪次 {current_round} 無符合條件的生成圖。篩選後的列表: {current_round_image_infos}")
            self.state["GATE2"] = "没有"
            self.state["GATE_REASON2"] = f"當前輪次 {current_round} 無符合條件的生成圖。"
            # outer_prompt 是舊的邏輯，這裡應該不需要再處理它，因為 UnifiedImageGenerationTask 不依賴 outer_prompt
            # if not isinstance(self.state.get("outer_prompt"), list):
            #     self.state["outer_prompt"] = []
            # return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"], "outer_prompt": self.state["outer_prompt"]}
            return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"]}


        # 不再需要手動組合路徑，直接從 state 讀取
        image_paths_for_tool = []
        image_filenames_for_prompt_list = []

        current_round_image_infos.sort(key=lambda x: x.get("id_in_round", 0))

        for img_info in current_round_image_infos:
            image_path = img_info.get("path")
            filename = img_info.get("filename")

            # 主要檢查 path 欄位
            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                image_paths_for_tool.append(image_path)
                image_filenames_for_prompt_list.append(f"{os.path.basename(filename)} (ID: {img_info.get('id_in_round')})")
            else:
                print(f"⚠️ GateCheck2: 圖片文件在 state 提供的路徑 '{image_path}' 中未找到。ImgInfo: {img_info}")


        if not image_paths_for_tool:
            print(f"⚠️ GateCheck2: 當前輪次 {current_round} 所有圖片文件均未找到或路徑無效。")
            self.state["GATE2"] = "没有"
            self.state["GATE_REASON2"] = f"當前輪次 {current_round} 所有圖片文件均未找到或路徑無效。"
            # outer_prompt 處理同上
            # if not isinstance(self.state.get("outer_prompt"), list):
            #     self.state["outer_prompt"] = []
            # return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"], "outer_prompt": self.state["outer_prompt"]}
            return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"]}

        image_list_str_for_prompt = "\n".join(image_filenames_for_prompt_list)

        gate2_prompt_content = active_config.gate_check2_img_recognition_prompt_template.format(
            image_list_str=image_list_str_for_prompt,
            advice_text=advice_text,
            llm_output_language=active_language,
            current_round=current_round 
        )

        analysis_result_str = img_recognition.invoke({
            "image_paths": image_paths_for_tool,
            "prompt": gate2_prompt_content,
        })

        result_content = analysis_result_str.strip() if isinstance(analysis_result_str, str) else ""
        lines = [line.strip() for line in result_content.splitlines() if line.strip()]
        
        best_id_from_llm = "没有"
        reason_from_llm = ""

        if lines:
            first_line = lines[0]
            if "没有" in first_line or "no" in first_line.lower():
                best_id_from_llm = "没有"
            else:
                # 從 filename (ID: X) 中提取 ID
                # 例如 "gemini_gen_xxxx.png (ID: 1)" -> 提取 1
                id_matches = re.findall(r'\(ID:\s*(\d+)\)', first_line)
                if id_matches: # 如果是直接提供ID
                    try:
                        best_id_from_llm = int(id_matches[0])
                    except ValueError:
                         print(f"⚠️ GateCheck2: 無法從LLM回覆的第一行解析ID (格式不符): '{first_line}'")
                         best_id_from_llm = "没有"
                else: # 嘗試從純數字中提取
                    digit_matches = re.findall(r'\b\d+\b', first_line)
                    if digit_matches:
                        try:
                            best_id_from_llm = int(digit_matches[0])
                        except ValueError:
                            print(f"⚠️ GateCheck2: 無法從LLM回覆的第一行解析數字ID: '{first_line}'")
                            best_id_from_llm = "没有"
                    else:
                        print(f"⚠️ GateCheck2: LLM 回覆的第一行未找到數字ID: '{first_line}'")
                        best_id_from_llm = "没有"
            
            if len(lines) >= 2:
                reason_from_llm = lines[1]
            elif best_id_from_llm != "没有":
                 reason_from_llm = "LLM 未提供選擇原因。"
            else: # best_id_from_llm 是 "没有"
                 reason_from_llm = lines[1] if len(lines) >= 2 else "LLM 未提供改進建議。"


        self.state["GATE2"] = best_id_from_llm
        self.state["GATE_REASON2"] = reason_from_llm

        # 舊的 outer_prompt 狀態更新邏輯已不再需要，因為 UnifiedImageGenerationTask 不依賴 outer_prompt
        # prompt_state_value = False if self.state["GATE2"] == "没有" else True
        # outer_prompt_list_for_state = self.state.get("outer_prompt", [])
        # if not isinstance(outer_prompt_list_for_state, list):
        #     outer_prompt_list_for_state = []
        # updated_outer_prompt = False
        # for prompt_entry in reversed(outer_prompt_list_for_state):
        #     if isinstance(prompt_entry, dict) and prompt_entry.get("round") == current_round:
        #         if "state" not in prompt_entry:
        #             prompt_entry["state"] = prompt_state_value
        #             updated_outer_prompt = True
        #             break 
        # if not updated_outer_prompt and outer_prompt_list_for_state:
        #     print(f"⚠️ GateCheck2: 未找到輪次 {current_round} 的外殼 prompt 來更新狀態。")
        # self.state["outer_prompt"] = outer_prompt_list_for_state

        print(f"【GateCheckCaseImage】已收到最佳評估結果：{self.state.get('GATE2')}，原因：{self.state.get('GATE_REASON2')} 😊")
        # return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"], "outer_prompt": self.state["outer_prompt"]}
        return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"]}

# 未來情境生成：使用 generate_gemini_image 生成方案細節和未來變化圖
class FutureScenarioGenerationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def _save_and_encode_image(self, image_bytes: bytes, absolute_filepath: str, extension: str, description: str, current_round: int, sub_id: str) -> dict:
        """輔助函數：處理工具返回的單個圖片字節和文件名，進行編碼並構建標準圖片資訊字典。
        absolute_filepath 應為圖片的絕對路徑。
        """
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/{extension};base64,{encoded_image}"
            
            return {
                "round": current_round,
                "id_in_round": sub_id, 
                "filename": os.path.basename(absolute_filepath), # 儲存純檔案名稱
                "image_url": image_url,
                "description": description,
                "path": absolute_filepath # 儲存絕對路徑
            }
        except Exception as e:
            print(f"⚠️ FutureScenario (_save_and_encode_image): 無法編碼圖片數據 for {absolute_filepath}: {e}")
            return {
                "round": current_round,
                "id_in_round": sub_id,
                "filename": os.path.basename(absolute_filepath) if absolute_filepath else "編碼失敗.png",
                "image_url": "無",
                "description": f"圖片數據編碼失敗: {description}",
                "error": str(e),
                "path": absolute_filepath if absolute_filepath else "編碼失敗路徑" 
            }

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state

        active_config = ensure_graph_overall_config(config)
        active_language = active_config.llm_output_language
        
        current_round = self.state.get("current_round", 0)
        design_advice_list_raw = self.state.get("design_advice", [])
        outer_prompt_list_raw = self.state.get("outer_prompt", [])
        case_images_list_raw = self.state.get("case_image", [])
        gate2_result_id = self.state.get("GATE2") 

        generated_future_images = []
        
        base_render_cache_dir = os.path.join(os.getcwd(), "output", "cache", "render_cache")
        os.makedirs(base_render_cache_dir, exist_ok=True)

        base_design_text_for_prompt = "一個具有創新性的木構造亭子。"
        design_advice_list = [item for item in design_advice_list_raw if isinstance(item, dict)]
        valid_current_round_advice = [
            adv for adv in design_advice_list 
            if adv.get("round") == current_round and adv.get("state") == True
        ]
        if valid_current_round_advice:
            base_design_text_for_prompt = valid_current_round_advice[0].get("proposal", base_design_text_for_prompt)
            print(f"FutureScenario: 使用來自 design_advice 的基礎設計文本: {base_design_text_for_prompt[:100]}...")
        else:
            outer_prompt_list = [item for item in outer_prompt_list_raw if isinstance(item, dict)]
            current_round_outer_prompts = [p for p in outer_prompt_list if p.get("round") == current_round]
            if current_round_outer_prompts:
                base_design_text_for_prompt = current_round_outer_prompts[-1].get("prompt", base_design_text_for_prompt)
                print(f"FutureScenario: 使用來自 outer_prompt (輪次 {current_round}) 的基礎設計文本: {base_design_text_for_prompt[:100]}...")
            else:
                print(f"FutureScenario: 未找到當前輪次有效的 design_advice 或 outer_prompt，使用預設設計文本。")
        
        base_image_bytes_for_input = None
        base_image_mime_type_for_input = "image/png" 
        base_image_filename_for_desc = "無基礎圖"
        image_inputs_for_tool = []

        print(f"FutureScenario: 嘗試查找 GATE2 ID: {gate2_result_id} (類型: {type(gate2_result_id)}) 在輪次 {current_round} 的基礎圖片。")
        if isinstance(gate2_result_id, int) and case_images_list_raw:
            case_images_list = [item for item in case_images_list_raw if isinstance(item, dict)]
            found_base_image = False
            for img_info in case_images_list:
                img_id_in_round = img_info.get("id_in_round")
                img_round = img_info.get("round")
                
                if img_round == current_round and img_id_in_round == gate2_result_id:
                    selected_path_from_case_image = img_info.get("path")
                    print(f"  FutureScenario: 找到候選圖片資訊: ID={img_id_in_round}, Round={img_round}, Path='{selected_path_from_case_image}'")

                    if isinstance(selected_path_from_case_image, str) and \
                       selected_path_from_case_image.strip() and \
                       selected_path_from_case_image.lower() not in ["無", "路徑錯誤或生成失敗", "none", "編碼失敗"] and \
                       not any(err_tag in selected_path_from_case_image.lower() for err_tag in ["失敗", "異常", "無效", "错误"]) and \
                       os.path.exists(selected_path_from_case_image):
                        try:
                            with open(selected_path_from_case_image, "rb") as f_img:
                                base_image_bytes_for_input = f_img.read()
                            
                            image_url_from_case = img_info.get("image_url")
                            if isinstance(image_url_from_case, str) and image_url_from_case.startswith("data:image/"):
                                base_image_mime_type_for_input = image_url_from_case.split(';')[0].split(':')[1]
                            elif selected_path_from_case_image.lower().endswith((".jpg", ".jpeg")):
                                base_image_mime_type_for_input = "image/jpeg"
                            elif selected_path_from_case_image.lower().endswith(".png"):
                                base_image_mime_type_for_input = "image/png"
                            
                            base_image_filename_for_desc = os.path.basename(selected_path_from_case_image)
                            image_inputs_for_tool = [{"data": base_image_bytes_for_input, "mime_type": base_image_mime_type_for_input}]
                            print(f"FutureScenario: ✅ 成功找到並加載基礎圖片: '{base_image_filename_for_desc}' (type: {base_image_mime_type_for_input}) 使用路徑: {selected_path_from_case_image}")
                            found_base_image = True
                            break 
                        except Exception as e_read:
                            print(f"FutureScenario: ⚠️ 嘗試讀取基礎圖片 '{selected_path_from_case_image}' 失敗: {e_read}")
                            base_image_bytes_for_input = None 
                            base_image_filename_for_desc = "讀取失敗"
                            image_inputs_for_tool = []
                    else:
                        print(f"  FutureScenario: 候選圖片路徑 '{selected_path_from_case_image}' 無效或文件不存在。")
            
            if not found_base_image:
                 print(f"FutureScenario: ℹ️ 在 case_image 輪次 {current_round} 中未找到 ID 為 {gate2_result_id} 的有效基礎圖片條目。")
        else:
            if not isinstance(gate2_result_id, int):
                 print(f"FutureScenario: Gate2 結果 '{gate2_result_id}' 不是有效的整數 ID。")
            if not case_images_list_raw:
                 print("FutureScenario: case_image 列表為空。")
        
        if not image_inputs_for_tool: 
            print(f"FutureScenario: ⚠️ 未找到或無法讀取有效的基礎圖片 (GATE2 ID: {gate2_result_id})，或基礎圖片列表為空。Phase 1 和 Phase 2 將不使用基礎圖片。")
            
        # --- Phase 1: Facade Detail and Construction Method Generation (Realigned with Phase 2 Logic) ---
        num_detail_images_to_generate = active_config.future_scenario_detail_image_count
        print(f"\n--- FutureScenario: Phase 1 - 生成立面細節與構造工法圖 (請求 {num_detail_images_to_generate} 張) ---")
        
        detail_prompt_template = active_config.future_scenario_detail_generation_prompt_template # Use the new unified template
        facade_detail_prompt_text = detail_prompt_template.format(
            base_design_description=base_design_text_for_prompt, # Always provide base text
            num_images=num_detail_images_to_generate,
            llm_output_language=active_language
        )
        
        if image_inputs_for_tool:
            facade_detail_prompt_text += "\nA base image has been provided; please show detail modifications on it or generate details inspired by it."
            print(f"  立面細節 Prompt (含基礎圖提示): {facade_detail_prompt_text[:200]}...")
        else:
            facade_detail_prompt_text += "\nNo base image was provided; generate details based on the text description."
            print(f"  立面細節 Prompt (純文字提示): {facade_detail_prompt_text[:200]}...")

        try:
            tool_result_details = generate_gemini_image.invoke({
                "prompt": facade_detail_prompt_text,
                "image_inputs": image_inputs_for_tool, 
                "i": num_detail_images_to_generate 
            })

            if tool_result_details.get("error"):
                print(f"  ⚠️ Phase 1 圖像生成失敗: {tool_result_details.get('error')}")
                for i_err in range(num_detail_images_to_generate):
                    generated_future_images.append({
                        "round": current_round, "id_in_round": f"detail_err_batch_img{i_err+1}",
                        "filename": f"細節生成失敗_img{i_err+1}.png", "image_url": "無", "path": "無",
                        "description": f"立面細節圖批次生成失敗: {tool_result_details.get('error')}",
                        "error": tool_result_details.get('error')
                    })
            else:
                returned_files_info_detail = tool_result_details.get("generated_files", [])
                returned_bytes_info_detail = tool_result_details.get("image_bytes", []) 
                
                print(f"  DEBUG Phase 1: 工具返回 {len(returned_files_info_detail)} 個文件資訊, {len(returned_bytes_info_detail)} 個字節項目。預期 {num_detail_images_to_generate} 個。")

                if returned_files_info_detail : 
                    print(f"  Phase 1 工具返回 {len(returned_files_info_detail)} 個文件資訊。")
                    if len(returned_files_info_detail) != num_detail_images_to_generate:
                        print(f"  ⚠️ Phase 1 警告: 工具返回的文件數量 ({len(returned_files_info_detail)}) 與預期 ({num_detail_images_to_generate}) 不符。")

                    for idx, file_info in enumerate(returned_files_info_detail):
                        filename_from_tool = file_info.get("filename") 
                        img_mime = file_info.get("file_type", "image/png") 
                        img_bytes = None
                        img_abs_path = None

                        if isinstance(filename_from_tool, str) and filename_from_tool.strip():
                            img_abs_path = os.path.join(base_render_cache_dir, os.path.basename(filename_from_tool))
                        else:
                            print(f"    ⚠️ Phase 1: 工具返回的第 {idx+1} 個文件資訊中檔名無效: '{filename_from_tool}'")
                            generated_future_images.append({
                                "round": current_round, "id_in_round": f"detail_badfilename_img{idx+1}", 
                                "filename": f"細節檔名無效{idx+1}.png", "image_url":"無", "path": "無", 
                                "description": f"細節圖 {idx+1} 檔名無效"
                            })
                            continue 

                        if returned_bytes_info_detail and idx < len(returned_bytes_info_detail) and isinstance(returned_bytes_info_detail[idx], dict):
                            img_bytes = returned_bytes_info_detail[idx].get("data")
                        
                        if not img_bytes and os.path.exists(img_abs_path):
                            print(f"    Phase 1: 字節數據未由工具直接提供，嘗試從路徑讀取: {img_abs_path}")
                            try:
                                with open(img_abs_path, "rb") as f_read_bytes:
                                    img_bytes = f_read_bytes.read()
                                print(f"      ✅ 成功從文件讀取字節: {os.path.basename(img_abs_path)}")
                            except Exception as e_read_manual:
                                print(f"      ⚠️ 從文件讀取字節失敗: {os.path.basename(img_abs_path)}, 錯誤: {e_read_manual}")
                                img_bytes = None 

                        if img_bytes: 
                            extension = img_mime.split('/')[-1] if '/' in img_mime else 'png'
                            desc_detail = (f"立面/構造細節圖 {idx+1}/{len(returned_files_info_detail)} "
                                           f"(基於: {base_image_filename_for_desc}, "
                                           f"Prompt類型: {'圖生文+圖調整' if image_inputs_for_tool else '純文生圖'}, " # Adjusted description
                                           f"原始描述: {base_design_text_for_prompt[:30]}...)")
                            
                            saved_image_info_detail = self._save_and_encode_image(
                                image_bytes=img_bytes, absolute_filepath=img_abs_path, extension=extension,
                                description=desc_detail, current_round=current_round,
                                sub_id=f"detail_img{idx+1}" # Consistent sub_id
                            )
                            generated_future_images.append(saved_image_info_detail)
                            print(f"    ✅ 成功處理細節圖: {saved_image_info_detail.get('filename')}") 
                        else:
                            err_reason = "無有效字節數據 (工具未提供且無法從文件讀取)"
                            if not os.path.exists(img_abs_path): 
                                err_reason = f"文件於路徑 {img_abs_path} 未找到或無法讀取"
                            print(f"    ⚠️ Phase 1 無法處理第 {idx+1} 個細節圖片 (檔名: {os.path.basename(filename_from_tool if filename_from_tool else '未知')}, 原因: {err_reason})。")
                            generated_future_images.append({
                                "round": current_round, "id_in_round": f"detail_nodata_img{idx+1}", 
                                "filename": os.path.basename(filename_from_tool) if filename_from_tool else f"細節數據無效{idx+1}.png", 
                                "image_url":"無", 
                                "path": img_abs_path if img_abs_path else "無效路徑", 
                                "description": f"細節圖 {idx+1} 數據無效或文件缺失 ({err_reason})"
                            })
                else: 
                    print(f"  ⚠️ Phase 1 圖像生成工具未返回任何文件資訊。")
                    for i_miss in range(num_detail_images_to_generate):
                         generated_future_images.append({"round": current_round, "id_in_round": f"detail_missing_all_files_img{i_miss+1}", "filename": f"細節文件資訊缺失{i_miss+1}.png", "image_url":"無", "path": "無", "description": f"細節圖 {i_miss+1} 所有文件資訊缺失"})
        except Exception as e:
            print(f"  💥 Phase 1 調用 generate_gemini_image 批處理異常: {e}")
            for i_exc in range(num_detail_images_to_generate):
                generated_future_images.append({
                    "round": current_round, "id_in_round": f"detail_exc_batch_img{i_exc+1}",
                    "filename": f"細節生成異常_img{i_exc+1}.png", "image_url": "無", "path": "無",
                    "description": f"立面細節圖批次生成異常: {e}", "error": str(e)
                })
        
        print(f"    ℹ️ Phase 1 細節圖像生成完成，延遲 3 秒...")
        time.sleep(5)

        # --- Phase 2: Aging Scenario Generation (10, 20, 30 years) - BATCH MODE ---
        print(f"\n--- FutureScenario: Phase 2 - 生成未來10、20、30年變化圖 (批次請求 3 張) ---")
        years_to_simulate = [10, 20, 30]
        num_aging_images_to_generate = len(years_to_simulate)
        
        aging_prompt_template_text = active_config.future_scenario_aging_generation_prompt_template
        
        final_aging_prompt_for_tool = aging_prompt_template_text.format(
            base_design_description=base_design_text_for_prompt, 
            num_images=num_aging_images_to_generate, 
            llm_output_language=active_language
        )
        final_aging_prompt_for_tool += (
            f"\nImportant: Generate exactly {num_aging_images_to_generate} images, "
            "representing the aging at 10, 20, and 30 years respectively. "
            "Maintain consistency in the base structure across the aging sequence. "
            "The images should be returned in the order of 10 years, then 20 years, then 30 years."
        )
        if image_inputs_for_tool:
            final_aging_prompt_for_tool += "\nA base image has been provided; please show aging modifications on it."
        else:
            final_aging_prompt_for_tool += "\nNo base image was provided; generate based on the text description."

        print(f"    未來老化場景 (批次) Prompt: {final_aging_prompt_for_tool[:300]}...")


        try:
            tool_result_aging_batch = generate_gemini_image.invoke({
                "prompt": final_aging_prompt_for_tool,
                "image_inputs": image_inputs_for_tool, 
                "i": num_aging_images_to_generate 
            })

            if tool_result_aging_batch.get("error"):
                print(f"    ⚠️ Phase 2 批次圖像生成失敗: {tool_result_aging_batch.get('error')}")
                for i_err_aging in range(num_aging_images_to_generate):
                    year_val_err = years_to_simulate[i_err_aging] if i_err_aging < len(years_to_simulate) else "unknown"
                    generated_future_images.append({
                        "round": current_round, "id_in_round": f"aging_batch_err_img{i_err_aging+1}_{year_val_err}yr",
                        "filename": f"老化批次失敗_img{i_err_aging+1}_{year_val_err}yr.png", "image_url": "無", "path": "無",
                        "description": f"老化圖批次生成失敗 ({year_val_err} yr): {tool_result_aging_batch.get('error')}",
                        "error": tool_result_aging_batch.get('error')
                    })
            else:
                returned_files_info_aging = tool_result_aging_batch.get("generated_files", [])
                returned_bytes_info_aging = tool_result_aging_batch.get("image_bytes", [])

                print(f"  DEBUG Phase 2: 工具返回 {len(returned_files_info_aging)} 個老化文件資訊, {len(returned_bytes_info_aging)} 個老化字節項目。預期 {num_aging_images_to_generate} 個。")

                if returned_files_info_aging: 
                    print(f"  Phase 2 工具返回 {len(returned_files_info_aging)} 個老化圖片文件資訊。")
                    if len(returned_files_info_aging) != num_aging_images_to_generate:
                         print(f"  ⚠️ Phase 2 警告: 工具返回的文件數量 ({len(returned_files_info_aging)}) 與預期 ({num_aging_images_to_generate}) 不符。")

                    for idx, file_info_aging in enumerate(returned_files_info_aging):
                        filename_from_tool_aging = file_info_aging.get("filename")
                        img_mime_aging = file_info_aging.get("file_type", "image/png")
                        img_bytes_aging = None
                        img_abs_path_aging = None
                        current_year_for_desc = years_to_simulate[idx] if idx < len(years_to_simulate) else f"batch_idx{idx+1}"


                        if isinstance(filename_from_tool_aging, str) and filename_from_tool_aging.strip():
                            img_abs_path_aging = os.path.join(base_render_cache_dir, os.path.basename(filename_from_tool_aging))
                        else:
                            print(f"    ⚠️ Phase 2: 工具返回的第 {idx+1} 個老化文件資訊中檔名無效: '{filename_from_tool_aging}'")
                            generated_future_images.append({
                                "round": current_round, "id_in_round": f"aging_badfilename_img{idx+1}_{current_year_for_desc}yr",
                                "filename": f"老化檔名無效{idx+1}_{current_year_for_desc}yr.png", "image_url": "無", "path": "無",
                                "description": f"{current_year_for_desc}年後變化圖檔名無效"
                            })
                            continue

                        if returned_bytes_info_aging and idx < len(returned_bytes_info_aging) and isinstance(returned_bytes_info_aging[idx], dict):
                            img_bytes_aging = returned_bytes_info_aging[idx].get("data")
                        
                        if not img_bytes_aging and os.path.exists(img_abs_path_aging):
                            print(f"    Phase 2: 字節數據未由工具直接提供 ({current_year_for_desc}yr)，嘗試從路徑讀取: {img_abs_path_aging}")
                            try:
                                with open(img_abs_path_aging, "rb") as f_read_bytes_aging:
                                    img_bytes_aging = f_read_bytes_aging.read()
                                print(f"      ✅ 成功從文件讀取字節 ({current_year_for_desc}yr): {os.path.basename(img_abs_path_aging)}")
                            except Exception as e_read_manual_aging:
                                print(f"      ⚠️ 從文件讀取字節失敗 ({current_year_for_desc}yr): {os.path.basename(img_abs_path_aging)}, 錯誤: {e_read_manual_aging}")
                                img_bytes_aging = None

                        if img_bytes_aging: 
                            extension_aging = img_mime_aging.split('/')[-1] if '/' in img_mime_aging else 'png'
                            desc_aging = (f"方案 {current_year_for_desc} 年後變化圖 "
                                          f"(基於: {base_image_filename_for_desc}, "
                                          f"Prompt類型: {'圖生圖' if image_inputs_for_tool else '文生圖'}, "
                                          f"原始描述: {base_design_text_for_prompt[:30]}...)")
                            
                            saved_image_info_aging = self._save_and_encode_image(
                                image_bytes=img_bytes_aging, absolute_filepath=img_abs_path_aging, extension=extension_aging,
                                description=desc_aging, current_round=current_round,
                                sub_id=f"aging_{current_year_for_desc}yr_img{idx+1}" 
                            )
                            generated_future_images.append(saved_image_info_aging)
                            print(f"    ✅ 成功處理 {current_year_for_desc} 年後變化圖: {saved_image_info_aging.get('filename')}")
                        else:
                            err_reason_aging = "無有效字節數據 (工具未提供且無法從文件讀取)"
                            if not os.path.exists(img_abs_path_aging): # Check again
                                err_reason_aging = f"文件於路徑 {img_abs_path_aging} 未找到或無法讀取"
                            print(f"    ⚠️ Phase 2 無法處理第 {idx+1} 個老化圖片 ({current_year_for_desc}yr, 檔名: {os.path.basename(filename_from_tool_aging if filename_from_tool_aging else '未知')}, 原因: {err_reason_aging})。")
                            generated_future_images.append({
                                "round": current_round, "id_in_round": f"aging_nodata_batch_img{idx+1}_{current_year_for_desc}yr", 
                                "filename": os.path.basename(filename_from_tool_aging) if filename_from_tool_aging else f"老化數據無效{idx+1}.png", 
                                "image_url":"無", 
                                "path": img_abs_path_aging if img_abs_path_aging else "無效路徑", 
                                "description": f"{current_year_for_desc}年後變化圖數據無效或文件缺失 ({err_reason_aging})"
                            })
                else: 
                    print(f"  ⚠️ Phase 2 圖像生成工具未返回任何老化圖片文件資訊。")
                    for i_miss_aging in range(num_aging_images_to_generate):
                        year_val_miss = years_to_simulate[i_miss_aging] if i_miss_aging < len(years_to_simulate) else f"batch_idx{i_miss_aging+1}"
                        generated_future_images.append({
                            "round": current_round, "id_in_round": f"aging_missing_all_files_img{i_miss_aging+1}_{year_val_miss}yr", 
                            "filename": f"老化文件資訊缺失{i_miss_aging+1}.png", "image_url":"無", "path": "無", 
                            "description": f"老化圖 {year_val_miss}yr 所有文件資訊缺失"
                        })
        except Exception as e_batch_aging:
            print(f"  💥 Phase 2 調用 generate_gemini_image 批處理老化場景異常: {e_batch_aging}")
            for i_exc_aging in range(num_aging_images_to_generate):
                year_val_exc = years_to_simulate[i_exc_aging] if i_exc_aging < len(years_to_simulate) else f"batch_idx{i_exc_aging+1}"
                generated_future_images.append({
                    "round": current_round, "id_in_round": f"aging_exc_batch_img{i_exc_aging+1}_{year_val_exc}yr",
                    "filename": f"老化生成異常_img{i_exc_aging+1}.png", "image_url": "無", "path": "無",
                    "description": f"老化圖批次生成異常 ({year_val_exc}yr): {e_batch_aging}", "error": str(e_batch_aging)
                })

        existing_future_images = self.state.get("future_image", [])
        if not isinstance(existing_future_images, list):
            existing_future_images = []
        
        images_from_other_rounds_future = [
            img for img in existing_future_images if isinstance(img, dict) and img.get("round") != current_round
        ]
        self.state["future_image"] = images_from_other_rounds_future + generated_future_images

        print(f"\n✅ FutureScenarioGenerationTask 完成，總共處理 {len(generated_future_images)} 張圖片資訊 (包含細節圖與老化圖)。")
        if generated_future_images:
            for idx, info in enumerate(generated_future_images):
                 print(f"  詳細圖片資訊 ({idx+1}): ID='{info.get('id_in_round')}', Filename='{info.get('filename')}', Path='{info.get('path')}', URL: {'有內容' if info.get('image_url') and info.get('image_url').startswith('data:image/') else info.get('image_url', '未定義')}")

        return {"future_image": self.state["future_image"]}

# 生成 3D =：根據 Glb 檔呼叫 LLM（使用圖片生成工具）生成 3D 
class Generate3DPerspective:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict): 
        if state is not None:
            self.state = state        
        
        active_config = ensure_graph_overall_config(config) 

        current_round = self.state.get("current_round", 0)
        case_images_raw = self.state.get("case_image", [])
        selected_image_id_from_gate2 = self.state.get("GATE2") 
        
        selected_image_full_path_for_3d = None

        print(f"Generate3DPerspective: 嘗試查找 GATE2 ID: {selected_image_id_from_gate2} (類型: {type(selected_image_id_from_gate2)}) 在輪次 {current_round} 的基礎圖片。")
        if isinstance(selected_image_id_from_gate2, int) and case_images_raw:
            case_images_list = [item for item in case_images_raw if isinstance(item, dict)]
            found_base_image_3d = False
            for img_info in case_images_list:
                img_id_in_round = img_info.get("id_in_round")
                img_round = img_info.get("round")
                # UnifiedImageGenerationTask 應該在 "path" 中存儲有效路徑
                raw_filename_path = img_info.get("path") 
                
                print(f"  檢查 case_image 項目: id_in_round={img_id_in_round}, round={img_round}, path='{raw_filename_path}' (類型: {type(raw_filename_path)})")

                if (img_round == current_round and
                    img_id_in_round == selected_image_id_from_gate2 and
                    isinstance(raw_filename_path, str) and
                    raw_filename_path not in ["無效路徑或錯誤", "無"] and # 排除佔位符
                    not any(err_placeholder in raw_filename_path.lower() for err_placeholder in 
                             ["prompt生成失敗", "工具錯誤_", "無文件_", "格式錯誤_", "調用異常_", "處理錯誤_"]) and
                    os.path.exists(raw_filename_path)): 
                    
                    selected_image_full_path_for_3d = raw_filename_path
                    print(f"ℹ️ Generate3DPerspective: ✅ 成功選中圖片 (來自GateCheck2 ID {selected_image_id_from_gate2}), "
                          f"絕對路徑 '{selected_image_full_path_for_3d}' 用於3D生成。")
                    found_base_image_3d = True
                    break
            if not found_base_image_3d:
                 print(f"⚠️ Generate3DPerspective: 雖然 GateCheck2 選擇了 ID {selected_image_id_from_gate2}, "
                       f"但在 case_image 輪次 {current_round} 中未找到對應的有效圖片文件路徑。")
        else:
            print(f"⚠️ Generate3DPerspective: GateCheck2 未提供有效的圖片 ID (GATE2: {selected_image_id_from_gate2}) "
                  f"或 case_image 為空，無法選擇用於3D生成的圖片。")


        if not selected_image_full_path_for_3d:
            print(f"⚠️ Generate3DPerspective: 未找到輪次 {current_round} 的有效渲染圖文件路徑用於3D生成。")
            if not isinstance(self.state.get("perspective_3D"), list): self.state["perspective_3D"] = []
            if not isinstance(self.state.get("model_3D"), list): self.state["model_3D"] = []
            no_result_entry = {"round": current_round, "status": "无有效渲染图片进行3D生成", "filename":"无", "path":"无"}
            self.state["perspective_3D"] = custom_add_messages(self.state.get("perspective_3D", []), [no_result_entry])
            self.state["model_3D"] = custom_add_messages(self.state.get("model_3D", []), [no_result_entry])
            return {"perspective_3D": self.state["perspective_3D"], "model_3D": self.state["model_3D"]}

        
        # 定義 3D 檔案的快取目錄
        base_3d_cache_dir = os.path.join(os.getcwd(), "output", "model_cache")
        os.makedirs(base_3d_cache_dir, exist_ok=True)

        gen_3d_output_dict = generate_3D.invoke({
            "image_path": str(selected_image_full_path_for_3d), 
            "current_round": current_round,
            # "prompt": active_config.llm_output_language # 移除，除非 generate_3D 工具明確需要此 prompt 鍵
        })

        video_filename_from_tool = "无生成结果"
        model_filename_from_tool = "无模型"
        video_path_from_tool = "无"
        model_path_from_tool = "无"

        if isinstance(gen_3d_output_dict, dict):
            # 假設 generate_3D 返回的 video 和 model 是包含 'filename' (可能是絕對路徑或僅檔名) 的字典或直接是路徑/檔名字符串
            video_output = gen_3d_output_dict.get("video")
            model_output = gen_3d_output_dict.get("model")

            # 輔助函數來解析檔名和路徑
            def process_tool_output(output, cache_dir):
                raw_path = None
                if isinstance(output, dict) and isinstance(output.get("filename"), str):
                    raw_path = output.get("filename")
                elif isinstance(output, str):
                    raw_path = output
                
                if raw_path and raw_path.strip():
                    filename = os.path.basename(raw_path)
                    # 如果工具回傳的不是絕對路徑，則將其與快取目錄組合
                    if os.path.isabs(raw_path):
                        return filename, raw_path
                    else:
                        return filename, os.path.join(cache_dir, filename)
                return None, None

            video_filename_from_tool, video_path_from_tool = process_tool_output(video_output, base_3d_cache_dir)
            model_filename_from_tool, model_path_from_tool = process_tool_output(model_output, base_3d_cache_dir)

            if not video_filename_from_tool:
                video_filename_from_tool = "返回格式无效(video)"
                video_path_from_tool = "无"
            
            if not model_filename_from_tool:
                model_filename_from_tool = "返回格式无效(model)"
                model_path_from_tool = "无"
            
            if gen_3d_output_dict.get("error"):
                 print(f"⚠️ Generate3DPerspective: 3D生成工具報告錯誤: {gen_3d_output_dict.get('error')}")
                 # 如果有錯誤，標記檔名並將路徑設為無效
                 video_filename_from_tool = f"工具錯誤_{video_filename_from_tool}" if video_filename_from_tool else "工具錯誤"
                 model_filename_from_tool = f"工具錯誤_{model_filename_from_tool}" if model_filename_from_tool else "工具錯誤"
                 video_path_from_tool = "无"
                 model_path_from_tool = "无"


        else:
            print(f"⚠️ Generate3DPerspective: 3D生成工具未返回字典。返回: {gen_3d_output_dict}")
            video_filename_from_tool = "工具返回格式错误"
            model_filename_from_tool = "工具返回格式错误"


        video_entry = {"round": current_round, "type": "video", "filename": video_filename_from_tool, "path": video_path_from_tool}
        model_entry = {"round": current_round, "type": "model", "filename": model_filename_from_tool, "path": model_path_from_tool}
        
        existing_perspective_3d_list = self.state.get("perspective_3D", [])
        if not isinstance(existing_perspective_3d_list, list):
            existing_perspective_3d_list = []
            
        existing_model_3d_list = self.state.get("model_3D", [])
        if not isinstance(existing_model_3d_list, list):
            existing_model_3d_list = []

        self.state["perspective_3D"] = custom_add_messages(existing_perspective_3d_list, [video_entry])
        self.state["model_3D"] = custom_add_messages(existing_model_3d_list, [model_entry])


        print(f"✅ 生成 3D 完成: 影片文件:{video_filename_from_tool} (路徑: {video_path_from_tool})、模型文件:{model_filename_from_tool} (路徑: {model_path_from_tool})")
        return {"perspective_3D": self.state["perspective_3D"], "model_3D": self.state["model_3D"]}

# 深度評估任務：呼叫 LLM（使用圖片辨識工具）對生成圖與未來情境圖進行深度評估 OK
class DeepEvaluationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def _extract_total_score(self, text_score_str: str) -> float:
        """從評估文本中提取總分。"""
        if not isinstance(text_score_str, str):
            return 0.0
            
        match = re.search(r"\*\*總分數:([\d.]+)\*\*", text_score_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        else:
            # 作為備用，查找文本中可能出現的所有數字並取最大值
            numbers = re.findall(r"(\d+(?:\.\d+)?)", text_score_str)
            if numbers:
                try:
                    return max(map(float, numbers))
                except ValueError:
                    return 0.0
            return 0.0

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state      

        active_config = ensure_graph_overall_config(config)
        active_language = active_config.llm_output_language
        
        # OUTPUT_EVAL_DIR = "./output/" # 已在 DeepEvalTask 内部处理
        # os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True) 

        current_round = self.state.get("current_round", 0) 
        eval_results_list_raw = self.state.get("evaluation_result", []) 
        if not isinstance(eval_results_list_raw, list):
            eval_results_list_raw = []
            
        future_img_list_raw = self.state.get("future_image", []) 
        perspective_3d_list_raw = self.state.get("perspective_3D", []) 
        design_advice_list_raw = self.state.get("design_advice", []) 

        future_img_list = [item for item in future_img_list_raw if isinstance(item, dict)]
        perspective_3d_list = [item for item in perspective_3d_list_raw if isinstance(item, dict)] # 包含 3D 影片/模型信息
        design_advice_list = [item for item in design_advice_list_raw if isinstance(item, dict)]

        valid_advices = [
            advice for advice in design_advice_list
            if advice.get("round") == current_round and advice.get("state") == True 
        ]
        advice_text = "無目標" 
        if valid_advices:
            advice_text = valid_advices[0].get("proposal", "無目標")
        else:
            print(f"⚠️ DeepEvaluationTask: 未找到輪次 {current_round} 且 state 為 True 的有效設計建議。")

        # 提取有效圖片路徑用於 img_recognition
        valid_future_image_paths_for_eval = []
        future_image_filenames_for_log = [] # 用於日誌記錄
        if future_img_list:
            for img_info in future_img_list:
                # 假設 "path" 字段存儲了由 FutureScenarioGenerationTask 驗證過的絕對路徑
                file_path = img_info.get("path") 
                img_filename = img_info.get("filename", "未知文件名") # 用於日誌

                if isinstance(file_path, str) and \
                   file_path not in ["無效路徑或錯誤", "無", "細節數據無效", "老化數據無效", "細節列表無效", "老化列表無效"] and \
                   not any(err_tag in file_path for err_tag in ["失敗", "異常", "無效"]) and \
                   os.path.exists(file_path):
                    valid_future_image_paths_for_eval.append(file_path)
                    future_image_filenames_for_log.append(os.path.basename(img_filename))
                else:
                    print(f"⚠️ DeepEvaluationTask: 從 future_image 中過濾掉無效條目: path='{file_path}', filename='{img_filename}'")
        
        img_keywords_content = "無有效未來圖片可供分析關鍵字。"
        img_eval_text = "無有效未來圖片可供評估。"

        if valid_future_image_paths_for_eval:
            print(f"ℹ️ DeepEvaluationTask: 使用 {len(valid_future_image_paths_for_eval)} 張有效未來圖片進行評估: {future_image_filenames_for_log}")
            try:
                keyword_prompt_for_img = active_config.deep_eval_keyword_img_recognition_prompt_template.format(
                    llm_output_language=active_language
                )
                img_key_output_str = img_recognition.invoke({
                    "image_paths": valid_future_image_paths_for_eval, 
                    "prompt": keyword_prompt_for_img
                })        
                img_keywords_content = img_key_output_str.strip() if isinstance(img_key_output_str, str) else "圖片關鍵詞生成失敗或為空。"
                print(f"  基於圖片生成的關鍵詞：{img_keywords_content[:200]}...")

                if "無法識別" in img_keywords_content or not img_keywords_content.strip() : # 檢查是否有意義的關鍵詞
                     print(f"  ⚠️ 關鍵詞生成可能未成功，關鍵詞內容為: '{img_keywords_content}'")
                     # 可以選擇不進行後續的圖片評估，或者讓LLM嘗試評估
                
                # 即使關鍵詞生成不佳，也嘗試進行圖片評估
                img_eval_prompt_content = active_config.deep_eval_img_eval_img_recognition_prompt_template.format(
                    rag_msg=img_keywords_content, # rag_msg 可以是空字符串或提示信息
                    llm_output_language=active_language
                )
                img_eval_output_str = img_recognition.invoke({
                    "image_paths": valid_future_image_paths_for_eval,
                    "prompt": img_eval_prompt_content
                })
                img_eval_text = img_eval_output_str.strip() if isinstance(img_eval_output_str, str) else "圖片評估工具返回空。"
            except Exception as e_img_rec:
                print(f"❌ DeepEvaluationTask: 圖片辨識過程中發生錯誤: {e_img_rec}")
                img_keywords_content = f"圖片辨識錯誤: {e_img_rec}"
                img_eval_text = f"圖片評估錯誤: {e_img_rec}"
        else:
            print("⚠️ DeepEvaluationTask: 無有效未來圖片傳遞給 img_recognition。")
        

        # 提取有效3D影片/模型路徑用於 video_recognition
        valid_perspective_3d_paths_for_eval = []
        perspective_3d_filenames_for_log = []
        if perspective_3d_list:
            for p3d_info in perspective_3d_list:
                # 假設 "path" 字段存儲了由 Generate3DPerspective 驗證過的絕對路徑
                file_path = p3d_info.get("path")
                p3d_filename = p3d_info.get("filename", "未知3D文件")

                if isinstance(file_path, str) and \
                   file_path not in ["無效路徑或錯誤", "無"] and \
                   not any(err_tag in file_path for err_tag in ["失敗", "異常", "無效", "错误"]) and \
                   os.path.exists(file_path):
                    valid_perspective_3d_paths_for_eval.append(file_path)
                    perspective_3d_filenames_for_log.append(os.path.basename(p3d_filename))
                else:
                    print(f"⚠️ DeepEvaluationTask: 從 perspective_3D 中過濾掉無效條目: path='{file_path}', filename='{p3d_filename}'")

        vid_eval_text = "無有效3D模型/影片可供評估。"
        vid_total_score = 0.0  # 初始化影片評估總分

        if valid_perspective_3d_paths_for_eval:
            print(f"ℹ️ DeepEvaluationTask: 使用 {len(valid_perspective_3d_paths_for_eval)} 個有效3D文件進行評估: {perspective_3d_filenames_for_log}")
            
            vid_eval_prompt_content = active_config.deep_eval_vid_eval_video_recognition_prompt_template.format(
                rag_msg=img_keywords_content, 
                llm_output_language=active_language
            )

            all_vid_eval_texts = []
            # 迭代處理每一個有效的 3D 檔案
            for video_path in valid_perspective_3d_paths_for_eval:
                try:
                    print(f"  - 正在評估文件: {os.path.basename(video_path)}")
                    # 修正：使用 'video_path' (單數) 並傳入單一路徑
                    vid_eval_output_str = video_recognition.invoke({ 
                        "video_path": video_path, 
                        "prompt": vid_eval_prompt_content
                    })
                    
                    current_vid_eval_text = vid_eval_output_str.strip() if isinstance(vid_eval_output_str, str) else f"影片評估工具對 {os.path.basename(video_path)} 返回空。"
                    
                    # 從本次評估中提取分數並累加
                    current_score = self._extract_total_score(current_vid_eval_text)
                    vid_total_score += current_score
                    
                    # 為報告添加標題以便區分
                    all_vid_eval_texts.append(f"--- 評估報告: {os.path.basename(video_path)} ---\n{current_vid_eval_text}\nScore from this file: {current_score}")

                except Exception as e_vid_rec:
                    error_text = f"影片/3D模型 '{os.path.basename(video_path)}' 辨識過程中發生錯誤: {e_vid_rec}"
                    print(f"❌ {error_text}")
                    all_vid_eval_texts.append(error_text)

            if all_vid_eval_texts:
                vid_eval_text = "\n\n".join(all_vid_eval_texts)

        else:
            print("⚠️ DeepEvaluationTask: 無有效3D影片/模型路徑傳遞給 video_recognition。")


        img_total_score = self._extract_total_score(img_eval_text)
        # vid_total_score 已在迴圈中計算完成
        all_score_for_round = img_total_score + vid_total_score

        current_eval_result_entry = {
            "current_round": current_round,            
            "eval_result_image": img_eval_text,
            "eval_result_video": vid_eval_text,
            "total_score_calculated": all_score_for_round
        }
        
        updated_eval_results_list = custom_add_messages(eval_results_list_raw, [current_eval_result_entry])
        self.state["evaluation_result"] = updated_eval_results_list

        current_eval_count_entry = {str(current_round): all_score_for_round}
        eval_count_list_raw = self.state.get("evaluation_count", [])
        if not isinstance(eval_count_list_raw, list):
            eval_count_list_raw = []
        updated_eval_count_list = custom_add_messages(eval_count_list_raw, [current_eval_count_entry])
        self.state["evaluation_count"] = updated_eval_count_list

        md_content_full = f"""# Evaluation Result for Round {current_round}

## Image Evaluation (Based on {future_img_list[0].get('filename') or 'N/A'})
{img_eval_text}
Score: {img_total_score}

## Video/3D Model Evaluation (Based on {perspective_3d_list[0].get('filename') or 'N/A'})
{vid_eval_text}
Score: {vid_total_score}

## Combined Score for Round: {all_score_for_round}
"""
        eval_file_path_md = os.path.join("./output/", f"eval_result_{current_round}.md")
        try:
            with open(eval_file_path_md, "w", encoding="utf-8") as f:
                f.write(md_content_full)
            print(f"✅ 評估報告已儲存至: {eval_file_path_md}")
        except IOError as e:
            print(f"❌ 無法儲存評估報告: {e}")

        self.state["current_round"] = current_round + 1  
        print(f"✅ 深度評估完成，進入下一輪次: {self.state['current_round']}")
        print(f"📌 本輪總評分: {all_score_for_round}")
        return {
            "evaluation_result": self.state["evaluation_result"],
            "evaluation_count": self.state["evaluation_count"],
            "current_round": self.state["current_round"]
        }

# 評估檢查任務：根據評估次數決定流程路由（參考條件分支範本邏輯）
class EvaluationCheckTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state
        
        active_config = ensure_graph_overall_config(config) # 處理 config

        current_iteration_count = self.state.get("current_round", 0) # current_round 代表已完成的輪次，下一輪是 current_round + 1
        max_rounds = active_config.max_evaluation_rounds

        # current_round 從0開始計數。如果 max_rounds 是3，
        # 當 current_round 是 0, 1, 2 時，表示還可以繼續迭代。
        # 當 current_round 變成 3 時，表示已經完成了3輪，應該結束。
        if current_iteration_count < max_rounds:
            self.state["evaluation_status"] = "NO"
            print(f"EvaluationCheckTask：目前已完成 {current_iteration_count} 輪評估，未達到最大輪數 {max_rounds}，將返回 RAGdesignThinking 執行下一輪。")
        else:
            self.state["evaluation_status"] = "YES"
            print(f"EvaluationCheckTask：目前已完成 {current_iteration_count} 輪評估，已達到最大輪數 {max_rounds}，流程結束。")
        return {"evaluation_status": self.state["evaluation_status"]}

# 總評估任務(用戶可介入)
class FinalEvaluationTask:
    def __init__(self, state: dict, short_term=None, long_term=None): 
        self.state = state
        self.short_term = short_term if short_term is not None else get_short_term_memory()
        self.long_term = long_term if long_term is not None else get_long_term_store()

    def run(self, state: GlobalState, config: GraphOverallConfig | dict):
        if state is not None:
            self.state = state

        active_config = ensure_graph_overall_config(config)
        current_llm = active_config.llm_config.get_llm()
        active_language = active_config.llm_output_language

        eval_results_list_final = self.state.get("evaluation_result", []) 
        eval_counts_list_final = self.state.get("evaluation_count", []) 
        
        if not isinstance(eval_results_list_final, list): eval_results_list_final = []
        if not isinstance(eval_counts_list_final, list): eval_counts_list_final = []

        short_memory_content = self.short_term.retrieve_all() if hasattr(self.short_term, "retrieve_all") else ""
        long_memory_content = self.long_term.retrieve_all() if hasattr(self.long_term, "retrieve_all") else ""
        current_round_for_final_eval = self.state.get("current_round", "未知最終輪次") 

        eval_results_formatted_str = ""
        for res_item in eval_results_list_final: 
            if isinstance(res_item, dict):
                round_num = res_item.get("current_round", "未知輪次")
                eval_results_formatted_str += f"\n輪次 {round_num}："
                eval_results_formatted_str += f"\n - 圖片評估：{res_item.get('eval_result_image', '無')}" 
                eval_results_formatted_str += f"\n - 3D 視角評估：{res_item.get('eval_result_video', '無')}\n" 
        
        eval_counts_formatted_str = ""
        for count_dict_item in eval_counts_list_final: 
             if isinstance(count_dict_item, dict):
                for round_key_str, score_val in count_dict_item.items(): 
                    eval_counts_formatted_str += f"輪次 {round_key_str}：總分 {score_val}\n"

        summary_prompt_content_final = active_config.final_evaluation_summary_prompt_template.format(
            eval_results_formatted=eval_results_formatted_str if eval_results_formatted_str else "無評估結果",
            eval_counts_formatted=eval_counts_formatted_str if eval_counts_formatted_str else "無評分結果",
            short_memory=short_memory_content if short_memory_content else "無短期記憶",
            long_memory=long_memory_content if long_memory_content else "無長期記憶",
            current_round=current_round_for_final_eval,
            llm_output_language=active_language
        )

        llm_response_msg_final = current_llm.invoke([SystemMessage(content=summary_prompt_content_final)])
        final_text_output = llm_response_msg_final.content if hasattr(llm_response_msg_final, "content") else "LLM總評估生成失敗。"

        self.state["final_evaluation"] = final_text_output

        print("✅ 總評估任務完成！")
        print(f"📌 總評估結果:\n{final_text_output}")
        return {"final_evaluation": self.state["final_evaluation"]}

# =============================================================================
# 建立工作流程圖 (Graph Setup)
# =============================================================================
workflow = StateGraph(GlobalState, config_schema=GraphOverallConfig)

initial_state = {
    "設計目標x設計需求x方案偏好": [],
    "design_summary": "",
    "analysis_img": "",
    "site_analysis": "",
    "design_advice": [],
    "case_image": [],
    "outer_prompt": [],
    "future_image": [],
    "perspective_3D": [],
    "model_3D": [],
    "GATE1": "初始值",
    "GATE2": "初始值",
    "GATE_REASON1": "",
    "GATE_REASON2": "",
    "current_round": 0,
    "evaluation_count": [],
    "evaluation_status": "",
    "evaluation_result": [],
    "final_evaluation": ""
}

question_task = QuestionTask(initial_state)
site_analysis_task = SiteAnalysisTask(initial_state)
rag_thinking = RAGdesignThinking(initial_state)
gate_check1 = GateCheck1(initial_state)
shell_prompt_task = OuterShellPromptTask(initial_state) # 註釋掉
image_render_task = CaseScenarioGenerationTask(initial_state) # 註釋掉
# unified_image_gen_task = UnifiedImageGenerationTask(initial_state) # 新增
gate_check2 = GateCheck2(initial_state)
future_scenario_task = FutureScenarioGenerationTask(initial_state)
generate_p3d_task = Generate3DPerspective(initial_state)
deep_evaluation_task = DeepEvaluationTask(initial_state)
evaluation_check_task = EvaluationCheckTask(initial_state)
final_eval_task = FinalEvaluationTask(initial_state)

workflow.set_entry_point("question_summary")

workflow.add_node("question_summary", question_task.run)
workflow.add_node("analyze_site", site_analysis_task.run)
workflow.add_node("designThinking", rag_thinking.run)
workflow.add_node("GateCheck1", gate_check1.run)
workflow.add_node("shell_prompt", shell_prompt_task.run) # 註釋掉
workflow.add_node("image_render", image_render_task.run) # 註釋掉
# workflow.add_node("img_generation", unified_image_gen_task.run) # 新增
workflow.add_node("GateCheck2", gate_check2.run)
workflow.add_node("future_scenario", future_scenario_task.run) # 恢復獨立節點
workflow.add_node("generate_3D", generate_p3d_task.run)       # 恢復獨立節點
workflow.add_node("deep_evaluation", deep_evaluation_task.run)
workflow.add_node("evaluation_check", evaluation_check_task.run)
workflow.add_node("final_eval", final_eval_task.run)

workflow.add_edge("question_summary", "analyze_site")
workflow.add_edge("analyze_site", "designThinking")
workflow.add_edge("designThinking", "GateCheck1")
workflow.add_edge("shell_prompt", "image_render") # 註釋掉
workflow.add_edge("image_render", "GateCheck2") # 註釋掉
# workflow.add_edge("img_generation", "GateCheck2") # 新增
workflow.add_edge("future_scenario", "generate_3D") # 恢復邊
workflow.add_edge("generate_3D", "deep_evaluation") # 恢復邊
workflow.add_edge("deep_evaluation", "evaluation_check")
workflow.add_edge("final_eval", END)

workflow.add_conditional_edges(
    "GateCheck1",
    lambda state: "YES" if state.get("GATE1") == "有" else "NO",
    {
        "YES": "shell_prompt",  # 修改：指向新節點
        "NO": "designThinking"  
    }
)

workflow.add_conditional_edges(
    "GateCheck2",
    lambda state: "YES" if isinstance(state.get("GATE2"), int) else "NO",
    { 
        "YES": "future_scenario", # 修改：GateCheck2 的 YES 分支指向 future_scenario
        "NO": "shell_prompt" 
    }
)

workflow.add_conditional_edges("evaluation_check",lambda state: state["evaluation_status"],
    { "NO": "designThinking",   "YES": "final_eval"  })

graph = workflow.compile()

graph.name = "Multi-Agent System for Timber Pavilion Design"
