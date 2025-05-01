import os
import re
import ast
import json
import base64
from dotenv import load_dotenv
from langgraph.graph.state import StateGraph, START, END
from typing import List
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.memory import get_long_term_store, get_short_term_memory
from src.tools.img_recognition import img_recognition
# from src.tools.IMG_rag_tool import IMG_rag_tool
from src.tools.prompt_generation import prompt_generation
from src.tools.case_render_image import case_render_image
from src.tools.simulate_future_image import simulate_future_image
from src.tools.ARCH_rag_tool import ARCH_rag_tool
from src.tools.video_recognition import video_recognition
from src.tools.generate_3D import generate_3D


# 載入 .env 設定
load_dotenv()

# 使用 tools_memory 提供的短期記憶與長期記憶存儲
short_term = get_short_term_memory()
long_term = get_long_term_store()

# =============================================================================
# 建立 LLM 實例（不再設定 system_message 屬性）
# =============================================================================
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
    )

# 依需求產生不同用途的 LLM 實例
llm_with_img = llm.bind_tools([img_recognition])
# llm_with_3d = llm.bind_tools({"3D_recognition": tools["3D_recognition"]})
# llm_with_IMGrag = llm.bind_tools([IMG_rag_tool])
llm_with_ARCHrag = llm.bind_tools([ARCH_rag_tool])
llm_with_prompt = llm.bind_tools([prompt_generation])
llm_with_gen2 = llm.bind_tools([simulate_future_image])


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
    
    def run(self, state=None):
        if state is not None:
            self.state = state
        user_input = self.state["設計目標x設計需求x方案偏好"][0].content
        print("✅ 用戶的設計需求已記錄：", user_input)

        # Step 1: LLM 查看用戶輸入，生成關鍵詞
        keyword_prompt = (
            "請從用戶輸入文本生成中英文關鍵詞以便於檢索建築設計目標、設計需求、方案偏好等相關資訊。"
            "需要特別關注木構造、數位製造工法、傳統製造工法、木結構等項目。"
            "請使用用戶的輸入語言來回答"
            f"{user_input}"
        )
        keywords_msg = llm.invoke([SystemMessage(content=keyword_prompt)])
        keywords = keywords_msg.content.strip()
        print("生成的關鍵詞：", keywords)

        # Step 2: 根據用戶輸入和關鍵詞構建 RAG prompt
        rag_prompt = (f"{keywords}")

        # 使用綁定工具的 llm_with_ARCHrag 進行 RAG 檢索
        RAG_msg = ARCH_rag_tool.invoke(rag_prompt)
        print("RAG檢索結果：", RAG_msg)

        # Step 3: 將 RAG 補充資訊與原始用戶輸入結合，生成最終總結報告
        summary_input = (
            "建築類型是Timber Curve Pavilion，以用戶的設計目標為主，根據補充資訊，"
            "說明設計方案可能要達成甚麼樣的設計決策方向以滿足用戶的設計目標。"
            "請使用用戶的輸入語言來回答"
            f"建築設計目標:{user_input}/n補充資訊:{RAG_msg}"
        )
        summary_msg = llm.invoke([SystemMessage(content=summary_input)])
        self.state["design_summary"] = f"用戶需求:{user_input}/n{summary_msg.content}"
        print("✅ 設計目標總結完成！")
        print(summary_msg.content)
        return {          
            "設計目標x設計需求x方案偏好": user_input,
            "design_summary": self.state["design_summary"]
            }

# 場地分析任務：讀取 JSON 並分析基地資訊，同時透過 LLM 呼叫進一步解讀資料 OK
class SiteAnalysisTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        base_map_path = "./input/2D/base_map.png"
        project_data_path = "./input/project_data.json"

        if not os.path.exists(base_map_path) or not os.path.exists(project_data_path):
            print("❌ 缺少場地分析所需的文件！")
            return {
                "site_analysis": self.state.get("site_analysis"),
                "design_advice": self.state.get("design_advice")
            }

        with open(project_data_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)

        geo_location = project_data.get("geoLocation", "未知地點")
        region = project_data.get("region", "未知區域")
        # north_direction = project_data.get("northDirection", "未知方位")

        # img_recognition的prompt
        prompt = f"""
        你是一個專業的都市視覺資訊分析工具，你的任務是分析使用者提供的基地圖片，辨識並標註都市環境物件，描述重要特徵，並生成視覺摘要。
        輸出結構化資料，包含物件標註 (類型、位置)、特徵描述 (建築、道路、綠地、水體、環境脈絡特徵) 。
        以圖片的上方為北方，初步推理基於方位來說，日照、熱環境、風環境、噪音、景觀、交通對於基地的影響。
        設計位置:{region}，經緯度:{geo_location}
        """

        # 調用工具進行場地分析
        analysis_img = img_recognition.invoke({
            "image_paths": base_map_path,
            "prompt": prompt,
        })

        # llm_with_ARCHrag的prompt
        LLMprompt = f"""
        作為建築師及空間分析專家，你擅長整合提供的資訊進行基地分析，基於圖片的辨識結果。
        查詢並整合地點、都市計畫規範、建築法規、氣候資料、人文歷史特色、其他特殊地質或都市情形、特殊氣候情形等背景資料。並整理為更深入的基地分析報告。
        設計位置:{region}，經緯度:{geo_location}。
        圖片辨識結果:{analysis_img}。
        """

        analysis_result = llm.invoke([SystemMessage(content=LLMprompt)])

        self.state["analysis_img"] = analysis_img
        self.state["site_analysis"] = analysis_result.content

        # 確保場地圖像存在
        if not os.path.exists(base_map_path):
            print(f"❌ 缺少基地圖像: {base_map_path}")
            return {
                "analysis_img": self.state.get("analysis_img"),
                "site_analysis": self.state.get("site_analysis"),
            }
        
        print("✅ 場地分析完成！")
        print(analysis_result.content)
        return {
            "analysis_img": self.state["analysis_img"],
            "site_analysis": self.state["site_analysis"],
        }

# 設計方案任務 OK
class RAGdesignThinking:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state        

        # 讀取設計參數
        design_goal = self.state.get("design_summary", "無目標")
        analysis_result = self.state.get("site_analysis", "無基地條件")
        current_round = self.state.get("current_round", 0)
        improvement = self.state.get("GATE_REASON1", "")

        # llm1 prompt
        prompt_keywords_preliminary = f"""
        你是一位經驗豐富的資深建築設計顧問，
        基於以下資料：
        設計目標：{design_goal}
        基地分析報告：{analysis_result}
        
        請為 Timber pavilion 設計提供具體且細緻的參考資料關鍵詞。
        **請列出RAG中英文關鍵字，格式為:中文(英文)**。描述相關案例、構造細節及製造工法。
        """
        response_kp = llm.invoke([SystemMessage(content=prompt_keywords_preliminary)])
        response_text = response_kp.content

        print("生成的關鍵詞：", response_text)

        # Step 2: 使用關鍵字進行 RAG 檢索以獲取參考資料
        rag_prompt = f"請根據關鍵字查詢相關案例、構造細節及製造工法、減碳及循環永續性、技術細節及研究理論：{response_text}。"
        RAG_msg = ARCH_rag_tool.invoke({"query": rag_prompt})
        print("RAG 檢索結果：", RAG_msg)

        # Step 3: 利用初步方案與 RAG 參考資料生成完整方案
        prompt_complete = f"""
        你是一位經驗豐富的資深建築設計顧問，根據以下方面生成完整的設計方案：
        主要設計決策針對**幾何形狀(比如方、圓、三角、錐型、塔型等)、外殼形式(比如平面、單曲面、雙曲面、自由曲面等)、木構造細節**。
        及次要設計決策針對日照、熱環境、風環境、噪音、景觀、基地周遭紋理等。專注於外殼設計，需要具有參數式設計的美感、高度創意性及前衛性。
        請綜合以下內容，以設計目標及改進建議為重點，提出一個完整、具備細節且具創新性、可行性的設計方案。
        **設計目標**：{design_goal}
        **改進建議**: {improvement}
        基地分析報告：{analysis_result}
        參考資料：{RAG_msg}
        """
        complete_response = llm.invoke([SystemMessage(content=prompt_complete)])
        complete_scheme = complete_response.content

        new_scheme_entry = {"round":int(current_round),"proposal":str(complete_scheme)}

        # 使用 custom_add_messages 累加存入 state (design_advice 也使用累加功能)
        existing_advice = self.state.get("design_advice", [])
        updated_advice = custom_add_messages(existing_advice, [new_scheme_entry])
        self.state["design_advice"] = updated_advice

        print("✅ 設計建議已完成！")
        print(f"最終設計建議：{self.state['design_advice']}")

        return {"design_advice": self.state["design_advice"]}

# GATE 檢查方案（請回答：有/沒有） OK
class GateCheck1:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        design_advice_raw = self.state.get("design_advice", [])
        site_analysis = self.state.get("site_analysis", "無基地條件")
        design_summary = self.state.get("design_summary", "無目標")

        def ensure_dict(item):
            if isinstance(item, dict):
                return item
            try:
                return json.loads(item)
            except Exception as e:
                print(f"⚠️ 無法解析項目: {item}，錯誤：{e}")
                return None

        # - formatted_current: 沒有 state 鍵的對象
        # - formatted_previous: 已包含 state 鍵的對象
        design_advice_list = []
        for item in design_advice_raw:
            d = ensure_dict(item)
            if d is not None:
                design_advice_list.append(d)

        # 使用新條件：沒有 "state" 鍵的對象作為 current proposals
        current_proposals = [
            advice for advice in design_advice_list if "state" not in advice
        ]
        historical_proposals = [
            advice for advice in design_advice_list if "state" in advice
        ]

        if not current_proposals:
            print("⚠️ 當前無符合條件的設計建議方案（未找到不含 state 鍵的對象）。")
            self.state["GATE1"] = "沒有"
            self.state["GATE_REASON1"] = "當前無符合條件的設計建議方案（未找到不含 state 鍵的對象）"
            return {"GATE1": self.state["GATE1"], "GATE_REASON1": self.state["GATE_REASON1"]}

        # 格式化輸出以供 prompt 使用
        formatted_current = json.dumps(current_proposals, ensure_ascii=False, indent=2)
        formatted_previous = json.dumps(historical_proposals, ensure_ascii=False, indent=2)

        prompt = f"""
        你是一位專業的建築方案評審員。
        請根據以下設計建議提供判斷及評比，須對於設計需求具有回應性，且與之前輪次的方案不過於接近。

        1.循環經濟潛力 (Circular Economy Potential): 方案是否展現朝向材料循環利用、永續木材來源的潛力？
        判斷點: 方案是否具有發展材料再利用、回收、模組化或組裝效率等處理計畫的機會 (即使沒有詳細計畫)？
        2.材料效率潛力 (Material Efficiency Potential): 方案是否展現減少材料浪費、提升材料利用率的潛力？
        判斷點: 方案是否具有發展規劃優化設計、數位製造、集成木材等方法的機會 (即使沒有具體數據)？
        3.製造效率潛力 (Manufacturing Efficiency Potential): 方案是否展現提升製造與施工效率的潛力？
        判斷點: 方案是否具有發展規劃預製化、模組化、自動化生產、簡化施工等策略的機會 (即使沒有詳細流程)？
        4.永續環保潛力 (Environmental Sustainability Potential): 方案是否展現降低環境足跡、符合永續環保原則的潛力？
        判斷點: 方案是否具有發展規劃木材的減少浪費、環境友善、減少污染的製造策略的機會 (即使沒有量化數據)？
        5.減碳潛力 (Carbon Reduction Potential): 方案是否展現碳封存、減少碳排放的潛力？
        判斷點: 方案是否具有發展規劃木構造的在地性、減碳效益、碳封存等策略的機會 (即使沒有碳排計算)？
        
        只有在符合設計需求的前提下，其他方面都具備潛力，才是"有"。反之就是"沒有。
        **請回覆兩行：第一行僅包含判斷後的"有"或"沒有"；第二行請說明改進的建議。**
        **設計需求**：{design_summary}
        當前輪次方案：{formatted_current}
        之前輪次方案：{formatted_previous}
        """.strip()
#        基地分析：{site_analysis}

        # 調用 LLM 並取得回覆
        llm_response = llm.invoke([SystemMessage(content=prompt)])
        response_lines = [line.strip() for line in llm_response.content.splitlines() if line.strip()]
        if not response_lines:
            print("⚠️ LLM 回覆為空，請檢查提示格式。")
            evaluation_result = "沒有"
            reason = "LLM 回覆為空，請檢查提示格式"
        else:
            evaluation_result = response_lines[0]
            reason = response_lines[1] if len(response_lines) > 1 else "空"

        # 判斷評估結果，決定 state 鍵值
        state_value = True if evaluation_result == "有" else False

        # 為每個當前方案字典新增 state 鍵
        for advice in current_proposals:
            advice["state"] = state_value

        # 覆蓋 design_advice[-1]：若存在則移除最後一個，再新增 current_proposals
        existing_advice = self.state.get("design_advice", [])
        if existing_advice:
            # 移除最後一個設計建議
            existing_advice.pop()
            updated_advice = existing_advice + current_proposals
        else:
            updated_advice = current_proposals

        self.state["design_advice"] = updated_advice

        # 最後 self.state["GATE1"] 僅返回評判結果（"有" 或 "没有"）
        self.state["GATE1"] = evaluation_result
        self.state["GATE_REASON1"] = reason

        print(f"【GateCheck】已收到評審結果：{evaluation_result}，原因：{reason}")
        return {"GATE1": self.state["GATE1"], "GATE_REASON1": self.state["GATE_REASON1"], "design_advice": self.state["design_advice"]}
        

# 外殼 Prompt 生成：呼叫 LLM（使用 prompt 生成工具）根據基地資訊與融合圖生成設計 prompt OK
class OuterShellPromptTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        # 取得累積的設計建議
        current_round = self.state.get("current_round", 0)
        design_advice_list = self.state.get("design_advice", [])
        improvement = self.state.get("GATE_REASON1", "")
        
        # 過濾出當前輪次且 state 為 True 的設計方案（必須是字典格式）
        valid_advices = [
            advice for advice in design_advice_list
            if isinstance(advice, dict) and advice.get("round") == current_round and advice.get("state") == True
        ]
        
        # 從有效的設計方案中取出 "proposal" 作為 advice_text
        if valid_advices:
            selected_advice = valid_advices[0]
            advice_text = selected_advice.get("proposal", "無目標")
        else:
            advice_text = "無目標"
        
        gpt_prompt = (
            f"作為建築師與Prompt engineering，請參考以下設計參考建議來推測未來此小型pavilion的樣貌。"
            f"使用英文 prompt，只需 positive prompt，要仔細、具體、使用專業的建築木構造設計語法，**最大token不可超過77**。"
            f"Prompt 主要根據設計提案描述此建築設計外觀、造型曲面型式、木構造細部設計及網格分割形式、整體風格與氛圍。"      
            f"**在具有細節且構造合理的情況下，需要避免木構造曲面、網格分割過度複雜**。"
            f"視角必須要看到建築整體，高質感，透視圖。**不生成內部隔間、家具、玻璃、人**。"
            f"設計提案: {advice_text}"
            f"改進建議:{improvement}"
        )

        gpt_output = llm.invoke([SystemMessage(content=gpt_prompt)])
        final_prompt = gpt_output.content if hasattr(gpt_output, "content") else "❌ GPT 生成失敗"

        lora_prompt = (
            f"請生成一個適合的 LoRA 權重數值，其數值必須在 0.3 到 0.7 之間。"
            f"權重越重（接近 0.7）表示生成結果會更趨於形式固定沒有創意性的曲面，但木網格構造清晰適合生成具有簍空木網格的構造；"
            f"權重越輕（接近 0.3）則生成結果會更具設計發散性但失去網格構造或網格不清晰，整體適合生成較為簡約的造型。"
            f"請根據設計提案動態生成適合的 LoRA 權重。**僅回答權重的數字**"
            f"設計提案: {final_prompt}"
        )

        gpt_output2 = llm.invoke([SystemMessage(content=lora_prompt)])
        lora_prompt = gpt_output2.content

        new_prompt_entry = {"round":int(current_round),"prompt": str(final_prompt),"lora":str(lora_prompt)}
        existing_prompts = self.state.get("outer_prompt", [])
        if not isinstance(existing_prompts, list):
            existing_prompts = []
        self.state["outer_prompt"] = custom_add_messages(existing_prompts, [new_prompt_entry])

        print("✅ 生成外殼 Prompt 完成！")
        print(f"📌 外殼 Prompt: {final_prompt}")
        return {"outer_prompt": self.state["outer_prompt"]}

# 方案情境生成：呼叫 LLM（使用圖片生成工具）根據外殼 prompt 與融合圖生成未來情境圖 OK
class CaseScenarioGenerationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        current_round = self.state.get("current_round", 0)
        outer_prompt = self.state.get("outer_prompt", [])

        # 篩選出不含 "state" 鍵的字典，並提取它們的 "prompt" 值
        prompt_values = [item["prompt"] for item in outer_prompt
                        if isinstance(item, dict) and "state" not in item]

        lora_values = [item["lora"] for item in outer_prompt
                        if isinstance(item, dict) and "state" not in item]

        prompt_str = " ".join(prompt_values)
        lora_str = " ".join(lora_values)

        # 循環四次生成圖片，並將返回的檔名以字典格式存放，格式例如：{1: "shell_result_{current_round}_1.png"}
        combined_images = []
        render_cache_dir = os.path.join(os.getcwd(), "output", "render_cache")
        for i in range(1, 5):  # 循環 1~4 次
            # 呼叫圖片生成工具，傳入當前輪次、outer_prompt 以及當前生成次數 i
            case_image_path = case_render_image.invoke({
                "current_round": current_round,
                "outer_prompt": prompt_str,
                "i": i,
                "strength":lora_str
            })
            
            # 從 render_cache 目錄中取得圖片檔案
            image_path = os.path.join(render_cache_dir, case_image_path)
            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{encoded_image}"
            else:
                image_url = "未生成"

            # 將每個生成結果以字典形式存放，key 為生成次數，值為包含檔名與 URL 的字典
            combined_images.append({i: case_image_path, "output": image_url})

        # 使用 custom_add_messages 累加存入 state["case_image"]
        existing_images = self.state.get("case_image", [])
        updated_images = custom_add_messages(existing_images, combined_images)
        self.state["case_image"] = updated_images

        print(f"✅ 未來情境圖生成完成，圖片資訊: {combined_images}")
        return {"case_image": self.state["case_image"]}    

# GATE 檢查方案（請回答：有/沒有） OK
class GateCheck2:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        # 获取当前轮次与生成的图片列表
        current_round = self.state.get("current_round", 0)
        case_images = self.state.get("case_image", [])
        design_advice_list = self.state.get("design_advice", [])
        
        # 過濾出當前輪次且 state 為 True 的設計方案（必須是字典格式）
        valid_advices = [
            advice for advice in design_advice_list
            if isinstance(advice, dict) and advice.get("round") == current_round and advice.get("state") == True
        ]
        
        # 從有效的設計方案中取出 "proposal" 作為 advice_text
        if valid_advices:
            selected_advice = valid_advices[0]
            advice_text = selected_advice.get("proposal", "無目標")
        else:
            advice_text = "無目標"

        # 提取每个字典中整数键对应的图片文件名
        image_dict = {}
        for item in case_images:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(key, int):
                        image_dict[key] = value

        # 检查是否有符合条件的图片
        if not image_dict:
            print("⚠️ 当前轮次无符合条件的生成图")
            self.state["GATE2"] = "没有"
            self.state["GATE_REASON2"] = "当前轮次无符合条件的生成图"
            return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"]}

        # 为每个图片文件名添加完整路径
        OUTPUT_SHELL_CACHE_DIR = "./output/render_cache"
        full_paths = {key: os.path.join(OUTPUT_SHELL_CACHE_DIR, filename) for key, filename in image_dict.items()}

        # 将图片文件名整理成多行字符串供 prompt 使用（只显示文件名，不含路径）
        image_list_str = "\n".join(image_dict.values())

        # 准备 prompt，要求 LLM 根据设计要求评估并选出最佳生成图
        prompt = f"""
        你是一位專業的建築圖像評審員，專精於從圖片評估建築構造與製造的可能性。
        **請根據以下條件進行嚴格評估。

        **優先項目**
        設計符合性與合理性： **需嚴格確保圖片符合設計提案所述外觀**。結構與造型需合理且符合預期，展現良好的建築設計邏輯。
        **圖片優劣評比項目**
        圖片品質與細節： 圖片必須清晰，細節表現良好沒有扭曲或透視錯誤。
        曲面簡潔度： 曲面線條是否簡潔流暢，避免過於複雜破碎。以有效率的使用材料和簡化製造流程。
        接合構造簡潔性： 木構件接合方式是否簡潔明瞭，避免過於複雜繁瑣。以提高循環使用潛力、降低組裝難度，並減少潛在的結構風險。
        表面處理完整性： 木材表面是否有塗層、封邊或其他保護處理，處理是否均勻完整，是否能看出針對基地環境氣候的防護考量。
        結構系統效率性： 結構系統設計是否有效利用材料特性，以較少材料達成所需效能。材料的使用邏輯是否能避免製造上的浪費。
        造型美觀協調性： 整體造型是否美觀，與周圍環境是否協調。

        **生成圖名順序： {image_list_str}
        **請回復兩行，優先檢查如果沒有任何圖片符合設計提案所述外觀，請僅回復「沒有」：
        第一行：僅回復最佳圖片文件名中的 id 數字部分 (整數)。（例如："shell_result_{current_round}_id.png"，則回覆 id）。
        第二行：回復「有」時綜合說明所有方案的優劣，並詳細解釋選擇此最佳方案的原因。如果回復「沒有」則說明改進建議。

        **設計提案：{advice_text} 
        """.strip()

        # 调用 img_recognition.invoke 处理所有图片
        analysis_result = img_recognition.invoke({
            "image_paths": list(full_paths.values()),
            "prompt": prompt,
        })

        result = analysis_result.strip() if isinstance(analysis_result, str) else ""
        # 将回复按行分割，解析第一行作为最佳图片 id，第二行作为选择原因
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            # 尝试解析第一行数字
            if first_line.isdigit():
                best_id = int(first_line)
                self.state["GATE2"] = best_id
            elif "没有" in first_line or "no" in first_line.lower():
                self.state["GATE2"] = "没有"
            else:
                digit_matches = re.findall(r'\d+', first_line)
                if digit_matches:
                    best_id = int(digit_matches[0])
                    self.state["GATE2"] = best_id
                else:
                    print("⚠️ 无法解析 LLM 回复中的最佳方案 id。")
                    self.state["GATE2"] = "没有"
            
            # 解析第二行作为选择原因，若有提供则存入 GATE_REASON2
            if len(lines) >= 2:
                self.state["GATE_REASON2"] = lines[1]
            else:
                self.state["GATE_REASON2"] = ""
        else:
            print("⚠️ LLM 回复为空，请检查 prompt 格式。")
            self.state["GATE2"] = "没有"
            self.state["GATE_REASON2"] = ""

        # 根據 GATE2 判斷評估結果：若為 "没有"，則 state_value 為 False，否則為 True
        state_value = False if self.state["GATE2"] == "没有" else True

        # 僅為 outer_prompt 列表中的最後一個對象新增 state 鍵
        outer_prompt = self.state.get("outer_prompt", [])
        if outer_prompt and isinstance(outer_prompt[-1], dict):
            outer_prompt[-1]["state"] = state_value
        # 將更新後的 outer_prompt 回寫回 state
        self.state["outer_prompt"] = outer_prompt

        # 將更新後的 outer_prompt 存回 state
        self.state["outer_prompt"] = outer_prompt                

        print(f"【GateCheckCaseImage】以收到最佳評估結果：{self.state.get('GATE2')}，原因：{self.state.get('GATE_REASON2')} 😊")
        return {"GATE2": self.state["GATE2"], "GATE_REASON2": self.state["GATE_REASON2"], "outer_prompt": self.state["outer_prompt"]}

# 未來情境生成：呼叫 LLM（使用圖片生成工具）根據外殼 prompt 與融合圖生成未來情境圖 OK
class FutureScenarioGenerationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        # 1️⃣ 獲取當前輪次、圖片列表與最佳方案 ID (gate2)
        current_round = self.state.get("current_round", 0)
        case_images = self.state.get("case_image", [])
        gate2 = self.state.get("GATE2", None)

        # 確保 gate2 為整數型態
        if not isinstance(gate2, int):
            print("⚠️ GATE2 的值無效或未找到")
            self.state["future_image"] = [{"future_image": "没有"}]
            return {"future_image": self.state["future_image"]}

        best_id = gate2

        # 2️⃣ 過濾符合條件的圖片：找到 key 為 best_id，且 value 以 "shell_result_{current_round}_" 為前綴的項目
        expected_prefix = f"shell_result_{current_round}_"
        result = None
        for item in case_images:
            if isinstance(item, dict) and best_id in item:
                case_image_path = item[best_id]
                if isinstance(case_image_path, str) and case_image_path.startswith(expected_prefix):
                    result = item
                    break

        # 3️⃣ 若未找到符合條件的圖片，則返回提示
        if not result:
            print(f"⚠️ 未找到符合条件的生成图，轮次：{current_round}，方案 ID：{best_id}")
            self.state["future_image"] = [{"future_image": "没有"}]
            return {"future_image": self.state["future_image"]}

        # 4️⃣ 更新狀態並返回結果：將結果包裝在列表中
        self.state["future_image"] = [result]
        print(f"✅ 未来情境图生成完成，图片保存为: {result}")
        return {"future_image": self.state["future_image"]}

# 生成 3D =：根據 Glb 檔呼叫 LLM（使用圖片生成工具）生成 3D 
class Generate3DPerspective:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state        

        # 1️⃣ 獲取當前輪次、圖片列表與最佳方案 ID (gate2)
        current_round = self.state.get("current_round", 0)
        case_images = self.state.get("case_image", [])
        gate2 = self.state.get("GATE2", None)

        # 確保 gate2 為整數型態
        if not isinstance(gate2, int):
            print("⚠️ GATE2 的值無效或未找到")
            self.state["perspective_3D"] = "没有"
            return {"perspective_3D": self.state["perspective_3D"]}
        best_id = gate2

        # 2️⃣ 過濾符合條件的圖片：找到 key 為 best_id，且 value 以 "shell_result_{current_round}_" 為前綴的項目
        expected_prefix = f"shell_result_{current_round}"
        selected_image = None
        for item in case_images:
            if isinstance(item, dict) and best_id in item:
                value = item[best_id]
                if isinstance(value, str) and value.startswith(expected_prefix):
                    selected_image = value
                    break

        # 若未找到符合條件的圖片，則返回提示
        if not selected_image:
            print(f"⚠️ 未找到符合条件的生成图，轮次：{current_round}，方案 ID：{best_id}")
            self.state["perspective_3D"] = "未找到符合条件的生成图"
            return {"perspective_3D": self.state["perspective_3D"]}

        # 3️⃣ 若選中的圖片路徑不是完整路徑，則補上目錄 "./output/render_cache"
        OUTPUT_SHELL_CACHE_DIR = "./output/render_cache"
        if not os.path.isabs(selected_image):
            selected_image = os.path.join(OUTPUT_SHELL_CACHE_DIR, selected_image)

        # 4️⃣ 呼叫 generate_3D 時，使用鍵 "image_path" 並傳入 selected_image
        object_file = generate_3D.invoke({
            "image_path": selected_image,
            "current_round": current_round,
        })
        object_video = object_file.get("video", "無生成結果")
        object_glb = object_file.get("model", "無模型")

        # 更新 3D 影片與模型資訊
        existing_3D = self.state.get("perspective_3D", [])
        updated_3D = custom_add_messages(existing_3D, object_video)
        self.state["perspective_3D"] = updated_3D

        existing_model = self.state.get("model_3D", [])
        updated_model = custom_add_messages(existing_model, object_glb)
        self.state["model_3D"] = updated_model

        print(f"✅ 生成 3D 位置: 影片:{object_video}、模型:{object_glb}")
        return {"perspective_3D": self.state["perspective_3D"], "model_3D": self.state["model_3D"]}

# class Generate3DPerspective顯示測試:
#     def __init__(self, state: GlobalState):
#         self.state = state

#     def encode_file_to_data_url(self, file_path, mime_type):
#         """讀取檔案並轉換成 data URL 格式"""
#         if os.path.exists(file_path):
#             with open(file_path, "rb") as f:
#                 encoded = base64.b64encode(f.read()).decode("utf-8")
#             return f"data:{mime_type};base64,{encoded}"
#         else:
#             return None

#     def run(self, state=None):
#         if state is not None:
#             self.state = state        

#         # 補上目錄 "./output/model_cache"，並指定測試檔案名稱
#         OUTPUT_MODEL_CACHE_DIR = "./output/model_cache"
#         selected_model = os.path.join(OUTPUT_MODEL_CACHE_DIR, "model_result_2.glb")
#         selected_mp4 = os.path.join(OUTPUT_MODEL_CACHE_DIR, "video_result_2.mp4")

#         # 將影片與模型轉換成 data URL，傳入對應的 MIME 類型
#         video_data_url = self.encode_file_to_data_url(selected_mp4, "video/mp4")
#         model_data_url = self.encode_file_to_data_url(selected_model, "model/gltf-binary")

#         self.state["perspective_3D_display"] = video_data_url if video_data_url else "無生成結果"
#         self.state["model_3D_display"] = model_data_url if model_data_url else "無模型"

#         print(f"✅ 生成 3D 位置: 影片:{selected_mp4}、模型:{selected_model}")
#         return {
#             "perspective_3D": self.state["perspective_3D_display"],
#             "model_3D": self.state["model_3D_display"]
#         }
    
# 深度評估任務：呼叫 LLM（使用圖片辨識工具）對生成圖與未來情境圖進行深度評估 OK
class DeepEvaluationTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state      

        OUTPUT_EVAL_DIR = "./output/"
        os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True) 

        current_round = self.state.get("current_round", 0) 
        eval_results_list = self.state.get("evaluation_result", [])  
        future_img = self.state.get("future_image", [])      
        perspective_3D = self.state.get("perspective_3D", [])
        design_advice_list = self.state.get("design_advice", [])
        
        # 過濾出當前輪次且 state 為 True 的設計方案（必須是字典格式）
        valid_advices = [
            advice for advice in design_advice_list
            if isinstance(advice, dict) and advice.get("round") == current_round and advice.get("state") == True
        ]
        
        # 從有效的設計方案中取出 "proposal" 作為 advice_text
        if valid_advices:
            selected_advice = valid_advices[0]
            advice_text = selected_advice.get("proposal", "無目標")
        else:
            advice_text = "無目標"

        # --- 提取 image ---
        expected_prefix = f"shell_result_{current_round}"
        images = []  # 存放符合條件的 image 字串
        # 遍歷 future_img 中的所有項目（每個項目皆為字典）
        for item in future_img:
            if isinstance(item, dict):
                for key, value in item.items():
                    # 檢查 key 為 int 且 value 為字串，且前綴符合
                    if isinstance(key, int) and isinstance(value, str) and value.startswith(expected_prefix):
                        images.append(value)
                        # 如果希望每個字典只提取一個符合條件的值，可以 break 退出內層迴圈
                        break

        # --- 提取 video ---
        expected_prefix2 = f"video_result_{current_round}"
        videos = []  # 存放符合條件的 video 字串
        # perspective_3D 預期為 list，每個元素皆為字串
        for item in perspective_3D:
            if isinstance(item, str) and item.startswith(expected_prefix2):
                videos.append(item)

        # --- 路徑組合 ---
        OUTPUT_SHELL_CACHE_DIR = "./output/render_cache"
        image_paths = [os.path.join(OUTPUT_SHELL_CACHE_DIR, img) for img in images] if images else []

        OUTPUT_3D_CACHE_DIR = "./output/model_cache"
        video_paths = [os.path.join(OUTPUT_3D_CACHE_DIR, vid) for vid in videos] if videos else []

        # **關鍵字生成方向：
        # 結合設計提案特性：請考量設計提案可能包含的元素，例如：建築類型、曲面形式 (例如：雙曲面、自由曲面、格柵曲面、薄殼曲面...)、材料種類 (例如：集成材、膠合木、CLT...)、構造工法、其他考量等。
        # 辨識圖片了解情況：假設已透過圖片辨識初步了解設計方案的視覺特徵，例如：曲面的複雜程度、結構系統的類型等。請根據這些可能的圖片資訊，生成更精確的關鍵字。   
        # 設計提案：{advice_text} 

        # Step 1: 關鍵詞 
        keyword_prompt = (f"""
            請生成適用於檢索參考做法的中英文關鍵字。**格式為:中文(英文)**
            檢索目標：根據圖片中的建築要素尋找關於曲面木構造建築的設計概念、方案、案例研究等資料。
            尋找曲面木構造在設計、材料、工法、循環性、永續性等方面的規範、技術指南、專家建議等參考資訊。 
            """
        )
        img_key_output = img_recognition.invoke({
            "image_paths": image_paths,
            "prompt": keyword_prompt
        })        
        keywords = img_key_output.strip() if isinstance(img_key_output, str) else ""
        print("生成的關鍵詞：", keywords)

        # Step 2: RAG prompt
        RAG_msg = ARCH_rag_tool.invoke(f"{keywords}")
        print("RAG檢索結果：", RAG_msg)

            # 數位製造背景知識: (例如：機械手臂木構加工原理、曲面分割與展開演算法、參數化設計在木構建築的應用、數位組裝流程與精度控制等相關文獻、技術指南、案例研究連結)
            # Timber Curve Frame Pavilion 設計規範: (例如：設計圖說、結構分析報告、材料選用說明、初步的製造流程規劃、設計目標與預期成果描述等)
            # 相關案例參考: (例如：已成功數位製造的曲面木構建築案例、類似結構形式的案例分析、數位製造工法應用案例等，可提供圖片或連結)

        img_prompt = (   
            f"針對 timber curve frame pavilion 設計方案渲染圖進行深入評估。"
            f"作為資深建築設計評審委員，請針對補充條件動態調整評估準則，並提供**公正且有鑑別度的評分**。"
            f"你的任務是基於以下評估準則，**客觀評估**其建築外殼設計的優劣。"
            f"""
            **造型與環境脈絡融合：總分10分
                評估建築造型是否能融入周圍環境脈絡，例如：自然景觀、都市紋理、地域文化。
                考量建築造型與環境的協調性、呼應性，以及對環境的尊重程度。
            **場所精神與使用者關注：總分10分
                評估建築設計是否能營造獨特的場所精神，回應使用者的需求與體驗。
                考量建築空間的氛圍、舒適度、機能性，以及對使用者情感和行為的影響。
            **材料及工法的環境及氣候應對程度：總分10分
                評估選用的木材材料和工法是否能有效應對當地環境及氣候條件。
                考量材料的永續性、環境友善性、氣候適應性，以及工法的合理性、效率性、材料損耗。
            **外殼系統的維護性與耐久性：總分10分
                評估當前構造形式的系統是否考量到後續的維護與長期耐久性。
                考量當前構造系統全生命週期的循環性。    
            **補充條件:{RAG_msg}
                
            評分標準:針對以上每個評估項目，根據方案表現給予 1.0 - 10.0 分評分 (1.0 = 極差, 10.0 = 極佳)。
            輸出格式:針對每個評估項目提供評分以及簡述評分理由。最後需計算加總得分並寫為**總分數:數字**
            """  
        )

        # 調用工具進行深度評估
        img_eval_output = img_recognition.invoke({
            "image_paths": image_paths,
            "prompt": img_prompt
        })

        ##3D辨識邏輯     
        ##還需要設定RAG木構造資料
        vid_prompt = (   
            f"針對 timber curve frame pavilion 設計方案模型進行深入評估。"
            f"作為專業的建築師、結構技師兼數位製造專家，請針對補充條件動態調整評估準則，並提供**公正且有鑑別度的評分**。"
            f"你的任務是基於以下評估準則，**客觀評估**其建築設計的優劣。"      
            f"""
            **I.整體構造系統之合理性與永續性:總分10分
                    **結構邏輯性:**  結構系統是否清晰、合理，能有效傳遞力流並抵抗外力？ (例如：抗彎、抗剪、抗扭能力評估)
                    **結構效率:**  結構系統是否能以最少的材料達成所需的跨度與承載力？ (例如：構材用量、跨度能力比值分析)
                    **材料永續性:**  結構材料選用是否符合永續發展原則？ (例如：可再生材料比例、碳足跡評估、生命週期評估 LCA)
                    **環境友善性:**  結構系統的生產、運輸、建造及拆解過程對環境的影響程度？ (例如：碳排放量、廢棄物產生量評估)
            **II.細部構造之機能性與整合性:總分10分
                    **機能實現度:**  細部構造是否能有效實現其預期機能？ (例如：連接強度、防水性能、氣密性能評估)
                    **構造整合性:**  細部構造與整體結構系統的協調性與整合程度？ (例如：力流傳遞的連續性、構造系統的完整性)
                    **節點設計:**  節點設計是否安全可靠、簡潔有效、易於製造與組裝？ (例如：節點力學性能分析、連接方式效率評估、組裝複雜度分析)
                    **介面協調性:**  細部構造與其他建築系統 (例如：外牆、屋面、設備) 的介面處理是否協調合理？ (例如：防水細節、保溫隔熱措施、管線整合方案)
            **III. 曲面造型與構成形式之技術可行性:總分10分
                    **幾何複雜度:**  曲面造型的幾何形式是否過於複雜，增加製造與建造難度？ (例如：曲率變化分析、曲面分格複雜度評估)
                    **製造技術:**  模型所展現的曲面構成形式，在現有製造技術條件下是否能實現？ (例如：CNC 加工可行性、熱壓成型可行性、積層製造可行性評估)
                    **組裝精度:**  模型所展現的曲面精度要求，在現場組裝條件下是否能達成？ (例如：構件加工精度要求、組裝誤差容許度分析)
                    **經濟性:**  曲面造型的實現是否會導致過高的製造成本與工期？ (例如：材料成本分析、加工成本估算、工期評估)
            **IV. 材料應用與結構邏輯之契合性:總分10分
                    **材料特性發揮:**  是否充分利用木材的力學性能 (例如：抗拉、抗壓、彈性模量)、紋理特性、輕質高強等優勢？ (例如：材料力學性能分析、材料選用合理性評估)
                    **結構邏輯清晰性:**  結構系統的設計是否清晰地展現了材料的力學特性與結構邏輯？ (例如：結構受力分析、力流傳遞路徑可視化)
                    **材料應用效率:**  材料的應用是否經濟高效，避免過度設計或材料浪費？ (例如：材料用量優化分析、構件尺寸合理性評估)
            **V. 製造流程與組裝可行性:總分10分
                    **組裝流程可行性:**  模型的組裝步驟是否清晰合理、易於理解與操作？ (例如：組裝步驟流程圖、組裝難度分析)
                    **製造效率優化:**  設計方案是否具有優化製造效率的潛力？ (例如：預製化程度評估、模組化設計分析、自動化生產應用潛力)
                    **材料損耗控制:**  製造過程中是否能有效控制材料損耗？ (例如：材料切割優化方案、剩料再利用策略)
                    **構件搬運:**  構件的尺寸、重量是否便於搬運與運輸？ (例如：構件尺寸限制分析、運輸成本估算、現場吊裝可行性)
            **VI. 美學表現與空間意象:總分10分
                    **造型美感:**  整體造型是否符合建築美學原則，具有視覺吸引力？ (例如：比例協調性評估、線條流暢度分析、形式美感評價)
                    **光影效果:**  整體空間是否展現良好的光影效果，提升空間的層次感與生動性？  (例如：使用者預期評價)


            **補充條件:{RAG_msg}

            評分標準:針對以上每個評估項目，根據方案表現給予 1.0 - 10.0 分評分 (1.0 = 極差, 10.0 = 極佳)。
            輸出格式:針對每個評估項目提供評分以及簡述評分理由。最後需計算加總得分並寫為**總分數:數字**
            """  
        )

        vid_eval_output = video_recognition.invoke({
            "video_paths": video_paths,
            "prompt": vid_prompt
        })

        img_eval_text = img_eval_output.strip() if isinstance(img_eval_output, str) else ""
        vid_eval_text = vid_eval_output.strip() if isinstance(vid_eval_output, str) else ""

        ##evaluation_count平均
        def extract_total_score(text):
            m = re.search(r"\*\*總分數:([\d.]+)\*\*", text)
            if m:
                try:
                    return float(m.group(1))
                except:
                    return 0.0
            else:
                # 如果沒有符合的總分數格式，取文本中所有浮點數，並返回最大的數字
                numbers = re.findall(r"(\d+(?:\.\d+)?)", text)
                if numbers:
                    return max(map(float, numbers))
                return 0.0
        img_total = extract_total_score(img_eval_text)
        vid_total = extract_total_score(vid_eval_text)
        all_score = img_total + vid_total

        # 組合本輪的評估結果，格式為 { current_round: 當前輪次, eval_result: img評語, eval_result2: video評語}
        current_eval_result = {
            "current_round": current_round,            
            "eval_result": img_eval_text,
            "eval_result2": vid_eval_text
        }
        eval_results_list =custom_add_messages(eval_results_list, current_eval_result)
        self.state["evaluation_result"] = eval_results_list

        # 組合本輪的平均分數，格式為 {current_round: average_score}，這裡將 current_round 作為字串鍵存入
        current_eval_count = {str(current_round): all_score}
        eval_count_list = self.state.get("evaluation_count", [])
        eval_count_list = custom_add_messages(eval_count_list, current_eval_count)
        self.state["evaluation_count"] = eval_count_list

        # ✅ 存入檔案
        # 組合 Markdown 格式內容，將評估結果寫入 Markdown 檔案
        md_content = f"""# Evaluation Result for Round {current_round}

        ## Image Evaluation
        {img_eval_text}

        ## Video Evaluation
        {vid_eval_text}

        ## Total Score: {all_score}
        """
        eval_file_path = os.path.join(OUTPUT_EVAL_DIR, f"eval_result_{current_round}.md")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.state["current_round"] = current_round + 1  
        print(f"✅ 深度評估完成，當前輪次: {current_round}")
        print(f"📌 評估結果: {current_eval_result}")
        print(f"📌 平均分數: {all_score}")
        return {
            "evaluation_result": self.state["evaluation_result"],
            "evaluation_count": self.state["evaluation_count"],
            "current_round": self.state["current_round"]
        }

# 評估檢查任務：根據評估次數決定流程路由（參考條件分支範本邏輯）
class EvaluationCheckTask:
    def __init__(self, state: GlobalState):
        self.state = state

    def run(self, state=None):
        if state is not None:
            self.state = state

        count = self.state.get("current_round", 0)
        if count < 3:
            self.state["evaluation_status"] = "NO"
            print(f"EvaluationCheckTask：評估次數 {count} 未達標，將返回 RAGdesignThinking 執行下一輪。")
        else:
            self.state["evaluation_status"] = "YES"
            print(f"EvaluationCheckTask：評估次數 {count} 達標，流程結束。")
        return {"evaluation_status": self.state["evaluation_status"]}

# 總評估任務(用戶可介入)
class FinalEvaluationTask:
    def __init__(self, state: dict, short_term=None, long_term=None):
        self.state = state
        # 若未傳入則使用預設的記憶管理器
        self.short_term = short_term if short_term is not None else get_short_term_memory()
        self.long_term = long_term if long_term is not None else get_long_term_store()

    def run(self, state=None):
        if state is not None:
            self.state = state

        # 從 state 中取得評估結果與評分（累加列表）
        eval_results = self.state.get("evaluation_result", [])
        eval_counts = self.state.get("evaluation_count", [])

        # 從記憶管理器讀取短期與長期記憶內容
        short_memory = self.short_term.retrieve_all() if hasattr(self.short_term, "retrieve_all") else ""
        long_memory = self.long_term.retrieve_all() if hasattr(self.long_term, "retrieve_all") else ""

        # 建立優化後的 prompt：
        # 要求 LLM 根據當前輪次直接指出哪個設計方案較好，並分析剩餘方案的優缺點，
        # 最後給出持續執行方案的深入建議。
        current_round = self.state.get("current_round", 0)
        summary_prompt = f"""請根據以下評估結果：      
        【評估結果】
        """
        for res in eval_results:
            round_num = res.get("current_round", "未知")
            summary_prompt += f"\n輪次 {round_num}："
            summary_prompt += f"\n - 圖片評估：{res.get('eval_result', '無')}"
            summary_prompt += f"\n - 3D 視角評估：{res.get('eval_result2', '無')}\n"
        
        summary_prompt += "\n【評分結果】\n"
        for count in eval_counts:
            for round_key, score in count.items():
                summary_prompt += f"輪次 {round_key}：總分 {score}\n"

        summary_prompt += "\n【記憶內容】\n"
        summary_prompt += "短期記憶：" + (short_memory if short_memory else "無") + "\n"
        summary_prompt += "長期記憶：" + (long_memory if long_memory else "無") + "\n\n"

        summary_prompt += f"""請綜合以上資訊，請直接指出在第 {current_round} 輪的方案表現最佳，
        並詳細說明該方案的優點，同時分析其他方案的優缺點。最後，請提供一個持續執行此方案的深入建議。
        """

        # 傳入的 prompt 必須是一個列表，且每個元素需為 BaseMessage，例如使用 SystemMessage 包裝
        llm_response = llm.invoke([SystemMessage(content=summary_prompt)])
        final_text = llm_response.content if hasattr(llm_response, "content") else llm_response

        # 將最終摘要存入 state 中
        self.state["final_evaluation"] = final_text

        print("✅ 總評估任務完成！")
        print(f"📌 總評估結果:\n{final_text}")
        return {"final_evaluation": self.state["final_evaluation"]}


    # def interactive_query(self, query: str):
    #     """
    #     當用戶詢問時，利用現有的最終評估摘要以及短期記憶來回覆問題。
    #     """
    #     # 從 state 或長期記憶中取出最終評估摘要
    #     final_eval = self.state.get("final_evaluation", "")

    #     if not final_eval:
    #         final_eval = self.long_term.get("final_evaluation") or ""

    #     interactive_prompt = (
    #         f"根據以下最終評估摘要，請回答用戶的問題：\n\n"
    #         f"{final_eval}\n\n"
    #         f"用戶問題：{query}\n"
    #         "請以中文詳盡回答："
    #     )
    #     response = llm.invoke({"prompt": interactive_prompt})
    #     answer = response.content if hasattr(response, "content") else response

    #     # 可將此交互內容存入短期記憶，方便後續對話追蹤
    #     self.short_term.save("final_evaluation_interaction", {"query": query, "answer": answer})

    #     return answer


# =============================================================================
# 建立工作流程圖 (Graph Setup)
# =============================================================================
workflow = StateGraph(GlobalState)
# workflow.config_schema = AssistantConfig

state = {
    "設計目標x設計需求x方案偏好": [],
    "design_summary": "",
    "analysis_img": "",
    "site_analysis": "",
    "design_advice": "",
    "case_image": "",
    "outer_prompt": "",
    "future_image": "",
    "perspective_3D": "",
    "model_3D": "",
    "GATE1": 1,
    "GATE2": 1,
    "GATE_REASON1": "",
    "GATE_REASON2": "",
    "current_round": 0,
    "evaluation_count": 0,
    "evaluation_status": "",
    "evaluation_result": "",
    "final_evaluation": ""
}

question_task = QuestionTask(state)
site_analysis_task = SiteAnalysisTask(state)
rag_thinking = RAGdesignThinking(state)
gate_check1 = GateCheck1(state)
shell_prompt = OuterShellPromptTask(state)
image_render = CaseScenarioGenerationTask(state)
gate_check2 = GateCheck2(state)
future_scenario = FutureScenarioGenerationTask(state)
generate_P3D = Generate3DPerspective(state)
deep_evaluation = DeepEvaluationTask(state)
evaluation_check = EvaluationCheckTask(state)
final_eval = FinalEvaluationTask(state)
# GenerateReactFlow = GenerateReactFlowTask(state)

workflow.set_entry_point("question_summary")

workflow.add_node("question_summary", question_task.run)
workflow.add_node("analyze_site", site_analysis_task.run)
workflow.add_node("designThinking", rag_thinking.run)
workflow.add_node("GateCheck1", gate_check1.run)
workflow.add_node("shell_prompt", shell_prompt.run)
workflow.add_node("image_render", image_render.run)
workflow.add_node("GateCheck2", gate_check2.run)
workflow.add_node("generate_3D", generate_P3D.run)
workflow.add_node("future_scenario", future_scenario.run)
workflow.add_node("deep_evaluation", deep_evaluation.run)
workflow.add_node("evaluation_check", evaluation_check.run)
workflow.add_node("final_eval", final_eval.run)
# workflow.add_node("GenerateReactFlow", GenerateReactFlow.run)

workflow.add_edge("question_summary", "analyze_site")
workflow.add_edge("analyze_site", "designThinking")
workflow.add_edge("designThinking", "GateCheck1")
workflow.add_edge("shell_prompt", "image_render")
workflow.add_edge("image_render", "GateCheck2")
workflow.add_edge("future_scenario", "deep_evaluation")
workflow.add_edge("generate_3D", "deep_evaluation")
workflow.add_edge("deep_evaluation", "evaluation_check")
# workflow.add_edge("deep_evaluation", "GenerateReactFlow")
workflow.add_edge("final_eval", END)

# GateCheck1 條件節點
workflow.add_conditional_edges(
    "GateCheck1",
    lambda state: "YES" if state.get("GATE1") == "有" else "NO",
    {
        "YES": "shell_prompt", "NO": "designThinking"  
    }
)

# GateCheck2 條件節點
workflow.add_conditional_edges(
    "GateCheck2",
    lambda state: "YES" if isinstance(state.get("GATE2"), int) else "NO",
    { "YES": "future_scenario", "NO": "shell_prompt" }
)
workflow.add_edge("GateCheck2", "generate_3D") 

workflow.add_conditional_edges("evaluation_check",lambda state: state["evaluation_status"],
    { "NO": "designThinking",   "YES": "final_eval"  })

# =============================================================================
# 構建並運行流程
# =============================================================================
graph = workflow.compile()

# graph.invoke(state)
graph.name = "Multi-Agent System"
