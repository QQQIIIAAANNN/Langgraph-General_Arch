import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Optional, Any

from src.configuration import ConfigManager

# 載入 .env 設定，以獲取 API 金鑰
load_dotenv()

# --- 從 configuration.py 參考的模型和語言 Literal ---
AllModelNamesLiteral = Literal[
    "DEFAULT(gpt-4o-mini)",
    # OpenAI
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o3-mini",
    # Google
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-flash-preview-04-17",
    # Anthropic
    "claude-3-5-sonnet-20240620",
]

SupportedLanguage = Literal[
    "繁體中文",
    "English",
    "日文"
]

# --- 集中存放 Prompt 模板文字 (並加入語言變數) ---
PROMPT_TEXT_TEMPLATES = {
    "QuestionTask": {
        "keyword": (
            "請從用戶輸入文本中提取核心設計意圖、關鍵技術需求（如木構造、數位製造等）和期望的建築特徵，"
            "生成一個簡潔、精煉的自然語言查詢，用於在知識庫和網路文獻中檢索相關的設計案例、構造細節、製造工法、"
            "減碳與循環永續性策略、技術細節及研究理論。請使用用戶的輸入語言來回答。\n"
            "用戶輸入：\n{user_input}\n語言: {llm_output_language}"
        ),
        "summary": (
            "以用戶的設計目標和指定的建築類型為主，並綜合以下檢索到的補充資訊，"
            "請深入分析並總結出一個清晰的設計方向和需要達成的關鍵設計決策，以滿足用戶的設計目標。"
            "重點關注如何融合補充資訊中的案例、構造、工法、永續性和技術理論。"
            "請使用用戶的輸入語言來回答。\n建築設計目標:\n{user_input}\n補充檢索資訊:\n{rag_msg}\n語言: {llm_output_language}"
        ),
    },

    "SiteAnalysisTask": {
        "img_recognition": """你是一個專業的都市視覺資訊分析工具，你的任務是分析使用者提供的基地圖片，辨識並標註都市環境物件，描述重要特徵，並生成視覺摘要。
輸出結構化資料，包含物件標註 (類型、位置)、特徵描述 (建築、道路、綠地、水體、環境脈絡特徵) 。
以圖片的上方為北方，初步推理基於方位來說，日照、熱環境、風環境、噪音、景觀、交通對於基地的影響。

設計位置:{region}，經緯度:{geo_location}\n語言: {llm_output_language}""",
        "rag_keywords_generation": """根據以下基地基本資訊和圖片初步辨識結果，請生成用於檢索更詳細基地背景資料的關鍵字。
重點檢索方向應包含：當地的**都市計畫規範、建築法規、氣候資料（溫度、濕度、降雨、盛行風向等）、日照軌跡、人文歷史特色、水文地質條件、周邊重要設施或建成環境、潛在的環境敏感點（如噪音源、污染源）**。
請提供精確且多樣化的中英文關鍵字，以便全面搜集資訊。格式為: 中文(英文)。

基地位置：{region}，經緯度：{geo_location}
圖片初步辨識摘要：{initial_img_analysis_summary}
語言: {llm_output_language}""",
        "llm_analysis": """作為建築師及空間分析專家，你擅長整合提供的資訊進行基地分析。
請基於以下基地圖片的辨識結果，以及透過關鍵字檢索到的補充背景資料，整理一份全面且深入的基地分析報告。
報告應涵蓋設計機遇與限制，特別關注日照、風環境、噪音、景觀、交通可達性、法規限制、氣候適應性、文化脈絡等方面。

設計位置:{region}，經緯度:{geo_location}。
圖片辨識結果:{analysis_img}
檢索到的補充背景資料:\n{rag_supplementary_info}
語言: {llm_output_language}""",
        "extract_site_info_from_user_input": """你是一個資訊提取助理。請從以下用戶的設計需求描述中，識別並提取專案的「地區」(例如城市、區域名稱) 和「地理位置/經緯度」。
如果用戶明確提到了地區或城市名稱，請將其作為「地區」"region"。
如果用戶明確提到了經緯度座標，請將其作為「地理位置/經緯度」"geo_location"。
如果只提到大概位置而沒有精確經緯度，請將「地理位置/經緯度」標註為「未知」。
如果兩者都未提及，或資訊不足以判斷，請將「地區」和「地理位置/經緯度」都標註為「未知」。

請嚴格以 JSON 格式回覆，且僅包含 "region" 和 "geo_location" 兩個鍵。

用戶設計需求描述：
{user_design_input}
語言: {llm_output_language}"""
    },

    "RAGdesignThinking": {
        "keywords": """你是一位經驗豐富的資深建築設計顧問。
基於以下初步整合的用戶設計目標和基地分析報告：
用戶設計目標總結：{design_goal_summary}
基地分析報告：{analysis_result}

請你提煉出至少三個核心的設計指導原則或創新的設計切入點，這些原則應直接回應上述目標和基地條件。
著重於所提議建築的幾何形態、外殼形式、所選構造方式的獨特性、與環境的互動方式，以及可能的數位製造或傳統工法的應用潛力。
提供的方向應具有启发性，能够引导下一步更具体的设计方案构思。
請使用用戶設定的語言: {llm_output_language}""",
        "rag_tool_query": "此 Prompt 不再於 RAGdesignThinking 中直接使用。",
        "complete_scheme": """你是一位經驗豐富的資深建築設計顧問，根據以下方面生成一個具體、創新且可行的建築設計方案：
主要設計決策針對**幾何形狀(比如方、圓、三角、錐型、塔型等)、外殼形式(比如平面、單曲面、雙曲面、自由曲面等)、主要構造細節**。
次要設計決策需回應日照、熱環境、風環境、噪音、景觀、基地周遭紋理等。
設計方案需專注於外殼設計，可展現參數式設計的美感、高度創意性及前衛性。

請綜合以下內容，以設計目標、改進建議和新的設計方向為重點，提出一個完整的設計方案。
**用戶設計目標總結**：{design_goal_summary}
**先前方案的改進建議 (若有)**: {improvement}
**基地分析報告**：{analysis_result}
**核心設計方向指引**：{design_directions} 

請確保方案的細節足夠豐富，能夠清晰傳達設計意圖。
請使用用戶設定的語言: {llm_output_language}""",
    },

    "GateCheck1": {
        "evaluation": """你是一位專業的建築方案評審員。
請根據以下設計建議提供判斷及評比，須對於設計需求具有回應性，且與之前輪次的方案不過於接近。

1.循環經濟潛力 (Circular Economy Potential): 方案是否展現朝向材料循環利用、永續木材來源的潛力？
判斷點: 方案是否具有發展材料再利用、回收、模組化或組裝效率等處理計畫的機會 (即使沒有詳細計畫)？
2.材料效率潛力 (Material Efficiency Potential): 方案是否展現減少材料浪費、提升材料利用率的潛力？
判斷點: 方案是否具有發展規劃優化設計、數位製造、集成木材等方法的機會 (即使沒有具體數據)？
3.製造效率潛力 (Manufacturing Efficiency Potential): 方案是否展現提升製造與施工效率的潛力？
判斷點: 方案是否具有發展規劃預製化、模組化、自動化生產、簡化施工等策略的機會 (即使沒有詳細流程)？
4.永續環保潛力 (Environmental Sustainability Potential): 方案是否展現降低環境足跡、符合永續環保原則的潛力？
判斷點: 方案是否具有發展規劃建材的減少浪費、環境友善、減少污染的製造策略的機會 (即使沒有量化數據)？
5.減碳潛力 (Carbon Reduction Potential): 方案是否展現碳封存、減少碳排放的潛力？
判斷點: 方案是否具有發展規劃構造的在地性、減碳效益、碳封存等策略的機會 (即使沒有碳排計算)？

只有在符合設計需求的前提下，其他方面都具備潛力，才是"有"。反之就是"沒有。
**請回覆兩行：第一行僅包含判斷後的"有"或"沒有"；第二行請說明改進的建議。**
**設計需求**：{design_summary}
當前輪次方案：{formatted_current}
之前輪次方案：{formatted_previous}\n語言: {llm_output_language}""",
    },

    "OuterShellPromptTask": {
        "gpt_generation": """作為建築師與Prompt engineering，請參考以下設計參考建議來推測未來此建築設計的樣貌。
使用英文 prompt，要仔細、具體、使用專業的曲面木構造建築設計語法。不要超過100個字、不用寫城市名稱。只能生成建築物的外觀，要能看到建築整體的高空鳥瞰圖。
Focus on architectural Appearance with photorealistic style. 
Prompt 主要根據設計提案描述此建築設計外觀、幾何造型或曲面型式、構造細部設計及網格分割形式、整體風格與氛圍。
**在具有細節且構造合理的情況下，需要避免構造、幾何面、曲面、網格分割過度複雜**。
**視角必須要看到建築整體，高質感，鳥瞰圖或透視圖為主**，**不要生成室內、內部隔間、家具、玻璃、人**。
開頭一定要寫:8K, detailed, best quality, architectural rendering.
設計提案: {advice_text}
改進建議:{improvement}\n語言: {llm_output_language}""",
        "lora_generation": """請生成一個適合的 LoRA 權重數值，其數值必須在 0.3 到 1 之間。
權重越重（接近 1）表示生成結果會更趨於形式固定沒有創意性的造型，但（例如木質）網格構造清晰適合生成具有簍空網格的構造；
權重越輕（接近 0.3）則生成結果會更具設計發散性但可能失去網格構造或網格不清晰，整體適合生成較為簡約的造型。
請根據設計提案動態生成適合的 LoRA 權重。**僅回答權重的數字**
設計提案: {final_prompt}\n語言: {llm_output_language}""",
    },

    "GateCheck2": {
        "img_evaluation": """你是一位專業的建築圖像評審員，專精於從圖片評估建築構造與製造的可能性。
**請根據以下條件進行評估。

**優先項目**
設計符合性與合理性：需確保圖片大致符合設計提案所述。結構與造型需合理且符合預期，展現良好的建築設計邏輯。
找到整理來說屬於最佳方案的情境圖，必須具有最好的建築設計品質、最合理的設計邏輯、最符合設計提案的設計。
**圖片優劣評比項目**
圖片品質與細節： 圖片必須清晰，細節表現良好沒有扭曲或透視錯誤。最好是能夠看到建築物整體的鳥瞰圖。

**生成圖名順序： {image_list_str}
**請回復兩行，優先檢查如果沒有任何圖片符合設計提案所述外觀，請僅回復「沒有」：
第一行：僅回復最佳圖片所屬的 id(id in round) 數字部分 (整數)。
第二行：如果「有」時綜合說明所有方案的優劣，並詳細解釋選擇此最佳方案的原因。如果回復「沒有」則說明改進建議。

**設計提案：{advice_text}\n語言: {llm_output_language}""",
    },  

    "FutureScenarioGenerationTask": {
        "detail_generation": """作為一個專業的建築視覺化AI，請根據以下設計描述生成一張高質量的建築效果圖。
重點展示該設計方案的**主要構造工法、節點細節、以及外殼表面的主要材質質感**。
風格要求：現代、寫實、注重細節。圖片應清晰，光線合理。這是第 {image_number}/{total_images} 張細節圖。
設計描述：
{base_design_description}
請使用語言: {llm_output_language}""",
        "aging_generation": """作為一個具備時間演化模擬能力的建築視覺化AI，請根據以下原始設計描述，一次性生成 {num_images} 張寫實風格的圖片，分別展示其在 **10年後、20年後、以及30年後** 的視覺外觀。
你需要細緻地刻畫材料老化、風化（例如木材變色、金屬鏽蝕、混凝土污漬）、植被生長與演替、以及可能的環境積累（如灰塵、雨痕）對建築主體造成的視覺影響。
**核心構圖、建築形態和主要結構必須在三張圖片中保持一致**，僅外殼材質老化或局部幾何形變等環境效果隨時間演進。
圖片應按 10年、20年、30年 的順序生成。請在基礎圖片上進行修改以反映這些變化。
一定要放: focused on the main building's 30-year transformation, depicting 30 years of aging, DO NOT generate text.** 
原始設計描述：
{base_design_description}
請使用語言: {llm_output_language}""",
        "facade_detail_construction": """作為一個專業的建築視覺化AI，請基於提供的基礎圖片和設計描述，生成 {num_images} 張細節圖。
這些圖片應重點展示該建築方案的**立面細節、特定視角的構造工法、以及外殼表面的主要材質質感和節點做法**。
請確保生成的細節與基礎圖片中的整體設計風格和形態保持一致。
風格要求：現代、寫實、注重細節，圖片應清晰，光線合理。DO NOT generate text or labels on the image.
基礎設計描述：
{base_design_description}
請使用語言: {llm_output_language}"""
    },

    "DeepEvaluationTask": {
        "keyword_img_recognition": """請生成適用於檢索參考做法的中英文關鍵字。**格式為:中文(英文)**
檢索目標：根據圖片中的建築要素尋找關於該建築類型及其構造（特別是木構造，如果適用）的設計概念、方案、案例研究等資料。
尋找該建築類型及其構造（特別是木構造，如果適用）在設計、材料、工法、循環性、永續性等方面的規範、技術指南、專家建議等參考資訊。\n語言: {llm_output_language}""",
        "rag_tool_query": "請根據以下關鍵字查詢相關資料：{keywords}\n語言: {llm_output_language}",
        "img_evaluation": """針對建築設計方案渲染圖進行深入評估。
作為資深建築設計評審委員，請針對補充條件動態調整評估準則，並提供**公正且有鑑別度的評分**。
你的任務是基於以下評估準則，**客觀評估**其建築外殼設計的優劣。
**造型與環境脈絡融合：總分10分
    評估建築造型是否能融入周圍環境脈絡，例如：自然景觀、都市紋理、地域文化。
    考量建築造型與環境的協調性、呼應性，以及對環境的尊重程度。
**場所精神與使用者關注：總分10分
    評估建築設計是否能營造獨特的場所精神，回應使用者的需求與體驗。
    考量建築空間的氛圍、舒適度、機能性，以及對使用者情感和行為的影響。
**材料及工法的環境及氣候應對程度：總分10分
    評估選用的主要建築材料（特別是木材，如果適用）和工法是否能有效應對當地環境及氣候條件。
    考量材料的永續性、環境友善性、氣候適應性，以及工法的合理性、效率性、材料損耗。
**外殼系統的維護性與耐久性：總分10分
    評估當前構造形式的系統是否考量到後續的維護與長期耐久性。
    考量當前構造系統全生命週期的循環性。    
**補充條件:{rag_msg}
            
評分標準:針對以上每個評估項目，根據方案表現給予 1.0 - 10.0 分評分 (1.0 = 極差, 10.0 = 極佳)。
輸出格式:針對每個評估項目提供評分以及簡述評分理由。最後需計算加總得分並寫為**總分數:數字**\n語言: {llm_output_language}""",
        "video_evaluation": """針對建築設計方案模型進行深入評估。
作為專業的建築師、結構技師兼數位製造專家，請針對補充條件動態調整評估準則，並提供**公正且有鑑別度的評分**。
你的任務是基於以下評估準則，**客觀評估**其建築設計的優劣。      
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
**III. 幾何造型與構成形式之技術可行性:總分10分
        **幾何複雜度:**  整體造型的幾何形式是否過於複雜，增加製造與建造難度？ (例如：曲率變化分析、造型分格複雜度評估)
        **製造技術:**  模型所展現的構成形式，在現有製造技術條件下是否能實現？ (例如：CNC 加工可行性、熱壓成型可行性、積層製造可行性評估)
        **組裝精度:**  模型所展現的幾何精度要求，在現場組裝條件下是否能達成？ (例如：構件加工精度要求、組裝誤差容許度分析)
        **經濟性:**  複雜造型的實現是否會導致過高的製造成本與工期？ (例如：材料成本分析、加工成本估算、工期評估)
**IV. 材料應用與結構邏輯之契合性:總分10分
        **材料特性發揮:**  是否充分利用主要結構材料（例如木材，如果適用）的力學性能 (例如：抗拉、抗壓、彈性模量)、紋理特性、輕質高強等優勢？ (例如：材料力學性能分析、材料選用合理性評估)
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
**補充條件:{rag_msg}
評分標準:針對以上每個評估項目，根據方案表現給予 1.0 - 10.0 分評分 (1.0 = 極差, 10.0 = 極佳)。
輸出格式:針對每個評估項目提供評分以及簡述評分理由。最後需計算加總得分並寫為**總分數:數字**\n語言: {llm_output_language}""",
    },

    "FinalEvaluationTask": {
        "summary": """請根據以下評估結果：
【評估結果】
{eval_results_formatted}
【評分結果】
{eval_counts_formatted}
【記憶內容】
短期記憶：{short_memory}
長期記憶：{long_memory}

請綜合以上資訊，請直接指出在第 {current_round} 輪的方案表現最佳，
並詳細說明該方案的優點，同時分析其他方案的優缺點。最後，請提供一個持續執行此方案的深入建議。\n語言: {llm_output_language}""",
    },
}

# --- 模擬 configuration.py 中的配置結構和讀取函數 ---
class _LocalSimplePromptConfig(BaseModel):
    template: str

class _LocalSimpleAgentConfig(BaseModel):
    prompts: Dict[str, _LocalSimplePromptConfig]

class _LocalSimpleFullConfig(BaseModel):
    agents: Dict[str, _LocalSimpleAgentConfig]

_local_base_default_prompts_obj_data = {"agents": {}}
for task_key, task_prompts in PROMPT_TEXT_TEMPLATES.items():
    _local_base_default_prompts_obj_data["agents"][task_key] = {"prompts": {}}
    for prompt_key, template_text in task_prompts.items():
        _local_base_default_prompts_obj_data["agents"][task_key]["prompts"][prompt_key] = {"template": template_text}

_local_base_default_config_obj = _LocalSimpleFullConfig(**_local_base_default_prompts_obj_data)

def get_base_default_prompt(task_key: str, prompt_key: str) -> str:
    """從本地模擬的配置對象中獲取指定的 Prompt 模板。"""
    try:
        agent_cfg = _local_base_default_config_obj.agents.get(task_key)
        if agent_cfg:
            prompt_config = agent_cfg.prompts.get(prompt_key)
            if prompt_config:
                return prompt_config.template
        print(f"警告: 在 T_config.py 中找不到預設 Prompt 模板: Task='{task_key}', Prompt='{prompt_key}'. 返回空字符串。")
        return ""
    except (KeyError, AttributeError, TypeError) as e:
        print(f"警告: 在 T_config.py 中讀取預設 Prompt 時發生錯誤 (Task='{task_key}', Prompt='{prompt_key}'): {e}. 返回空字符串。")
        return ""

class LLMConfig(BaseModel):
    """LLM 配置模型"""
    model_name: AllModelNamesLiteral = Field(default="gpt-4o-mini", description="LLM 模型名稱")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM 生成溫度 (0.0-2.0)")

    def get_llm(self) -> ChatOpenAI:
        """根據配置獲取 ChatOpenAI 實例"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")

        resolved_model_name = self.model_name
        if self.model_name == "DEFAULT(gpt-4o-mini)":
            resolved_model_name = "gpt-4o-mini"
        elif not (self.model_name.startswith("gpt-") or self.model_name.startswith("o1-")):
            print(f"警告: LLMConfig 目前主要為 OpenAI 模型優化。選擇的模型 '{self.model_name}' 可能不是 OpenAI 模型。將嘗試使用，但 ChatOpenAI 可能不支持。")
        
        return ChatOpenAI(
            model_name=resolved_model_name,
            openai_api_key=api_key,
            temperature=self.temperature
        )

class GraphOverallConfig(BaseModel):
    """圖的整體配置，包含 LLM 和直接的 Prompts"""
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    llm_output_language: SupportedLanguage = Field(
        default="繁體中文",
        title="LLM 輸出語言",
        description="指定 LLM 生成內容時應使用的主要語言。"
    )

    # 用於接收來自 Studio 的原始值。
    # Studio UI 看到的欄位名是 'run_site_analysis'。
    # 新的默認行為：如果 Studio 沒有傳遞這個值，我們假設要執行。
    run_site_analysis_raw_value: Optional[bool] = Field(
        default=True,  # <--- 修改點：默認為 True (執行)
        alias="run_site_analysis", 
        title="執行基地分析任務",
        description="如果取消勾選，將跳過基地分析任務；默認執行。" # 更新描述
    )
    
    # 最終給節點使用的、經過正確解釋的布林值
    run_site_analysis: bool = Field(
        default=True, # <--- 修改點：這個 default 與上面保持一致，但會被 validator 覆蓋
        exclude=True 
    )

    case_scenario_image_count: int = Field(
        default=4,
        title="方案情境生成圖片數量",
        description="指定生成圖片的數量 (例如，輸入 4 代表生成 4 張)。",
        ge=1, # 至少生成一張
        le=10 # 設定一個合理的上限，例如10張
    )
    max_evaluation_rounds: int = Field(
        default=3,
        title="最大評估迭代輪數",
        description="指定設計流程在結束前最多可以迭代的輪數。",
        ge=1, # 至少迭代一輪
        le=10 
    )

    @model_validator(mode='after')
    def _set_final_run_site_analysis(self) -> 'GraphOverallConfig':
        """
        根據 run_site_analysis_raw_value (來自 Studio) 來設定最終的 run_site_analysis 布林值。
        - Studio 未勾選 (傳遞 False) => 應為 False
        - Studio 勾選 (傳遞 True 或 None) => 應為 True
        - Studio 省略鍵 (導致 run_site_analysis_raw_value 為 default=True) => 應為 True
        """
        raw_value = self.run_site_analysis_raw_value

        if raw_value is False: # 只有當明確收到 False (來自未勾選的checkbox) 時，才不執行
            self.run_site_analysis = False
        else: 
            # 包括 raw_value 是 True, None (來自勾選的checkbox), 
            # 或因 default=True 而初始化的情況
            self.run_site_analysis = True
        
        # 打印以供調試
        print(f"DEBUG T_config._set_final_run_site_analysis: raw_value='{raw_value}' (type: {type(raw_value)}), resulting run_site_analysis='{self.run_site_analysis}'")
        return self

    # --- Prompts 直接定義在 GraphOverallConfig ---
    question_task_keyword_prompt_template: str = Field(
        default=get_base_default_prompt("QuestionTask", "keyword"),
        title="QuestionTask: 通用查詢生成",
        description="QuestionTask: 為 RAG 工具生成通用查詢的提示模板。包含變數: {user_input}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    question_task_summary_prompt_template: str = Field(
        default=get_base_default_prompt("QuestionTask", "summary"),
        title="QuestionTask: 總結報告生成 (整合檢索)",
        description="QuestionTask: 生成總結報告的提示模板，整合多方檢索資訊。包含變數: {user_input}, {rag_msg}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # SiteAnalysisTask Prompts
    site_analysis_extract_site_info_prompt_template: str = Field(
        default=get_base_default_prompt("SiteAnalysisTask", "extract_site_info_from_user_input"),
        title="SiteAnalysisTask: 從用戶輸入提取基地資訊",
        description="SiteAnalysisTask: 從用戶的初始設計需求中提取基地位置和經緯度的提示模板。變數: {user_design_input}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    site_analysis_img_recognition_prompt_template: str = Field(
        default=get_base_default_prompt("SiteAnalysisTask", "img_recognition"),
        title="SiteAnalysisTask: 基地圖片辨識",
        description="SiteAnalysisTask: 圖片辨識提示模板。包含變數: {region}, {geo_location}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    site_analysis_rag_keywords_prompt_template: str = Field(
        default=get_base_default_prompt("SiteAnalysisTask", "rag_keywords_generation"),
        title="SiteAnalysisTask: 基地RAG關鍵字生成",
        description="SiteAnalysisTask: 為基地背景資料檢索生成關鍵字的提示模板。包含變數: {region}, {geo_location}, {initial_img_analysis_summary}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    site_analysis_llm_prompt_template: str = Field(
        default=get_base_default_prompt("SiteAnalysisTask", "llm_analysis"),
        title="SiteAnalysisTask: 基地LLM分析 (整合RAG)",
        description="SiteAnalysisTask: LLM 分析提示模板，整合圖片辨識和RAG補充資訊。包含變數: {region}, {geo_location}, {analysis_img}, {rag_supplementary_info}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # RAGdesignThinking Prompts
    rag_design_thinking_keywords_prompt_template: str = Field(
        default=get_base_default_prompt("RAGdesignThinking", "keywords"),
        title="RAGdesignThinking: 設計方向生成",
        description="RAGdesignThinking: 生成設計方向/靈感的提示模板。包含變數: {design_goal_summary}, {analysis_result}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    rag_design_thinking_rag_prompt_template: str = Field(
        default=get_base_default_prompt("RAGdesignThinking", "rag_tool_query"),
        title="RAGdesignThinking: (已棄用) RAG檢索",
        description="此 Prompt 在 RAGdesignThinking 節點中不再用於觸發工具。",
        extra={'widget': {'type': 'textarea'}}
    )
    rag_design_thinking_complete_scheme_prompt_template: str = Field(
        default=get_base_default_prompt("RAGdesignThinking", "complete_scheme"),
        title="RAGdesignThinking: 完整方案生成 (基於設計方向)",
        description="RAGdesignThinking: 生成完整方案的提示模板。包含變數: {design_goal_summary}, {improvement}, {analysis_result}, {design_directions}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # GateCheck1 Prompt
    gate_check1_prompt_template: str = Field(
        default=get_base_default_prompt("GateCheck1", "evaluation"),
        title="GateCheck1: 初步設計評審",
        description="GateCheck1: 評審提示模板。包含變數: {design_summary}, {formatted_current}, {formatted_previous}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # OuterShellPromptTask Prompts
    outer_shell_gpt_prompt_template: str = Field(
        default=get_base_default_prompt("OuterShellPromptTask", "gpt_generation"),
        title="OuterShellPromptTask: 外殼Prompt生成",
        description="OuterShellPromptTask: 生成外殼 Prompt 的提示模板。包含變數: {advice_text}, {improvement}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    outer_shell_lora_prompt_template: str = Field(
        default=get_base_default_prompt("OuterShellPromptTask", "lora_generation"),
        title="OuterShellPromptTask: LoRA權重生成",
        description="OuterShellPromptTask: 生成 LoRA 權重的提示模板。包含變數: {final_prompt}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # GateCheck2 Prompt (for img_recognition tool)
    gate_check2_img_recognition_prompt_template: str = Field(
        default=get_base_default_prompt("GateCheck2", "img_evaluation"),
        title="GateCheck2: 生成圖像評審",
        description="GateCheck2: 圖像評審提示模板 (img_recognition 用)。包含變數: {image_list_str}, {advice_text}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # DeepEvaluationTask Prompts
    deep_eval_keyword_img_recognition_prompt_template: str = Field(
        default=get_base_default_prompt("DeepEvaluationTask", "keyword_img_recognition"),
        title="DeepEvaluationTask: 圖像關鍵詞(RAG用)",
        description="DeepEvaluationTask: 生成關鍵詞的提示模板 (img_recognition 用)。包含變數: {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    deep_eval_rag_prompt_template: str = Field(
        default=get_base_default_prompt("DeepEvaluationTask", "rag_tool_query"),
        title="DeepEvaluationTask: RAG檢索(深度評估)",
        description="DeepEvaluationTask: RAG 檢索提示模板 (用於 ARCH_rag_tool)。包含變數: {keywords}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    deep_eval_img_eval_img_recognition_prompt_template: str = Field(
        default=get_base_default_prompt("DeepEvaluationTask", "img_evaluation"),
        title="DeepEvaluationTask: 圖像深入評估",
        description="DeepEvaluationTask: 圖片評估提示模板 (img_recognition 用)。包含變數: {rag_msg}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    deep_eval_vid_eval_video_recognition_prompt_template: str = Field(
        default=get_base_default_prompt("DeepEvaluationTask", "video_evaluation"),
        title="DeepEvaluationTask: 3D模型深入評估",
        description="DeepEvaluationTask: 3D模型評估提示模板 (video_recognition 用)。包含變數: {rag_msg}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # FinalEvaluationTask Prompt
    final_evaluation_summary_prompt_template: str = Field(
        default=get_base_default_prompt("FinalEvaluationTask", "summary"),
        title="FinalEvaluationTask: 最終總結評估",
        description="FinalEvaluationTask: 總結評估提示模板。包含變數: {eval_results_formatted}, {eval_counts_formatted}, {short_memory}, {long_memory}, {current_round}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )

    # --- 新增 FutureScenarioGenerationTask 的 Prompt 字段 ---
    future_scenario_detail_generation_prompt_template: str = Field(
        default="""作為一個專業的建築視覺化AI，請根據以下原始設計描述，生成 {num_images} 張細節圖。
這些圖片應重點展示該建築方案的**立面細節、特定視角的構造工法、以及外殼表面的主要材質質感和節點做法**。
如果提供了基礎圖片，請在其上進行修改以反映這些細節。如果未提供基礎圖片，則完全基於文字描述生成。
請確保生成的細節與基礎圖片（如果提供）中的整體設計風格和形態保持一致。
風格要求：現代、寫實、注重細節，圖片應清晰，光線合理。DO NOT generate text or labels on the image.
原始設計描述：
{base_design_description}
請使用語言: {llm_output_language}""",
        title="FutureScenario: (通用)立面與構造細節生成",
        description="指導AI生成建築的立面細節和構造工法圖。可選地基於輸入圖片修改，或純文字生成。變數: {base_design_description}, {num_images}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    future_scenario_aging_generation_prompt_template: str = Field(
        default=get_base_default_prompt("FutureScenarioGenerationTask", "aging_generation"),
        title="FutureScenario: 未來老化場景生成 (批次)",
        description="指導AI一次性生成方案在10年、20年、30年後的老化外觀（共3張圖）。變數: {base_design_description}, {llm_output_language}",
        extra={'widget': {'type': 'textarea'}}
    )
    future_scenario_detail_image_count: int = Field(
        default=3,
        title="FutureScenario: 細節圖生成數量",
        description="指定 FutureScenarioGenerationTask 在 Phase 1 (立面/構造細節) 中生成的圖片數量。",
        ge=1,
        le=5 # 合理上限
    )

    # --- End of Prompts ---

    class Config:
        title = "設計T型圖配置 (扁平化)"
        description = "配置T型圖中LLM、Prompts及語言等參數。Prompt模板現在直接位於此配置下。"

# 為了方便在主腳本中直接使用預設配置
default_config_instance = GraphOverallConfig()

LLM_INSTANCE = default_config_instance.llm_config.get_llm()
PROMPTS_INSTANCE = default_config_instance # PROMPTS_INSTANCE 現在是 GraphOverallConfig 的實例
