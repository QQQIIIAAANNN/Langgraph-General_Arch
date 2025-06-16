 # LangGraph 智慧建築設計流程框架

這是一個基於 LangGraph 開發的智慧多代理 (Multi-Agent) 建築設計框架。系統旨在模擬一個由 AI 組成的建築設計團隊，自動化地執行從概念發想、資料搜集、方案設計、多方案比較、視覺化到最終評估的完整流程。

## 專案總覽

本框架利用大型語言模型 (LLM) 和生成式 AI 技術，將複雜的建築設計任務分解為一系列由專門化 AI 代理 (Agent) 執行的子任務。透過動態的工作流規劃與管理，系統能夠應對多樣的設計需求，並在過程中進行自我評估與修正，最終產出多樣化且高品質的設計成果。

## 核心功能

-   **動態工作流**：透過 `ProcessManagement` 代理，能根據使用者初始輸入，動態生成、規劃、並調整任務序列。
-   **多代理協作**：內建多個職能各異的 AI 代理，包括：
    -   `ArchRAGAgent`/`WebSearchAgent`: 建築法規、案例分析檢索。
    -   `ImageGenerationAgent`/`ModelRenderAgent`: 2D 概念圖、渲染圖生成。
    -   `RhinoMCPCoordinator`: 整合 Rhino 進行精準 3D 模型操作與分析。
    -   `PinterestMCPCoordinator`/`OSMMCPCoordinator`: 案例與地理圖資搜集。
    -   `EvaAgent`/`SpecialEvaAgent`/`FinalEvaAgent`: 多層次、多維度的設計方案評估。
    -   等等...
-   **可配置性**：可透過 `configuration` 對工作流、代理、模型、提示詞等進行詳細設定，以適應不同專案需求。
-   **多模型支援**：支援 OpenAI, Google Gemini, Anthropic Claude 等多種 LLM，可為不同代理指定不同模型。
-   **工作流視覺化**：能將複雜的設計決策過程與方案分支，自動生成為 Sankey (桑基) 圖，方便回顧與分析。
-   **記憶系統**：具備長期記憶 (LTM)，能夠在任務執行過程中學習並檢索相關知識。
-   **自動化報告**：能將最終的設計成果與評估過程，匯出為 `.docx` 格式的報告文件。

## 系統架構

系統主要由 `langgraph.json` 中定義的多個 Graph 組成，核心是 `General_Arch_graph`。

1.  **`General_Arch_graph`**:
    -   這是主要的、通用的建築設計工作流程。屬於層級化協作架構，代表廣域探索導向的同步化設計流程，並兼備目標導向最佳化設計探索流程。
    -   **`ProcessManagement` 節點**: 工作流的大腦，負責接收使用者需求、生成初始任務清單、處理失敗任務、以及在使用者中斷時調整流程。
    -   **`AssignTeamsSubgraph` (子圖)**: 根據任務目標，為其匹配最適合的代理，並準備執行所需的輸入。
    -   **`ToolAgent` / 各`MCPCoordinator`**: 實際執行任務的節點，調用外部工具或 API (如 Rhino, ComfyUI, Web Search 等)。
    -   **`EvaluationSubgraph` (子圖)**: 對任務產出進行評估。根據評估類型（標準、特殊比較、最終），會觸發不同的評估流程。
    -   **`save_final_summary`**: 在流程結束時，分析所有任務歷史，產生 Sankey 圖所需的數據結構，儲存最終報告。
    -   **`QA_Agent`**: 處理使用者問答環節。

2.  **`Design_T_Graph`**:
    -   一個特化的順序化協作架構，代表特定的目標導向最佳化設計探索流程。


## 如何執行

1.  **安裝依賴**:
    建議建立一個虛擬環境，並安裝所需的套件。
    ```bash
    pip install -r requirements.txt
    ```

2.  **啟動 LangGraph**:
    啟動虛擬環境，並使用 LangGraph CLI 來啟動服務。
    ```bash
    langgraph dev
    ```

3.  **與 Graph 互動**:
    -   服務啟動後，您可以透過 LangGraph 的 UI (通常在 `http://127.0.0.1:8000/graphs/`) 或 API 與其互動。
    -   選擇一個 Graph (例如 `General_Arch_graph`)，並提供您的初始設計需求（`user_input`）來開始一個新的流程。

4.  **其他安裝**:
    -   如果要完整啟用多代理的服務，建議還需額外安裝ComfyUI、Rhino mcp、Pinterest mcp、OSM mcp等工具。
    -   MCP等工具包已經在src/mcp內。安裝與啟動建議按照相關教學進行。
    -   ComfyUI: https://github.com/comfyanonymous/ComfyUI
    -   Rhino mcp: https://github.com/SerjoschDuering/rhino-mcp
    -   OSM mcp: https://github.com/jagan-shanmugam/open-streetmap-mcp
    -   Pinterest mcp: https://github.com/terryso/mcp-pinterest  

## 設定說明

1.  **環境變數 (`.env`)**
    請在專案根目錄下的`.env` 檔案，填入所需的 API Keys:
    ```env
    OPENAI_API_KEY="sk-..."
    GEMINI_API_KEY="..."
    ANTHROPIC_API_KEY="sk-ant-..."
    ```

2.  **流程設定 (`config.json`)**
    -   首次執行後，系統會根據 `src/configuration.py` 中的預設值生成 `config.json` 檔案。
    -   此檔案只會儲存與預設值**有差異**的設定。
    -   您可以手動修改此檔案或是直接在Langgraph Studio中調整configuration來客製化設定，例如：
        -   更換特定 Agent 使用的 LLM 模型 (`model_name`)。
        -   調整模型的 `temperature` 或 `max_tokens`。
        -   修改或實驗不同的 Prompt Template。
    -   若要還原所有設定，只需刪除 `config.json` 即可。
