\# LangGraph 自動化建築設計流程框架



這是一個基於 LangGraph 開發的智慧多代理 (Multi-Agent) 建築設計框架。系統模擬一個由 AI 組成的建築設計團隊，自動化地執行從概念發想、資料搜集、方案設計、多方案比較、視覺化到最終評估的完整流程。



\## 專案總覽



本框架利用大型語言模型 (LLM) 和生成式 AI 技術，將複雜的建築設計任務分解為一系列由專門化 AI 代理 (Agent) 執行的子任務。透過動態的工作流規劃與管理，系統能夠應對多樣的設計需求，並在過程中進行自我評估與修正，最終產出多樣化且高品質的設計成果。



\## 核心功能



\-   \*\*動態工作流\*\*：透過 `ProcessManagement` 代理，能根據使用者初始輸入，動態生成、規劃、並調整任務序列。

\-   \*\*可配置性\*\*：可透過 `config.json` 檔案對工作流、代理、模型、提示詞等進行詳細設定，以適應不同專案需求。

\-   \*\*多模型支援\*\*：支援 OpenAI, Google Gemini, Anthropic Claude 等多種 LLM，可為不同代理指定不同模型。

\-   \*\*工作流視覺化\*\*：能將複雜的設計決策過程與方案分支，自動生成為 Sankey (桑基) 圖，方便回顧與分析。

\-   \*\*記憶系統\*\*：具備長期記憶 (LTM)，能夠在任務執行過程中學習並檢索相關知識。

\-   \*\*自動化報告\*\*：能將最終的設計成果與評估過程，匯出為 `.docx` 格式的報告文件。



\## 系統架構



系統主要由 `langgraph.json` 中定義的多個 Graph 組成，核心是 `General\_Arch\_graph`。



1\.  \*\*`General\_Arch\_graph`\*\*:

&nbsp;   -   這是主要的、通用的建築設計工作流程。

&nbsp;   -   \*\*`ProcessManagement` 節點\*\*: 工作流的大腦，負責接收使用者需求、生成初始任務清單、處理失敗任務、以及在使用者中斷時調整流程。

&nbsp;   -   \*\*`AssignTeamsSubgraph` (子圖)\*\*: 根據任務目標，為其匹配最適合的代理，並準備執行所需的輸入。

&nbsp;   -   \*\*`EvaluationSubgraph` (子圖)\*\*: 對任務產出進行評估。根據評估類型（標準、特殊比較、最終），會觸發不同的評估流程。

&nbsp;   -   \*\*`save\_final\_summary`\*\*: 儲存最終報告。

&nbsp;   -   \*\*`QA\_Agent`\*\*: 處理使用者問答環節。



2\.  \*\*`Design\_T\_Graph`\*\*:

&nbsp;   -   一個特化的設計流程，可能代表一種特定的設計思考模型（例如 T 型思考）。它定義了一套與 `General\_Arch\_graph` 不同的節點與邏輯。



\## 設定說明



1\.  \*\*環境變數 (`.env`)\*\*

&nbsp;   在專案根目錄下建立 `.env` 檔案，並填入所需的 API Keys:

&nbsp;   ```env

&nbsp;   OPENAI\_API\_KEY="sk-..."

&nbsp;   GEMINI\_API\_KEY="..."

&nbsp;   ANTHROPIC\_API\_KEY="sk-ant-..."

&nbsp;   ```



2\.  \*\*流程設定 (`config.json`)\*\*

&nbsp;   -   首次執行後，系統會根據 `src/configuration.py` 中的預設值生成 `config.json` 檔案。

&nbsp;   -   此檔案只會儲存與預設值\*\*有差異\*\*的設定。

&nbsp;   -   您可以手動修改此檔案來客製化行為，例如：

&nbsp;       -   更換特定 Agent 使用的 LLM 模型 (`model\_name`)。

&nbsp;       -   調整模型的 `temperature` 或 `max\_tokens`。

&nbsp;       -   修改或實驗不同的 Prompt Template。

&nbsp;   -   若要還原所有設定，只需刪除 `config.json` 即可。



\## 如何執行



1\.  \*\*安裝依賴\*\*:

&nbsp;   建議建立一個虛擬環境，並安裝所需的套件。

&nbsp;   ```bash

&nbsp;   # (假設您有 requirements.txt)

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



2\.  \*\*啟動 LangGraph\*\*:

&nbsp;   使用 LangGraph CLI 來啟動服務。

&nbsp;   ```bash

&nbsp;   langgraph up

&nbsp;   ```



3\.  \*\*與 Graph 互動\*\*:

&nbsp;   -   服務啟動後，您可以透過 LangGraph 的 UI (通常在 `http://127.0.0.1:8000/graphs/`) 或 API 與其互動。

&nbsp;   -   選擇一個 Graph (例如 `General\_Arch\_graph`)，並提供您的初始設計需求（`user\_input`）來開始一個新的流程。

