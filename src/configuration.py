import os
import json
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# =============================================================================
# 核心配置模型
# =============================================================================

class ModelConfig(BaseModel):
    """Configuration for language models."""
    provider: Literal["openai", "google", "anthropic"] = Field("openai", description="Provider of the model (e.g., openai, google)")
    model_name: str = Field("gpt-4o-mini", description="""
        The name of the model to use.
        Available Options:
        OpenAI: 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1-mini', 'o3-mini'.
        Google: 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest', 'gemini-1.0-pro', 'Gemini 2.0 Flash', 'Gemini 2.5 Pro Experimental'.
        Anthropic: 'claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'.
        (Note: Pricing is indicative and may change. Ensure the selected model is available in your region and API plan.)
        """)
    temperature: float = Field(0.7, description="Temperature setting for the model (0.0-1.0)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for the response (None for default)")

class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding models."""
    provider: Literal["openai"] = Field("openai", description="Provider of the embedding model (currently only OpenAI supported)")
    model_name: str = Field("text-embedding-3-small", description="""
        The name of the embedding model to use for LTM.
        Available OpenAI Options: 'text-embedding-3-small' ($0.02/M Tok), 'text-embedding-3-large' ($0.13/M Tok).
        (Note: Pricing is indicative and may change.)
        """)
    # Note: Add parameters like dimensions if needed for specific models

class MemoryConfig(BaseModel):
    """Configuration for memory systems (LTM and potentially STM)."""
    long_term_memory: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig, description="Configuration for the Long-Term Memory embedding model.")
    # Parameters for potential future explicit chunking implementation
    # chunk_size: int = Field(1000, description="Target size for text chunks (currently not used by LTM saving).")
    # chunk_overlap: int = Field(200, description="Overlap between text chunks (currently not used by LTM saving).")
    retriever_k: int = Field(5, description="Number of relevant documents to retrieve from LTM.")

class ToolConfig(BaseModel):
    """Configuration for individual tools."""
    enabled: bool = Field(True, description="Whether the tool is enabled")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Default parameters specific to the tool (can be overridden by task inputs)")

class PromptConfig(BaseModel):
    """Configuration for prompts."""
    template: str = Field(..., description="The prompt template string")
    input_variables: List[str] = Field(default_factory=list, description="Variables expected by the prompt template")

class AgentConfig(BaseModel):
    """Configuration for an individual agent."""
    agent_name: str = Field(..., description="Unique name of the agent")
    description: str = Field("", description="Description of the agent's role")
    llm: ModelConfig = Field(default_factory=ModelConfig, description="LLM configuration for this agent")
    prompts: Dict[str, PromptConfig] = Field(default_factory=dict, description="Prompt templates used by the agent (key is prompt name)")
    tools: Dict[str, ToolConfig] = Field(default_factory=dict, description="Tool configurations available to the agent (key is tool name)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Other specific parameters for the agent (e.g., max_retries)")

class WorkflowConfig(BaseModel):
    """Global configuration for the workflow."""
    name: str = Field("Architecture Design Workflow", description="Name of the workflow")
    description: str = Field("An AI-powered workflow for architectural design", description="Description of the workflow")
    output_directory: str = Field("./output/Cache", description="Directory for output files") # Adjusted default
    debug_mode: bool = Field(False, description="Enable debug mode with extra logging")
    # max_iterations removed, graph logic controls flow
    llm_output_language: str = Field("繁體中文", description="Default output language for LLMs (e.g., '繁體中文', 'English')")
    interrupt_before_pm: bool = Field(True, description="Enable interrupt before ProcessManagement node runs.") # Defaulting to True

class FullConfig(BaseModel):
    """Root model for the entire configuration."""
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agents: Dict[str, AgentConfig] = Field(default_factory=dict, description="Dictionary of all agent configurations, keyed by agent name")


# =============================================================================
# Configuration Manager (使用上一個回應中修改後的版本，包含 _deep_update/diff)
# =============================================================================
class ConfigManager:
    """Manages loading, saving, and accessing workflow and agent configurations."""

    def __init__(self, config_file: str = "config.json"):
        """Initialize the config manager."""
        self.config_file = config_file
        self.default_config = self._create_default_config() # 先載入預設配置
        self.config = self._load_config() # 再載入 config.json 並覆蓋預設值

    def _load_config(self) -> FullConfig:
        """Load configuration from file, merging with defaults."""
        config_from_file = None # 初始化為 None

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    # Handle empty file case
                    content = f.read()
                    if not content:
                        print(f"Config file '{self.config_file}' is empty. Using defaults.")
                        return self.default_config
                    config_dict = json.loads(content)
                    config_from_file = FullConfig(**config_dict) # 從檔案載入配置
            except json.JSONDecodeError as json_e:
                 print(f"Error decoding JSON from config file '{self.config_file}': {json_e}. Using defaults.")
                 return self.default_config
            except Exception as e:
                print(f"Error loading config file '{self.config_file}': {e}. Using defaults.")
                return self.default_config # 載入失敗時返回預設配置

        if config_from_file:
            # 將檔案配置合併到預設配置，config_from_file 優先
            merged_config_dict = self.default_config.model_dump() # 取得預設配置的字典
            # 使用 model_dump(exclude_unset=True) 確保只合併檔案中明確設定的值
            file_config_dict = config_from_file.model_dump(exclude_unset=True)
            self._deep_update_dict(merged_config_dict, file_config_dict) # 深度合併
            return FullConfig(**merged_config_dict) # 使用合併後的字典創建 FullConfig
        else:
            print(f"Config file '{self.config_file}' not found. Using defaults.")
            return self.default_config # 如果 config.json 不存在，直接返回預設配置

    def _create_default_config(self) -> FullConfig:
        """Create default configuration using Pydantic model defaults."""
        # --- Define default values directly here for clarity ---
        default_workflow = WorkflowConfig(
            name="Architecture Design Workflow",
            description="An AI-powered workflow for architectural design",
            output_directory="./output/Cache",
            debug_mode=False,
            llm_output_language="繁體中文",
            interrupt_before_pm=True
        )
        default_memory = MemoryConfig(
            long_term_memory=EmbeddingModelConfig(provider="openai", model_name="text-embedding-3-small"),
            # chunk_size=1000, # Removed as it's not used currently
            # chunk_overlap=200, # Removed as it's not used currently
            retriever_k=5
        )

        # Default LLM Config (can be reused)
        default_llm = ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7)

        # Specific LLM Configs
        gemini_flash_llm = ModelConfig(provider="google", model_name="Gemini 2.0 Flash", temperature=0.7) # <<< Define Gemini 2.0 Flash config
        low_temp_gemini_flash_llm = ModelConfig(provider="google", model_name="Gemini 2.0 Flash", temperature=0.2) # <<< Define low-temp Gemini 2.0 Flash for EvaAgent

        default_agents = {
                "process_management": AgentConfig(
                    agent_name="process_management",
                    description="Responsible for planning and managing the task workflow.",
                    llm=ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7),
                    prompts={
                        "create_workflow": PromptConfig(
                            template="""
You are a meticulous and **detail-oriented** workflow planner specializing in architecture and design tasks. Your goal is to break down a user's request into a **sequence of granular, executable task objectives** for a team of specialized agents, ensuring smooth data flow between steps.

**User Request** (**Strictly Follow**): {user_input}

Analyze the request and generate a **complete, logical, and DETAILED sequence of tasks** to fulfill it. Pay close attention to dependencies and insert necessary intermediate processing steps.

**Key Planning Principles:**
1.  **Dependencies:** If Task B requires data generated by Task A, ensure Task A comes first.
2.  **Intermediate Processing:** If raw output of Task A isn't directly usable by Task B, **INSERT an `LLMTaskAgent` task** between them (e.g., summarize, reformat, generate prompts).
3.  **Evaluation Task Assignment (CRITICAL):**
    *   **Standard Evaluation (Pass/Fail):** Assign to `EvaAgent`. Set a descriptive `task_objective` (e.g., "Evaluate Task 3 output based on standard criteria").
    *   **Special Evaluation (Multi-Option Comparison/Score):** Assign to `SpecialEvaAgent`. Set `task_objective="special_evaluation"`.
    *   **Final Evaluation (Holistic Score):** Assign to `FinalEvaAgent`. Set `task_objective="final_evaluation"`.
4.  **Clarity:** Each task objective must be specific and actionable.
5.  **`requires_evaluation` for ALL Evaluation Agents:** **CRITICAL:** If `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, OR `FinalEvaAgent`, you **MUST** set `requires_evaluation` to `true`.

For **each** task in the sequence, you MUST specify:
1.  `description`: High-level goal for this step (in {llm_output_language}).
2.  `task_objective`: Specific outcome and method needed.
    *   Use `"final_evaluation"` or `"special_evaluation"` for the respective agent modes.
    *   Otherwise, describe the specific goal (e.g., "Evaluate task N based on standard criteria").
    *   **For `ModelRenderAgent` (Handles existing images):**
        *   If the goal is photorealistic rendering of an existing model view/image (e.g., from Rhino): Set objective like "Photorealistic rendering of the provided image(s) of the architectural scheme: [scheme details]."
        *   If the goal is future scenario simulation based on an existing image: Set objective like "Simulate future scenario for the provided image(s). Context: [user goals, site conditions, architectural scheme details for simulation]."
    *   **For `ImageGenerationAgent` (Generates new images from text/concepts):**
        *   If generating a new visual concept from a description: Set objective like "Generate an image representing [concept description, style, mood]."
        *   If exploring multiple *distinct visual options* for a design element (e.g., different facade styles): **Create SEPARATE `ImageGenerationAgent` tasks for EACH distinct option.** Each task objective should clearly define ONE option. Example Task 1: "Generate image for facade option A: modern style with glass and steel." Example Task 2: "Generate image for facade option B: classical style with stone and arches."
        *   If generating *multiple variations of the SAME concept/option*: Use a single `ImageGenerationAgent` task with a clear objective for that one concept, and its `i` parameter will be set later by the `prepare_tool_inputs_node` if needed (though typically `i` is for minor variations, not distinct design schemes).
    *   **For `RhinoMCPCoordinator`:**
        *   If the task involves geometric modeling, modification, or precise analysis: Define the specific Rhino operations needed.
        *   If the task involves **functional layout, programmatic blocking, or quantitative/qualitative spatial arrangement**: Clearly state this objective. For example: "Develop a functional block layout for the ground floor based on [program requirements], and provide a top-down parallel projection screenshot." or "Perform a solar access analysis on the south facade and output results."
3.  `inputs`: JSON object suggesting initial data needs (e.g., `{{"prompt": "..."}}`). Use placeholders like `{{output_from_task_id_xyz.key}}`. Indicate file needs clearly. **Do not invent paths.** Use `{{"}}` if no input suggestion applies. NEVER use `null`.
4.  `requires_evaluation`: Boolean (`true`/`false`). **MUST be `true` if `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent`.**
5.  `selected_agent`: **(CRITICAL)** The **exact name** of the agent from the list below. Mandatory for every task. **Use `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent` for evaluation tasks.**

--- BEGIN AGENT CAPABILITIES ---
**Available Agent Capabilities & *Primary* Expected Prepared Input Keys:**
*   `ArchRAGAgent`: Information Retrieval (Regulations, Expertise) -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`. (Also used internally by ModelRenderAgent).
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: **Generates NEW images from text descriptions, or edits existing images.** Ideal for visual exploration, concept art, mood boards, or creating textures/details. -> Needs `prompt`, optional `image_inputs` (list of paths for editing), optional `i` (count for variations of the *same* prompt).
*   `WebSearchAgent`: Web Search (Cases, General Info) -> Needs `prompt`.
*   `ModelRenderAgent`: **Processes EXISTING images (e.g., from Rhino, or previous generations) for photorealistic rendering or future scenario simulation.** -> Needs `outer_prompt` (initial context for the task type - photorealism or future simulation), `image_inputs` (list of paths to *existing* images), `is_future_scenario` (boolean). Internally uses `ImageRecognitionAgent` to generate a final English ComfyUI prompt.
*   `Generate3DAgent`: 3D MESH Model (.glb) Generation (Comfy3D) from an *existing* image -> Needs `image_path` (string).
*   `RhinoMCPCoordinator`: **Parametric/Precise 3D Modeling, Functional Layout, and Quantitative/Qualitative Analysis (Rhino).** Ideal for multi-step Rhino operations, tasks requiring precise coordinates/dimensions, **functional blocking and arrangement, programmatic analysis. If task involves visual output of layouts or plans, **request a top-down parallel projection screenshot.** -> Needs `user_request` (string command), optional `initial_image_path` (string path).
*   `PinterestMCPCoordinator`: **Pinterest Image Search & Download.** -> Needs `keyword` (string), optional `limit` (int).
*   `OSMMCPCoordinator`: **Map Screenshot Generation (OpenStreetMap).** -> Needs `user_request` (string: address or "lat,lon").
*   `LLMTaskAgent`: **General Text Tasks.** -> Needs `prompt`.
*   `EvaAgent`: **Standard Evaluation (Pass/Fail).** Requires `requires_evaluation: true`.
*   `SpecialEvaAgent`: **Special Evaluation (Multi-Option Comparison/Score).** Requires `requires_evaluation: true`.
*   `FinalEvaAgent`: **Final Evaluation (Holistic Score).** Requires `requires_evaluation: true`.
--- END AGENT CAPABILITIES ---

Return the entire workflow as a **single, valid JSON list** object. Do NOT include any explanatory text before or after the JSON list. Ensure perfect JSON syntax, detailed steps, and that **every task includes the `selected_agent` and correct `requires_evaluation` key.**
Respond in {llm_output_language}.
""",
                        input_variables=["user_input", "llm_output_language"]
                    ),
                    "failure_analysis": PromptConfig(
                        template="""
Context: A task in an automated workflow has FAILED. Analyze the provided context to determine the failure type and suggest the best course of action aimed at RESOLVING the failure.
**Failed Task Details:**
Agent Originally Assigned: {selected_agent_name}
Task Description: {task_description}
Original Task Objective: {task_objective}
Original Task Inputs (Suggestion): {inputs_json}

Last Execution Error Log: {execution_error_log}
Last Feedback Log (Includes Eval Results like 'Assessment: Fail' or 'Assessment: Score (X/10)' if applicable): {feedback_log}

--- BEGIN AVAILABLE AGENT CAPABILITIES (for Fallback Task Generation) ---
*   `ArchRAGAgent`: Information Retrieval -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`.
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: Generates NEW images from text descriptions, or edits existing images. -> Needs `prompt`, optional `image_inputs` (list of paths).
*   `WebSearchAgent`: Web Search -> Needs `prompt`.
*   `ModelRenderAgent`: Processes EXISTING images for photorealistic rendering or future scenario simulation. -> Needs `outer_prompt`, `image_inputs` (list of paths), `is_future_scenario` (boolean).
*   `Generate3DAgent`: 3D MESH Model Generation (Comfy3D) from an existing image. -> Needs `image_path` (string).
*   `RhinoMCPCoordinator`: Parametric/Precise 3D Modeling, Functional Layout, Analysis (Rhino). For functional layouts, request top-down parallel view screenshots. -> Needs `user_request`, optional `initial_image_path`.
*   `PinterestMCPCoordinator`: Pinterest Image Search & Download -> Needs `keyword`, optional `limit`.
*   `OSMMCPCoordinator`: Map Screenshot Generation (OpenStreetMap) -> Needs `user_request` (address).
*   `LLMTaskAgent`: General Text Tasks -> Needs `prompt`.
*   `EvaAgent`: Standard Pass/Fail Evaluation. Requires `requires_evaluation: true`.
*   `SpecialEvaAgent`: Multi-Option Comparison/Score. Requires `requires_evaluation: true`.
*   `FinalEvaAgent`: Final Holistic Score. Requires `requires_evaluation: true`.
--- END AVAILABLE AGENT CAPABILITIES ---
**Analysis Steps & Action Selection:**

1.  **Identify Failure Type:** Determine if 'evaluation' (check `feedback_log` for "Assessment: Fail" - Note: `SpecialEvaAgent`/`FinalEvaAgent` scores don't represent failure this way) or 'execution'.
2.  **Determine Action:** Based on failure type, logs, max retries, and original inputs/objective, choose ONE action.

**Action Options:**

*   **`FALLBACK_GENERAL`:** Propose a new task/sequence to fix the issue or try an alternative approach **aimed at achieving the original objective `{task_objective}`**.
    *   **If Failure Type is 'evaluation' (Only for `EvaAgent` - Standard Pass/Fail)**: **MUST** return `new_tasks_list` with **TWO** tasks:
        1.  **Task 1 (Alternative Generation/Execution):** Analyze `feedback_log` and `original Task Objective`. Choose the **most appropriate agent** (NOT evaluation agent) from the list to achieve the objective based on feedback. Define a **new `task_objective`**. Suggest inputs `{{"}}`. Set `requires_evaluation=false`.
        2.  **Task 2 (Re-Evaluation):** Create an `EvaAgent` task. Set `selected_agent="EvaAgent"`, `task_objective="Evaluate outcome of the alternative task"`, `inputs={{}}`, `requires_evaluation=true`.
    *   **If Failure Type is 'execution'**: Generate **ONLY the necessary task(s)** to overcome the error and achieve the original objective. Return this as `new_task` (for a single task) or `new_tasks_list` (for a short sequence if needed).
        *   Analyze `execution_error_log` and `original Task Objective`.
        *   Choose the **most appropriate agent** from the list. Consider retrying `{selected_agent_name}` with corrected inputs if the error suggests it, or select an alternative if `{selected_agent_name}` seems unsuitable.
        *   Define objective(s) for the fallback task(s).
        *   Suggest inputs `{{"}}`.
        *   Keep original `requires_evaluation` (`{original_requires_evaluation}`) **ONLY IF** the fallback task still logically needs it. **DO NOT automatically add a separate evaluation task.**
    *   Output JSON: `{{"action": "FALLBACK_GENERAL", ...}}`

*   **`MODIFY`:** Modify the *current* failed task and retry it with `{selected_agent_name}`. Appropriate for minor input errors.
    *   Output JSON: `{{"action": "MODIFY", "modify_description": "...", "modify_objective": "..."}}`

*   **`SKIP`:** Skip the failed task.
    *   Output JSON: `{{"action": "SKIP"}}`

**Instructions:**
*   If **Is Max Retries Reached?** is `True`, **MUST** choose `SKIP`.
*   Choose **only one action**. Provide **only** the single JSON output.
*   Ensure agent selections/objectives address the failure and prioritize the original objective (`{task_objective}`).
*   **REMEMBER:** Only 'evaluation' failures require the fixed two-task structure. 'Execution' failures require only the task(s) needed to fix the error.
*   Use language: {llm_output_language}
""",
                        input_variables=[
                            "failure_context", "is_max_retries", "max_retries",
                            "selected_agent_name", "task_description", "task_objective",
                            "inputs_json",
                            "execution_error_log",
                            "feedback_log",
                            "llm_output_language",
                            "original_requires_evaluation"
                        ]
                    ),
                    "process_interrupt": PromptConfig(
                        template="""
You are a meticulous workflow manager reacting to a user interrupt during task execution.
**User Interrupt Request** (**Strictly Follow**): {interrupt_input}
**Current Task Index (Point of Interruption):** {current_task_index}
**Full Current Task Sequence (JSON):**
```json
{tasks_json}
```
**Task About to Execute (at Current Index):**
```json
{current_task_json}
```

--- BEGIN AGENT CAPABILITIES (For planning new tasks if needed) ---
*   `ArchRAGAgent`: Information Retrieval -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`.
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: Generates NEW images from text descriptions, or edits existing images. -> Needs `prompt`, optional `image_inputs` (list of paths).
*   `WebSearchAgent`: Web Search -> Needs `prompt`.
*   `ModelRenderAgent`: Processes EXISTING images for photorealistic rendering or future scenario simulation. -> Needs `outer_prompt`, `image_inputs` (list of paths), `is_future_scenario` (boolean).
*   `Generate3DAgent`: 3D MESH Model Generation (Comfy3D) from an existing image. -> Needs `image_path`.
*   `RhinoMCPCoordinator`: Parametric/Precise 3D Modeling, Functional Layout, Analysis (Rhino). For functional layouts, request top-down parallel view screenshots. -> Needs `user_request`, optional `initial_image_path`.
*   `PinterestMCPCoordinator`: Pinterest Image Search & Download -> Needs `keyword`, optional `limit`.
*   `OSMMCPCoordinator`: Map Screenshot Generation (OpenStreetMap) -> Needs `user_request` (address).
*   `LLMTaskAgent`: General Text Tasks -> Needs `prompt`.
*   `EvaAgent`: Evaluation Agent (standard pass/fail, special comparison scoring, final holistic scoring).
*   `final_evaluation`: Objective for final holistic review.
*   `special_evaluation`: Objective for multi-option comparison.
--- END AGENT CAPABILITIES ---
**Your Task:** Choose ONE action based on the interrupt:

1.  **`PROCEED`**: Interrupt doesn't require plan changes. Continue from `current_task_index`. Output: `{{"action": "PROCEED"}}`
2.  **`INSERT_TASKS`**: Insert new tasks **after** `current_task_index`. Generate a list of new tasks (TaskState structure: `description`, `task_objective`, `selected_agent`, suggested `inputs`, `requires_evaluation`). Use Agent Capabilities. Output: `{{"action": "INSERT_TASKS", "insert_tasks_list": [ {{...task1...}}, ... ] }}`
3.  **`REPLACE_TASKS`**: Redesign workflow from `current_task_index` onwards (preserves completed tasks before index). Generate a **new sequence** for remaining tasks. Output: `{{"action": "REPLACE_TASKS", "new_tasks_list": [ {{...task1...}}, ... ] }}`
4.  **`CONVERSATION`**: Interrupt is unclear, ambiguous, requires discussion, or explicitly asks to discuss. Routes to QA Agent. Output: `{{"action": "CONVERSATION"}}`

**Instructions:**
*   Analyze interrupt: Is it a clear plan change (`INSERT_TASKS`/`REPLACE_TASKS`), minor clarification (`PROCEED`), or needs discussion (`CONVERSATION`)?
*   Ensure generated tasks follow TaskState structure (including appropriate `task_objective` like `"special_evaluation"` if needed).
*   Return ONLY the single JSON object for your chosen action. No explanations outside JSON.
Respond in {llm_output_language}.
""",
                        input_variables=[
                            "user_input", "interrupt_input", "current_task_index",
                            "tasks_json", "current_task_json", "llm_output_language"
                        ]
                    )
                },
                parameters={
                    "max_tasks": 20,
                    "max_retries": 3
                }
            ),
            "assign_agent": AgentConfig(
                agent_name="assign_agent",
                description="Prepares precise inputs for specialized agents based on task objectives and context.",
                llm=gemini_flash_llm,
                prompts={
                    "prepare_tool_inputs_prompt": PromptConfig(
                        template="""
You are an expert input preprocessor for specialized AI tools. Your goal is to take a high-level task objective, the overall user request, task history/outputs, and information about the selected tool, then generate the precise JSON input dictionary containing ONLY the keys REQUIRED by that specific tool, using standardized keys.

**Selected Tool/Agent:** `{selected_agent_name}`
**Tool Description:** {agent_description}
**Current Task Description:** {task_description}
**Current Task Objective (CRITICAL - READ CAREFULLY):** {task_objective}
**Overall Workflow Goal (User Request, please prioritize this):** {user_input}

**Workflow History Summary (Context):**
{aggregated_summary}

**Aggregated Outputs from ALL Previously COMPLETED Tasks (JSON String):**
```json
{aggregated_outputs_json}
```
**Aggregated Files from ALL Previously COMPLETED Tasks (JSON String, base64 data removed, includes 'source_task_id'):**
```json
{aggregated_files_json}
```
**Context from Previous Attempt (if applicable):** {error_feedback}

**Standardized Input Keys Reference & Tool Requirements:**
*   `prompt`: (String) REQUIRED for: `ArchRAGAgent`, `WebSearchAgent`, `ImageRecognitionAgent`, `VideoRecognitionAgent`, `ImageGenerationAgent`, `LLMTaskAgent`.
*   `image_paths`: (List[String]) REQUIRED for `ImageRecognitionAgent`. Find VALID paths in `aggregated_files_json`.
*   `video_paths`: (List[String]) REQUIRED for `VideoRecognitionAgent`. Find VALID paths in `aggregated_files_json`.
*   `image_path`: (String) REQUIRED for `Generate3DAgent`. Find VALID path in `aggregated_files_json`.
*   **For `ModelRenderAgent` (Processes EXISTING images):**
    *   `outer_prompt`: (String) REQUIRED. Initial contextual prompt (generated based on `is_future_scenario`).
    *   `image_inputs`: (List[String]) REQUIRED. List of FULL FILE PATHS of *existing* images to process (from `aggregated_files_json`).
    *   `is_future_scenario`: (Boolean) REQUIRED. `true` if simulating future, `false` if photorealistic rendering.
*   **For `ImageGenerationAgent` (Generates NEW images):**
    *   `prompt`: (String) REQUIRED. Detailed textual description for generating a NEW image or for editing.
    *   `image_inputs`: (List[String], Optional) For image editing tasks. List of FULL FILE PATHS from `aggregated_files_json`.
    *   `i`: (Integer, Default: 1) Optional. Number of variations to generate for the *same* `prompt`.
*   **For `RhinoMCPCoordinator`:**
    *   `user_request`: (String) REQUIRED. Detailed command for Rhino. If the task objective involves functional layout, programmatic blocking, or visualization of plans, ensure the request asks Rhino to "capture a top-down parallel projection view" or "set view to Top and capture parallel projection" as part of its final steps if a visual is needed.
    *   `initial_image_path`: (String, Optional) For `RhinoMCPCoordinator`. Find VALID path if needed.
*   `keyword`: (String) REQUIRED for `PinterestMCPCoordinator`.
*   `limit`: (Integer, Optional, Default: 10) For `PinterestMCPCoordinator`.
*   `user_request` (for `OSMMCPCoordinator`): (String) REQUIRED. Address or "lat,lon".
**Note:** Evaluation Agents (`EvaAgent`, `SpecialEvaAgent`, `FinalEvaAgent`) do not use this node.

**Instructions:**
1.  Analyze the **Current Task Objective/Description**, **Selected Tool/Agent (`{selected_agent_name}`)**, and **Overall Goal**.
2.  **Generate/Extract Required Textual Inputs & Flags**:
    *   **If `{selected_agent_name}` is `ModelRenderAgent`:**
        *   Carefully examine the **`Current Task Objective`** (case-insensitive search for keywords).
        *   Set `is_future_scenario` to `true` if the objective contains terms like "future", "scenario", "simulate", "simulation", "predict", "forecast". Otherwise, set to `false` (for photorealistic rendering).
        *   If `is_future_scenario` is `true`: Generate an `outer_prompt` that synthesizes user goals, site conditions, and architectural scheme details relevant to simulating a future scenario on an *existing* image. Example: "Simulate the building's appearance in 20 years, considering [user goal], on a [site condition], with [scheme details]. The focus is on [specific aspect like material weathering]."
        *   If `is_future_scenario` is `false`: Generate an `outer_prompt` that describes the architectural scheme and requests analysis of the *existing* image's perspective for photorealistic rendering. Example: "Render the architectural design: [key scheme elements]. Analyze the image perspective (e.g., eye-level, aerial) to ensure a high-quality photorealistic output of this existing view."
    *   **If `{selected_agent_name}` is `ImageGenerationAgent`:**
        *   The `Current Task Objective` should clearly define the **single, specific visual concept or design option** to be generated from scratch or edited.
        *   Generate a `prompt` that is a **highly detailed and specific textual description** for this single concept. Incorporate elements from the `Overall Workflow Goal` and `Workflow History Summary`. Describe the desired scene, style, composition, lighting, materials, mood, and specific architectural elements for this *one* option.
        *   If the `Current Task Objective` seems to ambiguously describe multiple distinct options, focus on the first or most prominent one for the `prompt`, as this agent task is intended for one detailed generation/edit at a time. (The Process Management Agent should have created separate tasks for distinct options.)
    *   **If `{selected_agent_name}` is `RhinoMCPCoordinator`:**
        *   Synthesize a clear `user_request` for Rhino.
        *   **Crucially, if the `Current Task Objective` implies creating a plan, layout, or requires a top-down view for analysis (e.g., "functional blocking", "layout design", "site plan generation"), append a clear instruction to the `user_request` for Rhino to "capture a top-down parallel projection screenshot of the result" or similar phrasing to ensure the correct view is generated by Rhino's `capture_viewport` tool.**
    *   For other agents, generate/extract text (`prompt`, `user_request`, `keyword`) as usual.
3.  **Determine and Validate Required File Paths/Filenames**:
    *   If `{selected_agent_name}` REQUIRES file inputs (`image_paths`, `video_paths`, `image_path`, `image_inputs` for `ModelRenderAgent` or `ImageGenerationAgent` if editing):
        *   Carefully parse `aggregated_files_json`.
        *   For `image_inputs` (`ModelRenderAgent`): This MUST be a LIST of full file paths to *existing* images.
        *   For `image_inputs` (`ImageGenerationAgent`, optional for editing): This MUST be a LIST of full file paths.
        *   Verify paths are valid. If required files are missing/invalid, RETURN ERROR JSON.
4.  **Handle Specific Parameters**: If `{selected_agent_name}` is `ImageGenerationAgent`, check if `i` (image count for variations of the same prompt) is suggested by `Current Task Objective` or `initial_plan_suggestion_json` and include it if valid.
5.  **Handle Optional Inputs**: As per standard logic.
6.  **Construct Final JSON**: Create the JSON dictionary containing ONLY the keys explicitly REQUIRED or validly determined OPTIONAL keys for **`{selected_agent_name}`**.
7.  **Error Handling**: If REQUIRED inputs are missing/invalid (e.g., `image_inputs` is empty for `ModelRenderAgent`), return error JSON.

**Output:** Return ONLY the final JSON input dictionary for `{selected_agent_name}`, or the error JSON. No other text.
Language: {llm_output_language}
""",
                        input_variables=[
                            "selected_agent_name", "agent_description", "user_input",
                            "task_objective", "task_description",
                            "aggregated_summary",
                            "aggregated_outputs_json",
                            "aggregated_files_json",
                            "error_feedback", "llm_output_language"
                        ]
                    )
                },
                parameters={
                     "specialized_agents_description": {
                        "ArchRAGAgent": "Retrieves information from the architectural knowledge base based on a query.",
                        "ImageRecognitionAgent": "Analyzes and describes the content of one or more images based on a prompt. Also used internally by ModelRenderAgent to generate final English ComfyUI prompts.",
                        "VideoRecognitionAgent": "Analyzes the content of one or more videos or 3D models based on a prompt.",
                        "ImageGenerationAgent": "Generates NEW images from detailed text descriptions, or edits existing images. Ideal for visual concept generation, style exploration, and creating specific artistic renditions from scratch. Takes a `prompt` for the core description, optionally `image_inputs` (list of paths) for editing context, and optionally `i` for generating multiple variations of the *same* described concept.",
                        "WebSearchAgent": "Performs a web search to find grounded information, potentially including relevant images and sources.",
                        "ModelRenderAgent": "Handles advanced image processing: either photorealistic rendering of existing model views OR simulation of future scenarios on images, using ComfyUI. Requires: `outer_prompt` (initial context for the task type - photorealism or future simulation), `image_inputs` (list of image paths), and `is_future_scenario` (boolean flag: true for future simulation, false for photorealistic rendering). Internally uses ImageRecognitionAgent to refine the `outer_prompt` into a final English ComfyUI prompt for each image.",
                        "Generate3DAgent": "Generates a 3D model and preview video from a single input image using ComfyUI diffusion.",
                        "RhinoMCPCoordinator": "Coordinates complex tasks within Rhino 3D, including parametric modeling, precise geometric operations, **functional layout design, programmatic blocking, and quantitative/qualitative spatial analysis.** Ideal for multi-step Rhino operations. **If the task involves visualizing plans or layouts, instruct it to capture top-down parallel projection screenshots.**",
                        "PinterestMCPCoordinator": "Searches for images on Pinterest based on a keyword and downloads them.",
                        "OSMMCPCoordinator": "Generates a map screenshot for a given address using OpenStreetMap and geocoding.",
                        "LLMTaskAgent": "Handles general text-based tasks like analysis, summarization, reformatting, complex reasoning, or generating prompts.",
                        "EvaAgent": "Performs standard (Pass/Fail) evaluation of a preceding task's output.",
                        "SpecialEvaAgent": "Performs special evaluation comparing multiple options/artifacts, providing a score and selection.",
                        "FinalEvaAgent": "Performs a final, holistic evaluation of the entire workflow, providing a score.",
                     },
                     "max_retries": 3
                }
            ),
            "tool_agent": AgentConfig(
                agent_name="tool_agent",
                description="Executes the specific tool chosen for a task.",
                llm=ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7), # Keep Tool Agent error handling as OpenAI unless specified
                prompts={
                    "error_handling": PromptConfig(
                        template="""
An error occurred while executing the task '{task_type}':
{error}

Please analyze the error and provide:
1. A brief explanation of what might have gone wrong.
2. Suggestions on how the input prompt or task parameters could be fixed for a retry.
Respond concisely in {llm_output_language}.
""",
                        input_variables=["task_type", "error", "llm_output_language"]
                    )
                },
                tools={},
                parameters={}
            ),
            "eva_agent": AgentConfig(
                agent_name="eva_agent",
                description="Evaluates the output of tasks based on criteria (Standard Pass/Fail, Special Comparison/Score, Final Holistic Score).",
                llm=low_temp_gemini_flash_llm, # <<< Update Evaluation Agent LLM
                prompts={
                    "prepare_evaluation_inputs": PromptConfig(
                        template="""
You are preparing inputs for a **standard Pass/Fail evaluation task** (`EvaAgent`).

**Current Task Objective (Standard Eval):** {current_task_objective}
**Previously Executed Task - Description:** {task_description}
**Previously Executed Task - Objective:** {task_objective}

**IMPORTANT: The following JSON strings contain ALL the outputs and files produced by the PREVIOUS task. Base your output ONLY on parsing these strings.**

**Previous Task's Outputs (JSON String):**
```json
{evaluated_task_outputs_json}
```
**Previous Task's Output Files (JSON String):**
```json
{evaluated_task_output_files_json}
```

**Your Goal:**
1. Parse the provided outputs/files JSON from the PREVIOUS task.
2. Format a structured JSON object for standard evaluation tools (LLM, Image/Video Rec).
3. **Determine if gathering external sources (RAG/Search) is necessary for generating *meaningful* evaluation criteria for this specific task.**

**Required Output JSON Format for Standard Evaluation Tools:**
- `evaluation_target_description`: (String) The description of the task being evaluated.
- `evaluation_target_objective`: (String) The objective of the task being evaluated.
- `evaluation_target_outputs_json`: (String) A JSON string representation of the structured outputs parsed from `evaluated_task_outputs_json`. Must be valid JSON.
- `evaluation_target_image_paths`: (List[String]) FULL paths to IMAGE files extracted.
- `evaluation_target_video_paths`: (List[String]) FULL paths to VIDEO files extracted.
- `evaluation_target_other_files`: (List[Dict]) Info about other non-image/video files extracted.
- `needs_detailed_criteria`: (Boolean) Set to `true` if RAG/Search context would likely **significantly improve** the quality or relevance of pass/fail criteria for *this specific task objective* (e.g., evaluating complex code generation, niche technical report). Set to `false` for common tasks where default criteria (completeness, relevance, basic quality) are likely sufficient (e.g., summarizing simple text, generating a standard diagram).

**Instructions:**
1. Set `evaluation_target_description` and `evaluation_target_objective` from the previous task's info.
2. Parse `evaluated_task_outputs_json`. If empty/invalid, use `{{"}}`. Re-stringify as JSON for the output field.
3. Parse `evaluated_task_output_files_json`. If empty/invalid, use `[]`. Populate the file path lists (`_image_paths`, `_video_paths`, `_other_files`).
4. **Analyze the previous task's objective and description to decide the value for `needs_detailed_criteria`.** Default to `false` if unsure.
5. Construct the final JSON object including `needs_detailed_criteria`.
6. **Error Check:** If the *previous task's objective* clearly implies a specific output type (e.g., "generate image") that is missing from the corresponding parsed list, return error JSON: `{{"error": "Missing expected output files/data based on objective."}}`.

**Output:** Return ONLY the final JSON input dictionary for the evaluation tools, or the error JSON. No other text.
Language: {llm_output_language}
""",
                        input_variables=[
                            "current_task_objective",
                            "task_description", "task_objective",
                            "evaluated_task_outputs_json",
                            "evaluated_task_output_files_json",
                            "llm_output_language"
                        ]
                    ),
                    "generate_criteria": PromptConfig(
                        template="""
Based on the following task description and context, define specific, measurable criteria for a **standard Pass/Fail assessment** (`EvaAgent`).

**Task Description:** {task_description}
**Task Objective:** {task_objective}
**Overall Workflow Goal:** {overall_goal}
**Context from RAG/Search:**
{rag_context}
{search_context}

**Instructions:**
*   Review the task description, objective, and overall goal.
*   If RAG/Search context is available and relevant, incorporate insights into the criteria.
*   Generate 3-5 specific, actionable criteria tailored to THIS task for a standard Pass/Fail evaluation. Focus on completeness, quality, and relevance to the objective.
*   Output ONLY the criteria as a numbered list.

Respond in {llm_output_language}.
""",
                        input_variables=["task_description", "task_objective", "overall_goal", "rag_context", "search_context", "llm_output_language"]
                    ),
                    "evaluation": PromptConfig(
                        template="""
You are evaluating the results of a design task based on specific criteria. Determine the outcome based on the **Selected Evaluation Agent Name** provided. **Your response MUST be a single, valid JSON object and nothing else.**

**Selected Evaluation Agent Name:** `{selected_agent}` (Determines evaluation type: `EvaAgent`=Standard, `SpecialEvaAgent`=Special, `FinalEvaAgent`=Final)
**Task Being Evaluated - Description:** {evaluation_target_description}
**Task Being Evaluated - Objective:** {evaluation_target_objective}

**Specific Evaluation Criteria/Rubric for this Task:**
{specific_criteria}

--- BEGIN Inputs for Assessment ---

**(IF Standard Evaluation - `EvaAgent`) Primary Results:**
*   Structured Outputs (JSON): ```json\n{evaluation_target_outputs_json}\n```
*   Generated Image Files: {evaluation_target_image_paths_str}
*   Generated Video Files: {evaluation_target_video_paths_str}
*   Other Generated Files: {evaluation_target_other_files_str}

**(IF Final or Special Evaluation - `FinalEvaAgent` or `SpecialEvaAgent`) Context & Artifacts for Review:**
*   Full Workflow Summary: {full_task_summary}
*   Key Artifact Summary (Paths/Info prepared by previous step):
       *   Key Images: {evaluation_target_key_image_paths_str}
       *   Key Videos: {evaluation_target_key_video_paths_str}
       *   Other Artifacts Summary: {evaluation_target_other_artifacts_summary_str}

**Feedback from Visual Analysis Tools (if applicable):**
*   Image Tool Feedback:
```text
{image_tool_feedback}
```
*   Video Tool Feedback:
```text
{video_tool_feedback}
```
--- END Inputs for Assessment ---

**Instructions:** Perform the **FINAL evaluation** based on the `{selected_agent}` type, **considering ALL provided inputs above (text, file info, criteria, AND visual tool feedback)**.

*   **IF `{selected_agent}` is `EvaAgent` (Standard Evaluation):**
       *   Assess results against criteria, incorporating insights from visual tool feedback if available. Determine overall "Pass" or "Fail".
       *   Provide feedback explaining Pass/Fail based on criteria AND visual analysis. Suggest improvements if "Fail".
       *   **Output Format:** Return JSON: `{{"assessment": "Pass" or "Fail", "feedback": "...", "improvement_suggestions": "...", "assessment_type": "Standard"}}`

   *   **IF `{selected_agent}` is `FinalEvaAgent` (Final Evaluation):**
       *   Review **Full Summary, Key Artifacts, and Visual Tool Feedback** against the holistic criteria/rubric.
       *   Assign a holistic score (1-10) based strictly on the rubric and all evidence.
       *   Provide feedback justifying the score based on the rubric and visual analysis. Suggest overall improvements.
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10)", "assessment_type": "Final", "feedback": "Rubric-based justification incorporating visual feedback. Holistic feedback.", "improvement_suggestions": "Overall improvement ideas."}}`

   *   **IF `{selected_agent}` is `SpecialEvaAgent` (Special Evaluation):**
       *   Review **Full Summary, Key Artifacts representing options, and Visual Tool Feedback** related to those options. Compare options using the **detailed comparative criteria/rubric provided in `{specific_criteria}`**.
       *   For each option, assess it against each dimension defined in the rubric.
       *   Assign a comparative score (1-10) reflecting the *best* option's quality and fit, based strictly on the rubric and all evidence (including visual feedback and alignment with `{user_input}`). **Identify the single best option** (e.g., by task ID or filename or descriptive name if available).
       *   Provide detailed feedback explaining the comparison across the specified dimensions (e.g., Green Building, Aesthetics, Functionality, Cost). Justify the score rationale and the selection based on the rubric, visual analysis, and how well options meet the `{user_input}`.
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10)", "assessment_type": "Special", "selected_option_identifier": "Identifier of the best option", "feedback": "Detailed rubric-based comparison across dimensions (e.g., Cost, Functionality, Green Building), incorporating visual feedback and alignment with user goal.", "improvement_suggestions": "N/A or specific if one option is close but needs minor tweak"}}`

**Instructions (Continued):**
*   Your response **MUST** be only the single, valid JSON object required for the agent type. No other text.
*   Adhere **strictly** to the output format and criteria/rubric. Base your final judgment on the **totality of the evidence provided, especially `{user_input}` and `{specific_criteria}`**.
Respond in {llm_output_language}.
""",
                        input_variables=[
                            "selected_agent",
                            "evaluation_target_description", "evaluation_target_objective",
                            "specific_criteria", "llm_output_language",
                            "evaluation_target_outputs_json",
                            "evaluation_target_image_paths_str",
                            "evaluation_target_video_paths_str",
                            "evaluation_target_other_files_str",
                            "full_task_summary",
                            "user_input", # Added user_input here for direct access during evaluation
                            "evaluation_target_key_image_paths_str",
                            "evaluation_target_key_video_paths_str",
                            "evaluation_target_other_artifacts_summary_str",
                            "image_tool_feedback",
                            "video_tool_feedback"
                        ]
                    ),
                    "prepare_final_evaluation_inputs": PromptConfig(
                        template="""
You are preparing inputs for the **FINAL HOLISTIC EVALUATION (`FinalEvaAgent`)** or a **SPECIAL MULTI-OPTION COMPARISON EVALUATION (`SpecialEvaAgent`)**.

**Current Task Agent:** `{selected_agent}`
**Current Task Objective:** {current_task_objective}
**Overall Workflow Goal (User Request - CRITICAL for identifying key comparison aspects):** {user_input}
**Full Workflow Task Summary (History - Source for outputs/options & contextual analysis like site/case studies):**
{full_task_summary}
**Aggregated Outputs from Completed Tasks (JSON String):**
```json
{aggregated_outputs_json}
```
**Aggregated Files from Completed Tasks (JSON String):**
```json
{aggregated_files_json}
```

**Your Goal:**
1. Consolidate critical information and artifacts into a structured JSON object for the final/special evaluation nodes.
2. Filter artifacts to focus on the *most relevant* final outputs (`FinalEvaAgent`) or the specific options being compared (`SpecialEvaAgent`).
3. **Determine if gathering external sources (RAG/Search) is necessary for generating the detailed rubric/criteria (Usually YES for Special/Final).**

**Required Input Format for Final/Special Evaluation:**
- `evaluation_target_description`: (String) Set based on agent: "Final Workflow Review" or "Special Multi-Option Comparison of [briefly describe what is being compared, e.g., facade designs]".
- `evaluation_target_objective`: (String) Pass through the `current_task_objective`.
- `evaluation_target_full_summary`: (String) Pass through the provided Full Workflow Task Summary.
- `evaluation_target_key_image_paths`: (List[String]) Identify and list FULL PATHS of the **most important output images representing the options or final result**.
- `evaluation_target_key_video_paths`: (List[String]) Identify and list FULL PATHS of the **most important output videos representing the options or final result**.
- `evaluation_target_other_artifacts_summary`: (String) Briefly summarize other key artifacts relevant to the agent type (e.g., textual descriptions of each option if comparing).
- `needs_detailed_criteria`: (Boolean) Set to `true`. Final and Special evaluations almost always benefit from RAG/Search context for generating a comprehensive rubric or comparative criteria.

**Instructions:**
1. Set `evaluation_target_description` based on `{selected_agent}`. If `SpecialEvaAgent`, try to briefly mention what options are being compared using info from `current_task_objective` or `full_task_summary`. Pass through `current_task_objective`.
2. Pass through `full_task_summary`.
3. Parse `aggregated_files_json` and `aggregated_outputs_json`. Analyze `full_task_summary` and `user_input` to understand context and identify the options/artifacts for comparison if `{selected_agent}` is `SpecialEvaAgent`.
4. Iterate through parsed files and outputs, identifying key images/videos/text descriptions relevant to `{selected_agent}`. Add their `path` (for files) or summarized content (for text) to the appropriate list/summary field.
5. Create the `evaluation_target_other_artifacts_summary` string, ensuring it captures key distinguishing features if comparing options.
6. **Set `needs_detailed_criteria` to `true` for these evaluation types.**
7. Construct the final JSON object using ONLY the specified keys.

**Output:** Return ONLY the final JSON input dictionary. No other text.
Language: {llm_output_language}
""",
                        input_variables=[
                            "selected_agent",
                            "current_task_objective",
                            "user_input", "full_task_summary",
                            "aggregated_outputs_json",
                            "aggregated_files_json",
                            "llm_output_language"
                        ]
                    ),
                    "generate_final_criteria": PromptConfig(
                        template="""
You are defining the HOLISTIC evaluation criteria/rubric for a completed workflow (**`FinalEvaAgent`**) OR defining comparative criteria/rubric for choosing the best among multiple options (**`SpecialEvaAgent`**).

**Current Task Agent:** `{selected_agent}`
**Current Task Objective:** {current_task_objective}
**Overall Workflow Goal (User Request - CRITICAL SOURCE for evaluation priorities):** {user_input}
**Full Workflow Task Summary (History - contains analysis like site/case studies, and outputs of options):**
{full_task_summary}
**Input Artifacts Provided for Review (prepared paths/summaries of options/final results):**
```json
{final_eval_inputs_json}
```
**Context from RAG/Search (if gathered, e.g., specific green building standards, cost benchmarks):**
{rag_context}
{search_context}

**Your Task:** Generate specific criteria and a **detailed scoring rubric/guideline** suitable for the evaluation agent type.

**IF `{selected_agent}` is `FinalEvaAgent`:**
*   **Criteria Should Address:** Goal Achievement (alignment with `{user_input}`), Quality of Final Artifacts, Process Efficiency/Logic (Optional). Incorporate insights from RAG/Search if relevant.
*   **Scoring Rubric:** Define a **detailed** 1-10 scale rubric with clear descriptions for each score range (e.g., 1-3, 4-6, 7-8, 9-10) relating to goal achievement and artifact quality, informed by context.

**IF `{selected_agent}` is `SpecialEvaAgent`:**
*   **Comparative Criteria MUST Address Multiple Dimensions:**
    *   **Identify Key Dimensions from `{user_input}` and `{full_task_summary}`:** Determine which aspects are most important for comparing the options (e.g., user explicitly mentioned "low cost" or "innovative facade").
    *   **Standard Architectural Dimensions (Consider if relevant):**
        *   `Green Building Potential`: e.g., passive design features, material choices, estimated energy performance (qualitative or simple rating like Low/Med/High).
        *   `Aesthetics/Form`: e.g., visual appeal, innovation, contextual fit, style adherence.
        *   `Functionality/Programmatic Fit`: e.g., how well the option meets stated functional needs, space efficiency, circulation.
        *   `Estimated Cost/Constructability`: e.g., qualitative assessment (Low/Med/High cost), or if possible, a rough comparative cost estimation or ranking based on materials/complexity.
        *   `Specific Standards/Client Requirements`: If mentioned in `{user_input}` or RAG/Search context (e.g., "must achieve LEED Gold potential," "prioritize local materials").
*   **Develop Discriminatory Metrics/Descriptions for Each Dimension:** For each chosen dimension, describe how to compare the options. Aim for more than just "good/bad."
    *   Example for Cost: "Option A appears most cost-effective (simple form, standard materials). Option B is likely highest cost (complex geometry, custom facade). Option C is mid-range."
    *   Example for Green Building: "Option A: Good solar shading, potential for natural ventilation (High Potential). Option B: Large glass areas, might require significant HVAC (Low Potential)."
*   **Scoring Rubric & Selection:**
    *   Define a **detailed** 1-10 scale rubric for *comparative* assessment. The score reflects how well the *best* option performs *overall* considering the weighted importance of the criteria.
    *   Describe characteristics for different score ranges, clearly linking back to the defined dimensions.
    *   Specify how to rank options and **identify the single best option** based on the comprehensive comparison.
    *   The final feedback should clearly explain the reasoning for the selection across the different dimensions.

**Output:** Output ONLY the criteria and the detailed scoring rubric/guideline as clear text, tailored to the agent type and informed by available context. Emphasize quantifiable or clearly comparable metrics where possible for `SpecialEvaAgent`.
Respond in {llm_output_language}.
""",
                        input_variables=[
                            "selected_agent",
                            "current_task_objective",
                            "user_input", "full_task_summary", "final_eval_inputs_json",
                            "rag_context", "search_context",
                            "llm_output_language"
                        ]
                    ),
                },
                parameters={}
            ),
             "qa_agent": AgentConfig(
                agent_name="qa_agent",
                description="Handles user interaction in the QA phase after task execution.",
                llm=ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7), # Keep QA as OpenAI unless specified
                prompts={
                    "qa_prompt": PromptConfig(
                        template="""
                        您目前正在與使用者進行問答。
                        目前的任務摘要：
                        {task_summary}

                        從長期記憶中檢索到的相關資訊如下：
                        {retrieved_ltm_context}

                        目前的短期問答記錄如下 (最近 {window_size} 輪)：
                        {chat_history}

                        使用者的最後一個問題/陳述是：'{last_user_query}'

                        請綜合以上所有資訊（目標、長期記憶、短期問答記錄），回答使用者最後的問題或陳述。

                        **你的回答指示：**
                        1.  如果使用者意圖結束對話 (例如，說謝謝、再見、沒問題了等)，請 **只** 回覆 `TERMINATE`。
                        2.  如果使用者意圖執行一個全新的、需要重新規劃的任務、或是不屬於當前任務的需求 (例如，"幫我做..."、"新任務..."、"幫我生成圖片..."等)，請回覆 `NEW_TASK:` 並接著清晰地、完整地總結這個新任務的目標。
                        3.  如果使用者表示沒事了，並且想要繼續執行*當前*的任務流程 (例如，"繼續執行..."、"幫我繼續..."等)，請 **只** 回覆 `RESUME_TASK`。
                        4.  否則，請正常回答使用者的問題。如果使用者沒有明確問題，可以嘗試根據對話記錄和任務摘要提供相關資訊或詢問是否需要進一步幫助。
                        你的回答語言應為：{llm_output_language}
                        """,
                        input_variables=[
                            "task_summary", "retrieved_ltm_context", "window_size",
                            "chat_history", "last_user_query", "llm_output_language"
                        ]
                    ),
                },
                parameters={
                    "memory_window_size": 10
                }
            )
        }        

        return FullConfig(workflow=default_workflow, memory=default_memory, agents=default_agents)

    def _deep_update_dict(self, target_dict, update_dict):
        """Recursively updates target_dict with values from update_dict."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and isinstance(target_dict.get(key), dict):
                # Ensure the target key exists before recursing
                if key not in target_dict:
                    target_dict[key] = {}
                self._deep_update_dict(target_dict[key], value) # 遞迴合併子字典
            elif value is not None: # Only update if the value in update_dict is not None
                 target_dict[key] = value # 覆蓋目標字典的值

    def save_config(self, config: Optional[FullConfig] = None) -> None:
        """Save the differences between the provided/current config and the default config."""
        config_to_save = config if config is not None else self.config
        if config_to_save:
            try:
                # Calculate the differences from the default config
                config_diff = self._deep_diff_dict(self.default_config.model_dump(), config_to_save.model_dump())

                # Clean up empty dictionaries potentially created by diff
                def clean_empty(d):
                    if isinstance(d, dict):
                        return {k: v for k, v in ((k, clean_empty(v)) for k, v in d.items()) if v}
                    return d

                cleaned_diff = clean_empty(config_diff)

                if not cleaned_diff:
                     print("No configuration changes detected compared to defaults. Skipping save.")
                     # Optionally delete config.json if it exists and diff is empty
                     if os.path.exists(self.config_file):
                         try:
                             os.remove(self.config_file)
                             print(f"Removed empty or default config file: {self.config_file}")
                         except OSError as e:
                             print(f"Error removing config file {self.config_file}: {e}")
                     return # Exit save function

                with open(self.config_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(cleaned_diff, indent=4, ensure_ascii=False)) # 儲存差異部分
                print(f"Configuration changes saved to '{self.config_file}'")
            except Exception as e:
                print(f"Error saving config to '{self.config_file}': {e}")
        else:
            print("Error: No configuration object available to save.")

    def _deep_diff_dict(self, dict1, dict2):
        """Recursively finds the differences between dict1 and dict2, returning a dict of differences (only values present in dict2 that differ from dict1)."""
        diff_dict = {}
        # Iterate through keys in dict2 (the potentially modified one)
        for key, value2 in dict2.items():
            value1 = dict1.get(key) # Get corresponding value from dict1 (defaults)

            if isinstance(value2, dict) and isinstance(value1, dict):
                # Recursively compare sub-dictionaries
                sub_diff = self._deep_diff_dict(value1, value2)
                if sub_diff: # Only add if there are differences in the sub-dict
                    diff_dict[key] = sub_diff
            elif value1 != value2:
                # Add if the key doesn't exist in dict1 or the values are different
                diff_dict[key] = value2
        return diff_dict

    # --- Getters ---
    def get_full_config(self) -> FullConfig:
        """Return the entire configuration object."""
        return self.config

    def get_workflow_config(self) -> WorkflowConfig:
        """Get the workflow configuration."""
        return self.config.workflow

    def get_memory_config(self) -> MemoryConfig:
         """Get the memory configuration."""
         return self.config.memory

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.config.agents.get(agent_name)

    def get_llm_config(self, agent_name: str) -> Optional[ModelConfig]:
         """Get LLM configuration for a specific agent."""
         agent_config = self.get_agent_config(agent_name)
         return agent_config.llm if agent_config else None

    def get_prompt_template(self, agent_name: str, prompt_name: str) -> Optional[str]:
         """Get a specific prompt template string."""
         agent_config = self.get_agent_config(agent_name)
         prompt_config = agent_config.prompts.get(prompt_name) if agent_config else None
         return prompt_config.template if prompt_config else None

    def get_tool_config(self, agent_name: str, tool_name: str) -> Optional[ToolConfig]:
         """Get configuration for a specific tool within an agent."""
         agent_config = self.get_agent_config(agent_name)
         tool_cfg = agent_config.tools.get(tool_name) if agent_config else None
         return tool_cfg

    # --- Updaters (Example - add more as needed) ---
    def update_agent_llm_config(self, agent_name: str, llm_config: ModelConfig) -> bool:
        """Update the LLM configuration for a specific agent and save."""
        agent_config = self.get_agent_config(agent_name)
        if agent_config:
            agent_config.llm = llm_config
            self.save_config()
            return True
        return False

    def update_memory_embedding_model(self, embedding_config: EmbeddingModelConfig) -> None:
         """Update the LTM embedding model configuration and save."""
         self.config.memory.long_term_memory = embedding_config
         self.save_config()

# --- Instantiate ConfigManager ONCE at module level to get BASE defaults for Schema ---
# This instance specifically accesses the programmatically defined defaults
# It does NOT load config.json for defining the schema defaults.
_schema_defaults_loader = ConfigManager()
# Access the config object created purely from _create_default_config()
_base_default_config_obj = _schema_defaults_loader.default_config

# Helper function to safely get base default prompts
def get_base_default_prompt(agent_name: str, prompt_name: str) -> str:
    """Gets the base default prompt template defined in code."""
    try:
        # Access the structure defined in _create_default_config
        # Add checks for existence of agent and prompts dict
        agent = _base_default_config_obj.agents.get(agent_name)
        if agent and agent.prompts:
             prompt_config = agent.prompts.get(prompt_name)
             if prompt_config and prompt_config.template:
                 return prompt_config.template
        # Fallback if not found
        print(f"Warning: Base default prompt not found or empty for {agent_name}/{prompt_name}. Using empty string for schema default.")
        return ""
    except (KeyError, AttributeError, TypeError) as e: # Catch potential errors during access
        print(f"Warning: Error accessing base default prompt for {agent_name}/{prompt_name}: {e}. Using empty string for schema default.")
        return ""

# =============================================================================
# Runtime Configuration Schema for LangGraph (MODIFIED - Using Single Literal)
# =============================================================================
# --- Define SINGLE Literal for ALL model names ---
AllModelNamesLiteral = Literal[
    # OpenAI
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o3-mini",
    # Google
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "Gemini 2.0 Flash",
    "Gemini 2.5 Pro Experimental",
    # Anthropic
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

# --- SupportedLanguage Literal (保持不變) ---
SupportedLanguage = Literal[
    "繁體中文",
    "English",
    "简体中文"
]

class ConfigSchema(BaseModel):
    """Configuration schema exposed in LangGraph Studio."""

    # --- Global Language ---
    global_llm_output_language: SupportedLanguage = Field(
        default="繁體中文",
        title="Global: Output Language",
        description="Global default language for LLM responses."
    )

    # --- Memory Config ---
    retriever_k: int = Field(
         default=5,
         title="Memory: Retriever K",
         description="Number of documents to retrieve from LTM (long-term memory)",
         gt=0
    )

    # --- Process Management Agent LLM Config (Using Single Literal) ---
    pm_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gpt-4o-mini",
        title="Process Management: Model Name",
        description="Select the LLM Model for the Process Management Agent."
    )
    pm_temperature: float = Field(
        default=0.7,
        title="Process Management: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Process Management Agent (0.0-1.0)."
    )
    pm_max_tokens: Optional[int] = Field(
        default=None,
        title="Process Management: Max Tokens",
        gt=0,
        description="Max Tokens for the Process Management Agent (Optional)."
    )

    # --- Assign Agent LLM Config (Using Single Literal) ---
    aa_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gpt-4o-mini",
        title="Assign Agent: Model Name",
        description="Select the LLM Model for the Assign Agent."
    )
    aa_temperature: float = Field(
        default=0.7,
        title="Assign Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Assign Agent (0.0-1.0)."
    )
    aa_max_tokens: Optional[int] = Field(
        default=None,
        title="Assign Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the Assign Agent (Optional)."
    )

    # --- Tool Agent LLM Config (Using Single Literal) ---
    ta_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gpt-4o-mini",
        title="Tool Agent: Model Name (Errors)",
        description="Select the LLM Model for the Tool Agent (used for error analysis)."
    )
    ta_temperature: float = Field(
        default=0.7,
        title="Tool Agent: Temperature (Errors)",
        ge=0.0, le=1.0,
        description="Temperature for the Tool Agent's error analysis (0.0-1.0)."
    )
    ta_max_tokens: Optional[int] = Field(
        default=None,
        title="Tool Agent: Max Tokens (Errors)",
        gt=0,
        description="Max Tokens for the Tool Agent's error analysis (Optional)."
    )

    # --- Evaluation Agent LLM Config (Using Single Literal) ---
    ea_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gpt-4o-mini",
        title="Evaluation Agent: Model Name",
        description="Select the LLM Model for the Evaluation Agent."
    )
    ea_temperature: float = Field(
        default=0.2,
        title="Evaluation Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Evaluation Agent (0.0-1.0)."
    )
    ea_max_tokens: Optional[int] = Field(
        default=None,
        title="Evaluation Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the Evaluation Agent (Optional)."
    )

    # --- Prompts ---
    ea_prepare_final_evaluation_inputs_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "prepare_final_evaluation_inputs"),
        title="Evaluation Agent: Prepare FINAL/SPECIAL Eval Inputs Prompt",
        description="Prompt to gather outputs/artifacts for FINAL holistic review or SPECIAL multi-option comparison. (LTM Context removed). Clear field to use runtime default.", # Updated description
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluation_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluation"),
        title="Evaluation Agent: Evaluation Prompt (Handles ALL Types)",
        description="The core prompt for performing evaluation (Standard, Final, or Special). (LTM Context removed). Clear field to use runtime default.", # Updated description
        extra={'widget': {'type': 'textarea'}}
    )

    # --- Validator (MODIFIED: Target only one field for testing) ---
    @field_validator('pm_temperature') # <<< MODIFIED: Temporarily validate only one field >>>
    @classmethod
    def check_temperature(cls, v: float, info: FieldValidationInfo): # Signature remains the same
        # 核心邏輯不變
        if not (0.0 <= v <= 1.0):
             raise ValueError(f'{info.field_name} must be between 0.0 and 1.0')
        return v


# =============================================================================
# Initialize LLM Function (remains the same - logic still valid)
# =============================================================================
def initialize_llm(llm_config_dict: Dict[str, Any]) -> Any:
    """Initializes the LangChain LLM based on a configuration dictionary, inferring the provider."""
    # --- Infer Provider from Model Name ---
    model_name = llm_config_dict.get("model_name", "gpt-4o-mini")
    provider = "openai" # Default provider

    # Updated inference logic
    lower_model_name = model_name.lower() # Use lower case for case-insensitive matching
    if "gemini" in lower_model_name:
        provider = "google"
    elif "claude" in lower_model_name:
        provider = "anthropic"
    # Keep openai check broad enough for gpt-* and o*-mini
    elif "gpt" in lower_model_name or lower_model_name.startswith("o"):
         provider = "openai"
    else:
        print(f"Warning: Could not infer provider for model '{model_name}'. Assuming 'openai'.")

    print(f"Inferred Provider: {provider} for Model: {model_name}")

    temperature = llm_config_dict.get("temperature", 0.7)

    # Validate temperature type and range
    try:
        temperature = float(temperature)
        if not (0.0 <= temperature <= 1.0):
            print(f"Warning: Invalid temperature {temperature} received for {provider}/{model_name}, using default 0.7.")
            temperature = 0.7
    except (ValueError, TypeError):
        print(f"Warning: Invalid type for temperature ('{temperature}') for {provider}/{model_name}, using default 0.7.")
        temperature = 0.7

    max_tokens = llm_config_dict.get("max_tokens")

    common_params = {"temperature": temperature}
    if max_tokens is not None:
        try:
            max_tokens_int = int(max_tokens)
            if max_tokens_int > 0:
                common_params["max_tokens"] = max_tokens_int
            else:
                 print(f"Warning: max_tokens ({max_tokens}) for {provider}/{model_name} must be > 0, ignoring.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid type for max_tokens ('{max_tokens}') for {provider}/{model_name}, ignoring.")

    print(f"Initializing LLM: Provider={provider}, Model={model_name}, Params={common_params}")

    try:
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"): print("Warning: OPENAI_API_KEY environment variable not set.")
            # Pass model_name as 'model'
            return ChatOpenAI(model=model_name, **common_params)
        elif provider == "google":
            if not os.getenv("GOOGLE_API_KEY"): print("Warning: GOOGLE_API_KEY environment variable not set.")
            google_params = {"temperature": temperature}
            if "max_tokens" in common_params:
                google_params["max_output_tokens"] = common_params["max_tokens"]
            # Pass model_name as 'model'
            return ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True, **google_params)
        elif provider == "anthropic":
             if not os.getenv("ANTHROPIC_API_KEY"): print("Warning: ANTHROPIC_API_KEY environment variable not set.")
             # Pass model_name as 'model'
             return ChatAnthropic(model=model_name, **common_params)
        else:
            print(f"Warning: Unsupported LLM provider '{provider}' despite inference attempt. Using default OpenAI.")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    except Exception as e:
        print(f"ERROR initializing LLM for provider {provider}, model {model_name}: {e}")
        print("Falling back to default OpenAI model.")
        if not os.getenv("OPENAI_API_KEY"): print("Warning: OPENAI_API_KEY environment variable not set for fallback.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# =============================================================================
# Example Usage (Optional)
# =============================================================================
# ...