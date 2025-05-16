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
        OpenAI: 'gpt-4o-mini', 'gpt-4o', 'o1-mini', 'o3-mini'.
        Google: 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest', 'gemini-1.0-pro', 'Gemini-2.0*Flash', 'gemini-2.5-pro-exp-03-25', 'gemini-2.5-flash-preview-04-17'.
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
    output_directory: str = Field("./output/cache", description="Directory for output files") # Adjusted default
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
            output_directory="./output/cache",
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

        default_agents = {
                "process_management": AgentConfig(
                    agent_name="process_management",
                    description="Responsible for planning and managing the task workflow.",
                    llm=default_llm,
                    prompts={
                        "create_workflow": PromptConfig(
                            template="""
You are a meticulous and **detail-oriented** workflow planner specializing in architecture and design tasks. Your goal is to break down a user's request into a **sequence of granular, executable task objectives** for a team of specialized agents, ensuring smooth data flow between steps.

**User Request** (**Strictly Follow**): {user_input}

Analyze the request and generate a **complete, logical, and DETAILED sequence of tasks** to fulfill it. Pay close attention to dependencies and insert necessary intermediate processing steps.

**Key Planning Principles:**
1.  **Dependencies:** If Task B requires data generated by Task A, ensure Task A comes first.
2.  **Intermediate Processing:** If raw output of Task A isn't directly usable by Task B, **INSERT an `LLMTaskAgent` task** between them (e.g., summarize, reformat, generate prompts).
3.  **Evaluation Task Assignment (CRITICAL - Follow these distinctions precisely):**
    *   **`EvaAgent` (Standard Evaluation - Pass/Fail for a SINGLE preceding task's output):**
        *   **Use Case:** Assign to `EvaAgent` ONLY when you need to evaluate the **direct output of the immediately preceding single task**. This is for a simple pass/fail check on one specific result.
        *   **Task Objective:** Set a descriptive `task_objective` (e.g., "Evaluate Task 3's generated floor plan based on standard criteria for residential layouts.").
        *   **Requires `requires_evaluation: true`**.
    *   **`SpecialEvaAgent` (Special Evaluation - Compare MULTIPLE distinct options/artifacts):**
        *   **Use Case:** Assign to `SpecialEvaAgent` when **multiple, separate, parallel tasks have generated distinct design options** (e.g., Task 2 generated Facade Option A, Task 3 generated Facade Option B, Task 4 generated Facade Option C). This agent will compare these options (A, B, C) based on their textual, visual (images/videos), and other outputs to select the best or score them comparatively.
        *   **Task Objective:** Set `task_objective="special_evaluation"`. The description should clarify what options are being compared (e.g., "Compare facade options A, B, and C").
        *   **Requires `requires_evaluation: true`**.
    *   **`FinalEvaAgent` (Final Evaluation - Holistic review of multi-stage/iterated results OR overall project):**
        *   **Use Case 1 (Iterated Design):** Assign to `FinalEvaAgent` when a previous `SpecialEvaAgent` (or user) selected a base option (e.g., Option A), and subsequent tasks have **iterated or elaborated upon that selected option** (e.g., Task 5 detailed Option A into A-1, Task 6 refined A-1 into A-2). `FinalEvaAgent` then holistically reviews these developed results (A-1, A-2).
        *   **Use Case 2 (Overall Project Review):** Assign to `FinalEvaAgent` at the very end of a complex, multi-stage workflow to provide a holistic score and assessment of the entire project's outcome against the original user request.
        *   **Task Objective:** Set `task_objective="final_evaluation"`. The description should reflect the scope (e.g., "Final holistic evaluation of the refined 'Option A' development" or "Final comprehensive review of the entire 'Mountain Cabin Design' project").
        *   **Requires `requires_evaluation: true`**.
4.  **Clarity:** Each task objective must be specific and actionable.
5.  **`requires_evaluation` for ALL Evaluation Agents:** **CRITICAL:** If `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, OR `FinalEvaAgent`, you **MUST** set `requires_evaluation` to `true`.

For **each** task in the sequence, you MUST specify:
1.  `description`: High-level goal for this step.
2.  `task_objective`: Specific outcome and method needed.
    *   Use `"final_evaluation"` or `"special_evaluation"` for the respective agent modes as defined above.
    *   Otherwise, describe the specific goal.
    *   **For `ModelRenderAgent` (Handles existing images):**
        *   If the goal is photorealistic rendering of an existing model view/image (e.g., from Rhino): Set objective like "Photorealistic rendering of the provided image(s) of the architectural scheme: [scheme details]."
        *   If the goal is future scenario simulation based on an existing image: Set objective like "Simulate future scenario for the provided image(s). Context: [user goals, site conditions, architectural scheme details for simulation]."
    *   **For `ImageGenerationAgent` (Generates new images from text/concepts):**
        *   If generating a new visual concept from a description: Set objective like "Generate an image representing [concept description, style, mood]."
        *   If exploring multiple *distinct visual options* for a design element (e.g., different facade styles): Use a single `ImageGenerationAgent` task with a clear objective that defines all options to be explored. Ensure the input prompt clearly differentiates between options to avoid generating similar designs. Example: "Generate multiple distinct facade options including: Option A - modern style with glass and steel; Option B - classical style with stone and arches." Each generated image should correspond to one distinct option. These options would then typically be evaluated by a `SpecialEvaAgent`.
        *   If generating *multiple variations of the SAME concept/option*: Use a single `ImageGenerationAgent` task with a clear objective for that one concept, ensuring the input provides sufficient guidance to generate diverse variations.
    *   **For `RhinoMCPCoordinator`:**
        *   If the task involves geometric modeling, modification, or precise analysis: Define the specific Rhino operations needed.
        *   If the task involves **functional layout, programmatic blocking, or quantitative/qualitative spatial arrangement**: Clearly state this objective. For example: "Develop a functional block layout for the ground floor based on [program requirements], and provide a top-down parallel projection screenshot." or "Perform a solar access analysis on the south facade and output results."
3.  `inputs`: JSON object suggesting initial data needs (e.g., `{{"prompt": "..."}}`). Use placeholders like `{{output_from_task_id_xyz.key}}`. Indicate file needs clearly. **Do not invent paths.** Use `{{"}}` if no input suggestion applies. NEVER use `null`.
4.  `requires_evaluation`: Boolean (`true`/`false`). **MUST be `true` if `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent`.**
5.  `selected_agent`: **(CRITICAL)** The **exact name** of the agent from the list below. Mandatory for every task. **Use `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent` for evaluation tasks, adhering STRICTLY to the use case definitions above.**

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
*   `EvaAgent`
*   `SpecialEvaAgent`
*   `FinalEvaAgent`
--- END AGENT CAPABILITIES ---

Return the entire workflow as a **single, valid JSON list** object. Do NOT include any explanatory text before or after the JSON list. Ensure perfect JSON syntax, detailed steps, and that **every task includes the `selected_agent` and correct `requires_evaluation` key, adhering to the specific use cases for evaluation agents.**
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
1.  **Identify Failure Type:** Determine if 'evaluation' (Primarily for `EvaAgent` - Standard Pass/Fail which evaluates *content*):
        *   **MUST** return `new_tasks_list` with **TWO** tasks:
            1.  **Task 1 (Alternative Generation/Execution):** Analyze `feedback_log` and `original Task Objective`. Choose the **most appropriate agent** (NOT an evaluation agent) from the list to achieve the objective based on feedback. Define a **new `task_objective`**. Suggest inputs `{{"}}`. Set `requires_evaluation=false`.
            2.  **Task 2 (Re-Evaluation):** Create an `EvaAgent` task. Set `selected_agent="EvaAgent"`, `task_objective="Evaluate outcome of the alternative task"`, `inputs={{}}`, `requires_evaluation=true`.
    *   **If Failure Type is 'execution' (This applies to ALL agents, including `SpecialEvaAgent`/`FinalEvaAgent` if their internal process fails):**
        *   **If Original Agent was `SpecialEvaAgent` OR `FinalEvaAgent`:**
            *   Generate **ONLY ONE new task**: a retry of the original evaluation.
            *   Set `selected_agent` to `{selected_agent_name}` (i.e., `SpecialEvaAgent` or `FinalEvaAgent`).
            *   Set `task_objective` to its original value (e.g., `"special_evaluation"` or `"final_evaluation"`).
            *   Set `description` to something like "Retry {selected_agent_name} for task: {task_description}".
            *   Suggest `inputs={{}}`.
            *   Set `requires_evaluation=true`.
            *   Return this single task in `new_tasks_list`: `[{{"action": "FALLBACK_GENERAL", "new_tasks_list": [{{...the single retry evaluation task...}}]}}]`
        *   **Else (Original Agent was NOT `SpecialEvaAgent`/`FinalEvaAgent`):**
            *   Generate **ONLY the necessary task(s)** to overcome the error and achieve the original objective.
            *   Analyze `execution_error_log` and `original Task Objective`.
            *   Choose the **most appropriate agent** from the list. Consider retrying `{selected_agent_name}` with corrected inputs if the error suggests it, or select an alternative if `{selected_agent_name}` seems unsuitable.
            *   Define objective(s) for the fallback task(s).
            *   Suggest inputs `{{"}}`.
            *   Keep original `requires_evaluation` (`{original_requires_evaluation}`) **ONLY IF** the fallback task still logically needs it. **DO NOT automatically add a separate evaluation task.**
            *   Return this as `new_task` (for a single task) or `new_tasks_list` (for a short sequence if needed) within the JSON: `{{"action": "FALLBACK_GENERAL", ...}}`

*   **`MODIFY`:** Modify the *current* failed task and retry it with `{selected_agent_name}`. Appropriate for minor input errors.
    *   Output JSON: `{{"action": "MODIFY", "modify_description": "...", "modify_objective": "..."}}`

*   **`SKIP`:** Skip the failed task.
    *   Output JSON: `{{"action": "SKIP"}}`

**Instructions:**
*   If **Is Max Retries Reached?** is `True`, **MUST** choose `SKIP`.
*   Choose **only one action**. Provide **only** the single JSON output.
*   Ensure agent selections/objectives address the failure and prioritize the original objective (`{task_objective}`).
*   **REMEMBER:**
    *   Only 'evaluation' failures of *content* by `EvaAgent` require the fixed two-task structure.
    *   'Execution' failures (including internal failures of `SpecialEvaAgent`/`FinalEvaAgent` processes) should generate only the task(s) needed to fix or retry the error.
    *   For `SpecialEvaAgent`/`FinalEvaAgent` failures, the fallback is a direct retry of that evaluation agent itself.
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
                llm=default_llm,
                prompts={
                    "prepare_tool_inputs_prompt": PromptConfig(
                        template="""
You are an expert input preprocessor for specialized AI tools. Your goal is to take a high-level task objective, the overall user request, task history/outputs, and information about the selected tool, then generate the precise JSON input dictionary containing ONLY the keys REQUIRED by that specific tool, using standardized keys.

**Selected Tool/Agent:** `{selected_agent_name}`
**Tool Description:** {agent_description}
**Current Task Description:** {task_description}
**Current Task Objective (CRITICAL - READ CAREFULLY):** {task_objective}
**Overall Workflow Goal (User Request, please prioritize this):** {user_input}
**User Budget Limit (if provided, typically in TWD/NTD unless otherwise specified):** {user_budget_limit}

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
**Latest Evaluation Results (if applicable):**
```json
{latest_evaluation_results_json}
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
    *   `prompt`: (String) REQUIRED. Detailed textual description for generating a NEW image or for editing. Focus on photorealistic style. **Add DO NOT generate text in the prompt.**
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
1.  Analyze the **Current Task Objective/Description**, **Selected Tool/Agent (`{selected_agent_name}`)**, **Overall Goal**, and **User Budget Limit (if provided)**.
2.  **Generate/Extract Required Textual Inputs & Flags**:
    *   **Budget Awareness (for ALL agents)**: If a `user_budget_limit` is provided, incorporate cost-consciousness into prompts and parameters as appropriate.
        * For `LLMTaskAgent`: If generating design proposals or specifications, explicitly reference materials and construction methods that align with the budget.
        * For `RhinoMCPCoordinator`: If modeling, include budget considerations in the `user_request` to guide appropriate spatial dimensions and geometric forms.
        * For `ImageGenerationAgent`、`ModelRenderAgent`: Suggest materials/finishes/forms in the `prompt` that align with the budget class (luxury, mid-range, economical).**Add DO NOT generate text in the prompt.**
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
8.  **When there is a need to iterate on the next step based on the selected solution, pay attention to the items in the Evaluation Results, and organize the solution-related files and information as task inputs.**

**Output:** Return ONLY the final JSON input dictionary for `{selected_agent_name}`, or the error JSON. No other text.
Language: {llm_output_language}
""",
                            input_variables=[
                                "selected_agent_name", "agent_description", "user_input",
                                "task_objective", "task_description",
                                "aggregated_summary",
                                "aggregated_outputs_json",
                                "aggregated_files_json",
                                "error_feedback", "llm_output_language", "user_budget_limit",
                                "latest_evaluation_results_json"
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
                    llm=default_llm, # Keep Tool Agent error handling as OpenAI unless specified
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
                llm=default_llm, 
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
**Overall Workflow Goal (User Request):** {user_input}

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
*   **Options Data (for Special/Final, IF `evaluate_with_text_agent` is the current evaluator. This JSON comes from the *prepare_evaluation_inputs_node*'s output, specifically its `options_data` field, which itself was derived from prior tasks. This data is crucial for the LLM to perform its text-based evaluation across multiple options.):**
```json
{options_data_for_llm_eval_json}
```

**Feedback from Visual Analysis Tools (if applicable, these are summaries from prior parallel evaluation branches):**
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
       *   Review **Full Summary, Key Artifacts, Visual Tool Feedback, and `options_data_for_llm_eval_json` (if provided by LLM evaluation path and relevant)** against the holistic criteria/rubric.
       *   **Crucially, when assessing aspects like cost, functionality, or adherence to original intent, refer to `raw_outputs_for_llm_parsing.mcp_internal_messages` within each option in `options_data_for_llm_eval_json`. Early AI messages from MCP agents (e.g., RhinoMCP's plan) are key.**
       *   Assign a holistic score (1-10) based strictly on the rubric and all evidence.
       *   Provide feedback justifying the score based on the rubric and visual analysis. Suggest overall improvements.
       *   The `detailed_option_scores` in your output should be the list of option evaluations generated by *this LLM evaluation path* if it was responsible for evaluating options based on `options_data_for_llm_eval_json`. If this is a final review *after* parallel visual tools have run, this field might be an aggregation or reference existing scores. For LLM-based option evaluation, ensure each item in `detailed_option_scores` has "option_id", "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text".
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10)", "assessment_type": "Final", "detailed_option_scores": [{{...option1_scores...}}, ...], "feedback": "Rubric-based justification incorporating visual feedback. Holistic feedback.", "improvement_suggestions": "Overall improvement ideas."}}`

   *   **IF `{selected_agent}` is `SpecialEvaAgent` (Special Evaluation):**
       *   Review **`options_data_for_llm_eval_json` (CRITICAL: use this as the primary source for options to evaluate via text/LLM). Also consider Full Summary, Key Artifacts (if different from options_data), and Visual Tool Feedback (if this LLM call is *after* visual tools and needs to consolidate).** Compare options using the **detailed comparative criteria/rubric provided in `{specific_criteria}`**.
       *   **Crucially, when assessing aspects like cost, functionality, or adherence to original intent, refer to `raw_outputs_for_llm_parsing.mcp_internal_messages` within each option in `options_data_for_llm_eval_json`. Early AI messages from MCP agents (e.g., RhinoMCP's plan) are key.**
       *   For each option parsed from `options_data_for_llm_eval_json`, assess it against each dimension defined in the rubric. Provide scores for "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", estimate `estimated_cost`, and `green_building_potential_percentage`. Each of these should be from the LLM's textual/conceptual evaluation of the option.
       *   Assign a comparative score (1-10) reflecting the *best* option's quality and fit, based strictly on the rubric and all evidence (including visual feedback and alignment with `{user_input}`). **Identify the single best option** (e.g., by its `option_id` from `options_data_for_llm_eval_json`).
       *   Provide detailed feedback explaining the comparison across the specified dimensions.
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10)", "assessment_type": "Special", "selected_option_identifier": "Identifier of the best option", "detailed_option_scores": [ {{"option_id": "id1", "user_goal_responsiveness_score_llm": <score>, "aesthetics_context_score_llm": <score>, "functionality_flexibility_score_llm": <score>, "durability_maintainability_score_llm": <score>, "estimated_cost": <cost>, "green_building_potential_percentage": <percentage>, "llm_feedback_text":"option specific feedback based on text/concept evaluation"}}, ... ], "feedback": "Detailed rubric-based comparison across dimensions (e.g., User Goal Responsiveness, Aesthetics/Context, Functionality/Flexibility, Durability/Maintainability, Cost, Green Building), incorporating visual feedback if available and relevant to text evaluation, and alignment with user goal.", "improvement_suggestions": "N/A or specific if one option is close but needs minor tweak"}}`

**Instructions (Continued):**
*   Your response **MUST** be only the single, valid JSON object required for the agent type. No other text.
*   Adhere **strictly** to the output format and criteria/rubric. Base your final judgment on the **totality of the evidence provided, especially `{user_input}` and `{specific_criteria}`**.
*   For `SpecialEvaAgent` and `FinalEvaAgent` when processing `options_data_for_llm_eval_json`: The `detailed_option_scores` list you generate MUST be a list of dictionaries. Each dictionary represents one option and MUST contain the keys: `option_id` (string), `user_goal_responsiveness_score_llm` (float), `aesthetics_context_score_llm` (float), `functionality_flexibility_score_llm` (float), `durability_maintainability_score_llm` (float), `estimated_cost` (float), `green_building_potential_percentage` (float, 0-100), and `llm_feedback_text` (string).
Respond in {llm_output_language}.
""",
                        input_variables=[
                            "selected_agent",
                            "evaluation_target_description", "evaluation_target_objective",
                            "specific_criteria", "llm_output_language", "user_input",
                            "evaluation_target_outputs_json",
                            "evaluation_target_image_paths_str",
                            "evaluation_target_video_paths_str",
                            "evaluation_target_other_files_str",
                            "full_task_summary",
                            "evaluation_target_key_image_paths_str",
                            "evaluation_target_key_video_paths_str",
                            "evaluation_target_other_artifacts_summary_str",
                            "options_data_for_llm_eval_json", 
                            "image_tool_feedback",
                            "video_tool_feedback"
                        ]
                    ),
                    "prepare_final_evaluation_inputs": PromptConfig(
                        template="""
You are preparing inputs for the **FINAL HOLISTIC EVALUATION (`FinalEvaAgent`)** or a **SPECIAL MULTI-OPTION COMPARISON EVALUATION (`SpecialEvaAgent`)**.

**Current Task Agent:** `{selected_agent}`
**Current Task Objective (Overall Goal for this Evaluation):** {current_task_objective}
**Overall Workflow User Request (CRITICAL for identifying key comparison aspects, architecture type, and number of options):** {user_input}
**Budget Limit Overall (if provided by user for the entire project):** {budget_limit_overall}

**Full Workflow Task Summary (History - Source for outputs/options & contextual analysis like site/case studies):**
{full_task_summary}
**Aggregated Outputs from ALL Completed Tasks (JSON String - Parse this carefully. Pay ATTENTION to outputs from image generation tasks, as their textual descriptions/prompts are KEY to linking images to options):**
```json
{aggregated_outputs_json}
```
**Aggregated Files from ALL Completed Tasks (JSON String - Correlate files with options. Use 'source_task_id' to link files to tasks. CRITICALLY, use the 'description' field within each file object in this JSON, as it now contains structured info like 'SourceAgent: ...; TaskDesc: ...', to link files to options):**
```json
{aggregated_files_json}
```
**Previously identified Options Data (JSON String - If a prior step already structured some options, use as a starting point or reference. This might be from ProcessManagement's initial plan if it detailed options. If empty, you must derive options from scratch.):**
```json
{options_data_json}
```

**Your Goal:**
1.  **Identify Distinct Design Options:**
    *   Analyze the `Overall Workflow User Request` for explicit mentions of multiple options (e.g., "design three concepts", "compare facade A and B").
    *   Carefully parse `Aggregated Outputs from ALL Completed Tasks`. Look for tasks whose outputs represent distinct design concepts. A single task might output multiple options (e.g., an LLM generating text for Option A, Option B, Option C in its output). **You MUST treat each such concept as a separate option.**
    *   The `full_task_summary` can also provide clues about which tasks were intended to generate options.
    *   If `options_data_json` is not empty, it might contain a pre-defined list of options. Validate and augment this list.
    *   **CRITICAL: For each distinct design option identified, assign a NEW, SEQUENTIAL `option_id` (e.g., "option_1", "option_2", "option_3", ... or "option_A", "option_B", "option_C", ...). Do NOT reuse task IDs as option_ids if a task generated multiple options.**

2.  **For EACH distinct design option identified, extract or infer the following:**
    *   `option_id`: (String) **Your NEWLY ASSIGNED sequential identifier for this option (e.g., "option_1").**
    *   `description`: (String) **The short, descriptive THEME or NAME of this specific design option (e.g., "Modern Glass Pavilion", "Biophilic Community Center", "Cost-Effective Modular Housing").** Infer this from the option's content in `aggregated_outputs_json` or user request.
    *   `architecture_type`: (String) **The primary architectural type for THIS option (e.g., "Residential Tower", "Museum", "Urban Park").** Infer this from the `Overall Workflow User Request` or the option's specific content.
    *   `textual_summary_from_outputs`: (String) **CRITICAL: A concise summary of the textual outputs from `aggregated_outputs_json` that are EXCLUSIVELY related to THIS SPECIFIC design option. If a single task output in `aggregated_outputs_json` contains descriptions for multiple options, you MUST isolate and extract ONLY the text relevant to the current option being processed. Do NOT include text from other options.**
    *   `image_paths`: (List[Dict]) **CRITICAL LOGIC REQUIRED for associating images to options:**
        *   For the CURRENT design option you are processing (identified by its `description` or theme, e.g., "Modern Glass Pavilion"):
        *   Iterate through each file object in the `aggregated_files_json` list. 
        *   Each file object's `description` field is now a structured string, typically like: `"SourceAgent: <AgentName>; TaskDesc: <Description of the task that created this file>; ImageNum: <X/Y>; PromptHint: <...>"`. (The exact fields like `ImageNum` or `PromptHint` may vary depending on the `SourceAgent`.)
        *   **Your primary matching strategy: Parse the structured `description` string of each file. Extract the value associated with the `TaskDesc:`and `ImageNum` key (this is the `<Description of the task that created this file>`). Compare this extracted `TaskDesc` string with the theme/description of the CURRENT design option you are processing.**
        *   If the extracted `TaskDesc` from the file's `description` closely matches or is clearly intended for the current design option's theme, then this file (image) belongs to this option.
        *   You can also refer to `full_task_summary` to understand the sequence and context of tasks that generated these files, which will help confirm these relationships.**
        *   The `source_task_id` in the file object can confirm which task generated the file, and the `TaskDesc` (extracted as above) tells you the *purpose* of that task at generation time, which is key for matching to an option theme.
        *   List all image file info (`{{"path": "...", "filename": "..."}}`) that you can confidently associate with THIS specific design option using this `TaskDesc` matching.
        *   If no images can be confidently associated, this list MUST be empty `[]`.
    *   `video_paths`: (List[Dict]) **CRITICAL**: Apply the exact same logic as for `image_paths`. Parse each video file's structured `description` in `aggregated_files_json`. Extract the `TaskDesc:` value, and match it to the current design option's theme. If it matches, include its info (`{{"path": "...", "filename": "..."}}`). If no videos are associated, this list must be empty `[]`.
    *   `other_relevant_files`: (List[Dict]) List of other file info relevant to this option (e.g., text documents, spreadsheets) identified using similar contextual matching if possible, or broader association if their `TaskDesc` (extracted from their structured `description`) generally aligns with the option's theme.
    *   `initial_estimated_cost`: (Float, Optional) If any task output for this option mentioned a cost.
    *   `initial_green_building_percentage`: (Float, Optional) If any task output for this option mentioned green building metrics.

3.  **Consolidate into `options_data` List:** Create a list where each element is a dictionary structured as defined in step 2 for each identified option. **VERY IMPORTANT: Verify that for each option, `textual_summary_from_outputs` is exclusive, `image_paths` and `video_paths` ONLY contain files belonging to that specific option (using the `TaskDesc` matching logic described above), and each option has its unique sequential `option_id`.**
4.  **Summarize Other Artifacts (Key Overall Media):** Create `evaluation_target_key_image_paths`, `evaluation_target_key_video_paths`, and `evaluation_target_other_artifacts_summary` by selecting the MOST representative files/outputs across all identified options OR from the overall project if it's a `FinalEvaAgent` on a single final result. These "key" paths are for overall summary, distinct from per-option paths in `options_data`.
5.  **Determine `needs_detailed_criteria`**: Set to `true`.
6.  **Pass through `budget_limit_overall`**.

**Required Output JSON Format (Your JSON Output):**
- `evaluation_target_description`: (String) Set based on agent: "Final Workflow Review of [Project Name/Theme from user_input]" or "Special Multi-Option Comparison of [e.g., Facade Designs for Project X]".
- `evaluation_target_objective`: (String) Pass through the `current_task_objective`.
- `evaluation_target_full_summary`: (String) Pass through the provided `full_task_summary`.
- `evaluation_target_key_image_paths`: (List[String]) FULL PATHS of the most important output images for OVERALL summary.
- `evaluation_target_key_video_paths`: (List[String]) FULL PATHS of the most important output videos for OVERALL summary.
- `evaluation_target_other_artifacts_summary`: (String) Briefly summarize other key artifacts.
- `needs_detailed_criteria`: (Boolean) `true`.
- `options_data`: (List[Dict]) **MANDATORY & CRITICAL**: Your list of option dictionaries, structured as per step 2 above. If no distinct options are found despite the user request implying them, this might be an empty list, but clearly log this. **Ensure each option's `textual_summary_from_outputs` is specific to it, and its specific image and video files (identified via `TaskDesc` matching from the file's own `description` field) are listed under its `image_paths` and `video_paths` keys respectively, and each has a unique sequential `option_id`.**
- `budget_limit_overall`: (Float or Null) Parsed from input.

**Instructions:**
1.  **Prioritize `Overall Workflow User Request` and `Aggregated Outputs from ALL Completed Tasks` for identifying options and their details. Crucially, for media files, parse the structured `description` field within each file object in `aggregated_files_json` to extract the `TaskDesc:` value. Use this extracted `TaskDesc` to link media files to the correct design options by matching it to the current option's theme/description.**
2.  Ensure each dictionary in your `options_data` list accurately reflects one distinct design option with its **unique sequential `option_id`**, its specific theme, architecture type, and **its OWN EXCLUSIVE textual summary, and accurately associated image files (`image_paths`) and video files (`video_paths`) based on the described `TaskDesc` (extracted from the file's structured `description`) matching logic.**
3.  The number of items in `options_data` should match the number of distinct design concepts you can identify and reasonably extract from the inputs.

**Output:** Return ONLY the final JSON input dictionary. No other text.
Language: {llm_output_language}
""",
                        input_variables=[
                            "selected_agent",
                            "current_task_objective",
                            "user_input", "full_task_summary",
                            "aggregated_outputs_json",
                            "aggregated_files_json",
                            "llm_output_language",
                            "options_data_json", 
                            "budget_limit_overall" 
                        ]
                    ),
                    "generate_final_criteria": PromptConfig(
                        template="""
You are defining the HOLISTIC evaluation criteria/rubric for a completed workflow (**`FinalEvaAgent`**) OR defining comparative criteria/rubric for choosing the best among multiple options (**`SpecialEvaAgent`**).

**Current Task Agent:** `{selected_agent}`
**Current Task Objective:** {current_task_objective}
**Overall Workflow Goal (User Request - CRITICAL SOURCE for evaluation priorities):** {user_input}
**Architecture Type for primary focus (if applicable, e.g., for tailoring rubric):** {architecture_type}

**Full Workflow Task Summary (History - contains analysis like site/case studies, and outputs of options):**
{full_task_summary}
**Parsed Options Data (JSON string representing a list of option dictionaries, including summaries, image/video paths if any, cost/green estimates if any, AND `raw_outputs_for_llm_parsing` which may contain `mcp_internal_messages` from agents like RhinoMCPCoordinator):**
```json
{options_to_evaluate_json}
```
**Context from RAG/Search (if gathered, e.g., specific green building standards, cost benchmarks):**
{rag_context}
{search_context}

**Your Task:** Generate specific criteria and a **detailed scoring rubric/guideline** suitable for the evaluation agent type.

**CRITICAL FOR COST & FEASIBILITY ASSESSMENT (if applicable):**
*   Carefully examine the `raw_outputs_for_llm_parsing.mcp_internal_messages` within each option in `{options_to_evaluate_json}`.
*   Pay special attention to early messages from AI agents (e.g., `RhinoMCPCoordinator`'s initial plan or intent messages). These messages might reveal crucial details about proposed materials, construction complexity, or specific geometric goals that directly impact cost and feasibility.
*   Incorporate these insights when defining criteria for '早期成本效益估算' and '機能性與適應彈性'.

**IF `{selected_agent}` is `FinalEvaAgent`:**
*   **Criteria Should Address:** Goal Achievement (alignment with `{user_input}`), Quality of Final Artifacts (from `{options_to_evaluate_json}` or summary), Process Efficiency/Logic (Optional). Incorporate insights from RAG/Search if relevant. Consider `{architecture_type}` if provided.
*   **Scoring Rubric:** Define a **detailed** 1-10 scale rubric with clear descriptions for each score range (e.g., 1-3, 4-6, 7-8, 9-10) relating to goal achievement and artifact quality, informed by context. The rubric should cover the six core dimensions:
    1.  **使用者目標回應性**: 設計是否有針對特定用途（住宅、學校、醫療、文化等）與業主目標做出清楚的空間與功能回應。
    2.  **美學與場域關聯性**: 初步檢視設計在造型語彙、尺度、材質的特殊性，以及與在地文化及環境脈絡間的關聯性。
    3.  **機能性與適應彈性**: 評估空間配置是否合理、流線否清楚，並兼具未來調整的可能性（如增建、模組化、分區調整等）。
    4.  **耐久性與維護性**: 材料、構造與造型操作形式的初步選擇是否符合耐用、易保養的原則？是否避免過度依賴昂貴維護或脆弱結構？
    5.  **早期成本效益估算**: 以面積、單位造價或歷史案例作為初步估價基礎，檢查方案是否合理落在預算範圍內。
    6.  **綠建築永續潛力**: 評估設計在生態、健康、節能、減廢等面向的潛力。

**IF `{selected_agent}` is `SpecialEvaAgent`:**
*   **Comparative Criteria MUST Address Multiple Dimensions based on `{options_to_evaluate_json}` (each item is an option):**
    *   **Identify Key Dimensions from `{user_input}`, `{full_task_summary}`, and the nature of options in `{options_to_evaluate_json}`.** Determine which aspects are most important for comparing the options (e.g., user explicitly mentioned "low cost," "innovative facade," or options clearly differ in style, function, etc.).
    *   **Core Architectural Dimensions for Comparison (Consider if relevant for the `{architecture_type}` and options):**
        1.  **使用者目標（建築類型）回應性**: How well each option addresses the specific use (residential, school, healthcare, cultural, etc.) and owner's goals with clear spatial and functional responses.
        2.  **美學與場域關聯性**: Uniqueness in form, scale, materials, and connection to local culture and environmental context.
        3.  **機能性與適應彈性**: Rationality of spatial layout, clarity of circulation, and potential for future adjustments (extensions, modularity, rezoning). **Consider `mcp_internal_messages` for feasibility and alignment with Rhino's planned actions.**
        4.  **耐久性與維護性**: Choice of materials, construction methods, and forms for durability and ease of maintenance; avoidance of over-reliance on expensive upkeep or fragile structures.
        5.  **早期成本效益估算**: Based on area, unit cost, or historical examples, check if the scheme is reasonably within budget. Relate to budget if provided. **Crucially, use `mcp_internal_messages` to infer material choices or construction complexity that would affect cost.**
        6.  **綠建築永續潛力**: Potential in areas like ecology, health, energy saving, and waste reduction. Use estimated green building percentages if available, or qualitatively assess passive design, materials, etc.
    *   The rubric should allow scoring each option from 1-10 on these dimensions.
*   **Develop Discriminatory Metrics/Descriptions for Each Dimension:** For each chosen dimension, describe how to compare the options. Aim for more than just "good/bad."
    *   Example for 早期成本效益估算: "Option A (Cost: $X) - Most cost-effective, aligns with initial MCP intent for simple construction. Option B (Cost: $Y) - Highest cost, MCP messages indicated complex custom geometry. Option C (Cost: $Z) - Mid-range."
    *   Example for 綠建築永續潛力: "Option A (Green %: X) - Good solar shading. Option B (Green %: Y) - Large glass areas."
*   **Scoring Rubric & Selection:**
    *   The main output of the evaluation LLM using this rubric will be per-option scores for these dimensions (e.g., `user_goal_responsiveness_score_llm`, `aesthetics_context_score_llm`, etc. for each option).
    *   The overall `assessment` ("Score (1-10)") for the `SpecialEvaAgent` task should reflect the quality/fit of the *best* option identified after comparing all options against this rubric.
    *   The rubric should guide the LLM to provide a `selected_option_identifier`.
    *   The final feedback should clearly explain the reasoning for the selection across the different dimensions.

**Output:** Output ONLY the criteria and the detailed scoring rubric/guideline as clear text, tailored to the agent type and informed by available context. Emphasize quantifiable or clearly comparable metrics where possible for `SpecialEvaAgent`. The rubric should be detailed enough for an LLM to assign scores (1-10) to each option across the defined dimensions.
Respond in {llm_output_language}.
""",
                        input_variables=[
                            "selected_agent",
                            "current_task_objective",
                            "user_input", "full_task_summary", 
                            "options_to_evaluate_json", # This comes from current_task.task_inputs.options_data
                            "architecture_type", # Added: To help tailor criteria/rubric
                            "rag_context", "search_context",
                            "llm_output_language"
                        ]
                    ),
                    "evaluate_option_with_image_tool": PromptConfig(
                        template="""You are an expert architecture design evaluator.
Evaluate the following design option based on the provided text, images (if any), and specific criteria.

**Option ID:** {option_id}
**Option Description:**
{option_description}

**Textual Summary from Option's Outputs:**
{textual_summary}
{architecture_type_info}
**Initial Estimated Cost (if known, for reference):** {initial_estimated_cost}
**Initial Green Building Potential % (if known, for reference):** {initial_green_building_percentage}

**Specific Evaluation Criteria/Rubric (Apply this to the image content):**
{specific_criteria}

**Image Paths (THESE are the images to evaluate):** {image_paths_str}

**Green Building Context (for estimating `green_building_potential_percentage`):**
The green building assessment considers four major categories:
1.  **Ecology (Max 27 points):** Focuses on biodiversity, green coverage, and site water retention.
2.  **Health (Max 25 points):** Considers indoor environmental quality, water resources, and waste management.
3.  **Energy Saving (Max 32 points):** Includes building envelope performance, HVAC efficiency, and lighting efficiency for daily energy use.
4.  **Waste Reduction (Max 16 points):** Targets CO2 emission reduction and construction/demolition waste reduction.
A total of 100 points are possible. Your estimate should reflect how well the VISUALS suggest performance in these areas.

**Your Task:**
1.  Analyze all provided information (text, and ESPECIALLY the images at `image_paths_str`).
2.  Based on the VISUALS and textual context, for each of the design-focused dimensions below, provide a score from 1 to 10 (decimals allowed) and a brief justification referencing visual evidence:
    *   `user_goal_responsiveness_score_llm`: How well the VISUALS fulfill user goals and architectural type requirements.
    *   `aesthetics_context_score_llm`: Visual appeal, uniqueness, and contextual fit evident in the IMAGES.
    *   `functionality_flexibility_score_llm`: Functionality, layout, and adaptability as can be INFERRED PURELY FROM THE IMAGES.
    *   `durability_maintainability_score_llm`: Implied durability and ease of maintenance from VISUALS of materials and forms.
3.  Provide your best estimate for:
    *   `estimated_cost`: A numerical value for the total project cost, INFERRED FROM VISUAL COMPLEXITY AND IMPLIED MATERIALS in the images. (This aligns with '早期成本效益估算')
    *   `green_building_potential_percentage`: A numerical value (0-100) representing its overall green building potential, as SUGGESTED BY THE VISUALS (e.g., solar panels visible, green roofs, window types, shading). (This aligns with '綠建築永續潛力')
4.  Provide overall textual feedback (`llm_feedback_text`) summarizing your assessment of this option based PRIMARILY ON THE IMAGES.

**Output Format (Strictly JSON):**
Return a single JSON object with the following keys:
"option_id": "{option_id}",
"user_goal_responsiveness_score_llm": <float_score_1_to_10>,
"aesthetics_context_score_llm": <float_score_1_to_10>,
"functionality_flexibility_score_llm": <float_score_1_to_10>,
"durability_maintainability_score_llm": <float_score_1_to_10>,
"estimated_cost": <float_cost_value>,
"green_building_potential_percentage": <float_percentage_0_to_100>,
"llm_feedback_text": "<string_detailed_feedback_focused_on_visuals>"

Respond in {llm_output_language}.
""",
                        input_variables=[
                            "option_id", "option_description", "textual_summary",
                            "architecture_type_info", "initial_estimated_cost",
                            "initial_green_building_percentage", "specific_criteria",
                            "image_paths_str", "llm_output_language"
                        ]
                    ),
                    "evaluate_option_with_video_tool": PromptConfig(
                        template="""You are an expert architecture design evaluator.
Evaluate the following design option based on the provided text, video, and specific criteria.

**Option ID:** {option_id}
**Option Description:**
{option_description}

**Textual Summary from Option's Outputs:**
{textual_summary}
{architecture_type_info}
**Initial Estimated Cost (if known, for reference):** {initial_estimated_cost}
**Initial Green Building Potential % (if known, for reference):** {initial_green_building_percentage}

**Specific Evaluation Criteria/Rubric (Apply this to the video content):**
{specific_criteria}

**Video Path for this evaluation (THIS is the video to evaluate):** {video_path_to_eval}

**Green Building Context (for estimating `green_building_potential_percentage`):**
The green building assessment considers four major categories:
1.  **Ecology (Max 27 points):** Focuses on biodiversity, green coverage, and site water retention.
2.  **Health (Max 25 points):** Considers indoor environmental quality, water resources, and waste management.
3.  **Energy Saving (Max 32 points):** Includes building envelope performance, HVAC efficiency, and lighting efficiency for daily energy use.
4.  **Waste Reduction (Max 16 points):** Targets CO2 emission reduction and construction/demolition waste reduction.
A total of 100 points are possible. Your estimate should reflect how well the VIDEO CONTENT suggests performance in these areas.

**Your Task:**
1.  Analyze all provided information (text, and ESPECIALLY the video at `video_path_to_eval`).
2.  Based on the VIDEO CONTENT and textual context, for each of the design-focused dimensions below, provide a score from 1 to 10 (decimals allowed) and a brief justification referencing visual evidence from the video (cite timestamps if helpful):
    *   `user_goal_responsiveness_score_llm`: How well the VIDEO CONTENT fulfills user goals and architectural type requirements.
    *   `aesthetics_context_score_llm`: Visual appeal, uniqueness, and contextual fit evident in the VIDEO.
    *   `functionality_flexibility_score_llm`: Functionality, layout, and adaptability as can be INFERRED PURELY FROM THE VIDEO (layout, spatial qualities, etc.).
    *   `durability_maintainability_score_llm`: Implied durability and ease of maintenance from VISUALS of materials and forms in the VIDEO.
3.  Provide your best estimate for:
    *   `estimated_cost`: A numerical value for the total project cost, INFERRED FROM VISUAL COMPLEXITY AND IMPLIED MATERIALS in the video. (This aligns with '早期成本效益估算')
    *   `green_building_potential_percentage`: A numerical value (0-100) representing its overall green building potential, as SUGGESTED BY THE VIDEO CONTENT (e.g., solar panels visible, green roofs, window types, shading). (This aligns with '綠建築永續潛力')
4.  Provide overall textual feedback (`llm_feedback_text`) summarizing your assessment of this option based PRIMARILY ON THE VIDEO. Include key timestamps if they support your points.

**Output Format (Strictly JSON):**
Return a single JSON object with the following keys:
"option_id": "{option_id}",
"user_goal_responsiveness_score_llm": <float_score_1_to_10>,
"aesthetics_context_score_llm": <float_score_1_to_10>,
"functionality_flexibility_score_llm": <float_score_1_to_10>,
"durability_maintainability_score_llm": <float_score_1_to_10>,
"estimated_cost": <float_cost_value>,
"green_building_potential_percentage": <float_percentage_0_to_100>,
"llm_feedback_text": "<string_detailed_feedback_focused_on_video_content_with_timestamps>"

Respond in {llm_output_language}.
""",
                        input_variables=[
                            "option_id", "option_description", "textual_summary",
                            "architecture_type_info", "initial_estimated_cost",
                            "initial_green_building_percentage", "specific_criteria",
                            "video_path_to_eval", "llm_output_language"
                        ]
                    )
                },
                parameters={}
            ),
            "qa_agent": AgentConfig(
                agent_name="qa_agent",
                description="Handles user interaction in the QA phase after task execution.",
                llm=default_llm, # Keep QA as OpenAI unless specified
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
    "DEFAULT(gpt-4o-mini)",  # New option to use code-defined default
    # OpenAI
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
    "claude-3-5-sonnet-20240620", # Corrected name
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
        description="Global default language for LLM responses. Select 'USE_DEFAULT_MODEL' for an agent's model to use its code-defined default LLM."
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
        default="gemini-2.0-flash",
        title="Process Management: Model Name",
        description="Select LLM Model. 'USE_DEFAULT_MODEL' applies code default."
    )
    pm_temperature: float = Field(
        default=_base_default_config_obj.agents.get("process_management").llm.temperature 
        if _base_default_config_obj.agents.get("process_management") else 0.7,
        title="Process Management: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Process Management Agent (0.0-1.0)."
    )
    pm_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("process_management").llm.max_tokens 
        if _base_default_config_obj.agents.get("process_management") else None,
        title="Process Management: Max Tokens",
        gt=0,
        description="Max Tokens for the Process Management Agent (Optional)."
    )

    # --- Assign Agent LLM Config (Using Single Literal) ---
    aa_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gemini-2.5-pro-exp-03-25",
        title="Assign Agent: Model Name",
        description="Select LLM Model. 'USE_DEFAULT_MODEL' applies code default."
    )
    aa_temperature: float = Field(
        default=_base_default_config_obj.agents.get("assign_agent").llm.temperature 
        if _base_default_config_obj.agents.get("assign_agent") else 0.7,
        title="Assign Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Assign Agent (0.0-1.0)."
    )
    aa_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("assign_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("assign_agent") else None,
        title="Assign Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the Assign Agent (Optional)."
    )

    # --- Tool Agent LLM Config (Using Single Literal) ---
    ta_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="DEFAULT(gpt-4o-mini)",
        title="Tool Agent: Model Name (Errors)",
        description="Select LLM Model for error analysis. 'USE_DEFAULT_MODEL' applies code default."
    )
    ta_temperature: float = Field(
        default=_base_default_config_obj.agents.get("tool_agent").llm.temperature 
        if _base_default_config_obj.agents.get("tool_agent") else 0.7,
        title="Tool Agent: Temperature (Errors)",
        ge=0.0, le=1.0,
        description="Temperature for the Tool Agent's error analysis (0.0-1.0)."
    )
    ta_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("tool_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("tool_agent") else None,
        title="Tool Agent: Max Tokens (Errors)",
        gt=0,
        description="Max Tokens for the Tool Agent's error analysis (Optional)."
    )

    # --- Evaluation Agent LLM Config (Using Single Literal) ---
    ea_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gemini-2.5-pro-exp-03-25",
        title="Evaluation Agent: Model Name",
        description="Select LLM Model. 'USE_DEFAULT_MODEL' applies code default."
    )
    ea_temperature: float = Field(
        default=_base_default_config_obj.agents.get("eva_agent").llm.temperature 
        if _base_default_config_obj.agents.get("eva_agent") else 0.2,
        title="Evaluation Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the Evaluation Agent (0.0-1.0)."
    )
    ea_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("eva_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("eva_agent") else None,
        title="Evaluation Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the Evaluation Agent (Optional)."
    )

    # --- QA Agent LLM Config ---
    qa_model_name: AllModelNamesLiteral = Field(
        default="DEFAULT(gpt-4o-mini)",
        title="QA Agent: Model Name",
        description="Select LLM Model for the QA Agent. 'USE_DEFAULT_MODEL' applies code default."
    )
    qa_temperature: float = Field(
        default=_base_default_config_obj.agents.get("qa_agent").llm.temperature 
        if _base_default_config_obj.agents.get("qa_agent") else 0.7,
        title="QA Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the QA Agent (0.0-1.0)."
    )
    qa_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("qa_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("qa_agent") else None,
        title="QA Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the QA Agent (Optional)."
    )

    # --- Prompts ---
    aa_prepare_tool_inputs_prompt: str = Field(
        default=get_base_default_prompt("assign_agent", "prepare_tool_inputs_prompt"), # Use the updated default function
        title="Assign Agent: Prepare Tool Inputs Prompt",
        description="Prompt for Assign Agent to prepare structured inputs for the next tool/agent. Includes: {user_input}, {workflow_history_summary}, {aggregated_outputs_json}, {aggregated_files_json}, {task_objective}, {task_description}, {selected_agent_name}, {agent_description}, {error_feedback}, {user_budget_limit}, {latest_evaluation_results_json}, {llm_output_language}. Clear field to use runtime default.", # Updated description to include new variable
        extra={'widget': {'type': 'textarea'}}
    )

    ea_prepare_final_evaluation_inputs_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "prepare_final_evaluation_inputs"),
        title="Evaluation Agent: Prepare FINAL/SPECIAL Eval Inputs Prompt",
        description="Prompt to gather outputs/artifacts for FINAL holistic review or SPECIAL multi-option comparison. Clear field to use runtime default.",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluation_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluation"),
        title="Evaluation Agent: Evaluation Prompt (Handles ALL Types)",
        description="The core prompt for performing evaluation (Standard, Final, or Special). Clear field to use runtime default.",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluate_option_with_image_tool_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluate_option_with_image_tool"),
        title="Evaluation Agent: Image Tool Option Evaluation Prompt",
        description="Prompt for image recognition tool to evaluate a single design option (generates initial scores). Clear field to use runtime default.",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluate_option_with_video_tool_prompt: str = Field(
         default=get_base_default_prompt("eva_agent", "evaluate_option_with_video_tool"),
         title="Evaluation Agent: Video Tool Option Evaluation Prompt",
         description="Prompt for video recognition tool to evaluate a single design option (generates initial scores). Clear field to use runtime default.",
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
def initialize_llm(llm_config_dict: Dict[str, Any], agent_name_for_default_lookup: Optional[str] = None) -> Any:
    """Initializes the LangChain LLM based on a configuration dictionary, inferring the provider."""
    # --- Handle "USE_DEFAULT" ---
    model_name_from_config = llm_config_dict.get("model_name", "gpt-4o-mini")

    if model_name_from_config == "DEFAULT(gpt-4o-mini)":
        if agent_name_for_default_lookup:
            default_agent_llm_config = None
            if _base_default_config_obj and _base_default_config_obj.agents:
                agent_default = _base_default_config_obj.agents.get(agent_name_for_default_lookup)
                if agent_default and agent_default.llm:
                    default_agent_llm_config = agent_default.llm
            
            if default_agent_llm_config:
                print(f"Info: '{agent_name_for_default_lookup}' is using its code-defined default LLM: {default_agent_llm_config.model_name} (Temp: {default_agent_llm_config.temperature})")
                # Override the input dict with the agent's default LLM config
                llm_config_dict = default_agent_llm_config.model_dump()
            else:
                print(f"Warning: 'USE_DEFAULT' specified for agent '{agent_name_for_default_lookup}', but its default LLM config not found in code. Falling back to 'gpt-4o-mini'.")
                llm_config_dict = {"model_name": "gpt-4o-mini", "temperature": 0.7} # Sensible fallback
        else:
            print("Warning: 'USE_DEFAULT' specified but no 'agent_name_for_default_lookup' provided. Falling back to 'gpt-4o-mini'.")
            llm_config_dict = {"model_name": "gpt-4o-mini", "temperature": 0.7} # Sensible fallback
    
    # Now, model_name is the actual model_name to use, either from original config or default
    model_name = llm_config_dict.get("model_name", "gpt-4o-mini") # Re-get after potential override

    # --- Infer Provider from Model Name ---
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