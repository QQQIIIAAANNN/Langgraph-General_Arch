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
            chunk_size=1000,
            chunk_overlap=200,
            retriever_k=5
        )

        # Default LLM Config (can be reused)
        default_llm = ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7)

        default_agents = {
                "process_management": AgentConfig(
                    agent_name="process_management",
                    description="Responsible for planning and managing the task workflow.",
                    llm=default_llm.copy(), # Use copy to avoid modification issues
                    prompts={
                        "create_workflow": PromptConfig(  # MODIFIED
                            template="""
You are a meticulous and **detail-oriented** workflow planner specializing in architecture and design tasks. Your goal is to break down a user's request into a **sequence of granular, executable task objectives** for a team of specialized agents, ensuring smooth data flow between steps.

User Request: {user_input}

Analyze the request and generate a **complete, logical, and DETAILED sequence of tasks** to fulfill it. Pay close attention to dependencies and insert necessary intermediate processing steps.

**Key Planning Principles:**
1.  **Dependencies:** If Task B requires data generated by Task A (e.g., a file path, summarized text), ensure Task A comes first.
2.  **Intermediate Processing:** If the raw output of Task A is not directly usable as input for Task B, **YOU MUST INSERT an `LLMTaskAgent` task between them.** This intermediate task should perform actions like summarizing, extracting, reformatting, generating prompts, or analyzing results for the next step.
3.  **Multiple Options:** Plan multiple generation tasks *before* planning an evaluation if comparison is needed.
4.  **Clarity:** Each task objective must be specific and actionable.
5.  **Evaluation:** Tasks needing analysis or assessment must be assigned to `EvaAgent` with `requires_evaluation=true`.

For **each** task in the sequence, you MUST specify:
1.  `description`: High-level goal for this step (in {llm_output_language}).
2.  `task_objective`: Specific outcome and method needed from the agent.
3.  `inputs`: JSON object suggesting initial data needs (e.g., `{{"prompt": "..."}}`, `{{"image_path": "{{output_from_task_id_abc.filename}}"}}`, `{{"user_request": "..."}}`). Use placeholders like `{{output_from_task_id_xyz.key}}` where applicable. For files, suggest the key (`image_paths`, `video_paths`, `image_path`, `render_image`, `initial_image_path`). If a filename is known *from a specific previous output*, suggest it. Otherwise, indicate the need (e.g., `"REQUIRES_IMAGE_FROM_PREVIOUS_STEP"`). **Do not invent paths.** Use `{{"}}` if no input suggestion applies. NEVER use `null`.
4.  `requires_evaluation`: Boolean (`true`/`false`). Set to `true` for tasks needing analysis/assessment.
5.  `selected_agent`: **(CRITICAL)** The **exact name** of the agent from the list below best suited for the `task_objective`. Mandatory for every task.

--- BEGIN AGENT CAPABILITIES ---
**Available Agent Capabilities & *Primary* Expected Prepared Input Keys:**
*   `ArchRAGAgent`: Information Retrieval (Regulations, Expertise) -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`.
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: General Image Generation/Editing (Gemini) -> Needs `prompt`, optional `image_inputs`. Good for visual exploration, detailed textures.
*   `WebSearchAgent`: Web Search (Cases, General Info) -> Needs `prompt`.
*   `CaseRenderAgent`: Architectural Rendering (ComfyUI) -> Needs `outer_prompt`, `i` (int), `strength` (string "0.0"-"0.8").
*   `Generate3DAgent`: 3D MESH Model (.glb) Generation (Comfy3D) -> Needs `image_path` (string). Suited for **exploratory form-finding or generating detailed but less precise geometry**.
*   `RhinoMCPCoordinator`: **Parametric/Precise 3D Modeling (Rhino)** -> Needs `user_request` (string command), optional `initial_image_path` (string path). Ideal for **multi-step Rhino operations, tasks requiring precise coordinates, dimensions, spatial relationships, or modifications to existing geometry**. Coordinates internal planning and tool calls within Rhino.
*   `SimulateFutureAgent`: Future Scenario Simulation (ComfyUI) -> Needs `outer_prompt`, `render_image` (string filename).
*   `LLMTaskAgent`: **Intermediate Processing & General Text Tasks** -> Needs `prompt`. Crucial for summarizing, extracting, reformatting, or generating prompts for other agents.
*   `EvaAgent`: **General Evaluation Agent** -> Used for evaluating task outputs and final holistic review. Requires inputs prepared by `prepare_evaluation_inputs` or `prepare_final_evaluation_inputs`.
*   `final_evaluation`: Task objective for final holistic review, typically assigned to `EvaAgent`.
--- END AGENT CAPABILITIES ---

**Example Intermediate Step (RAG -> LLM -> Image Gen):**
1.  Task: `ArchRAGAgent` to find info. Inputs suggestion: `{{"prompt": "..."}}`
2.  Task: `LLMTaskAgent` to process RAG output. Inputs suggestion: `{{"prompt": "Summarize Task 1 output {{output_from_task_id_1.content}} and create an image prompt."}}`
3.  Task: `ImageGenerationAgent` using LLM's output. Inputs suggestion: `{{"prompt": "{{output_from_task_id_2.content}}"}}`

Return the entire workflow as a **single, valid JSON list** object. Do NOT include any explanatory text before or after the JSON list. Ensure perfect JSON syntax, detailed steps, and that **every task includes the `selected_agent` key.**
Respond in {llm_output_language}.
""",
                            input_variables=["user_input", "llm_output_language"]
                        ),
                        "failure_analysis": PromptConfig( # MODIFIED
                            template="""
Context: A task in an automated workflow has FAILED. Analyze the provided context to determine the failure type and suggest the best course of action aimed at RESOLVING the failure.

**Failure Context:** {failure_context}

**Failed Task Details:**
Agent Originally Assigned: {selected_agent_name}
Task Description: {task_description}
Original Task Objective: {task_objective}
Original Task Inputs (Suggestion): {inputs_json}

Last Execution Error Log: {execution_error_log}
Last Feedback Log (Includes Eval Results like 'Assessment: Fail' if applicable): {feedback_log}

--- BEGIN AVAILABLE AGENT CAPABILITIES (for Fallback Task Generation) ---
*   `ArchRAGAgent`: Information Retrieval -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`.
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: Image Generation/Editing (Gemini) -> Needs `prompt`, optional `image_inputs`.
*   `WebSearchAgent`: Web Search -> Needs `prompt`.
*   `CaseRenderAgent`: Architectural Rendering (ComfyUI) -> Needs `outer_prompt`, `i` (int), `strength` (string "0.0"-"0.8").
*   `Generate3DAgent`: 3D MESH Model Generation (Comfy3D) -> Needs `image_path` (string).
*   `RhinoMCPCoordinator`: Parametric/Precise 3D Modeling (Rhino) -> Needs `user_request`, optional `initial_image_path`.
*   `SimulateFutureAgent`: Future Scenario Simulation (ComfyUI) -> Needs `outer_prompt`, `render_image` (string filename).
*   `LLMTaskAgent`: Intermediate Processing & General Text Tasks -> Needs `prompt`.
--- END AVAILABLE AGENT CAPABILITIES ---

**Analysis Steps & Action Selection:**

1.  **Identify Failure Type:** Determine if 'evaluation' (check `feedback_log` for "Assessment: Fail") or 'execution'.
2.  **Determine Action:** Based on failure type, logs, max retries, and original inputs/objective, choose ONE action.

**Action Options:**

*   **`FALLBACK_GENERAL`:** Propose a new task/sequence to fix the issue or try an alternative approach.
    *   **If Failure Type is 'evaluation'**: MUST return `new_tasks_list` with TWO tasks.
        1.  **Task 1 (Alternative Generation/Execution):** Create a task based on `feedback_log`. Choose an appropriate agent **EXCEPT `EvaAgent`**. If `feedback_log` mentions geometric precision/modification issues, consider `RhinoMCPCoordinator`. If it needs image generation, consider `ImageGenerationAgent` or `CaseRenderAgent`. If text processing, use `LLMTaskAgent`. Define a **new `task_objective`** addressing the feedback. Suggest inputs `{{"}}`. Set `requires_evaluation=false`.
        2.  **Task 2 (Re-Evaluation):** Create an `EvaAgent` task. Set `selected_agent="EvaAgent"`, `task_objective="Evaluate outcome of the alternative task"`, `inputs={{}}`, `requires_evaluation=true`.
    *   **If Failure Type is 'execution'**: Return `new_task` or `new_tasks_list`. Analyze `execution_error_log`. If it's an input error (e.g., bad path, wrong parameter), consider suggesting `LLMTaskAgent` first to prepare better inputs for the original agent, or switch to `RhinoMCPCoordinator` if precision is key. Define objective(s) to overcome the error. Suggest inputs `{{"}}`. Keep original `requires_evaluation` (`{original_requires_evaluation}`) unless logical.
    *   Output JSON: `{{"action": "FALLBACK_GENERAL", ...}}`

*   **`MODIFY`:** Modify the *current* failed task and retry it.
    *   Appropriate for minor input errors (typos, slightly wrong parameters) identifiable in `execution_error_log` or `feedback_log`, or technical glitches.
    *   Suggest modifications via `modify_description` or `modify_objective` in the output JSON.
    *   Output JSON: `{{"action": "MODIFY", "modify_description": "...", "modify_objective": "..."}}`

*   **`SKIP`:** (Available Always). Skip the failed task.
    *   Output JSON: `{{"action": "SKIP"}}`

**Instructions:**
*   If **Is Max Retries Reached?** is `True`, you **MUST** choose `SKIP`.
*   Choose **only one action**.
*   Provide **only** the single JSON output for your chosen action.
*   Ensure agent selections and objectives directly address the failure analysis.
*   Use language: {llm_output_language}
""",
                            input_variables=[
                                "failure_context", "is_max_retries", "max_retries",
                                "selected_agent_name", "task_description", "task_objective",
                                "inputs_json", # Added original inputs for context
                                "execution_error_log",
                                "feedback_log",
                                "llm_output_language",
                                "original_requires_evaluation"
                            ]
                        ),
                        "process_interrupt": PromptConfig( # MODIFIED
                            template="""
You are a meticulous workflow manager reacting to a user interrupt during task execution. Analyze the user's interrupt request in the context of the overall goal and the currently planned task sequence. Decide the most appropriate action to take.

**Overall Workflow Goal (User Request):** {user_input}
**User Interrupt Request:** {interrupt_input}
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
*   `ImageGenerationAgent`: Image Generation/Editing (Gemini) -> Needs `prompt`, optional `image_inputs`.
*   `WebSearchAgent`: Web Search -> Needs `prompt`.
*   `CaseRenderAgent`: Architectural Rendering (ComfyUI) -> Needs `outer_prompt`, `i`, `strength`.
*   `Generate3DAgent`: 3D MESH Model Generation (Comfy3D) -> Needs `image_path`.
*   `RhinoMCPCoordinator`: Parametric/Precise 3D Modeling (Rhino) -> Needs `user_request`, optional `initial_image_path`.
*   `SimulateFutureAgent`: Future Scenario Simulation (ComfyUI) -> Needs `outer_prompt`, `render_image`.
*   `LLMTaskAgent`: Intermediate Processing & General Text Tasks -> Needs `prompt`.
*   `EvaAgent`: General Evaluation Agent.
*   `final_evaluation`: Task objective for final holistic review.
--- END AGENT CAPABILITIES ---

**Your Task:** Choose ONE action based on the interrupt:

1.  **`PROCEED`**: Interrupt doesn't require plan changes. Continue from `current_task_index`. Output: `{{"action": "PROCEED"}}`
2.  **`INSERT_TASKS`**: Insert new tasks **after** `current_task_index`. Generate a list of new tasks (TaskState structure: `description`, `task_objective`, `selected_agent`, suggested `inputs`, `requires_evaluation`). Use Agent Capabilities. Output: `{{"action": "INSERT_TASKS", "insert_tasks_list": [ {{...task1...}}, ... ] }}`
3.  **`REPLACE_TASKS`**: Redesign workflow from `current_task_index` onwards (preserves completed tasks before index). Generate a **new sequence** for remaining tasks. Output: `{{"action": "REPLACE_TASKS", "new_tasks_list": [ {{...task1...}}, ... ] }}`
4.  **`CONVERSATION`**: Interrupt is unclear, ambiguous, requires discussion, or explicitly asks to discuss. Routes to QA Agent. Output: `{{"action": "CONVERSATION"}}`

**Instructions:**
*   Analyze interrupt: Is it a clear plan change (`INSERT_TASKS`/`REPLACE_TASKS`), minor clarification (`PROCEED`), or needs discussion (`CONVERSATION`)?
*   Ensure generated tasks follow TaskState structure.
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
                    description="Selects the appropriate specialized agent for a given task objective.",
                    llm=default_llm.copy(),
                    prompts={
                        # "select_agent_prompt": REMOVED (Handled by PM)
                        "prepare_tool_inputs_prompt": PromptConfig( # MODIFIED
                            template="""
You are an expert input preprocessor for specialized AI tools. Your goal is to take a high-level task objective, the overall user request, task history/outputs, and information about the selected tool, then generate the precise JSON input dictionary required by that specific tool using standardized keys.

**Selected Tool:** `{selected_agent_name}`
**Tool Description:** {agent_description}
**Current Task Description:** {task_description}
**Current Task Objective:** {task_objective}
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
*   `prompt`: (String) Textual query/instruction. REQUIRED for: `ArchRAGAgent`, `WebSearchAgent`, `ImageRecognitionAgent`, `VideoRecognitionAgent`, `ImageGenerationAgent`, `LLMTaskAgent`.
*   `image_paths`: (List[String]) FULL paths to existing input images. REQUIRED for `ImageRecognitionAgent`. **Find VALID paths by parsing `aggregated_files_json`.**
*   `video_paths`: (List[String]) FULL paths to existing input videos. REQUIRED for `VideoRecognitionAgent`. **Find VALID paths by parsing `aggregated_files_json`.**
*   `image_path`: (String) FULL path to a SINGLE existing input image. REQUIRED for `Generate3DAgent`. **Find VALID path by parsing `aggregated_files_json`.**
*   `outer_prompt`: (String) Textual prompt. REQUIRED for `CaseRenderAgent`, `SimulateFutureAgent`.
*   `i`: (Integer, Default: 1) Number of images (> 0). For `CaseRenderAgent`.
*   `strength`: (String, Default: "0.5") Rendering strength ("0.0"-"1.0"). For `CaseRenderAgent`.
*   `render_image`: (String) FILENAME of the base image (in RENDER_CACHE_DIR). REQUIRED for `SimulateFutureAgent`. **Find correct FILENAME by parsing `aggregated_files_json` (look for files from previous renders).**
*   `image_inputs`: (List of Strings, Optional) Input image paths/data for `ImageGenerationAgent`. Find paths in `aggregated_files_json`.
*   `user_request`: (String) **Specific instruction/command for the MCP tool.** REQUIRED for `RhinoMCPCoordinator`. Generate based on `task_objective`, `task_description`, and `aggregated_outputs_json` if needed.
*   `initial_image_path`: (String, Optional) FULL path to an image to be used as initial input/reference by `RhinoMCPCoordinator`. **Find VALID path by parsing `aggregated_files_json`** if the objective requires it.

**Instructions:**
1.  Analyze the **Current Task Objective/Description**, **Selected Tool**, and **Overall Goal**.
2.  **Generate/Extract `prompt`, `outer_prompt`, or `user_request`**:
    *   Generate the primary textual input based on the Objective/Description/Goal.
    *   **For `RhinoMCPCoordinator`**: Synthesize a clear, actionable command for Rhino based on the task objective and description. This is the `user_request`.
    *   If the task involves analyzing previous *textual* results, parse `aggregated_outputs_json` and incorporate relevant info into the prompt/user_request.
    *   **For `ImageGenerationAgent`**: If based on previous text, parse `aggregated_outputs_json` to create a rich image generation prompt.
3.  **Determine and Validate File Paths/Filenames (CRITICAL - Use Aggregated Data ONLY)**:
    *   If the tool requires file inputs (`image_paths`, `video_paths`, `image_path`, `render_image`, `initial_image_path`):
        *   **Parse `aggregated_files_json`.** Identify files potentially relevant to the `{task_objective}`. Use `source_task_id` and context if needed.
        *   Select the most appropriate file(s).
        *   For path keys (`image_paths`, `video_paths`, `image_path`, `initial_image_path`), extract the `path` value.
        *   For `render_image`, extract the `filename`.
        *   **Verify paths exist** using the `path` value (or constructed path for `render_image`).
        *   **If a required path/filename cannot be found/validated, RETURN THE ERROR JSON (step 7).** Do NOT invent paths.
4.  **Handle Specific Parameters (`i`, `strength`, `top_k`)**: Extract/validate from suggestion/objective.
5.  **Handle Retries**: Use `{error_feedback}` to modify inputs if appropriate.
6.  **Construct Final JSON**: Create the final JSON dictionary using only the required standardized keys for `{selected_agent_name}`.
7.  **Error Handling**: If critical inputs are missing/invalid (esp. unvalidated paths), return error JSON: `{{"error": "Missing/Invalid critical input [specify] for {selected_agent_name}, could not find/validate required data in aggregated history."}}`

**Output:** Return ONLY the final JSON input dictionary for the tool, or the error JSON. No other text.
Language: {llm_output_language}
""",
                            input_variables=[
                                "selected_agent_name", "agent_description", "user_input",
                                "task_objective", "task_description",
                                # "initial_plan_suggestion_json", # Removed as LLM should derive from objective/history
                                "aggregated_summary",
                                "aggregated_outputs_json",
                                "aggregated_files_json",
                                "error_feedback", "llm_output_language"
                            ]
                        )
                    },
                    parameters={
                         "specialized_agents_description": { # Ensure this matches create_workflow list
                            "ArchRAGAgent": "Retrieves information from the architectural knowledge base based on a query.",
                            "ImageRecognitionAgent": "Analyzes and describes the content of one or more images based on a prompt.",
                            "VideoRecognitionAgent": "Analyzes the content of one or more videos or 3D models based on a prompt.",
                            "ImageGenerationAgent": "Generates or edits images using AI based on a textual description and optional input images.",
                            "WebSearchAgent": "Performs a web search to find grounded information, potentially including relevant images and sources.",
                            "CaseRenderAgent": "Renders architectural images using ComfyUI based on a prompt, count, and strength.",
                            "Generate3DAgent": "Generates a 3D model and preview video from a single input image using ComfyUI diffusion.",
                            "RhinoMCPCoordinator": "Coordinates complex tasks within Rhino 3D using planning and multiple tool calls. Ideal for multi-step Rhino operations, tasks requiring precise coordinates/dimensions, or modifications to existing geometry.",
                            "SimulateFutureAgent": "Simulates future scenarios on a previously rendered image using ComfyUI based on a prompt.",
                            "LLMTaskAgent": "Handles general text-based tasks like analysis, summarization, reformatting, complex reasoning, or generating prompts.",
                            "EvaAgent": "Evaluates task outputs or the entire workflow based on criteria.",
                         },
                         "max_retries": 3
                    }
                ),
                "tool_agent": AgentConfig(
                    agent_name="tool_agent",
                    description="Executes the specific tool chosen for a task.",
                    llm=default_llm.copy(),
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
                    description="Evaluates the output of tasks based on criteria.",
                    llm=default_llm.copy(update={"temperature": 0.2}),
                    prompts={
                        "prepare_evaluation_inputs": PromptConfig(
                            template="""
You are preparing inputs for an evaluation task. Your goal is to structure the outputs of the *previously executed task* for evaluation tools.

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

**Your Goal:** Parse the directly provided outputs and files JSON strings from the PREVIOUS task above, extracting the necessary information to format a structured JSON object for the subsequent evaluation tools (LLM, Image Recognition, Video Recognition).

**Required Output JSON Format for Evaluation Tools:**
- `evaluation_target_description`: (String) The description of the task being evaluated (i.e., the previous task's description).
- `evaluation_target_objective`: (String) The objective of the task being evaluated (i.e., the previous task's objective).
- `evaluation_target_outputs_json`: (String) A JSON string representation of the **structured outputs** parsed from `evaluated_task_outputs_json`. Must be valid JSON.
- `evaluation_target_image_paths`: (List[String]) FULL paths to IMAGE files extracted from `evaluated_task_output_files_json`.
- `evaluation_target_video_paths`: (List[String]) FULL paths to VIDEO files extracted from `evaluated_task_output_files_json`.
- `evaluation_target_other_files`: (List[Dict]) Info about other non-image/video files extracted from `evaluated_task_output_files_json`.

**Instructions:**
1. Set `evaluation_target_description` and `evaluation_target_objective` using the provided description and objective of the **previous task**.
2. **Parse** the JSON string `evaluated_task_outputs_json`. If parsing fails or the string is empty/null, use **an empty dictionary**. **Re-stringify** the parsed (or default empty) dictionary as valid JSON for the `evaluation_target_outputs_json` field in your output.
3. **Parse** the JSON string `evaluated_task_output_files_json`. If parsing fails or the string is empty/null, use **an empty list**.
4. Initialize `evaluation_target_image_paths`, `evaluation_target_video_paths`, and `evaluation_target_other_files` as empty lists.
5. Iterate through the list parsed from `evaluated_task_output_files_json`:
    - If a file dictionary contains a `path` and a `type`:
        - If `type` starts with "image/", add its `path` to `evaluation_target_image_paths`.
        - If `type` starts with "video/", add its `path` to `evaluation_target_video_paths`.
        - Otherwise, add the complete file dictionary object to `evaluation_target_other_files`.
6. Construct the final JSON object using the keys specified in "Required Output JSON Format".
7. **Error Check:** After constructing the JSON, review the `task_objective` (of the previous task). If the objective clearly implies a specific output type (e.g., "generate an image", "analyze video") that is *missing* from the corresponding parsed list (`evaluation_target_image_paths` or `evaluation_target_video_paths`), THEN AND ONLY THEN, return an error JSON: `{{"error": "Missing expected output files/data based on objective and provided outputs."}}`. Do NOT return an error simply because the provided JSON strings were empty if the objective didn't mandate specific outputs.

**Output:** Return ONLY the final JSON input dictionary for the evaluation tools, or the error JSON. No other text.
Language: {llm_output_language}
""",
                            input_variables=[
                                "task_description", "task_objective",
                                "evaluated_task_outputs_json",
                                "evaluated_task_output_files_json",
                                "llm_output_language"
                            ]
                        ),
                        "generate_criteria": PromptConfig(
                            template="""
Based on the following task description and potentially relevant context, define specific, measurable criteria to evaluate the task's output.

**Task Description:** {task_description}
**Task Objective:** {task_objective}
**Context from RAG/Search:**
{rag_context}
{search_context}

**Default Principles:**
1.  Completeness: Does the output address all aspects of the task description?
2.  Quality: Is the output accurate, coherent, and well-presented (considering the task type)?
3.  Relevance: Is the output directly relevant to the task?

**Generate 3-5 specific criteria tailored to THIS task.** Output ONLY the criteria as a numbered list.
Respond in {llm_output_language}.
""",
                            input_variables=["task_description", "rag_context", "search_context", "llm_output_language"]
                        ),
                        "evaluation": PromptConfig(
                            template="""
You are evaluating the results of a design task based on specific criteria. Determine if the results PASS or FAIL based *primarily* on the criteria and the overall task objective.

**Task Being Evaluated - Description:** {evaluation_target_description}
**Task Being Evaluated - Objective:** {evaluation_target_objective}

**Results to Evaluate:**
*   **Structured Outputs (JSON):** ```json\n{evaluation_target_outputs_json}\n```
*   **Generated Image Files:** {evaluation_target_image_paths_str}
*   **Generated Video Files:** {evaluation_target_video_paths_str}
*   **Other Generated Files:** {evaluation_target_other_files_str}
**(Note: Actual image/video content is evaluated by specialized tools if applicable, this prompt focuses on overall compliance and textual outputs).**

**Specific Evaluation Criteria for this Task:**
{specific_criteria}

**Instructions:**
1.  **Assess:** Evaluate the results against the Specific Evaluation Criteria, considering their relevance to the Task Objective.
2.  **Determine Assessment:** Decide on an overall "Pass" or "Fail".
    *   **"Pass":** The output *substantially meets* the core Task Objective and the *most critical* evaluation criteria. Minor flaws on less important criteria may be acceptable.
    *   **"Fail":** The output has *significant flaws*, *fails to meet the core Task Objective*, or *misses critical evaluation criteria*, preventing the workflow from meaningfully progressing.
3.  **Provide Feedback:** Explain your reasoning, addressing the most important criteria and overall objective achievement. Clearly state why it Passed or Failed based on the definition above.
4.  **Suggest Improvements (if Fail):** Offer concrete suggestions focused on addressing the critical flaws or unmet core objectives.

**Output Format:** Return a JSON object with keys: "assessment" ("Pass" or "Fail"), "feedback", "improvement_suggestions". Ensure valid JSON.
Respond in {llm_output_language}.

**SPECIAL INSTRUCTIONS FOR FINAL EVALUATION:**
If the Task Description indicates this is a **Final Evaluation** of the entire workflow:
*   Disregard the specific "Results to Evaluate" section above.
*   Instead, review the **Full Workflow Summary** and **LTM Context** provided below.
*   Your goal is to provide a holistic assessment score (e.g., 1-10) and qualitative feedback on the overall process and final outcome compared to the initial goal.
*   Modify the output JSON to: `{{"assessment": "Score (1-10)", "feedback": "Holistic review feedback.", "improvement_suggestions": "Overall workflow improvement ideas."}}`

**Full Workflow Summary (if available):**
{full_task_summary}

**LTM Context (if available):**
{ltm_context}
""",
                           input_variables=[
                               "evaluation_target_description", "evaluation_target_objective",
                               "evaluation_target_outputs_json", "evaluation_target_image_paths_str",
                               "evaluation_target_video_paths_str", "evaluation_target_other_files_str",
                               "specific_criteria", "llm_output_language",
                               "full_task_summary", "ltm_context"
                           ]
                        ),
                        "prepare_final_evaluation_inputs": PromptConfig(
                            template="""
You are preparing inputs for the FINAL HOLISTIC EVALUATION of an entire completed workflow.

**Overall Workflow Goal:** {user_input}
**Full Workflow Task Summary (History - Primary source for outputs):**
{full_task_summary}
**Aggregated Outputs from Completed Tasks (JSON String):**
```json
{aggregated_outputs_json}
```
**Aggregated Files from Completed Tasks (JSON String):**
```json
{aggregated_files_json}
```

**Your Goal:** Consolidate all critical information and artifacts generated throughout the workflow into a structured JSON object for the final evaluation nodes, primarily using the `full_task_summary` and the aggregated JSON strings.

**Required Input Format for Final Evaluation:**
- `evaluation_target_description`: (String) Set this to "Final Workflow Review and Scoring".
- `evaluation_target_objective`: (String) Set this to "final_evaluation".
- `evaluation_target_full_summary`: (String) Pass through the provided Full Workflow Task Summary.
- `evaluation_target_ltm_context`: (String) Pass through the provided LTM context placeholder.
- `evaluation_target_key_image_paths`: (List[String]) Identify and list the FULL PATHS of the **most important final output images** by parsing the `aggregated_files_json` and cross-referencing with the `full_task_summary`. Focus on images representing the final design or key milestones. If none, use an empty list.
- `evaluation_target_key_video_paths`: (List[String]) Identify and list the FULL PATHS of the **most important final output videos** (e.g., walkthroughs, previews) by parsing the `aggregated_files_json` and cross-referencing with the `full_task_summary`. If none, use an empty list.
- `evaluation_target_other_artifacts_summary`: (String) Briefly summarize any other key final artifacts mentioned in `full_task_summary` or found in `aggregated_files_json` (e.g., 3D model filenames, final reports).

**Instructions:**
1. Set fixed values for `evaluation_target_description` and `evaluation_target_objective`.
2. Pass through `full_task_summary` and `ltm_context`.
3. Parse the `aggregated_files_json` string.
4. Iterate through the parsed files:
    - Identify key **final** images (check description/type) and add their `path` to `evaluation_target_key_image_paths`.
    - Identify key **final** videos (check description/type) and add their `path` to `evaluation_target_key_video_paths`.
    - Note other significant final artifacts (e.g., models, reports).
5. Create the `evaluation_target_other_artifacts_summary` string based on the non-image/video final artifacts found.
6. Construct the final JSON object using ONLY the keys specified in "Required Input Format for Final Evaluation".

**Output:** Return ONLY the final JSON input dictionary. No other text.
Language: {llm_output_language}
""",
                            input_variables=[
                                "user_input", "full_task_summary",
                                "aggregated_outputs_json",
                                "aggregated_files_json",
                                "ltm_context",
                                "llm_output_language"
                            ]
                        ),
                        "generate_final_criteria": PromptConfig(
                            template="""
You are defining the HOLISTIC evaluation criteria and scoring rubric for a completed design workflow.

**Overall Workflow Goal:** {user_input}
**Full Workflow Task Summary (History):**
{full_task_summary}
**Input Artifacts Provided for Final Review (prepared paths/summaries):**
```json
{final_eval_inputs_json}
```

**Your Task:** Generate specific criteria for a final, overall assessment of the workflow's success, including a scoring guideline.

**Criteria Should Address:**
1.  **Goal Achievement:** How well does the final output (represented by the artifacts) meet the original `user_input`?
2.  **Quality of Final Artifacts:** Assess the likely quality, completeness, and relevance of the key artifacts produced (images, videos, text summaries mentioned).
3.  **Process Efficiency (Optional):** Briefly comment if the task summary suggests an efficient or overly complex path was taken.
4.  **Overall Score:** Define a simple scoring rubric (e.g., 1-10 scale) where specific score ranges correspond to levels of success (e.g., 1-4 = Major Issues, 5-7 = Acceptable, 8-10 = Excellent).

**Output:** Output ONLY the criteria and scoring rubric as clear text.
Respond in {llm_output_language}.
""",
                            input_variables=[
                                "user_input", "full_task_summary", "final_eval_inputs_json", "llm_output_language"
                            ]
                        ),
                    },
                    parameters={}
                ),
                 "qa_agent": AgentConfig(
                    agent_name="qa_agent",
                    description="Handles user interaction in the QA phase after task execution.",
                    llm=default_llm.copy(),
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
                            1.  如果使用者表示要結束對話 (例如，說謝謝、再見、沒問題了等)，請 **只** 回覆 `TERMINATE`。
                            2.  如果使用者要求執行一個全新的、需要重新規劃的任務 (例如，"現在幫我做..."、"我們來換個主題..." 等)，請回覆 `NEW_TASK:` 並接著清晰地、完整地總結這個新任務的目標。
                            3.  **如果使用者表示沒事了，並且想要繼續執行*當前*的任務流程 (例如，"沒事了，繼續吧"、"請繼續執行任務")，請 **只** 回覆 `RESUME_TASK`。**
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
        template = _base_default_config_obj.agents[agent_name].prompts[prompt_name].template
        return template if template else "" # Return "" if template is None or empty
    except (KeyError, AttributeError, TypeError):
        print(f"Warning: Base default prompt not found for {agent_name}/{prompt_name}. Using empty string for schema default.")
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

    # --- QA Agent LLM Config (Using Single Literal) ---
    qa_model_name: AllModelNamesLiteral = Field( # Use the single Literal
        default="gpt-4o-mini",
        title="QA Agent: Model Name",
        description="Select the LLM Model for the QA Agent."
    )
    qa_temperature: float = Field(
        default=0.7,
        title="QA Agent: Temperature",
        ge=0.0, le=1.0,
        description="Temperature for the QA Agent (0.0-1.0)."
    )
    qa_max_tokens: Optional[int] = Field(
        default=None,
        title="QA Agent: Max Tokens",
        gt=0,
        description="Max Tokens for the QA Agent (Optional)."
    )

    # --- Prompts (Update default values, update descriptions) ---
    pm_create_workflow_prompt: str = Field(
        default=get_base_default_prompt("process_management", "create_workflow"), # 使用基礎預設值
        title="Process Management: Create Workflow Prompt",
        description="The prompt for creating the initial workflow. Clear this field to use the default loaded at runtime (incl. config.json overrides).", # 更新描述
        extra={'widget': {'type': 'textarea'}}
    )
    pm_failure_analysis_prompt: str = Field(
        default=get_base_default_prompt("process_management", "failure_analysis"), # 使用基礎預設值
        title="Process Management: Failure Analysis Prompt",
        description="The prompt for analyzing max retry failures and suggesting MODIFY, INSERT_PRE, or FALLBACK_GENERAL. Clear this field to use the default loaded at runtime.", # 更新描述
        extra={'widget': {'type': 'textarea'}}
    )
    pm_process_interrupt_prompt: str = Field(
        default=get_base_default_prompt("process_management", "process_interrupt"), # Use base default
        title="Process Management: Process Interrupt Prompt",
        description="The prompt for analyzing user interrupts and deciding how to modify the task sequence (PROCEED, INSERT_TASKS, REPLACE_TASKS). Clear to use default.",
        extra={'widget': {'type': 'textarea'}}
    )
    aa_select_agent_prompt: str = Field(
        default=get_base_default_prompt("assign_agent", "select_agent_prompt"), # 使用基礎預設值
        title="Assign Agent: Select Agent Prompt",
        description="The prompt for selecting the appropriate specialized agent. Clear this field to use the default loaded at runtime (incl. config.json overrides).", # 更新描述
        extra={'widget': {'type': 'textarea'}}
    )
    ta_error_handling_prompt: str = Field(
        default=get_base_default_prompt("tool_agent", "error_handling"), # 使用基礎預設值
        title="Tool Agent: Error Handling Prompt",
        description="The prompt for analyzing tool execution errors. Clear this field to use the default loaded at runtime (incl. config.json overrides).", # 更新描述
        extra={'widget': {'type': 'textarea'}}
    )
    ea_prepare_evaluation_inputs_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "prepare_evaluation_inputs"), # Gets the updated default
        title="Evaluation Agent: Prepare Eval Inputs Prompt",
        description="Prompt to prepare inputs for REGULAR evaluation using the PREVIOUS task's outputs. Clear to use default.", # Updated description
        extra={'widget': {'type': 'textarea'}}
    )
    ea_generate_criteria_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "generate_criteria"),
        title="Evaluation Agent: Criteria Gen Prompt",
        description="The prompt for generating evaluation criteria. Clear to use default.",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluation_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluation"),
        title="Evaluation Agent: Evaluation Prompt (Handles Regular & Final)",
        description="The prompt for performing the evaluation (regular or final). Clear to use default.",
        extra={'widget': {'type': 'textarea'}}
    )
    qa_qa_prompt: str = Field(
        default=get_base_default_prompt("qa_agent", "qa_prompt"), # 使用基礎預設值
        title="QA Agent: QA Prompt",
        description="The prompt for answering user questions. Clear this field to use the default loaded at runtime (incl. config.json overrides).", # 更新描述
        extra={'widget': {'type': 'textarea'}}
    )

    # --- Add new Eva final prompts ---
    ea_prepare_final_evaluation_inputs_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "prepare_final_evaluation_inputs"), # Gets the updated default
        title="Evaluation Agent: Prepare FINAL Eval Inputs Prompt",
        description="Prompt to gather all workflow outputs for FINAL holistic review using aggregated results. Clear to use default.", # Updated description
        extra={'widget': {'type': 'textarea'}}
    )
    ea_generate_final_criteria_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "generate_final_criteria"),
        title="Evaluation Agent: Generate FINAL Criteria Prompt",
        description="Prompt to generate holistic criteria and scoring for final review. Clear to use default.",
        extra={'widget': {'type': 'textarea'}}
    )

    # --- Validator (保持不變) ---
    @field_validator('pm_temperature', 'aa_temperature', 'ta_temperature', 'ea_temperature', 'qa_temperature')
    @classmethod
    def check_temperature(cls, v):
        if not (0.0 <= v <= 1.0):
             # This check might be redundant due to ge/le but kept for safety
             raise ValueError('Temperature must be between 0.0 and 1.0')
        return v


# =============================================================================
# Initialize LLM Function (保持不變 - 邏輯仍然有效)
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