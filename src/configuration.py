import os
import json
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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
        Google: 'Gemini-2.0*Flash', 'gemini-2.5-pro-exp-03-25', 'gemini-2.5-flash-preview-04-17'.
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
    output_directory: str = Field("./output", description="Directory for output files") # Adjusted default
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
            output_directory="./output",
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
        # LLM config for Sankey Structure Agent
        sankey_llm = ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.1, max_tokens=None) # Default for sankey, max_tokens=None

        default_agents = {
                "process_management": AgentConfig(
                    agent_name="process_management",
                    description="Responsible for planning and managing the task workflow.",
                    # llm=default_llm,
                    prompts={
                        "create_workflow": PromptConfig(
                            template="""
You are a meticulous and **detail-oriented** workflow planner specializing in architecture and design tasks. Your goal is to break down a user's request into a **sequence of granular, executable task objectives** for a team of specialized agents, ensuring smooth data flow between steps.

**User Request** (**Strictly Follow**): {user_input}

Analyze the request and generate a **complete, logical, and DETAILED sequence of tasks** to fulfill it. Pay close attention to dependencies and insert necessary intermediate processing steps.

**Key Planning Principles:**
1.  **Dependencies:** If Task B requires data generated by Task A, ensure Task A comes first.
2.  **Multi-Option Design Strategy:**
    *   If the user requests multiple distinct design options (e.g., Option A, Option B):
        *   **`RhinoMCPCoordinator`:** Plan as a **single task**. The `task_objective` and `inputs.user_request` must instruct Rhino to generate all distinct options within one session, each organized (e.g., on separate top-level layers like "OptionA_Description", "OptionB_Description"). Ensure screenshots are planned for each option after its generation within this single task.
        *   **`ImageGenerationAgent`:** If each distinct design option from Rhino requires its own set of visual explorations or style variations, plan **separate `ImageGenerationAgent` tasks (or sequences of tasks) for each distinct design option.** This also applies if a design option needs future scenario simulation based on its generated image.
        *   **`SpecialEvaAgent`:** If a specific design option (e.g., Option A, after being visually explored by `ImageGenerationAgent`) has produced multiple internal sub-variants (e.g., Option A - Style 1, Option A - Style 2), and you need to select the best sub-variant *for that specific Option A*, then plan a **separate `SpecialEvaAgent` task for Option A** to evaluate its sub-variants. Repeat for other main options (Option B, etc.) if they also have sub-variants needing evaluation.
3.  **Evaluation Task Assignment (CRITICAL - Follow these distinctions precisely):**
    *   **`EvaAgent` (Standard Evaluation - Pass/Fail for a SINGLE preceding task's output):**
        *   **Use Case:** Assign to `EvaAgent` ONLY when you need to evaluate the **direct output of the immediately preceding single task**. This is for a simple pass/fail check on one specific result.
        *   **Task Objective:** Set a descriptive `task_objective` (e.g., "Evaluate Task 3's generated floor plan based on standard criteria for residential layouts.").
        *   **Requires `requires_evaluation: true`**.
    *   **`SpecialEvaAgent` (Special Evaluation - Compare INTERNAL BRANCHES/ITERATIONS of a SINGLE design concept for pruning/convergence):**
        *   **Use Case:** Assign to `SpecialEvaAgent` when a **single design concept has undergone multiple internal iterations or has developed several distinct sub-branches** (e.g., Option A evolved into A.1, A.2, A.3), and you need to compare these internal variations to select the best one to carry forward for *that specific concept*. This is for pruning and converging within a design lineage.
        *   **Task Objective:** Set `task_objective="special_evaluation"`. The description should clarify which specific internal branches of a concept are being compared (e.g., "Compare internal iterations A.1, A.2, and A.3 of Facade Concept X to select the most promising branch.").
        *   **Requires `requires_evaluation: true`**.
    *   **`FinalEvaAgent` (Final Evaluation - Holistic review of ALL distinct final options AND overall project process):**
        *   **Use Case 1 (Overall Best Selection):** Assign to `FinalEvaAgent` to conduct a comprehensive comparison of **all major, distinct, and complete design alternatives** that have been developed throughout the project, selecting the overall best one against the original user request.
        *   **Use Case 2 (Process Review & Learning):** This agent will also provide a holistic review of the **entire iterative development path** of the project, analyzing decision points, iterations, and outcomes to offer suggestions for future improvements or similar projects.
        *   **Task Objective:** Set `task_objective="final_evaluation_and_process_review"`. The description should reflect the scope (e.g., "Final comprehensive review of all proposed building designs (Options X, Y, Z) and iterative process analysis against original user request.").
        *   **Requires `requires_evaluation: true`**.
3.  **Clarity:** Each task objective must be specific and actionable.
4.  **`requires_evaluation` for ALL Evaluation Agents:** **CRITICAL:** If `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, OR `FinalEvaAgent`, you **MUST** set `requires_evaluation` to `true`.

For **each** task in the sequence, you MUST specify:
1.  `description`: High-level goal for this step.
2.  `task_objective`: Specific outcome and method needed.
    *   Use `"final_evaluation_and_process_review"` or `"special_evaluation"` for the respective agent modes as defined above.
    *   Otherwise, describe the specific goal.
    *   **For `ModelRenderAgent` (Handles existing images for photorealistic rendering):**
        *   Set objective like "Photorealistic rendering of the provided image(s) of the architectural scheme: [scheme details]."
    *   **For `ImageGenerationAgent` (Generates new images, edits, or simulates future scenarios):**
        *   If generating a new visual concept from a description: Set objective like "Generate an image representing [concept description, style, mood]."
        *   If exploring multiple *distinct visual options* for a design element (e.g., different facade styles): Create SEPARATE `ImageGenerationAgent` tasks for EACH distinct option. 
        *   If generating *multiple variations of the SAME concept/option*: Use a single `ImageGenerationAgent` task with a clear objective for that one concept, ensuring the input provides sufficient guidance to generate diverse variations. If you need exactly 3 variations of the same concept, specify this in the task objective and inputs.
        *   If generating *multiple variations of the multiple concept/option*: Create SEPARATE `ImageGenerationAgent` task with a clear objective for each concept, and specify the number of variations needed in the task objective and inputs.
        *   **If simulating a future scenario based on an existing image:** Set objective like "Simulate future scenario for the image [e.g., `output_from_task_id_xyz.filename`] of architectural scheme [scheme details]. Context: [user goals, site conditions for simulation]." Ensure `image_inputs` will reference the correct existing image file.
    *   **For `RhinoMCPCoordinator`:**
        *   `initial_image_path`: (String, Optional) For `RhinoMCPCoordinator`. Find VALID path if needed. 不要將圖片路徑放入`user_request`中。
        *   If the task involves geometric modeling, modification, or precise analysis: Define the specific Rhino operations needed.
        *   If the task involves **functional layout, programmatic blocking, or quantitative/qualitative spatial arrangement**: Clearly state this objective.
        *   **When handling multiple distinct design options (as per "Multi-Option Design Strategy" principle), ensure the `user_request` to Rhino clearly instructs it to generate ALL options, manage them on separate layers, and capture necessary views for each.**
3.  `inputs`: JSON object suggesting initial data needs (e.g., `{{"prompt": "..."}}`). Use placeholders like `{{output_from_task_id_xyz.key}}`. Indicate file needs clearly. **Do not invent paths.** Use `{{"}}` if no input suggestion applies. NEVER use `null`.
4.  `requires_evaluation`: Boolean (`true`/`false`). **MUST be `true` if `selected_agent` is `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent`.**
5.  `selected_agent`: **(CRITICAL)** The **exact name** of the agent from the list below. Mandatory for every task. **Use `EvaAgent`, `SpecialEvaAgent`, or `FinalEvaAgent` for evaluation tasks, adhering STRICTLY to the use case definitions above.**

--- BEGIN AGENT CAPABILITIES ---
**Available Agent Capabilities & *Primary* Expected Prepared Input Keys:**
*   `ArchRAGAgent`: Information Retrieval (Regulations, Expertise) -> Needs `prompt`, optional `top_k`.
*   `ImageRecognitionAgent`: Image Analysis -> Needs `image_paths` (list), `prompt`. (Also used internally by ModelRenderAgent).
*   `VideoRecognitionAgent`: Video/3D Model Analysis -> Needs `video_paths` (list), `prompt`.
*   `ImageGenerationAgent`: **Generates NEW images from text descriptions, edits existing images, or simulates future scenarios based on existing images.** Ideal for visual exploration, concept art, mood boards, or creating textures/details. -> Needs `prompt`, optional `image_inputs` (list of paths for editing; REQUIRED for future scenario simulation), optional `i` (count for variations of the *same* prompt).
*   `WebSearchAgent`: Web Search (**Textual Information**, Cases, General Info) -> Needs `prompt`.
*   `ModelRenderAgent`: **Processes EXISTING images (e.g., from Rhino, or previous generations) for photorealistic rendering.** -> Needs `outer_prompt` (context for photorealistic rendering), `image_inputs` (list of paths to *existing* images). Internally uses `ImageRecognitionAgent` to generate a final English ComfyUI prompt.
*   `Generate3DAgent`: 3D MESH Model (.glb) Generation (Comfy3D) from an *existing* image -> Needs `image_path` (string).
*   `RhinoMCPCoordinator`: **Parametric/Precise 3D Modeling, Functional Layout, and Quantitative/Qualitative Analysis (Rhino).** Ideal for multi-step Rhino operations, tasks requiring precise coordinates/dimensions, **functional blocking and arrangement, programmatic analysis. If task involves visual output of layouts or plans, **request a top-down parallel projection screenshot.** -> Needs `user_request` (string command), optional `initial_image_path` (string path).
*   `PinterestMCPCoordinator`: **Pinterest Image Search & Download for case studies.** -> Needs `keyword` (string), optional `limit` (int). Should be used in conjunction with ImageRecognitionAgent tasks to understand design approaches and learning points.
*   `OSMMCPCoordinator`: **Map Screenshot Generation (OpenStreetMap).** -> Needs `user_request` (string: address or "lat,lon"). Should be used in conjunction with ImageRecognitionAgent tasks to understand site conditions.
*   `LLMTaskAgent`: **General Text Tasks.** -> Needs `prompt`. -> Use for summarize, reformat, generate prompts. **IMPORTANT: Can ONLY process text data.**
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
**Task About to Execute (at Current Index) or Recently Completed (if at end of list/interrupt on completed):**
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
*   `LLMTaskAgent`: Text Tasks (including summarization, report generation, text analysis). -> Needs `prompt`.
*   `EvaAgent`: Evaluation Agent (standard pass/fail, special comparison scoring, final holistic scoring).
*   `final_evaluation`: Objective for final holistic review.
*   `special_evaluation`: Objective for multi-option comparison.
--- END AGENT CAPABILITIES ---
**Your Task:** Choose ONE action based on the interrupt:

1.  **`PROCEED`**: Interrupt doesn't require plan changes, or is a minor comment not requiring action. Continue from `current_task_index`. Output: `{{"action": "PROCEED"}}`
2.  **`INSERT_TASKS`**: Insert new tasks **after** `current_task_index` if the user requests a new, distinct action (e.g., "summarize this for me", "generate a new image of X", "search for Y", "perform analysis Z") that isn't directly and fully addressed by the immediate next task, OR if the current task is already completed and the user wants a new follow-up action. Generate a list of new tasks (TaskState structure: `description`, `task_objective`, `selected_agent`, suggested `inputs`, `requires_evaluation`). Use Agent Capabilities. Output: `{{"action": "INSERT_TASKS", "insert_tasks_list": [ {{...task1...}}, ... ] }}`
3.  **`REPLACE_TASKS`**: Redesign workflow from `current_task_index` onwards (preserves completed tasks before index). Generate a **new sequence** for remaining tasks. Output: `{{"action": "REPLACE_TASKS", "new_tasks_list": [ {{...task1...}}, ... ] }}`
4.  **`CONVERSATION`**: Interrupt is unclear, ambiguous, requires discussion, or explicitly asks to discuss/chat. Routes to QA Agent. Output: `{{"action": "CONVERSATION"}}`

**Instructions:**
*   Analyze interrupt: Is it a clear plan change (`INSERT_TASKS`/`REPLACE_TASKS`), a minor clarification/comment not requiring plan change (`PROCEED`), or needs discussion (`CONVERSATION`)?
*   If the user's interrupt is a request for a distinct action (like "summarize", "generate", "search", "analyze"), and either (a) the current task index points to an already completed task, or (b) the upcoming task (if any) does not fulfill this request, then **prefer `INSERT_TASKS`**. For summarization, use `LLMTaskAgent`.
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
                    # llm=default_llm,
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
*   `prompt`: (String) REQUIRED for: `ArchRAGAgent`, `WebSearchAgent`, `ImageRecognitionAgent`, `VideoRecognitionAgent`, `LLMTaskAgent`:**不要先預設概念或方案內容**. 
*   `image_paths`: (List[String]) REQUIRED for `ImageRecognitionAgent`. Find VALID paths in `aggregated_files_json`.
*   `video_paths`: (List[String]) REQUIRED for `VideoRecognitionAgent`. Find VALID paths in `aggregated_files_json`.
*   `image_paths`: (List[String]) For `Generate3DAgent`. REQUIRED. List of FULL FILE PATHS of one or more *existing* images to process from `aggregated_files_json`. If the objective is to process a single image, this list will contain one path. If multiple, list all relevant paths.
*   **For `ModelRenderAgent` (Processes EXISTING images for architectural rendering):**
    *   `outer_prompt`: (String) REQUIRED. **ENGLISH prompt**. THIS MUST BE a highly detailed, comma-separated list of CONCISE, VISUAL-ONLY keywords, phrases, and very short descriptive clauses for an image generation model. **嚴禁使用負面詞彙如「舊」、「老」、「old」等。**
    *   `image_inputs`: (List[String]) REQUIRED. List of FULL FILE PATHS of *existing* images to process (from `aggregated_files_json`).
*   **For `ImageGenerationAgent` (Generates NEW images, edits existing images, or simulates future scenarios based on existing images):**
    *   `prompt`: (String) REQUIRED. **ENGLISH prompt**.
        *   For NEW images or editing: Detailed textual description. Focus on architectural Appearance with photorealistic style. **Add DO NOT generate text in the prompt.**
        *   For future scenario simulation (if `task_objective` indicates this and `image_inputs` are provided by `ModelRenderAgent`): THIS MUST BE a highly detailed, comma-separated list of CONCISE, VISUAL-ONLY phrases, clauses, **specifically describing the predicted appearance after 30 years.** **Add DO NOT generate text in the prompt.**
    *   `image_inputs`: (List[String], Optional for editing, REQUIRED for future scenario simulation based on existing images). List of FULL FILE PATHS from `aggregated_files_json`.
    *   `i`: (Integer, Default: 3) REQUIRED. Number of variations to generate for the `prompt` (applies when generating new images or variations of a future scenario).
*   **For `RhinoMCPCoordinator`:**
    *   `user_request`: (String) REQUIRED. Detailed command for Rhino. If the task objective involves: 1) multi-floor spatial planning or other model types - request "capture a perspective projection from an aerial viewpoint"; 2) models intended for subsequent rendering - request "capture a two_point perspective view with camera positioned at human eye level".
    *   `initial_image_path`: (String, Optional) For `RhinoMCPCoordinator`. 如果有提到需要時(比如說參考某圖等)請將找到相關的圖片路徑。
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
        * For `ImageGenerationAgent`: Suggest materials/finishes/forms in the `prompt` that align with the budget class (luxury, mid-range, economical).**Add DO NOT generate text in the prompt.**
    *   **If `{selected_agent_name}` is `ModelRenderAgent`:**
        *   Carefully examine the **`Current Task Objective`**, **`Overall Workflow Goal`**, and **`Workflow History Summary`**.
        *   **CRITICAL STEP 1: Translate ALL relevant information from the task objective, overall goal, and history into a list of CONCRETE, VISUAL-ONLY ENGLISH descriptors for architectural rendering.** Convert abstract concepts and jargon into what they would *look like* visually.
        *   **CRITICAL STEP 2: Combine these visual descriptors into a single, comma-separated string for the `outer_prompt`.一定要放:8K, detailed, best quality, architectural rendering, New building**
        *   **Include Visual Descriptor Categories (Base List):** **一定要記得視角、樓層**
            *   **Architectural Style:** (e.g., modern, futuristic, minimalist)
            *   **Building/Element Type:** (e.g., residential building, office tower, museum)
            *   **Forms:** (e.g., flowing organic shapes, sharp geometric forms, modular units, cantilever)
            *   **Materials & Textures:** (e.g., smooth concrete, glass curtain wall, steel beams, lush vertical greenery, metallic surfaces, translucent panels)
            *   **Specific Features:** (e.g., large windows, skylight, solar panels, balcony, courtyard)
            *   **Lighting & Atmosphere:** (e.g., soft ambient light, dramatic shadows, harsh sunlight, warm interior glow, foggy, rainy, clear sky, atmosphere)
            *   **Time of Day:** (e.g., dawn, morning, noon, afternoon, sunset, dusk, night)
            *   **Environment/Context:** (e.g., urban street view, natural landscape, park setting, forest backdrop, waterfront, foggy)
            *   **Camera View Angle:** (e.g., person perspective, high angle perspective, aerial view, street view, two point perspective)  
        *   **ENSURE the final `outer_prompt` output is *only* the comma-separated string of visual descriptors. DO NOT include abstract terms that haven't been translated, or conversational text.**
    *   **If `{selected_agent_name}` is `ImageGenerationAgent`:**
        *   **If the `task_objective` indicates future scenario simulation (e.g., "simulate 30 years later") using `image_inputs`:**
            *   Carefully examine the **`Current Task Objective`**, **`Overall Workflow Goal`**, **`Workflow History Summary`**, **relevant site/climate conditions from history/inputs**, and the **original design's geometric features/materials (implied by task context)**.
            *   **CRITICAL STEP 1 (Future Scenario - 30 Years Later): Translate the current state and anticipated influences into a list of CONCRETE, VISUAL-ONLY ENGLISH descriptors. These descriptors MUST specifically focus on the *visual changes* and *resulting appearance* of the **main building** (the primary architectural structure) after 30 years.**
                *   **Refer to the "image paths" out of `ModelRenderAgent` section as a input for `image_inputs`**
                *   **THEN, ADD SPECIFIC DESCRIPTORS FOR 30-YEAR CHANGES TO THE MAIN BUILDING, such as:**
                    *   **Material Aging/Weathering on the main building:** (e.g., `main building's aged concrete with water stains`, `main building with patinated copper roofing`, `main building showing weathered timber with visible grain`, `main building's oxidized steel elements`, `main building with faded paintwork`)
                    *   **Vegetation Evolution on/around the main building:** (e.g., `mature climbing ivy partially covering main building's facade`, `overgrown landscaping near the main building base`, `large trees casting new shadows on the main building`)
                    *   **Environmental Accumulation on the main building:** (e.g., `light dust accumulation on main building's horizontal surfaces`, `streaks from rain runoff on main building's glass`)
                    *   **Subtle Structural Settling of the main building (if plausible):** (e.g., `minor, non-critical hairline cracks in the main building's plaster`)
                    *   **Technological Patina/Integration on the main building:** (e.g., `main building with slightly outdated solar panel models`, `visible retrofitted environmental sensors on the main building`)
                    *   **Signs of Use/Maintenance on the main building (if applicable):** (e.g., `areas of repointed brickwork on the main building`, `worn pathways leading to the main building`)
            *   **CRITICAL STEP 2 (Future Scenario - 30 Years Later): Combine these visual descriptors (30-year changes to the main building) into a `prompt`.一定要放: focused on the main building's 30-year transformation, depicting 30 years of aging, DO NOT generate text.** 
            *   **ENSURE the final `prompt` output is *only* the string of visual descriptors. DO NOT include abstract terms that haven't been translated, or conversational text.**
        *   **Else (for generating new images or general editing):**
            *   **When generating multiple images for the one design concept (e.g., `i > 1`), the `prompt` MUST be crafted to explore distinct variations, sub-themes, or contextual scenarios branching from the core concept. For example, if the core concept is "a modern residential tower" and `i=3`, the prompt should guide the generation of three visually distinct interpretations, such as "Scenario 1:...", "Scenario 2:...", and "Scenario 3:...". Explicitly state the differentiating factors for each variation within the prompt if generating multiple images, avoiding mere repetition. Ensure each variation maintains photorealistic style and avoids text generation.**
            *   **When generating multiple images for *different* design concepts (i.e., multiple `ImageGenerationAgent` tasks), ensure each task's prompt describes completely different architectural materials, geometric forms, spatial experiences, architectural atmospheres, and design approaches to create diverse visual representations of each respective concept.**
            *   **When editing images, ensure to clearly identify the specific area that needs modification, and focus only on changing that part while keeping the rest unchanged. The overall composition should be maintained as much as possible. The `prompt` MUST include: "8K, detailed, best quality, only modify [specific part] of the image leaving everything else unchanged, the modified [specific area] should appear as [detailed description of desired changes]..." Clearly define the boundaries of what should be changed and what should remain intact.**
    *   **If `{selected_agent_name}` is `RhinoMCPCoordinator`:**
        *   Synthesize a clear `user_request` for Rhino.
        *   **Crucially, if the `Current Task Objective` implies creating a plan, layout, or requires a specific view, append a clear instruction to the `user_request` for Rhino to ensure the correct projection mode and viewpoint is used when capturing screenshots with Rhino's `capture_viewport` tool. For multiple design options, ensure each option is captured with appropriate screenshots. Always include screenshot instructions for all generated designs.**
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
                        ),
                    },
                    parameters={
                         "specialized_agents_description": {
                            "ArchRAGAgent": "Retrieves information from the architectural knowledge base based on a query.",
                            "ImageRecognitionAgent": "Analyzes and describes the content of one or more images based on a prompt. Also used internally by ModelRenderAgent to generate final English ComfyUI prompts.",
                            "VideoRecognitionAgent": "Analyzes the content of one or more videos or 3D models based on a prompt.",
                            "ImageGenerationAgent": "Generates NEW images from detailed text descriptions, or edits existing images. Ideal for visual concept generation, style exploration, and creating specific artistic renditions from scratch. Takes a `prompt` for the core description, optionally `image_inputs` (list of paths) for editing context, and optionally `i` for generating multiple variations of the *same* described concept.",
                            "WebSearchAgent": "Performs a web search to find grounded information, potentially including relevant images and sources.",
                            "ModelRenderAgent": "Handles advanced image processing: either photorealistic rendering of existing model views OR simulation of future scenarios on images, using ComfyUI. Requires: `outer_prompt` (initial context for the task type - photorealism or future simulation), `image_inputs` (list of image paths), and `is_future_scenario` (boolean flag: true for future simulation, false for photorealistic rendering). Internally uses ImageRecognitionAgent to refine the `outer_prompt` into a final English ComfyUI prompt for each image.",
                            "Generate3DAgent": "Generates a 3D model (.glb) and preview video from one or more input images using ComfyUI diffusion. Expects a list of image paths under the key `image_paths`.",
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
                    # llm=default_llm, 
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
                # llm=default_llm, 
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

   *   **IF `{selected_agent}` is `FinalEvaAgent` (Final Evaluation - Overall Best Selection & Process Review):**
       *   **Part 1: Overall Best Selection:**
           *   Review **Full Summary, Key Artifacts, Visual Tool Feedback, and `options_data_for_llm_eval_json` (if provided, representing distinct final design options)** against the holistic criteria/rubric for project outcome.
           *   **Crucially, when assessing aspects like cost, functionality, or adherence to original intent for each option, refer to `raw_outputs_for_llm_parsing.mcp_internal_messages` within each option in `options_data_for_llm_eval_json`.**
           *   Assign a holistic score (1-10) based strictly on the rubric and all evidence, reflecting the quality of the **best overall design option(s)**.
           *   Provide `feedback` justifying this score based on the rubric and visual analysis for the selected best option(s).
           *   The `detailed_option_scores` in your output should be the list of option evaluations (for all final options considered) generated by *this LLM evaluation path* if it was responsible for evaluating options based on `options_data_for_llm_eval_json`. Each item should have the standard option score fields.
       *   **Part 2: Iterative Development Process Review:**
           *   Based on the `{full_task_summary}`, the evolution implied by `{options_data_for_llm_eval_json}` (if it shows stages or distinct final products of different paths), and the "Iterative Development Process Review" guidelines from the `{specific_criteria}`, generate a textual review.
           *   This review should analyze strengths, weaknesses, decision points, and lessons learned from the entire design process.
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10 for best option)", "assessment_type": "Final", "selected_option_identifier": "Identifier of the overall best option(s)", "detailed_option_scores": [{{...option1_scores...}}, ...], "feedback": "Rubric-based justification for the best option's score, incorporating visual feedback.", "iteration_review_and_suggestions": "Detailed textual review of the iterative development process, including strengths, weaknesses, and suggestions for future projects.", "improvement_suggestions": "Overall improvement ideas for the project outcome or future similar endeavors."}}`

   *   **IF `{selected_agent}` is `SpecialEvaAgent` (Special Evaluation - Comparing INTERNAL BRANCHES/ITERATIONS of a single concept):**
       *   The primary goal is to **compare multiple internal development branches or iterative versions WITHIN a single overarching design concept/option** to select the best path forward for *that specific concept*.
       *   Review **`options_data_for_llm_eval_json` (CRITICAL: use this as the primary source for the internal branches/iterations to evaluate via text/LLM). Also consider Full Summary (for context of the parent concept), Key Artifacts (if relevant to the branches), and Visual Tool Feedback (if this LLM call is *after* visual tools and needs to consolidate).** Compare these internal branches/iterations using the **detailed comparative criteria/rubric provided in `{specific_criteria}`** (which are tailored for comparing variations of a single concept).
       *   **Crucially, when assessing aspects like cost, functionality, or adherence to original intent for each branch/iteration, refer to `raw_outputs_for_llm_parsing.mcp_internal_messages` within each item in `options_data_for_llm_eval_json`.**
       *   For each branch/iteration parsed from `options_data_for_llm_eval_json`, assess it against each dimension defined in the rubric. Provide scores for "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", estimate `estimated_cost`, and `green_building_potential_percentage`. Each of these should be from the LLM's textual/conceptual evaluation of that specific branch/iteration.
       *   Assign a comparative score (1-10) reflecting the quality and fit of the **single best branch/iteration** identified, based strictly on the rubric and all evidence (including visual feedback and alignment with the parent concept's goals derived from `{user_input}`).
       *   **Identify the single best branch/iteration** (e.g., by its `option_id` from `options_data_for_llm_eval_json`).
       *   Provide detailed `feedback` explaining the comparison across the specified dimensions, justifying why the selected branch/iteration is superior for advancing the parent concept.
       *   **Output Format:** Return JSON: `{{"assessment": "Score (1-10 for best branch)", "assessment_type": "Special", "selected_option_identifier": "Identifier of the best branch/iteration", "detailed_option_scores": [ {{"option_id": "id_branch1", ...scores... ,"llm_feedback_text":"branch specific feedback"}}, ... ], "feedback": "Detailed rubric-based comparison of branches/iterations across dimensions, justifying the selection of the best one to carry forward for the parent concept. Incorporate visual feedback if relevant.", "improvement_suggestions": "N/A or specific if the selected branch needs minor refinement"}}`

**Instructions (Continued):**
*   Your response **MUST** be only the single, valid JSON object required for the agent type. No other text.
*   Adhere **strictly** to the output format and criteria/rubric. Base your final judgment on the **totality of the evidence provided, especially `{user_input}` and `{specific_criteria}`**.
*   For `SpecialEvaAgent` and `FinalEvaAgent` when processing `options_data_for_llm_eval_json`: The `detailed_option_scores` list you generate MUST be a list of dictionaries. Each dictionary represents one option (or branch/iteration for SpecialEvaAgent) and MUST contain the keys: `option_id` (string), `user_goal_responsiveness_score_llm` (float), `aesthetics_context_score_llm` (float), `functionality_flexibility_score_llm` (float), `durability_maintainability_score_llm` (float), `estimated_cost` (float), `green_building_potential_percentage` (float, 0-100), and `llm_feedback_text` (string).
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
You are preparing inputs for the **FINAL HOLISTIC EVALUATION (`FinalEvaAgent`)** OR a **SPECIAL MULTI-OPTION COMPARISON EVALUATION (`SpecialEvaAgent`)**.

**Current Task Agent:** `{selected_agent}`
**Current Task Objective (Overall Goal for this Evaluation - CRITICAL for identifying the scope of comparison):** {current_task_objective}
**Overall Workflow User Request (Context for overall project goals):** {user_input}
**Budget Limit Overall (if provided by user for the entire project):** {budget_limit_overall}

**Full Workflow Task Summary (History - Source for outputs/options & contextual analysis like site/case studies):**
{full_task_summary}
**Aggregated Outputs from ALL Completed Tasks (JSON String - Parse this carefully. Pay ATTENTION to outputs from image generation tasks, as their textual descriptions/prompts are KEY to linking images to options/branches):**
```json
{aggregated_outputs_json}
```
**Aggregated Files from ALL Completed Tasks (JSON String - Correlate files with options/branches. Use 'source_task_id' to link files to tasks. CRITICALLY, use the 'description' field within each file object in this JSON, as it now contains structured info like 'SourceAgent: ...; TaskDesc: ...', to link files to options/branches):**
```json
{aggregated_files_json}
```
**Previously identified Options Data (JSON String - If a prior step already structured some options, use as a starting point or reference. This might be from ProcessManagement's initial plan if it detailed options. If empty, you must derive options from scratch.):**
```json
{options_data_json}
```

**Your Goal (Context-Dependent based on `{selected_agent}`):**

**IF `{selected_agent}` is `FinalEvaAgent`:**
1.  **Identify ALL Distinct, Final Design Options/Concepts** developed throughout the project. These are top-level, independent design proposals.
2.  For each final option, extract its details (description, type, summaries, associated media files, cost/green estimates).
3.  Construct `options_data` as a list of these complete, distinct, final design options. Each option should have a unique `option_id` (e.g., "final_concept_A", "final_concept_B").

**IF `{selected_agent}` is `SpecialEvaAgent`:**
**Your primary focus is dictated by the `{current_task_objective}`.**
1.  **Identify the EXACT PARENT Design Concept/Option** that this `SpecialEvaAgent` task is focused on evaluating. **You MUST derive this ParentConceptID strictly from the `{current_task_objective}`.** For example, if `{current_task_objective}` is "Compare internal iterations of **Facade Concept X**", then `ParentConceptID` is "FacadeConceptX". If it's "Select best detail for **Roof Design Alpha**'s variations", then `ParentConceptID` is "RoofDesignAlpha". Do NOT infer this from other sources if the objective is clear.
2.  **VERY IMPORTANT: Filter all subsequent information gathering (from `{full_task_summary}`, `{aggregated_outputs_json}`, `{aggregated_files_json}`) to find ONLY the INTERNAL BRANCHES, ITERATIONS, or SUB-VARIATIONS that were developed *specifically and exclusively* as children or refinements of THIS EXACT `ParentConceptID` identified in step 1.**
    *   Look for tasks in `{full_task_summary}` whose descriptions explicitly state they are iterating on, refining, or are sub-parts of this `ParentConceptID`.
    *   Examine outputs in `{aggregated_outputs_json}` for mentions that tie them directly to this `ParentConceptID`.
    *   **IGNORE ALL OTHER design concepts, options, or branches that DO NOT DIRECTLY BELONG to this `ParentConceptID`.**
3.  For EACH identified internal branch/iteration that BELONGS to the `ParentConceptID`:
    *   Assign a unique `option_id` that explicitly links it to the parent and denotes it as a branch/iteration. **Use a format like: `{{ParentConceptID}}_branch_{{sequential_number_or_letter}}` (e.g., "FacadeConceptX_branch_1", "FacadeConceptX_branch_2", "RoofDesignAlpha_branch_A").**
    *   Extract its specific `description` (e.g., "Facade Concept X - Branch 1: Material Study with Glass", "Roof Design Alpha - Branch A: Integrated Solar Panels"). This description should clearly reflect it's a branch of the `ParentConceptID`.
    *   Determine its `architecture_type` (usually same as parent).
    *   Gather its `textual_summary_from_outputs` (specific to this branch, derived from outputs related ONLY to this branch of the `ParentConceptID`).
    *   Associate relevant `image_paths`, `video_paths`, and `other_relevant_files` that were generated FOR THIS SPECIFIC BRANCH/ITERATION of the `ParentConceptID`. Use the `TaskDesc` matching logic very carefully to isolate files belonging only to this branch.
    *   Extract any `initial_estimated_cost` or `initial_green_building_percentage` specific to this branch.
4.  Construct `options_data` as a list of these identified internal branches/iterations. **This list MUST ONLY contain items belonging to the SINGLE `ParentConceptID` derived from `{current_task_objective}`. If no such branches are found for that specific `ParentConceptID`, the `options_data` list should be empty.**

**CRITICAL LOGIC FOR BOTH AGENTS (Information Extraction & File Association for each option/branch):**
For **EACH** distinct design option (for `FinalEvaAgent`) or internal branch/iteration (for `SpecialEvaAgent`, belonging to the correct `ParentConceptID`) identified, you MUST extract or infer the following:
    *   `option_id`: (String) **Your NEWLY ASSIGNED identifier. For `FinalEvaAgent`, use sequential IDs like "final_concept_A". For `SpecialEvaAgent`, use the `{{ParentConceptID}}_branch_{{counter}}` format described above, ensuring `{{ParentConceptID}}` matches the one from `{current_task_objective}`.**
    *   `description`: (String) **The short, descriptive THEME or NAME of this specific option/branch (e.g., "Modern Glass Pavilion", "Facade X - Branch 1: Material Study A").** Infer from content in `{aggregated_outputs_json}` or objectives.
    *   `architecture_type`: (String) **The primary architectural type for THIS option/branch (e.g., "Residential Tower", "Facade Detail").** Infer from `{user_input}` or specific content.
    *   `textual_summary_from_outputs`: (String) **CRITICAL: A concise summary of textual outputs from `{aggregated_outputs_json}` EXCLUSIVELY related to THIS option/branch. If a single task output contains info for multiple items, isolate ONLY the relevant text.**
    *   `image_paths`: (List[Dict]) **CRITICAL LOGIC REQUIRED for associating images:**
        *   For the CURRENT option/branch you are processing:
        *   Iterate through each file object in the `{aggregated_files_json}` list.
        *   Each file object's `description` field is a structured string, typically like: `"SourceAgent: <AgentName>; TaskDesc: <Description of the task that created this file>; ImageNum: <X/Y>; PromptHint: <...>"`.
        *   **Primary Matching Strategy: Parse the structured `description` string of each file. Extract the value associated with the `TaskDesc:` key (this is the `<Description of the task that created this file>`). Compare this extracted `TaskDesc` string with the theme/description of the CURRENT option/branch.**
        *   If the extracted `TaskDesc` closely matches or is clearly intended for the current option/branch's theme, AND this option/branch belongs to the correct `ParentConceptID` (for SpecialEvaAgent), then this file (image) belongs to it.
        *   Refer to `{full_task_summary}` to understand task sequence and context to confirm relationships. The `source_task_id` in the file object confirms which task generated it.
        *   List all image file info (`{{"path": "...", "filename": "..."}}`) confidently associated. If none, this list MUST be empty `[]`.
    *   `video_paths`: (List[Dict]) **CRITICAL**: Apply the exact same logic as for `image_paths`. Parse each video file's structured `description` in `{aggregated_files_json}`, extract `TaskDesc:`, and match to the current option/branch theme.
    *   `other_relevant_files`: (List[Dict]) List of other file info (text docs, spreadsheets) relevant to this option/branch, identified using similar `TaskDesc` matching.
    *   `initial_estimated_cost`: (Float, Optional) If any task output for this option/branch mentioned a cost.
    *   `initial_green_building_percentage`: (Float, Optional) If any task output for this option/branch mentioned green building metrics.

**Output Requirements (Common for both agents, but `options_data` content differs as described above):**
- `evaluation_target_description`: (String) Set based on agent: "Final Workflow Review of [Project Name/Theme]" or "Special Comparison of Branches for [Parent Concept Theme from objective]". The Parent Concept Theme MUST match the `ParentConceptID` derived from `{current_task_objective}`.
- `evaluation_target_objective`: (String) Pass through `{current_task_objective}`.
- `evaluation_target_full_summary`: (String) Pass through `{full_task_summary}`.
- `evaluation_target_key_image_paths`, `evaluation_target_key_video_paths`, `evaluation_target_other_artifacts_summary`: Summarize overall key media. For `SpecialEvaAgent`, these should ideally be from the parent concept or the most representative branches of that parent.
- `needs_detailed_criteria`: (Boolean) `true`.
- `options_data`: (List[Dict]) **CRITICAL**: Your list of option/branch dictionaries.
    *   For `FinalEvaAgent`: List of distinct, final design options.
    *   For `SpecialEvaAgent`: List of internal branches/iterations of a single parent concept, **where this parent concept is STRICTLY determined by `{current_task_objective}`**. All items in this list must belong to that single parent. Use `option_id`s like `{{ParentConceptID}}_branch_{{counter}}`.
    *   **ENSURE unique `option_id`s, exclusive textual summaries, and correctly associated media files for each item.**
- `budget_limit_overall`: (Float or Null) Parsed from input.

**Instructions:**
1.  Carefully determine if you are preparing for `FinalEvaAgent` or `SpecialEvaAgent` based on `{selected_agent}`.
2.  **If `SpecialEvaAgent`, your MOST IMPORTANT first step is to precisely identify the single `ParentConceptID` from `{current_task_objective}`. All subsequent data gathering for `options_data` MUST be filtered to only include branches of THIS `ParentConceptID`.**
3.  Follow the specific goal instructions above for identifying and structuring items for `options_data`.
4.  Follow the "CRITICAL LOGIC FOR BOTH AGENTS (Information Extraction & File Association...)" section to populate the details for each item in `options_data`.
5.  Prioritize `{current_task_objective}` for `SpecialEvaAgent` to identify the parent concept (`ParentConceptID`) and its branches. Ensure `option_id`s for branches clearly reflect this parentage using the `{{ParentConceptID}}_branch_{{counter}}` format.
6.  Use the `TaskDesc` matching logic for file association rigorously, ensuring files are linked to the correct branch of the correct `ParentConceptID` (for `SpecialEvaAgent`).

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
*   **Criteria Should Address:**
    1.  **Overall Project Outcome:** Alignment with `{user_input}`, quality of the final selected design option(s) based on `{options_to_evaluate_json}`. This involves assessing how well the proposed final designs meet the project's core objectives and requirements.
    2.  **Iterative Development Process Review:** Effectiveness of the design evolution throughout the project (based on `{full_task_summary}` and evolution implied by `{options_to_evaluate_json}` if it represents stages), appropriateness of decision-making at key iteration points, and overall efficiency or insights gained from the process.
*   Incorporate insights from RAG/Search if relevant (e.g., for benchmarking quality or process). Consider `{architecture_type}` for tailoring outcome expectations.
*   **Scoring Rubric & Guidelines:**
    *   For **Overall Project Outcome**: Define a detailed 1-10 scale rubric for the six core architectural dimensions (使用者目標回應性, 美學與場域關聯性, 機能性與適應彈性, 耐久性與維護性, 早期成本效益估算, 綠建築永續潛力) to assess the best final option(s) identified.
    *   For **Iterative Development Process Review**: Provide guidelines on what aspects to comment on. This is primarily for generating textual feedback. The review should cover:
        *   Clarity and logic of the iterative steps.
        *   Effectiveness of choices made at each significant iteration or branching point.
        *   Identification of any particularly strong or weak paths taken during development.
        *   Lessons learned that could be applied to future projects.
        *   Suggestions for improving the design process itself.

**IF `{selected_agent}` is `SpecialEvaAgent`:**
*   **Comparative Criteria MUST Address Multiple Dimensions based on `{options_to_evaluate_json}` (each item is an option, representing an internal branch/iteration of a single concept):**
    *   **Identify Key Dimensions from `{user_input}` (for the overarching concept), `{full_task_summary}`, and the nature of branches/iterations in `{options_to_evaluate_json}`.** Determine which aspects are most important for comparing these internal variations (e.g., one branch explored material A, another material B; one iteration refined form, another functional layout).
    *   **Core Architectural Dimensions for Comparison (Apply these to each branch/iteration):**
        1.  **使用者目標（概念核心）回應性**: How well each branch/iteration develops the core user goals and architectural intent of the parent design concept.
        2.  **美學與場域關聯性**: Visual development, refinement of form, scale, materials within the branch, and how it relates to the intended context of the parent concept.
        3.  **機能性與適應彈性**: How the branch explores or refines spatial layout, circulation, or adaptability for the parent concept. **Consider `mcp_internal_messages` for feasibility and alignment with Rhino's planned actions for this branch.**
        4.  **耐久性與維護性**: Implied durability and maintenance considerations of the material/structural choices explored in this branch.
        5.  **早期成本效益估算**: Relative cost implications of the design choices made within this branch. Relate to overall budget if applicable. **Crucially, use `mcp_internal_messages` to infer material choices or construction complexity that would affect cost for this branch.**
        6.  **綠建築永續潛力**: Potential in areas like ecology, health, energy saving, and waste reduction as explored or implied by this branch.
    *   The rubric should allow scoring each branch/iteration from 1-10 on these dimensions to determine which is the most promising to carry forward.
*   **Develop Discriminatory Metrics/Descriptions for Each Dimension:** For each chosen dimension, describe how to compare the branches/iterations.
    *   Example for 早期成本效益估算: "Branch A (Implied Cost: Low) - Uses simpler construction as per MCP intent. Branch B (Implied Cost: High) - MCP messages indicated complex custom geometry."
    *   Example for 綠建築永續潛力: "Branch A (Green Potential: Good) - Better solar shading. Branch B (Green Potential: Fair) - Larger unshaded glass areas."
*   **Scoring Rubric & Selection:**
    *   The main output of the evaluation LLM using this rubric will be per-branch/iteration scores for these dimensions.
    *   The overall `assessment` ("Score (1-10)") for the `SpecialEvaAgent` task should reflect the quality/fit of the *best branch/iteration* identified.
    *   The rubric should guide the LLM to provide a `selected_option_identifier` (which would be the ID of the best branch/iteration).
    *   The final feedback should clearly explain the reasoning for the selection across the different dimensions.

**Output:** Output ONLY the criteria and the detailed scoring rubric/guideline as clear text, tailored to the agent type and informed by available context. Emphasize quantifiable or clearly comparable metrics where possible for `SpecialEvaAgent`. The rubric should be detailed enough for an LLM to assign scores (1-10) to each option/branch across the defined dimensions.
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
                # llm=default_llm, 
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
                        2.  如果使用者意圖執行一個與**目前所有任務上下文和成果完全無關的、全新的、需要從頭重新規劃的任務** (例如，用戶突然說 "現在我們來設計一個公園" 而之前都在設計建築)，請回覆 `NEW_TASK:` 並接著清晰地、完整地總結這個新任務的目標。
                        3.  如果使用者想要**繼續當前的任務流程**，或者提出**對現有任務/成果的修改、調整、迭代、或基於現有成果的進一步請求** (例如，"繼續執行"、"幫我修改方案A的顏色"、"在目前的基礎上增加一個陽台"、"針對這個結果做進一步分析")，請 **只** 回覆 `RESUME_TASK`。
                        4.  否則，只要是**詢問類或閒聊類**的內容，請正常回答使用者的問題。如果使用者沒有明確問題，可以嘗試根據對話記錄和任務摘要提供相關資訊或詢問是否需要進一步幫助。

                        **當前任務可用的代理類型**
                            查詢資料類:"ArchRAGAgent"、"WebSearchAgent"、"PinterestMCPCoordinator"、"OSMMCPCoordinator"
                            圖片處理類:"ImageRecognitionAgent"、"ImageGenerationAgent"(更通用)、"ModelRenderAgent"
                            模型處理類:"VideoRecognitionAgent"(辨識3Dagent的結果)、"Generate3DAgent"、"RhinoMCPCoordinator"(建模)
                            文字處理類:"LLMTaskAgent"
                            評估類:"EvaAgent"(用於檢核是否通過)、"SpecialEvaAgent"(用於比較)、"FinalEvaAgent"(用於比較與總結)

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
            ),
            "sankey_structure_agent": AgentConfig(
                agent_name="sankey_structure_agent",
                description="Analyzes task history to generate a structured JSON for Sankey diagrams.",
                # llm=ModelConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.1, max_tokens=4000),
                prompts={
                    "generate_sankey_structure": PromptConfig(
                        template="""你是一位頂尖的視覺化流程分析師，專長是將複雜的工作流程數據轉換為清晰、直觀的Sankey圖。現在，你需要分析以下JSON格式的設計任務歷史記錄，並嚴格按照指示生成Sankey圖所需的節點和鏈接JSON結構。最終的圖表必須清晰地展示每一個獨立的設計方案分支，就像使用者手繪的草圖一樣。

**使用者最初的整體目標:** {user_input}

**完整的任務歷史 (JSON 陣列格式的任務物件):**
```json
{full_tasks_json}
```

**Sankey圖結構生成核心指南 (請極其嚴格地遵守，這對結果至關重要):**

**A. 關於節點 (Nodes):**
    1.  **核心原則：每個獨立的設計方案/分支的「最小單元」都是一個獨立的「成果節點」**:
        *   父節點通常來自於單一任務的多個結果；而子節點則是後續相同子任務的產出。要分辨清楚這些任務結果的父子關係。
        *   流程中的主要節點 **必須是** 具體的、獨立的設計成果的最小單元，例如「設計概念A」、「圖像集A的變體1」、「圖像集A的變體2」、「被選方案1」、「最終設計」。
        *   **關鍵識別**: 當一個任務的 `outputs` 字段包含一個列表，其中每個列表項代表一個獨立的設計成果項目 (例如 `outputs: {{ "generated_concepts": ["未來都市住宅", "生態友好辦公樓"] }}` 或 `outputs: {{ "image_sets_for_concept_X": [ {{ "id": "set1", "description": "外觀A"}}, {{"id": "set2", "description": "外觀B"}} ] }}`)，你 **必須** 為該列表中的 **每一個項目** 創建一個獨立的「成果節點」。**這點至關重要，請務必為每一個單獨產出的項目（例如列表中的每個元素，代表一個變體、一個圖、一個模型等）創建獨立節點，而不是將它們合併！例如，如果生成了3張不同的圖片作為分支，則必須創建3個獨立的圖片節點。**
    2.  **嚴禁將Agent作為獨立節點**: **絕對不要** 將Agent本身（如ImageGenerationAgent, LLMTaskAgent）創建為一個獨立的流程節點。
    3.  **起始節點**:
        *   `id`: "start_workflow"
        *   `label`: "流程起點"
        *   `type`: "workflow_control"
    4.  **成果節點的標籤 (`label`) (重要格式規定!)**:
        *   **必須嚴格遵循此格式 (僅包含成果類型和主要描述):** `成果類型簡稱: 主要描述`
        *   範例: `概: 未來都市住宅`
        *   範例: `圖: 希望號外觀`
    5.  **成果節點的類型 (`type`)**:
        *   `"concept_item"`: 初始的設計概念。
        *   `"image_collection_item"`: 由一個父成果分支產生的一組相關圖像中的一個獨立項目。
        *   `"rhino_model_item"`: Rhino產生的3D模型中的一個獨立項目。
        *   `"selected_option_item"`: 經過中間評估被選中的方案/成果。
        *   `"final_choice_item"`: 最終被確定的設計方案。
    6.  **成果節點的ID (`id`)**: 必須全局唯一。例如: `item_concept_taskABC_opt1`, `item_image_taskXYZ_setA_view1`。
    7.  **結束節點**:
        *   `id`: "end_workflow"
        *   `label`: "流程終點"
        *   `type`: "workflow_control"

**B. 關於鏈接 (Links) 和 Value 設定 (嚴格遵循流量守恆和分配原則):**
    1.  **基本原則概述**: 流量從 `start_workflow` 開始，逐級分配。父節點的流出流量均分給其直接產生的所有平級子節點。被選中的路徑將繼承其接收到的流量繼續發展。未選中的分支則以極小的流量、淺色鏈接到主路徑下游的共同匯聚點或流程終點，以保持視覺上的連接性和整潔性。
    2.  **起始流量**:
        *   `start_workflow` 節點是所有流程的起點，其初始流出總量視為 `1.0`。
        *   如果 `start_workflow` 直接連接到 N 個初始成果節點 (例如，N 個獨立的 `concept_item`)，則每一條從 `start_workflow` 到這些初始成果節點的鏈接，其 `value` **固定為 `1.0 / N`**。 (例如，若有2個初始概念，每個鏈接value為0.5；若有3个，則為0.333)
    3.  **後續分支的流量分配 (父到子)**:
        *   當一個已存在的父成果節點 P (它本身是由一條值為 `V_P_in` 的鏈接創建的，即它接收到的流入總量為 `V_P_in`) 作為源頭，在**單一個任務**中產生 M 個新的平級子成果節點 (C1, ..., CM) 時（例如，一個概念產生 M 個圖像變體）：
            *   則每一條從 P 到其直接子節點 Ci 的鏈接 (`P -> Ci`)，其 `value` **必須計算並設定為 `V_P_in / M`**。這適用於所有直接子節點，無論它們後續是否被選中。
    4.  **被選中/主路徑的流量延續 (子到下一階段)**:
        *   承接 B.3，如果 P 的某個子成果節點 Cj (它從P接收到的流量是 `V_P_in / M`) 被選中或作為主路徑繼續發展，產生了下一個成果節點 S_next (S_next 可以是 `selected_option_item`，也可以是其他類型的成果節點)：
            *   則從 Cj 到 S_next 的鏈接 (`Cj -> S_next`)，其 `value` **應繼承 Cj 所接收到的流量，即 `V_P_in / M`**。
            *   如果 S_next 再繼續發展到 S_next_next，則 `S_next -> S_next_next` 的鏈接 `value` 依此類推，繼續傳遞 `V_P_in / M`。
    5.  **未被選中分支的處理 (視覺匯合 - 極其重要!)**:
        *   **情境**: 父節點 P 產生了多個子成果 (C1, C2, ..., CM)，它們都根據 B.3 從 P 接收了值為 `V_P_in / M` 的流量。其中一個子成果 Cj (例如 C1) 被選中並根據 B.4 發展到了 S_next (C1 -> S_next)。
        *   **目標匯聚點 (S_converge) 的識別**:
            *   `S_converge` 通常是這個被選中的 S_next 節點本身，或者是 S_next 再進一步鏈接到的下一個主要階段性成果節點 (例如，由 `selected_option_item` 鏈接到的 `rhino_model_item` 或 `final_choice_item`)。LLM 需要根據流程邏輯判斷這個合適的下游共同匯聚點。
        *   **未選中分支的鏈接創建**:
            *   對於那些**未被選中的同級子成果 Ci (例如 C2, C3, ..., CM)**，它們 **必須** 創建一條**從 Ci 指向識別出的 S_converge 節點的鏈接** (即 `C2 -> S_converge`, `C3 -> S_converge`, 等)。
            *   **這些特定的鏈接 (`Ci -> S_converge`) 的屬性必須嚴格設定為**:
                *   `value`: **一個非常小的值** (例如 `0.02`)。此值遠小於主路徑流量，僅用於繪製細線，表示連接性。
                *   `color`: **一個非常淺的顏色** (例如 `'#E0E0E0'` 或 `'#DCDCDC'`)。
                *   `alpha`: **一個較低的透明度** (例如 `0.4`)。
        *   **完全無後續的分支**: 如果一個分支在產生後，既沒有被選中發展，也沒有產生任何子代（即它是一個終端分支但非最終方案），它也應該有一條具備上述小 `value`、淺色、低透明度屬性的鏈接，指向 `end_workflow`。
    6.  **最終選定方案到流程終點**:
        *   從任何 `final_choice_item` 節點 (假設它接收到的流入總值為 `V_final_in`) 到 `end_workflow` 節點的鏈接，其 `value` **應設為 `V_final_in`**，並使用主路徑的顏色和透明度。
    7.  **完整性**: 每條鏈接都必須有 `source` 和 `target` 屬性。
    8.  **鏈接顏色和透明度 (默認/主路徑)**: 除非根據 B.5 有特殊設定，主路徑鏈接的 `color` 可選 `'#A9A9A9'` (暗灰色) 或 `'#B0B0B0'`，`alpha` 為`0.7`。


**C. 輸出JSON格式 (嚴格遵守):**
一個JSON對象，包含 `nodes` 和 `links` 兩個鍵。
*   `nodes`: 節點對象列表。每個節點: `id`, `label` (遵循A.4格式 - 使用 `\\n` 換行), `type`。
*   `links`: 鏈接對象列表。每個鏈接: `source`, `target`, `value` (遵循B.2固定規則), `color` (可選), `alpha` (可選)。

**範例片段 (展示如何為任務產生的每個獨立分支創建節點及Value設定):**
假設任務歷史：
1.  `ConceptGen` 任務 (ID: task_001) 的 `outputs` 是 `{{ "generated_concepts": ["太空站概念A", "星際港概念B", "月球基地概念C"] }}`。產生了 **3個獨立概念**。
2.  `ImgGen` 任務 (ID: task_002) 以 "太空站概念A" 為輸入，其 `outputs` 是 `{{ "image_sets": [{{ "id":"set_A1", "desc":"希望號外觀"}}, {{"id":"set_A2", "desc":"探索號內部"}}] }}`。為概念A產生了 **2個獨立的圖像集分支**。
3.  `BranchEval` 任務 (ID: task_003) 評估 "希望號外觀" (set_A1) 和 "探索號內部" (set_A2)，並選擇了 "希望號外觀" (set_A1) 作為 "選定方案S1"。

```json
{{
  "nodes": [
    {{ "id": "start_workflow", "label": "流程起點", "type": "workflow_control" }},
    // 根據 task_001 的 outputs，創建三個獨立的 concept_item 節點
    {{ "id": "item_concept_task_001_A", "label": "概: 太空站概念A\\n(由 ConceptGen 生成)", "type": "concept_item"}},
    {{ "id": "item_concept_task_001_B", "label": "概: 星際港概念B\\n(由 ConceptGen 生成)", "type": "concept_item"}},
    {{ "id": "item_concept_task_001_C", "label": "概: 月球基地概念C\\n(由 ConceptGen 生成)", "type": "concept_item"}},
    // 根據 task_002 的 outputs，為 "太空站概念A" 創建兩個獨立的 image_collection_item 節點
    {{ "id": "item_image_collection_conceptA_set_A1", "label": "圖: 希望號外觀\\n(由 ImgGen 生成)", "type": "image_collection_item"}},
    {{ "id": "item_image_collection_conceptA_set_A2", "label": "圖: 探索號內部\\n(由 ImgGen 生成)", "type": "image_collection_item"}},
    // 根據 task_003 的選擇結果
    {{ "id": "item_selected_option_task_003_S1", "label": "選: 選定方案S1 (源自希望號外觀)\\n(由 BranchEval 選定)", "type": "selected_option_item"}},
    {{ "id": "item_final_choice_task_mno_F1", "label": "終: 最終方案F1 (來自S1)\\n(由 FinalEvaAgent 確認)", "type": "final_choice_item"}},
    {{ "id": "end_workflow", "label": "流程終點", "type": "workflow_control" }}
  ],
  "links": [
    // 規則 A: start_workflow 到 3 個初始概念，每個 value = 1 / 3 = 0.33
    {{ "source": "start_workflow", "target": "item_concept_task_001_A", "value": 0.33 }},
    {{ "source": "start_workflow", "target": "item_concept_task_001_B", "value": 0.33 }},
    {{ "source": "start_workflow", "target": "item_concept_task_001_C", "value": 0.33 }},
    // 規則 B1: "太空站概念A" (父) 分支出 2 個圖像集 (子)，每個 value = 0.33 / 2 = 0.165
    {{ "source": "item_concept_task_001_A", "target": "item_image_collection_conceptA_set_A1", "value": 0.165 }},
    {{ "source": "item_concept_task_001_A", "target": "item_image_collection_conceptA_set_A2", "value": 0.165 }},
    // 規則 B2: "希望號外觀" (set_A1) 被選中並發展為 "選定方案S1"
    {{ "source": "item_image_collection_conceptA_set_A1", "target": "item_selected_option_task_003_S1", "value": 0.165 }},
    {{ "source": "item_selected_option_task_003_S1", "target": "item_final_choice_task_mno_F1", "value": 0.165 }},
    {{ "source": "item_final_choice_task_mno_F1", "target": "end_workflow", "value": 0.165 }}
    // 規則 B3: "希望號外觀" (set_A2) 及其他未選中方案
    {{ "source": "item_image_collection_conceptA_set_A2", "target": "item_selected_option_task_003_S1", "value": 0.02 }},
    {{ "source": "item_concept_task_001_B", "target": "item_final_choice_task_mno_F1", "value": 0.02 }},
    {{ "source": "item_concept_task_001_C", "target": "item_final_choice_task_mno_F1", "value": 0.02 }},
  ]
}}
```

**關鍵要點 (更新並統一):**
*   **嚴格遵循全新的固定Value規則 (B.2)**: 這是最重要的。`value`根據路徑類型（初始、M個平級分支、單一路徑/被選中路徑延續、最終）賦予特定的計算值 (`1.0 / M` 或 `1.0`)。
*   **識別被選中的路徑**: LLM必須能夠從任務數據中判斷哪個分支被選中了，以便將被選中的路徑上連接到後續的節點。
*   **每一個生成結果都是一個獨立節點，請處理好其父子關係**: 持續強調這一點。
*   **ID唯一性與可追溯性**: 鼓勵LLM使用能反映來源的ID命名方式。
*   **聚焦成果流**: 再次強調圖的核心是「設計成果」的演變。
*   **標籤語言**: {llm_output_language}

請僅返回嚴格符合上述格式的JSON對象，不要包含任何額外解釋。
""",
                        input_variables=["user_input", "full_tasks_json", "llm_output_language"]
                    )
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
        agent_cfg = _base_default_config_obj.agents.get(agent_name) # Renamed for clarity
        if agent_cfg and agent_cfg.prompts:
             prompt_config = agent_cfg.prompts.get(prompt_name)
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
    "日文"
]

class ConfigSchema(BaseModel):
    """Configuration schema exposed in LangGraph Studio."""

    # --- 全局語言設定 ---
    global_llm_output_language: SupportedLanguage = Field(
        default="繁體中文",
        title="全局：輸出語言",
        description="LLM 回應的全局預設語言。選擇 'DEFAULT(gpt-4o-mini)' 作為代理的模型，以使用其代碼定義的預設 LLM。"
    )

    # --- 記憶配置 ---
    retriever_k: int = Field(
         default=5,
         title="記憶：檢索 K 值",
         description="從長期記憶 (LTM) 中檢索的文檔數量",
         gt=0
    )

    # --- 流程管理代理 LLM 配置 (使用單一字面值) ---
    pm_model_name: AllModelNamesLiteral = Field(
        default="gemini-2.0-flash",
        title="流程管理：模型名稱",
        description="選擇 LLM 模型。'USE_DEFAULT_MODEL' 應用代碼預設值。"
    )
    pm_temperature: float = Field(
        default=0.3,
        # default=_base_default_config_obj.agents.get("process_management").llm.temperature 
        # if _base_default_config_obj.agents.get("process_management") else 0.3,
        title="流程管理：溫度",
        ge=0.0, le=1.0,
        description="流程管理代理的溫度 (0.0-1.0)。"
    )
    pm_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("process_management").llm.max_tokens 
        if _base_default_config_obj.agents.get("process_management") else None,
        title="流程管理：最大標記數",
        gt=0,
        description="流程管理代理的最大標記數 (可選)。"
    )

    # --- 分配代理 LLM 配置 (使用單一字面值) ---
    aa_model_name: AllModelNamesLiteral = Field(
        default="gemini-2.0-flash",
        title="分配代理：模型名稱",
        description="選擇 LLM 模型。'USE_DEFAULT_MODEL' 應用代碼預設值。"
    )
    aa_temperature: float = Field(
        default=_base_default_config_obj.agents.get("assign_agent").llm.temperature 
        if _base_default_config_obj.agents.get("assign_agent") else 0.7,
        title="分配代理：溫度",
        ge=0.0, le=1.0,
        description="分配代理的溫度 (0.0-1.0)。"
    )
    aa_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("assign_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("assign_agent") else None,
        title="分配代理：Max Tokens",
        gt=0,
        description="分配代理的最大標記數 (可選)。"
    )

    # --- 工具代理 LLM 配置 (使用單一字面值) ---
    ta_model_name: AllModelNamesLiteral = Field(
        default="gpt-4o-mini",
        title="工具代理：模型名稱 (錯誤分析)",
        description="選擇用於錯誤自修復的 LLM 模型。'USE_DEFAULT_MODEL' 應用代碼預設值。"
    )
    ta_temperature: float = Field(
        default=_base_default_config_obj.agents.get("tool_agent").llm.temperature 
        if _base_default_config_obj.agents.get("tool_agent") else 0.7,
        title="工具代理：溫度 (錯誤分析)",
        ge=0.0, le=1.0,
        description="工具代理錯誤自修復分析的溫度 (0.0-1.0)。"
    )
    ta_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("tool_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("tool_agent") else None,
        title="工具代理：Max Tokens (錯誤分析)",
        gt=0,
        description="工具代理錯誤自修復分析的最大標記數 (可選)。"
    )

    # --- 評估代理 LLM 配置 (使用單一字面值) ---
    ea_model_name: AllModelNamesLiteral = Field(
        default="gemini-2.5-flash-preview-04-17",
        title="評估代理：模型名稱",
        description="選擇 LLM 模型。'USE_DEFAULT_MODEL' 應用代碼預設值。"
    )
    ea_temperature: float = Field(
        default=0.5,
        # default=_base_default_config_obj.agents.get("eva_agent").llm.temperature 
        # if _base_default_config_obj.agents.get("eva_agent") else 0.5,
        title="評估代理：溫度",
        ge=0.0, le=1.0,
        description="評估代理的溫度 (0.0-1.0)。"
    )
    ea_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("eva_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("eva_agent") else None,
        title="評估代理：Max Tokens",
        gt=0,
        description="評估代理的最大標記數 (可選)。"
    )

    # --- 桑基圖結構代理 LLM 配置 (新增) ---
    sankey_structure_model_name: AllModelNamesLiteral = Field(
        default="gemini-2.5-flash-preview-04-17",
        title="桑基圖結構代理：模型名稱",
        description="為桑基圖結構代理選擇 LLM 模型。'USE_DEFAULT_MODEL' 應用其代碼預設值。"
    )
    sankey_structure_temperature: float = Field(
        default=0.1,
        # default=_base_default_config_obj.agents.get("sankey_structure_agent").llm.temperature
        # if _base_default_config_obj.agents.get("sankey_structure_agent") else 0.1,
        title="桑基圖結構代理：溫度",
        ge=0.0, le=1.0,
        description="桑基圖結構代理的溫度 (0.0-1.0)。"
    )
    sankey_structure_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("sankey_structure_agent").llm.max_tokens
        if _base_default_config_obj.agents.get("sankey_structure_agent") else None,
        title="桑基圖結構代理：Max Tokens",
        gt=0,
        description="桑基圖結構代理的最大標記數 (可選)。"
    )

    # --- 問答代理 LLM 配置 ---
    qa_model_name: AllModelNamesLiteral = Field(
        default="gpt-4o-mini",
        title="問答代理：模型名稱",
        description="為問答代理選擇 LLM 模型。'USE_DEFAULT_MODEL' 應用代碼預設值。"
    )
    qa_temperature: float = Field(
        default=_base_default_config_obj.agents.get("qa_agent").llm.temperature 
        if _base_default_config_obj.agents.get("qa_agent") else 0.7,
        title="問答代理：溫度",
        ge=0.0, le=1.0,
        description="問答代理的溫度 (0.0-1.0)。"
    )
    qa_max_tokens: Optional[int] = Field(
        default=_base_default_config_obj.agents.get("qa_agent").llm.max_tokens 
        if _base_default_config_obj.agents.get("qa_agent") else None,
        title="問答代理：Max Tokens",
        gt=0,
        description="問答代理的最大標記數 (可選)。"
    )

    # --- 提示詞 ---
    aa_prepare_tool_inputs_prompt: str = Field(
        default=get_base_default_prompt("assign_agent", "prepare_tool_inputs_prompt"),
        title="分配代理：準備工具輸入提示詞",
        description="分配代理準備下一個工具/代理結構化輸入的提示詞。包含：{user_input}, {workflow_history_summary}, {aggregated_outputs_json}, {aggregated_files_json}, {task_objective}, {task_description}, {selected_agent_name}, {agent_description}, {error_feedback}, {user_budget_limit}, {latest_evaluation_results_json}, {llm_output_language}。清除欄位以使用運行時預設值。",
        extra={'widget': {'type': 'textarea'}}
    )

    ea_prepare_final_evaluation_inputs_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "prepare_final_evaluation_inputs"),
        title="評估代理：準備最終/特殊評估輸入提示詞",
        description="用於收集輸出/成果進行最終整體審查或特殊多選項比較的提示詞。清除欄位以使用運行時預設值。",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluation_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluation"),
        title="評估代理：評估提示詞 (處理所有類型)",
        description="進行評估的核心提示詞 (標準、最終或特殊)。清除欄位以使用運行時預設值。",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluate_option_with_image_tool_prompt: str = Field(
        default=get_base_default_prompt("eva_agent", "evaluate_option_with_image_tool"),
        title="評估代理：圖像工具選項評估提示詞",
        description="圖像識別工具評估單一設計選項的提示詞 (生成初始分數)。清除欄位以使用運行時預設值。",
        extra={'widget': {'type': 'textarea'}}
    )
    ea_evaluate_option_with_video_tool_prompt: str = Field(
         default=get_base_default_prompt("eva_agent", "evaluate_option_with_video_tool"),
         title="評估代理：視頻工具選項評估提示詞",
         description="視頻識別工具評估單一設計選項的提示詞 (生成初始分數)。清除欄位以使用運行時預設值。",
         extra={'widget': {'type': 'textarea'}}
    )
    # # --- Sankey Structure Agent Prompt (NEW) ---
    # sankey_structure_agent_prompt: str = Field(
    #     default=get_base_default_prompt("sankey_structure_agent", "generate_sankey_structure"),
    #     title="Sankey Structure Agent: Generate Structure Prompt",
    #     description="Prompt for the Sankey Structure Agent to analyze tasks and output structured node/link data. Clear field to use runtime default.",
    #     extra={'widget': {'type': 'textarea'}}
    # )

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
            if not OPENAI_API_KEY: print("Warning: OPENAI_API_KEY environment variable not set.")
            # Pass model_name as 'model'
            return ChatOpenAI(model=model_name, **common_params)
        elif provider == "google":
            # The Python variable GOOGLE_API_KEY holds the value from os.getenv("GEMINI_API_KEY")
            if not GOOGLE_API_KEY:
                print("Warning: GEMINI_API_KEY environment variable not set.") # MODIFIED Warning message

            google_params = {"temperature": temperature}
            if "max_tokens" in common_params:
                google_params["max_output_tokens"] = common_params["max_tokens"]

            if GOOGLE_API_KEY:
                google_params["google_api_key"] = GOOGLE_API_KEY

            # --- MODIFIED: Ensure model name is lowercase for Google ---
            google_model_name = model_name.lower()
            # Ensure it starts with "models/" if it doesn't already, as required by some Google APIs
            if not google_model_name.startswith("models/"):
                 google_model_name = "models/" + google_model_name.replace("models/", "") # Avoid double 'models/'

            print(f"Initializing Google LLM with formatted model name: {google_model_name}")
            return ChatGoogleGenerativeAI(model=google_model_name, convert_system_message_to_human=True, **google_params)
            # --- END MODIFIED ---

        elif provider == "anthropic":
            if not ANTHROPIC_API_KEY: print("Warning: ANTHROPIC_API_KEY environment variable not set.")

            # if ANTHROPIC_API_KEY:
            #     common_params["anthropic_api_key"] = ANTHROPIC_API_KEY

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