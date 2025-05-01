# =============================================================================
# 1. Imports
# =============================================================================
import os
import uuid
import json
import base64
from dotenv import load_dotenv
from typing import Dict, List, Any, Literal, Union, Optional, Tuple
from pathlib import Path
import traceback
import asyncio

# LangChain/LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# --- Tool Imports (Essential for Nodes) ---
from src.tools.ARCH_rag_tool import ARCH_rag_tool
from src.tools.img_recognition import img_recognition
from src.tools.video_recognition import video_recognition
from src.tools.gemini_image_generation_tool import generate_gemini_image
from src.tools.gemini_search_tool import perform_grounded_search
from src.tools.case_render_image import case_render_image
from src.tools.generate_3D import generate_3D
from src.tools.simulate_future_image import simulate_future_image

# --- Configuration & LLM Initialization ---
from src.configuration import ConfigManager, ModelConfig, MemoryConfig, initialize_llm, ConfigSchema

# --- <<< 修改：從 state.py 導入 >>> ---
from src.state import WorkflowState, TaskState # 從 state.py 導入狀態

# --- MCP Imports (Import MCPAgentState for internal use/type hinting) ---
try:
    from src.mcp_test import (
        MCPAgentState, # Keep for type hinting internal state if desired
        call_rhino_agent, agent_tool_executor,
        should_continue as mcp_should_continue
    )
    print("Successfully imported MCP components from src.mcp_test")
except ImportError as e:
    print(f"WARNING: Could not import MCP components from src.mcp_test: {e}")
    # Define MCPAgentState as a simple Dict if import fails, for type hinting robustness
    MCPAgentState = Dict[str, Any]
    async def call_rhino_agent(*args, **kwargs): raise NotImplementedError("MCP import failed")
    async def agent_tool_executor(*args, **kwargs): raise NotImplementedError("MCP import failed")
    def mcp_should_continue(*args, **kwargs): return END

# =============================================================================
# 2. 環境變數與組態載入 (Load static parts if needed)
# =============================================================================
load_dotenv()

# --- MODIFIED ---
# ConfigManager might still be used for *static* config parts like prompts
# or initial non-configurable settings.
# If prompts are static, load them here. Let's assume they are for now.
config_manager = ConfigManager("config.json")
_full_static_config = config_manager.get_full_config() # Load once for static parts
workflow_config_static = _full_static_config.workflow
# --- Get static prompts (example) ---
_pm_prompts = _full_static_config.agents.get("process_management", {}).prompts or {}
_aa_prompts = _full_static_config.agents.get("assign_agent", {}).prompts or {}
_ta_prompts = _full_static_config.agents.get("tool_agent", {}).prompts or {}
_ea_prompts = _full_static_config.agents.get("eva_agent", {}).prompts or {}
_qa_prompts = _full_static_config.agents.get("qa_agent", {}).prompts or {}
# Get static tool descriptions for AssignAgent
_aa_tool_descriptions = _full_static_config.agents.get("assign_agent", {}).parameters.get("tool_descriptions", {})
# Get static tool default parameters for ToolAgent
_ta_tool_configs_static = _full_static_config.agents.get("tool_agent", {}).tools or {}
_ta_file_handling_static = _full_static_config.agents.get("tool_agent", {}).parameters.get("file_handling", {})
# --- END MODIFIED ---

# --- Get static tool descriptions for AssignAgent ---
agent_descriptions = _full_static_config.agents.get("assign_agent", {}).parameters.get("specialized_agents_description", {})
# --- <<< NEW: Add RhinoMCPCoordinator Description if missing >>> ---
# Ensure the description is available for prepare_tool_inputs_node
if "RhinoMCPCoordinator" not in agent_descriptions:
     agent_descriptions["RhinoMCPCoordinator"] = "Coordinates complex tasks within Rhino 3D using planning and multiple tool calls. Ideal for multi-step Rhino operations or requests involving existing Rhino geometry analysis and modification. Takes the user request and optional image path."
# --- <<< END NEW >>> ---

# =============================================================================
# 3. 常數設定 (Mostly static, okay to load from config or define here)
# =============================================================================
OUTPUT_DIR = workflow_config_static.output_directory
RENDER_CACHE_DIR = os.path.join(OUTPUT_DIR, "render_cache")
MODEL_CACHE_DIR = os.path.join(OUTPUT_DIR, "model_cache")
os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
MEMORY_DIR = os.path.join("knowledge", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
LLM_OUTPUT_LANGUAGE_DEFAULT = workflow_config_static.llm_output_language # Get default from static config

# =============================================================================
# 4. LLM 與記憶體初始化 (Non-configurable parts, retriever becomes dynamic)
# =============================================================================

# Long-Term Memory (LTM) - Embedding part is likely static
# Use defaults from static config for embedding model
ltm_embedding_config = _full_static_config.memory.long_term_memory
# Currently only supports OpenAI embeddings based on EmbeddingModelConfig
if ltm_embedding_config.provider == "openai":
     embeddings = OpenAIEmbeddings(model=ltm_embedding_config.model_name)
else:
     print(f"Warning: Unsupported LTM embedding provider '{ltm_embedding_config.provider}'. Defaulting to OpenAI.")
     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=MEMORY_DIR,
    embedding_function=embeddings,
    collection_name="workflow_memory"
)
# --- Retriever is now created dynamically based on config['configurable']['retriever_k'] ---
# retriever = vectorstore.as_retriever(...) # Removed static retriever creation
ltm_memory_key = "ltm_context" # Define memory key for LTM
ltm_input_key = "input"        # Define input key for LTM

print(f"Long-term memory vector store initialized. Storing in: {MEMORY_DIR}")
print(f"LTM Retriever 'k' value will be determined at runtime from configuration.")

# Short-Term Memory (STM) for QA will be instantiated within the QA_Agent
# agent_descriptions 仍然需要從 configuration 或其定義處導入
agent_descriptions = _full_static_config.agents.get("assign_agent", {}).parameters.get("specialized_agents_description", {})
# --- <<< 結束修改 >>> ---

# =============================================================================
# 5. Helper Functions (Potentially moved from main graph or defined here)
# =============================================================================

def _set_task_failed(task: TaskState, error_message: str, node_name: str):
    """Sets task status to failed and logs the error."""
    print(f"{node_name} Error: {error_message}")
    task["status"] = "failed"
    task["error_log"] = f"[{node_name}] {error_message}"
    # Optionally clear outputs if needed
    task["outputs"] = {}
    task["output_files"] = []

def _filter_base64_from_files(file_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes base64 data from a list of file dictionaries for prompts."""
    filtered_list = []
    for file_dict in file_list:
        # Create a copy to avoid modifying the original dictionary in the state
        filtered_dict = file_dict.copy()
        # Remove the 'base64_data' key if it exists
        filtered_dict.pop("base64_data", None)
        filtered_list.append(filtered_dict)
    return filtered_list

def _update_task_state_after_tool(
    current_task: TaskState,
    outputs: Optional[Dict[str, Any]] = None,
    output_files: Optional[List[Dict[str, str]]] = None,
    error: Optional[Exception] = None,
    mcp_result: Optional[Dict[str, Any]] = None # <<< NEW: Specific field for MCP results
) -> TaskState:
    """Updates task state fields based on tool execution outcome, including MCP results."""
    if error:
        error_msg = f"Error during {current_task.get('selected_agent', 'Task')} execution: {str(error)}"
        print(error_msg)
        # Failure Path
        current_task["outputs"] = {} # Clear structured outputs
        current_task["output_files"] = [] # Clear files
        current_task["error_log"] = error_msg # Log the specific error
        current_task["status"] = "failed"
        # PM will handle retry increment and status check
        print(f"Task {current_task['task_id']}: Failure recorded. Status set to 'failed'.")
    else:
        # Success Path
        # Prioritize standard outputs/files if provided
        current_task["outputs"] = outputs if outputs is not None else {}
        current_task["output_files"] = output_files if output_files is not None else []

        # <<< NEW: Merge MCP results if available >>>
        if mcp_result:
            print(f"Merging MCP result into task outputs: {mcp_result}")
            # Add MCP result under a specific key to avoid conflicts
            current_task["outputs"]["mcp_final_result"] = mcp_result.get("message", "MCP completed without specific message.")
            # If MCP returned an image path/URI, add it to output_files
            saved_path = mcp_result.get("saved_image_path")
            saved_uri = mcp_result.get("saved_image_data_uri")
            if saved_path:
                # Attempt to create a file entry
                filename = os.path.basename(saved_path)
                file_entry = {
                    "filename": filename,
                    "path": saved_path,
                    "type": "image/png", # Assuming PNG, adjust if needed
                    "description": "Final screenshot from MCP agent.",
                }
                # Add base64 if URI available
                if saved_uri:
                    file_entry["base64_data"] = saved_uri
                current_task["output_files"].append(file_entry)
        # <<< END NEW >>>

        current_task["error_log"] = None
        current_task["feedback_log"] = None

        # Status Logic remains the same: "completed", let PM/Eva handle next steps
        current_task["status"] = "completed"
        print(f"Task {current_task['task_id']}: Execution successful. Status: {current_task['status']}")

    return current_task

# --- run_web_search_tool_node ---
# Placeholder for _save_file - needs actual implementation if saving web search images
def _save_file(data: bytes, prefix: str, ext: str, is_binary: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Saves data to a file in OUTPUT_DIR. Needs implementation."""
    # WARNING: Dummy implementation. Needs proper file saving logic.
    try:
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
        # Use imported OUTPUT_DIR
        filepath = os.path.join(OUTPUT_DIR, filename)
        mode = "wb" if is_binary else "w"
        encoding = None if is_binary else "utf-8"
        with open(filepath, mode, encoding=encoding) as f:
            f.write(data)
        print(f"Placeholder _save_file: Saved to {filepath}")
        return filepath, filename
    except Exception as e:
        print(f"Placeholder _save_file Error: Failed to save {prefix}.{ext}: {e}")
        return None, None

# Placeholder for _save_tool_output_file - needs actual implementation
def _save_tool_output_file(filename: str, cache_dir: str, mime_type: str, description: str) -> Optional[Dict[str, str]]:
    """Checks for file in cache, returns info dict. Needs implementation."""
    # WARNING: Dummy implementation. Needs proper file checking/handling.
    # Use imported cache_dir constant indirectly via argument
    try:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            print(f"Placeholder _save_tool_output_file: Found {filepath}")
            file_info = {"filename": filename, "path": filepath, "type": mime_type, "description": description}
             # Add base64 for images if needed for downstream tasks
            if mime_type.startswith("image/"):
                 try:
                     with open(filepath, "rb") as f: encoded_string = base64.b64encode(f.read()).decode('utf-8')
                     file_info["base64_data"] = f"data:{mime_type};base64,{encoded_string}"
                 except Exception as e: print(f"Warning: Could not encode image {filepath}: {e}")
            return file_info
        else:
            print(f"Placeholder _save_tool_output_file Error: File not found at {filepath}")
            return None
    except Exception as e:
        print(f"Placeholder _save_tool_output_file Error: {e}")
        return None

# =============================================================================
# 6. Subgraph Node Definitions
# =============================================================================

# --- Node: Prepare Tool Inputs ---
async def prepare_tool_inputs_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Uses LLM to prepare the 'task_inputs' field in the current TaskState.
    Includes logic for the new RhinoMCPCoordinator.
    """
    node_name = "Prepare Tool Inputs"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]

    if current_idx < 0 or current_idx >= len(tasks):
        print(f"{node_name} Error: Invalid task index {current_idx}.")
        return {"tasks": state.get("tasks", []), "current_task": state.get("current_task")}

    current_task = tasks[current_idx] # Work on the copy
    selected_agent_name = current_task.get('selected_agent')

    if not selected_agent_name:
        prep_error = "Input Preparation Failed: No agent selected for the task."
        _set_task_failed(current_task, prep_error, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks, "current_task": current_task.copy()}

    print(f"--- {node_name}: Preparing Inputs for Agent: {selected_agent_name} ---")

    # --- Aggregation Logic (remains the same) ---
    task_objective = current_task.get('task_objective', '')
    task_description = current_task.get('description', '')
    initial_plan_suggestion = current_task.get('task_inputs', {})
    overall_goal = state.get('user_input', "N/A")
    print(f"{node_name}: Aggregating outputs and files from previous tasks...")
    aggregated_outputs = {}
    aggregated_files_raw = []
    aggregated_summary_parts = ["Workflow History Summary:"]
    for i, task in enumerate(tasks[:current_idx]):
        task_id = task.get("task_id", f"task_{i}")
        task_desc = task.get('description', 'N/A')
        task_status = task.get("status")
        agent_used = task.get("selected_agent", "N/A")
        objective = task.get("task_objective", "N/A")
        aggregated_summary_parts.append(f"  Task {i} (ID: {task_id}, Agent: {agent_used}, Status: {task_status}): {task_desc} (Objective: {objective})")
        if task_status == "completed":
            task_outputs = task.get("outputs")
            if task_outputs: aggregated_outputs[task_id] = task_outputs; aggregated_summary_parts.append(f"    Outputs: {json.dumps(task_outputs, ensure_ascii=False, indent=None)}")
            task_files = task.get("output_files")
            if task_files:
                for file_dict in task_files:
                     if 'source_task_id' not in file_dict: file_dict['source_task_id'] = task_id
                aggregated_files_raw.extend(task_files); aggregated_summary_parts.append(f"    Files Generated: {[f.get('filename', 'N/A') for f in task_files]}")
        elif task_status == "failed" or task_status == "max_retries_reached":
             error_log = task.get('error_log', 'N/A'); feedback_log = task.get('feedback_log', 'N/A')
             aggregated_summary_parts.append(f"    Status: {task_status} - Error: {error_log[:100]}... Feedback: {feedback_log[:100]}...")
    filtered_aggregated_files = _filter_base64_from_files(aggregated_files_raw)
    aggregated_outputs_json = json.dumps(aggregated_outputs, ensure_ascii=False)
    aggregated_files_json = json.dumps(filtered_aggregated_files, ensure_ascii=False)
    aggregated_summary_str = "\n".join(aggregated_summary_parts)
    print(f"{node_name}: Aggregation complete. {len(aggregated_outputs)} output sets, {len(filtered_aggregated_files)} files (filtered).")
    # --- End Aggregation Logic ---

    error_feedback_str = "N/A (First Attempt)"
    last_error = current_task.get('error_log')
    last_feedback = current_task.get('feedback_log')
    if last_error or last_feedback:
        error_feedback_str = f"Context from Previous Attempt (Task {current_task['task_id']}):\nLast Error: {last_error or 'None'}\nLast Feedback/Analysis: {last_feedback or 'None'}"
        print(f"{node_name}: Including feedback for retry:\n{error_feedback_str[:200]}...")

    runtime_config = config["configurable"]
    prep_llm_config_dict = runtime_config.get("aa_llm", {})
    llm = initialize_llm(prep_llm_config_dict)
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    agent_description = agent_descriptions.get(selected_agent_name, "No description available.")

    prompt_config_obj = config_manager.get_prompt("assign_agent", "prepare_tool_inputs_prompt")
    prepare_inputs_prompt_template_str = prompt_config_obj.template if prompt_config_obj else None

    if not prepare_inputs_prompt_template_str:
        prep_error = "Input Preparation Failed: Missing 'prepare_tool_inputs_prompt' template!"
        _set_task_failed(current_task, prep_error, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks, "current_task": current_task.copy()}

    try:
        prompt_inputs = {
            "selected_agent_name": selected_agent_name, "agent_description": agent_description,
            "user_input": overall_goal, "task_objective": task_objective, "task_description": task_description,
            "initial_plan_suggestion_json": json.dumps(initial_plan_suggestion, ensure_ascii=False, indent=2),
            "aggregated_summary": aggregated_summary_str, "aggregated_outputs_json": aggregated_outputs_json,
            "aggregated_files_json": aggregated_files_json, "error_feedback": error_feedback_str,
            "llm_output_language": llm_output_language
        }
        required_format_keys = list(prompt_inputs.keys()) # Get keys dynamically
        missing_format_keys = [key for key in required_format_keys if key not in prompt_inputs] # Should be empty now
        if missing_format_keys: raise KeyError(f"Internal Error: Missing keys required for prompt formatting: {missing_format_keys}")

        print(f"{node_name}: Formatting prompt '{prepare_inputs_prompt_template_str[:50]}...'")
        prep_prompt = prepare_inputs_prompt_template_str.format(**prompt_inputs)
        print(f"{node_name}: Invoking LLM for agent {selected_agent_name}...")
        prep_response = await llm.ainvoke(prep_prompt)
        prep_content = prep_response.content.strip()

        if prep_content.startswith("```json"): prep_content = prep_content[7:-3].strip()
        elif prep_content.startswith("```"): prep_content = prep_content[3:-3].strip()

        try:
            prepared_inputs = json.loads(prep_content)
            if isinstance(prepared_inputs, dict) and "error" in prepared_inputs:
                err_msg = f"Input Preparation Failed (LLM): {prepared_inputs['error']}"
                _set_task_failed(current_task, err_msg, node_name)
            elif not isinstance(prepared_inputs, dict):
                err_msg = f"Input Prep Failed: LLM returned invalid format (expected dict). Content: {prep_content}"
                _set_task_failed(current_task, err_msg, node_name)
            else:
                # --- Validation (using imported constants and adding RhinoMCPCoordinator) ---
                agent_key_map = {
                    "ArchRAGAgent": ["prompt"], "WebSearchAgent": ["prompt"],
                    "ImageRecognitionAgent": ["image_paths", "prompt"], "VideoRecognitionAgent": ["video_paths", "prompt"],
                    "Generate3DAgent": ["image_path"], "CaseRenderAgent": ["outer_prompt", "i", "strength"],
                    "SimulateFutureAgent": ["outer_prompt", "render_image"], "ImageGenerationAgent": ["prompt"],
                    "LLMTaskAgent": ["prompt"],
                    # <<< NEW: Define required inputs for Rhino coordinator >>>
                    # LLM should extract the core request and optional image path here
                    "RhinoMCPCoordinator": ["user_request"] # initial_image_path is optional
                }
                required_keys = agent_key_map.get(selected_agent_name, [])
                missing_keys = []
                invalid_paths = []
                invalid_values = []

                for key in required_keys + (["initial_image_path"] if selected_agent_name == "RhinoMCPCoordinator" else []): # 檢查可選的 key
                    value = prepared_inputs.get(key)
                    is_optional_rhino_path = (selected_agent_name == "RhinoMCPCoordinator" and key == "initial_image_path")

                    # 檢查必需鍵是否存在且非空
                    if key in required_keys and (value is None or (isinstance(value, (str, list)) and not value)):
                         missing_keys.append(key)
                         continue
                    # 如果是可選的 Rhino 路徑，檢查是否存在且有效
                    elif is_optional_rhino_path and value and (not isinstance(value, str) or not os.path.exists(value)):
                         invalid_paths.append(f"{key}: '{value}' (Optional path provided but not found or invalid type)")
                         continue
                    elif not is_optional_rhino_path and value is None: # Skip if optional and not provided
                         continue

                    # --- Type/Value/Path Validation (對其他 key 的驗證邏輯保持不變) ---
                    if key in ["image_paths", "video_paths"] and isinstance(value, list):
                         for path in value:
                             if not isinstance(path, str) or not os.path.exists(path): invalid_paths.append(f"{key}: '{path}'")
                    elif key == "image_path" and isinstance(value, str):
                         if not os.path.exists(value): invalid_paths.append(f"{key}: '{value}'")
                    elif key == "render_image" and isinstance(value, str):
                        # 檢查 cache 或 workspace 路徑
                        full_path_cache = os.path.join(RENDER_CACHE_DIR, value)
                        full_path_output = os.path.join(OUTPUT_DIR, value) # 假設也可能在 output 根目錄
                        if not os.path.exists(full_path_cache) and not os.path.exists(full_path_output) and not os.path.exists(value):
                             invalid_paths.append(f"{key}: '{value}' (Not found in cache, output, or as absolute path)")
                    elif key == "i" and selected_agent_name == "CaseRenderAgent":
                         try: prepared_inputs[key] = int(value)
                         except (ValueError, TypeError): invalid_values.append(f"{key}: '{value}' (Must be int)")
                    elif key == "strength" and selected_agent_name == "CaseRenderAgent":
                        try:
                             strength_val = float(value)
                             if not (0.0 <= strength_val <= 1.0): # Strength range might be 0-1 for some tools
                                 print(f"Warning: CaseRenderAgent strength {strength_val} outside typical 0.0-0.8 range. Allowing 0.0-1.0.")
                             prepared_inputs[key] = str(strength_val) # Keep as string if tool expects string
                        except (ValueError, TypeError): invalid_values.append(f"{key}: '{value}' (Must be float-like string)")
                    elif key in ["prompt", "outer_prompt", "user_request"] and not isinstance(value, str):
                         invalid_values.append(f"{key}: Value must be a string.")

                if missing_keys or invalid_paths or invalid_values:
                    error_parts = []
                    if missing_keys: error_parts.append(f"Missing/empty required keys: {', '.join(missing_keys)}")
                    if invalid_paths: error_parts.append(f"Invalid/missing paths: {', '.join(invalid_paths)}")
                    if invalid_values: error_parts.append(f"Invalid values: {', '.join(invalid_values)}")
                    validation_err_msg = f"Input Validation Failed for {selected_agent_name}: {'. '.join(error_parts)}. LLM Output: {json.dumps(prepared_inputs, ensure_ascii=False)}"
                    _set_task_failed(current_task, validation_err_msg, node_name)
                else:
                    print(f"{node_name}: Inputs prepared and validated successfully: {list(prepared_inputs.keys())}")
                    current_task["task_inputs"] = prepared_inputs
                    current_task["error_log"] = None; current_task["feedback_log"] = None
                    current_task["status"] = "in_progress" # Ready for tool/coordinator execution

        except json.JSONDecodeError:
            err_msg = f"Input Prep Failed: Could not parse LLM JSON response. Raw content: '{prep_content}'"
            _set_task_failed(current_task, err_msg, node_name)

    except KeyError as ke:
         err_msg = f"Input Prep Failed: Formatting error, missing key {ke}."
         traceback.print_exc(); _set_task_failed(current_task, err_msg, node_name)
    except Exception as prep_e:
        err_msg = f"Input Prep Failed: Unexpected error: {prep_e}"
        traceback.print_exc(); _set_task_failed(current_task, err_msg, node_name)

    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}


# --- Tool Subgraph Router ---
def determine_tool_route(state: WorkflowState) -> str: # 返回值改為 str
    """Determines the next tool node based on the current task's selected agent."""
    print(f"--- Tool Subgraph Router Node ---")
    current_idx = state.get("current_task_index", -1)
    tasks = state.get("tasks", [])

    # 檢查索引有效性
    if current_idx < 0 or current_idx >= len(tasks):
        print("Tool Subgraph Router Error: Invalid task index."); return END # 直接返回 END

    current_task = tasks[current_idx]
    agent_name = current_task.get("selected_agent")

    # 檢查 agent_name 是否存在
    if not agent_name:
        print(f"Tool Subgraph Router Error: No 'selected_agent'. Routing to END."); return END # 直接返回 END

    print(f"--- Tool Subgraph Router: Routing for agent '{agent_name}' ---")

    node_mapping = {
        "ArchRAGAgent": "rag_agent", "ImageGenerationAgent": "image_gen_agent",
        "WebSearchAgent": "web_search_agent", "CaseRenderAgent": "case_render_agent",
        "Generate3DAgent": "generate_3d_agent", "SimulateFutureAgent": "simulate_future_agent",
        "VideoRecognitionAgent": "video_recognition_agent", "ImageRecognitionAgent": "image_recognition_agent",
        "LLMTaskAgent": "llm_task_agent",
        # <<< NEW: Add mapping for Rhino MCP Coordinator >>>
        "RhinoMCPCoordinator": "rhino_mcp_node" # 使用新節點名稱
    }
    target_node = node_mapping.get(agent_name)

    if target_node:
        print(f"--- Tool Subgraph Router: Target node: '{target_node}' ---"); return target_node # 返回目標節點名稱
    else:
        print(f"Tool Subgraph Router Error: Unknown agent name '{agent_name}'. Routing to END."); return END # 直接返回 END

# --- run_llm_task_node ---
async def run_llm_task_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running LLM Task Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])} # Return original if invalid
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        prompt_for_llm = inputs.get("prompt")
        if not prompt_for_llm: raise ValueError("Missing 'prompt' in task_inputs")

        runtime_config = config["configurable"]
        llm_config_dict = runtime_config.get("ta_llm", {})
        llm = initialize_llm(llm_config_dict)
        print(f"LLM Task Node: Invoking LLM ({llm.__class__.__name__})...") # Log LLM class
        response = await llm.ainvoke(prompt_for_llm)
        result_content = response.content.strip()
        print(f"LLM Task Node: Received response: {result_content[:100]}...")
        final_outputs = {"content": result_content}
        final_output_files = []
    except Exception as e:
        error_to_report = e
        final_outputs = None
        final_output_files = None

    # Use local helper to update task copy
    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=final_output_files, error=error_to_report)
    # Return the modified task list
    return {"tasks": tasks}

# --- run_rag_tool_node ---
def run_rag_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running RAG Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    final_outputs = None
    final_output_files = None

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        query = inputs.get("prompt")
        if not query: raise ValueError("Missing 'prompt' in prepared task_inputs")
        top_k = int(inputs.get("top_k", 5))
        print(f"RAG Node: Using query='{query}', top_k={top_k}")

        result = ARCH_rag_tool({"query": query, "top_k": top_k}) # Use imported tool
        print(f"RAG Node: Received result: {str(result)[:100]}...") # Handle non-string results if necessary
        if isinstance(result, str) and result.startswith("Error"): raise ValueError(f"Tool Error: {result}")

        final_outputs = {"content": result} # Assuming result is the content
        final_output_files = []
    except Exception as e:
        error_to_report = e
        # final_outputs/files remain None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=final_output_files, error=error_to_report)
    # Return modified task list and the current task
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_image_gen_tool_node ---
def run_image_gen_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Image Gen Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        prompt = inputs.get("prompt")
        if not prompt: raise ValueError("Missing 'prompt' in task_inputs")
        image_inputs = inputs.get("image_inputs") # Optional
        print(f"Image Gen Node: Using prompt='{prompt[:50]}...', image_inputs: {'Provided' if image_inputs else 'None'}")

        # Use imported tool and OUTPUT_DIR constant
        tool_result = generate_gemini_image({"prompt": prompt, "image_inputs": image_inputs or []})

        print(f"DEBUG: ImageGen tool_result: {json.dumps(tool_result, indent=2, ensure_ascii=False)}")

        if isinstance(tool_result, dict) and tool_result.get("error"): raise ValueError(f"Tool Error: {tool_result['error']}")

        text_response = tool_result.get("text_response")
        tool_generated_files = tool_result.get("generated_files", [])

        if tool_generated_files and isinstance(tool_generated_files, list):
            print(f"Image Gen Node: Tool reported {len(tool_generated_files)} generated files. Processing...")
            for i, file_info in enumerate(tool_generated_files):
                if not isinstance(file_info, dict):
                    print(f"Warning: Item {i} in generated_files is not a dictionary: {file_info}")
                    continue

                filename = file_info.get("filename")
                if not filename or not isinstance(filename, str):
                     print(f"  - Warning: Invalid or missing 'filename': {file_info}. Skipping.")
                     continue

                # Use imported OUTPUT_DIR
                expected_path = os.path.join(OUTPUT_DIR, filename)
                file_type = file_info.get("type", "image/png")
                description = file_info.get("description", f"Generated image {i+1} for: {prompt[:30]}...")

                print(f"  - Checking constructed path: {expected_path}")
                if os.path.exists(expected_path):
                    print(f"  - File FOUND at constructed path: {expected_path}")
                    processed_file_info = {"filename": filename, "path": expected_path, "type": file_type, "description": description}
                    if file_type.startswith("image/"):
                        try:
                            with open(expected_path, "rb") as image_file: encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                            processed_file_info["base64_data"] = f"data:{file_type};base64,{encoded_string}"
                            print(f"    - Added base64 data URI for image: {filename}")
                        except Exception as e: print(f"    - Warning: Could not read/encode image {expected_path}: {e}")
                    output_files_list.append(processed_file_info)
                else:
                    print(f"  - Warning: File '{filename}' not found at expected path '{expected_path}'. Skipping.")

        elif not tool_generated_files: print("Image Gen Node: Tool did not report any generated files.")
        else: print(f"Warning: 'generated_files' from tool is not a list: {tool_generated_files}")

        if text_response: final_outputs["text_response"] = text_response

    except Exception as e:
        error_to_report = e
        final_outputs = {} # Clear on error
        traceback.print_exc()

    if error_to_report is None and not output_files_list:
        no_image_error_msg = "Image Generation Tool ran but produced no processable output image files."
        print(f"Image Gen Node Warning: {no_image_error_msg}")
        error_to_report = ValueError(no_image_error_msg)

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=output_files_list or [], error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

def run_web_search_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Web Search Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        query = inputs.get("prompt")
        if not query: raise ValueError("Missing 'prompt' in task_inputs")
        print(f"Web Search Node: Using query='{query}'")

        # Use imported tool
        result = perform_grounded_search({"query": query})
        if isinstance(result, dict) and result.get("error"): raise ValueError(f"Tool Error: {result['error']}")
        if not isinstance(result, dict): raise ValueError(f"Tool returned unexpected result type: {type(result)}")

        returned_images = result.get("images", [])
        if returned_images:
            for i, img_info in enumerate(returned_images):
                img_bytes = img_info.get("data")
                mime_type = img_info.get("mime_type", "image/png")
                img_desc = img_info.get("description", f"Web search result {i+1} for '{query[:30]}...'")
                if isinstance(img_bytes, bytes):
                    file_ext = mime_type.split('/')[-1] if '/' in mime_type else 'png'
                    if file_ext == 'jpeg': file_ext = 'jpg'
                    # Use the placeholder _save_file (requires real implementation)
                    filepath, filename = _save_file(img_bytes, f"web_search_{i}", file_ext, is_binary=True)
                    if filepath:
                        output_files_list.append({"filename": filename, "path": filepath, "type": mime_type, "description": img_desc})

        final_outputs = {
            "text_content": result.get("text_content"), "grounding_sources": result.get("grounding_sources"),
            "search_suggestions": result.get("search_suggestions"),
        }
        if not final_outputs.get("text_content") and not output_files_list:
             print("Web Search Warning: Tool returned no text content or processable images.")
             # Decide if this is an error state or just an empty result
             # For now, let's consider it a non-error empty result

    except Exception as e:
        error_to_report = e
        final_outputs = None # Clear on error
        output_files_list = None # Clear on error

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=output_files_list, error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_case_render_tool_node ---
def run_case_render_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Case Render Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        outer_prompt = inputs.get("outer_prompt")
        i = inputs.get("i")
        strength = inputs.get("strength")
        if not outer_prompt or i is None or strength is None: raise ValueError("Missing required inputs ('outer_prompt', 'i', 'strength')")

        print(f"Case Render Node: Using outer_prompt='{outer_prompt[:50]}...', count(i)={i}, strength={strength}")

        # Use imported tool
        result = case_render_image({"outer_prompt": outer_prompt, "i": i, "strength": strength})
        if isinstance(result, dict) and "Error:" in result.get("error", ""): raise ValueError(f"Tool Error: {result['error']}")
        if not isinstance(result, str) or not result: raise ValueError(f"Tool returned unexpected/empty result: {result}")

        filenames = result.split(',')
        for filename in filenames:
            fn_stripped = filename.strip()
            if fn_stripped:
                # Use placeholder _save_tool_output_file and imported RENDER_CACHE_DIR
                file_info = _save_tool_output_file(fn_stripped, RENDER_CACHE_DIR, "image/png", f"ComfyUI render: {outer_prompt[:50]}...")
                if file_info: output_files_list.append(file_info)

        if not output_files_list: raise ValueError("Tool ran but failed to produce/locate output files.")
        final_outputs = {"generated_filenames": filenames}

    except Exception as e:
        error_to_report = e
        final_outputs = None
        output_files_list = None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=output_files_list, error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_generate_3d_tool_node ---
def run_generate_3d_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Generate 3D Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        image_path = inputs.get("image_path")
        if not image_path: raise ValueError("Missing 'image_path'")
        print(f"Generate 3D Node: Using image_path='{image_path}'")

        # Use imported tool
        result = generate_3D({"image_path": image_path})
        if isinstance(result, dict) and "error" in result: raise ValueError(f"Tool Error: {result['error']}")
        if not isinstance(result, dict): raise ValueError(f"Tool returned unexpected result type: {type(result)}")

        model_filename = result.get("model")
        video_filename = result.get("video")
        # Use placeholder _save_tool_output_file and imported MODEL_CACHE_DIR
        if model_filename:
            file_info = _save_tool_output_file(model_filename, MODEL_CACHE_DIR, "model/gltf-binary", f"3D model from {os.path.basename(image_path)}")
            if file_info: output_files_list.append(file_info)
        if video_filename:
            file_info = _save_tool_output_file(video_filename, MODEL_CACHE_DIR, "video/mp4", f"Preview video from {os.path.basename(image_path)}")
            if file_info: output_files_list.append(file_info)

        if not output_files_list: raise ValueError("Tool ran but produced no model or video file.")
        final_outputs = {"model_filename": model_filename, "video_filename": video_filename}

    except Exception as e:
        error_to_report = e
        final_outputs = None
        output_files_list = None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=output_files_list, error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_simulate_future_tool_node ---
def run_simulate_future_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Simulate Future Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        outer_prompt = inputs.get("outer_prompt")
        render_image_filename = inputs.get("render_image")
        if not outer_prompt: raise ValueError("Missing 'outer_prompt'")
        if not render_image_filename: raise ValueError("Missing 'render_image' filename")
        print(f"Simulate Future Node: Using outer_prompt='{outer_prompt[:50]}...', render_image='{render_image_filename}'")

        # Use imported tool
        result = simulate_future_image({"outer_prompt": outer_prompt, "render_image": render_image_filename})
        if isinstance(result, dict) and result.get("error"): raise ValueError(f"Tool Error: {result['error']}")
        if not isinstance(result, str) or not result: raise ValueError(f"Tool returned unexpected/empty result: {result}")

        output_filename = result
        # Use placeholder _save_tool_output_file and imported RENDER_CACHE_DIR
        file_info = _save_tool_output_file(output_filename, RENDER_CACHE_DIR, "image/png", f"Future simulation: {outer_prompt[:50]}...")
        if file_info: output_files_list.append(file_info)
        else: raise ValueError("Tool ran but failed to produce/locate output file.")

        final_outputs = {"generated_filename": output_filename}

    except Exception as e:
        error_to_report = e
        final_outputs = None
        output_files_list = None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=output_files_list, error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_video_recognition_tool_node ---
def run_video_recognition_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Video Rec Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        prompt = inputs.get("prompt")
        video_paths = inputs.get("video_paths")
        if not prompt: raise ValueError("Missing 'prompt'")
        if not video_paths or not isinstance(video_paths, list): raise ValueError("Missing or invalid 'video_paths'")

        print(f"Video Rec Node: Using prompt='{prompt[:50]}...', video_paths={video_paths}")

        # Use imported tool
        result = video_recognition({"video_paths": video_paths, "prompt": prompt})
        if not isinstance(result, str): raise ValueError(f"Tool returned non-string result: {type(result)}")
        # Check for common error strings
        if any(err_str in result for err_str in ["不支援", "錯誤", "Error"]): raise ValueError(f"Tool Error: {result}")

        final_outputs = {"analysis": result}

    except Exception as e:
        error_to_report = e
        final_outputs = None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=[], error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_image_recognition_tool_node ---
def run_image_recognition_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    print("--- Running Image Rec Tool Node (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict): raise ValueError("Invalid or missing 'task_inputs'")
        prompt = inputs.get("prompt")
        image_paths = inputs.get("image_paths")
        if not prompt: raise ValueError("Missing 'prompt'")
        if not image_paths or not isinstance(image_paths, list): raise ValueError("Missing or invalid 'image_paths'")

        print(f"Image Rec Node: Using prompt='{prompt[:50]}...', image_paths={image_paths}")

        # Use imported tool
        result = img_recognition({"image_paths": image_paths, "prompt": prompt})
        if not isinstance(result, str): raise ValueError(f"Tool returned non-string result: {type(result)}")
        if any(err_str in result for err_str in ["錯誤", "Error", "找不到"]): raise ValueError(f"Tool Error: {result}")

        final_outputs = {"description": result}

    except Exception as e:
        error_to_report = e
        final_outputs = None

    tasks[current_idx] = _update_task_state_after_tool(current_task, outputs=final_outputs, output_files=[], error=error_to_report)
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- <<< NEW: Rhino MCP Coordinator Node >>> ---
async def run_rhino_mcp_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes a Rhino task using the MCP agent logic within this node.
    Uses TaskState inputs and updates TaskState outputs/status.
    """
    node_name = "Rhino MCP Node" # Renamed
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])] # Operate on a copy

    if current_idx < 0 or current_idx >= len(tasks):
        print(f"{node_name} Error: Invalid task index {current_idx}.")
        return {"tasks": [t.copy() for t in state.get("tasks", [])], "current_task": state.get("current_task").copy() if state.get("current_task") else None}

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs")
    outer_error_to_report = None # Error related to this node's execution
    final_mcp_outcome = None   # To store the final outcome dict from the MCP loop

    try:
        # 1. Extract inputs from TaskState
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs' in the current task.")
        user_request = task_inputs.get("user_request")
        initial_image_path = task_inputs.get("initial_image_path") # Optional

        if not user_request:
            raise ValueError("Missing 'user_request' in task_inputs for RhinoMCPCoordinator")

        print(f"{node_name}: Starting MCP task for request: '{user_request[:100]}...'")
        if initial_image_path: print(f"  with initial image: {initial_image_path}")

        # 2. Construct Initial *Local* MCP State (mimics MCPAgentState)
        local_mcp_state: Dict[str, Any] = { # Use Dict for flexibility
            "messages": [],
            # Store initial request/image path locally if needed by mcp_should_continue etc.
            "initial_request": user_request,
            "initial_image_path": initial_image_path,
            "target_mcp": "rhino", # Implicitly rhino for this node
            "task_complete": False,
            "saved_image_path": None,
            "saved_image_data_uri": None
        }

        # Construct initial HumanMessage for the local MCP state
        initial_human_content = [{"type": "text", "text": user_request}]
        if initial_image_path and os.path.exists(initial_image_path):
            try:
                with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                initial_human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
                print(f"  Successfully encoded initial image for local MCP state.")
            except Exception as img_err:
                print(f"  Warning: Failed to read/encode initial image {initial_image_path} for local MCP state: {img_err}")
        local_mcp_state["messages"] = [HumanMessage(content=initial_human_content)]

        # 3. Run Internal MCP Loop
        max_mcp_steps = 15
        step_count = 0
        mcp_loop_error = None # Error specifically from the internal loop
        while step_count < max_mcp_steps:
            step_count += 1
            print(f"\n{node_name}: --- MCP Internal Loop Step {step_count} ---")
            # Pass the local state to MCP functions
            next_step = mcp_should_continue(local_mcp_state)
            print(f"  mcp_should_continue result: {next_step}")

            if next_step == END:
                print(f"{node_name}: MCP loop finished based on should_continue.")
                break
            elif next_step == "rhino_agent":
                print(f"  Calling call_rhino_agent...")
                # Pass the existing config down
                agent_output = await call_rhino_agent(local_mcp_state, config)
                local_mcp_state["messages"].extend(agent_output.get("messages", []))
                # Update local state with potential image path/uri from agent output
                if "saved_image_path" in agent_output: local_mcp_state["saved_image_path"] = agent_output["saved_image_path"]
                if "saved_image_data_uri" in agent_output: local_mcp_state["saved_image_data_uri"] = agent_output["saved_image_data_uri"]
            elif next_step == "agent_tool_executor":
                print(f"  Calling agent_tool_executor...")
                 # Pass the existing config down
                executor_output = await agent_tool_executor(local_mcp_state, config)
                local_mcp_state["messages"].extend(executor_output.get("messages", []))
                 # Update local state with potential image path from tool message content
                last_tool_msg = local_mcp_state["messages"][-1] if local_mcp_state["messages"] and isinstance(local_mcp_state["messages"][-1], ToolMessage) else None
                if last_tool_msg and last_tool_msg.name == "capture_viewport" and last_tool_msg.content.startswith("[IMAGE_FILE_PATH]:"):
                     local_mcp_state["saved_image_path"] = last_tool_msg.content.split(":", 1)[1]
            else:
                mcp_loop_error = ValueError(f"MCP loop error: Unknown step '{next_step}'")
                print(f"{node_name} Error: {mcp_loop_error}")
                break

            # --- Short delay ---
            await asyncio.sleep(1)

        # Check for timeout
        if step_count >= max_mcp_steps:
            print(f"{node_name} Warning: Reached max MCP steps ({max_mcp_steps}). Ending loop.")
            mcp_loop_error = TimeoutError("MCP task exceeded maximum steps.")

        # 4. Extract Final Result from *local* MCP State
        final_message_obj = local_mcp_state["messages"][-1] if local_mcp_state["messages"] else None
        final_mcp_outcome = {
            "message": "MCP task finished without a final message.", # Default message
            "saved_image_path": local_mcp_state.get("saved_image_path"),
            "saved_image_data_uri": local_mcp_state.get("saved_image_data_uri")
        }

        if isinstance(final_message_obj, AIMessage):
            final_mcp_outcome["message"] = final_message_obj.content
        elif isinstance(final_message_obj, ToolMessage):
             # Handle cases where loop ended after tool call
             if final_message_obj.name == "capture_viewport" and final_message_obj.content.startswith("[IMAGE_FILE_PATH]:"):
                  saved_path = final_message_obj.content.split(":", 1)[1]
                  final_mcp_outcome["message"] = f"MCP task completed with screenshot: {saved_path}"
                  # Path/URI already set in local_mcp_state
             elif final_message_obj.content.startswith("[Error"):
                  final_mcp_outcome["message"] = f"MCP task ended after tool call which reported error: {final_message_obj.content}"
                  # Report the tool error as the primary error if no loop error occurred yet
                  if not mcp_loop_error: mcp_loop_error = ValueError(f"MCP Tool Error: {final_message_obj.content}")
             else:
                  final_mcp_outcome["message"] = f"MCP task ended after tool '{final_message_obj.name}' call."

        if mcp_loop_error:
            final_mcp_outcome["message"] = f"MCP task failed: {str(mcp_loop_error)}"
            outer_error_to_report = mcp_loop_error # Propagate loop error to task status

    except Exception as e:
        error_msg = f"Error during {node_name} execution: {e}"
        print(error_msg)
        traceback.print_exc()
        outer_error_to_report = e # Set error for the main task state update
        # Ensure final_mcp_outcome reflects the error
        final_mcp_outcome = {
             "message": f"MCP coordination failed: {e}",
             "saved_image_path": None,
             "saved_image_data_uri": None
        }

    # 5. Update the main WorkflowState Task using the helper
    # Pass the extracted outcome to the helper via mcp_result
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=None, # Standard outputs are usually None for MCP coordinator itself
        output_files=None, # Output files are derived from mcp_result
        error=outer_error_to_report, # Report coordination error if any
        mcp_result=final_mcp_outcome # Pass the final MCP outcome dict
    )

    # Return the updated task list for WorkflowState
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}
# --- <<< END MODIFICATION >>> ---


# =============================================================================
# 7. Subgraph Definition & Compilation
# =============================================================================
tool_subgraph_builder = StateGraph(WorkflowState) # Use imported WorkflowState

# Add nodes using the functions defined above
tool_subgraph_builder.add_node("prepare_tool_inputs", prepare_tool_inputs_node)
tool_subgraph_builder.add_node("route_to_tool", determine_tool_route)
tool_subgraph_builder.add_node("rag_agent", run_rag_tool_node)
tool_subgraph_builder.add_node("image_gen_agent", run_image_gen_tool_node)
tool_subgraph_builder.add_node("web_search_agent", run_web_search_tool_node)
tool_subgraph_builder.add_node("case_render_agent", run_case_render_tool_node)
tool_subgraph_builder.add_node("generate_3d_agent", run_generate_3d_tool_node)
tool_subgraph_builder.add_node("simulate_future_agent", run_simulate_future_tool_node)
tool_subgraph_builder.add_node("video_recognition_agent", run_video_recognition_tool_node)
tool_subgraph_builder.add_node("image_recognition_agent", run_image_recognition_tool_node)
tool_subgraph_builder.add_node("llm_task_agent", run_llm_task_node)
# <<< NEW: Add Rhino MCP node with updated name >>>
tool_subgraph_builder.add_node("rhino_mcp_node", run_rhino_mcp_node)


# --- Routing logic after input prep (保持不變) ---
def route_after_input_prep(state: WorkflowState) -> Literal["route_to_tool", "finished"]:
    # ... (此函數邏輯保持不變) ...
    current_idx = state.get("current_task_index", -1)
    tasks = state.get("tasks", [])
    status = "unknown"
    if 0 <= current_idx < len(tasks):
        status = tasks[current_idx].get("status", "unknown")
    if status == "failed":
        print("--- Tool Subgraph: Routing to END due to input preparation error ---")
        return "finished"
    elif status == "in_progress":
        print("--- Tool Subgraph: Input preparation successful, routing to tool selection ---")
        return "route_to_tool"
    else:
        print(f"--- Tool Subgraph: Unexpected status '{status}' after input prep. Routing to END. ---")
        return "finished"

# Set Entry Point (保持不變)
tool_subgraph_builder.set_entry_point("prepare_tool_inputs")

# Conditional Edge from Input Prep to Router or END (保持不變)
tool_subgraph_builder.add_conditional_edges(
    "prepare_tool_inputs", 
    route_after_input_prep, 
    {
    "route_to_tool": "route_to_tool", 
    "finished": END
    }
)

# Conditional Edges from Router to Specific Tools or END (更新 Rhino 節點名稱)
tool_subgraph_builder.add_conditional_edges(
    "route_to_tool",
    determine_tool_route, # 使用更新後的函數
    {
        "rag_agent": "rag_agent", "image_gen_agent": "image_gen_agent",
        "web_search_agent": "web_search_agent", "case_render_agent": "case_render_agent",
        "generate_3d_agent": "generate_3d_agent", "simulate_future_agent": "simulate_future_agent",
        "video_recognition_agent": "video_recognition_agent", "image_recognition_agent": "image_recognition_agent",
        "llm_task_agent": "llm_task_agent",
        # <<< NEW: Add route to Rhino node with updated name >>>
        "rhino_mcp_node": "rhino_mcp_node",
        "finished": END # 確保 END 路徑存在
    }
)

# Edges from ALL Tool Nodes (including the new one) to END
# ... (為其他工具節點添加邊緣) ...
tool_subgraph_builder.add_edge("rag_agent", END)
tool_subgraph_builder.add_edge("image_gen_agent", END)
tool_subgraph_builder.add_edge("web_search_agent", END)
tool_subgraph_builder.add_edge("case_render_agent", END)
tool_subgraph_builder.add_edge("generate_3d_agent", END)
tool_subgraph_builder.add_edge("simulate_future_agent", END)
tool_subgraph_builder.add_edge("video_recognition_agent", END)
tool_subgraph_builder.add_edge("image_recognition_agent", END)
tool_subgraph_builder.add_edge("llm_task_agent", END)
# <<< NEW: Add edge from Rhino node with updated name >>>
tool_subgraph_builder.add_edge("rhino_mcp_node", END)


# Compile the subgraph and assign it to the original variable name (保持不變)
AssignTeamsSubgraph_Compiled = tool_subgraph_builder.compile()
AssignTeamsSubgraph_Compiled.name = "AssignTeamsSubgraph_Compiled"
assign_teams = AssignTeamsSubgraph_Compiled # Assign back to assign_teams for consistent import
print("Unified Tool Subgraph ('assign_teams') compiled successfully with Rhino MCP Coordinator.")

