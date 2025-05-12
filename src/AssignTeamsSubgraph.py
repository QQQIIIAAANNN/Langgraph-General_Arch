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
from src.tools.model_render_image import model_render_image
from src.tools.generate_3D import generate_3D
# --- MODIFIED: Remove simulate_future_image tool import ---
# from src.tools.simulate_future_image import simulate_future_image
# --- END MODIFIED ---

# --- Configuration & LLM Initialization ---
from src.configuration import ConfigManager, ModelConfig, MemoryConfig, initialize_llm, ConfigSchema

# --- <<< 修改：從 state.py 導入 >>> ---
from src.state import WorkflowState, TaskState # 從 state.py 導入狀態

# --- MCP Imports (Import MCPAgentState for internal use/type hinting) ---
try:
    from src.mcp_test import (
        MCPAgentState, # Keep for type hinting internal state if desired
        call_rhino_agent,
        # --- <<< ADDED: Import Pinterest and OSM agent functions >>> ---
        call_pinterest_agent,
        call_osm_agent, # <--- Add OSM import
        # --- <<< END ADDED >>> ---
        agent_tool_executor,
        should_continue as mcp_should_continue
    )
    print("Successfully imported MCP components from src.mcp_test (Rhino, Pinterest, OSM, Executor, ShouldContinue)") # Update print
except ImportError as e:
    print(f"WARNING: Could not import MCP components from src.mcp_test: {e}")
    # Define MCPAgentState as a simple Dict if import fails, for type hinting robustness
    MCPAgentState = Dict[str, Any]
    async def call_rhino_agent(*args, **kwargs): raise NotImplementedError("MCP import failed")
    async def call_pinterest_agent(*args, **kwargs): raise NotImplementedError("MCP import failed")
    # --- <<< ADDED: Dummy OSM function on import error >>> ---
    async def call_osm_agent(*args, **kwargs): raise NotImplementedError("MCP import failed")
    # --- <<< END ADDED >>> ---
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
# --- <<< NEW/UPDATED: Add/Ensure Agent Descriptions (including OSM) >>> ---
# Ensure the description is available for prepare_tool_inputs_node
if "RhinoMCPCoordinator" not in agent_descriptions:
     agent_descriptions["RhinoMCPCoordinator"] = "Coordinates complex tasks within Rhino 3D using planning and multiple tool calls. Ideal for multi-step Rhino operations or requests involving existing Rhino geometry analysis and modification. Takes the user request and optional image path."
if "PinterestMCPCoordinator" not in agent_descriptions:
    agent_descriptions["PinterestMCPCoordinator"] = "Searches for images on Pinterest based on keywords and downloads them. Ideal for finding visual references or inspiration. Takes the user request (keywords) as input."
# --- <<< ADDED: OSM Description >>> ---
if "OSMMCPCoordinator" not in agent_descriptions:
    agent_descriptions["OSMMCPCoordinator"] = "Generates a map screenshot for a given address using OpenStreetMap and geocoding. Takes the address string as input."
# --- <<< END ADDED >>> ---

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
    mcp_result: Optional[Dict[str, Any]] = None
) -> TaskState:
    """Updates task state fields based on tool execution outcome, including MCP results (handling multiple Pinterest files)."""
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
        current_task["outputs"] = outputs if outputs is not None else {}
        current_task["output_files"] = output_files if output_files is not None else []

        if mcp_result:
            print(f"Merging MCP result into task outputs...")
            mcp_history = mcp_result.get("mcp_message_history")
            if mcp_history:
                # --- Make MCP History serializable before storing ---
                serializable_history = []
                for msg in mcp_history:
                     if isinstance(msg, BaseMessage):
                         # Convert BaseMessage to a serializable dict representation
                         try:
                              # Use LangChain's recommended way or a custom representation
                              # Example simple representation:
                              msg_dict = {
                                   "type": msg.type,
                                   "content": repr(msg.content)[:500] + "..." if len(repr(msg.content)) > 500 else repr(msg.content), # Use repr for safety
                                   "additional_kwargs": msg.additional_kwargs,
                              }
                              if hasattr(msg, 'name'): msg_dict['name'] = msg.name
                              if hasattr(msg, 'tool_call_id'): msg_dict['tool_call_id'] = msg.tool_call_id
                              if hasattr(msg, 'tool_calls'): msg_dict['tool_calls'] = msg.tool_calls # Might still contain unserializable parts if complex
                              serializable_history.append(msg_dict)
                         except Exception as msg_ser_err:
                              print(f"  Warning: Could not serialize message {type(msg)}: {msg_ser_err}")
                              serializable_history.append({"type": "serialization_error", "original_type": type(msg).__name__})
                     else:
                          serializable_history.append(msg) # Append if already serializable

                current_task["outputs"]["mcp_internal_messages"] = serializable_history
                print(f"  Added {len(serializable_history)} messages from MCP internal history (serialized).")
                # --- End Serialization Fix ---


            # --- <<< MODIFIED: Handle Multiple Pinterest Paths >>> ---
            saved_paths_list = mcp_result.get("saved_image_paths") # Check for the list first
            if saved_paths_list and isinstance(saved_paths_list, list):
                print(f"  Processing {len(saved_paths_list)} saved image paths from MCP (likely Pinterest)...")
                # Ensure output_files list exists
                if "output_files" not in current_task or not isinstance(current_task["output_files"], list):
                     current_task["output_files"] = []

                processed_count = 0
                for saved_path in saved_paths_list:
                     if not saved_path or not isinstance(saved_path, str) or not os.path.exists(saved_path): # Added exists check here too
                          print(f"    - Warning: Skipping invalid or non-existent path: {saved_path}")
                          continue
                     filename = os.path.basename(saved_path)
                     mime_type = "image/png" # Default
                     ext = os.path.splitext(saved_path)[1].lower()
                     if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                     elif ext == ".gif": mime_type = "image/gif"
                     elif ext == ".webp": mime_type = "image/webp"

                     file_entry = {
                         "filename": filename,
                         "path": saved_path,
                         "type": mime_type,
                         "description": f"Downloaded image from MCP ({current_task.get('selected_agent','?')}).",
                     }
                     # Attempt to add base64 data
                     try:
                         with open(saved_path, "rb") as f: encoded_string = base64.b64encode(f.read()).decode('utf-8')
                         file_entry["base64_data"] = f"data:{mime_type};base64,{encoded_string}"
                         print(f"    - Added base64 data URI for: {filename}")
                     except Exception as e:
                         print(f"    - Warning: Could not read/encode image from path {saved_path}: {e}. base64_data will be missing.")

                     current_task["output_files"].append(file_entry)
                     processed_count += 1
                print(f"  Finished processing {processed_count} valid paths.")

            else:
                # Fallback to single path logic (for Rhino/OSM or if Pinterest failed partially)
                saved_path = mcp_result.get("saved_image_path")
                saved_uri = mcp_result.get("saved_image_data_uri")
                if saved_path:
                    print("  Processing single saved image path from MCP...")
                    filename = os.path.basename(saved_path)
                    mime_type = "image/png" # Default
                    # ... (mime type detection for single path) ...
                    ext = os.path.splitext(saved_path)[1].lower()
                    if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                    elif ext == ".gif": mime_type = "image/gif"
                    elif ext == ".webp": mime_type = "image/webp"

                    file_entry = {
                        "filename": filename,
                        "path": saved_path,
                        "type": mime_type,
                        "description": "Final screenshot/output from MCP agent.",
                    }
                    if saved_uri and saved_uri.startswith("data:"):
                        file_entry["base64_data"] = saved_uri
                    # ... (fallback base64 generation for single path) ...
                    elif os.path.exists(saved_path): # Only try if URI wasn't provided
                         try:
                              with open(saved_path, "rb") as f: encoded_string = base64.b64encode(f.read()).decode('utf-8')
                              file_entry["base64_data"] = f"data:{mime_type};base64,{encoded_string}"
                              print(f"  Constructed and stored data URI from single path.")
                         except Exception as e:
                              print(f"  Warning: Could not read/encode single image from path {saved_path}: {e}. base64_data will be missing.")

                    current_task["output_files"].append(file_entry)
                    print(f"  Added single MCP output '{filename}' (type: {mime_type}) to output_files.")
            # --- <<< END MODIFIED >>> ---

        current_task["error_log"] = None
        current_task["feedback_log"] = None
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

# --- Node: Prepare Tool Inputs (Modified for TypeError Fix) ---
async def prepare_tool_inputs_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Uses LLM to prepare the 'task_inputs' field in the current TaskState.
    Includes logic for Rhino, Pinterest, and OSM MCP Coordinators.
    FIXED: Handles non-serializable 'mcp_internal_messages' during summary generation.
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

    # --- Aggregation Logic ---
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
        error_log = task.get('error_log', 'N/A'); feedback_log = task.get('feedback_log', 'N/A')
        error_log_summary = error_log[:100] + "..." if error_log else "N/A"
        feedback_log_summary = feedback_log[:100] + "..." if feedback_log else "N/A"
        aggregated_summary_parts.append(f"  Task {i} (ID: {task_id}, Agent: {agent_used}, Status: {task_status}): {task_desc} (Objective: {objective})")
        aggregated_summary_parts.append(f"    Status: {task_status} - Error: {error_log_summary} Feedback: {feedback_log_summary}")
        if task_status == "completed":
            task_outputs = task.get("outputs")
            if task_outputs:
                # --- <<< 修改：過濾輸出以生成摘要 >>> ---
                outputs_for_summary = task_outputs.copy()
                outputs_for_summary.pop("mcp_internal_messages", None)
                outputs_for_summary.pop("grounding_sources", None)
                outputs_for_summary.pop("search_suggestions", None)
                # --- <<< 結束修改 >>> ---
                aggregated_outputs[task_id] = task_outputs # 儲存原始輸出
                try:
                    # 使用過濾後的字典生成摘要字串
                    summary_outputs_json = json.dumps(outputs_for_summary, ensure_ascii=False, indent=None, default=str) # 添加 default=str
                    aggregated_summary_parts.append(f"    Outputs: {summary_outputs_json[:500]}{'...' if len(summary_outputs_json)>500 else ''}") # 限制摘要長度
                except Exception as e:
                    print(f"    Warning: Could not JSON dump filtered outputs for task {task_id} summary: {e}. Skipping outputs in summary string.")
                    aggregated_summary_parts.append(f"    Outputs: [Could not serialize - check logs for task {task_id}]")
            task_files = task.get("output_files")
            if task_files:
                for file_dict in task_files:
                     if 'source_task_id' not in file_dict: file_dict['source_task_id'] = task_id
                aggregated_files_raw.extend(task_files); aggregated_summary_parts.append(f"    Files Generated: {[f.get('filename', 'N/A') for f in task_files]}")
        elif task_status == "failed" or task_status == "max_retries_reached":
             error_log = task.get('error_log', 'N/A'); feedback_log = task.get('feedback_log', 'N/A')
             error_log_summary = error_log[:100] + "..." if error_log else "N/A"
             feedback_log_summary = feedback_log[:100] + "..." if feedback_log else "N/A"
             aggregated_summary_parts.append(f"    Status: {task_status} - Error: {error_log_summary} Feedback: {feedback_log_summary}")
    filtered_aggregated_files = _filter_base64_from_files(aggregated_files_raw)
    # --- MODIFIED: Use the aggregated_outputs dict (which contains original task outputs) for the final JSON string ---
    try:
        aggregated_outputs_json = json.dumps(aggregated_outputs, ensure_ascii=False, default=lambda o: f"<Object {type(o).__name__}>", indent=2) # Use default handler for safety
    except TypeError:
         aggregated_outputs_json = json.dumps({"error": "Could not serialize aggregated outputs"}, ensure_ascii=False)
    # --- END MODIFIED ---
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

    # --- Updated Prompt Handling to use config_manager ---
    prepare_inputs_prompt_template_str = None
    aa_prompts_config = config_manager.get_agent_config("assign_agent").prompts
    if aa_prompts_config:
        prompt_config_obj = aa_prompts_config.get("prepare_tool_inputs_prompt")
        if prompt_config_obj:
            prepare_inputs_prompt_template_str = prompt_config_obj.template

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
            "aggregated_summary": aggregated_summary_str,
            "aggregated_outputs_json": aggregated_outputs_json, # Use the potentially handled JSON string
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
                # --- Validation ---
                agent_key_map = {
                    "ArchRAGAgent": ["prompt"], "WebSearchAgent": ["prompt"],
                    "ImageRecognitionAgent": ["image_paths", "prompt"], "VideoRecognitionAgent": ["video_paths", "prompt"],
                    "ModelRenderAgent": ["outer_prompt", "image_inputs", "is_future_scenario"], # Updated keys
                    "Generate3DAgent": ["image_path"],
                    "ImageGenerationAgent": ["prompt"], # image_inputs is optional for ImageGenerationAgent
                    "LLMTaskAgent": ["prompt"],
                    "RhinoMCPCoordinator": ["user_request"],
                    "PinterestMCPCoordinator": ["keyword"],
                    "OSMMCPCoordinator": ["user_request"]
                }
                optional_keys = []
                if selected_agent_name == "RhinoMCPCoordinator":
                    optional_keys.append("initial_image_path")
                elif selected_agent_name == "PinterestMCPCoordinator":
                    optional_keys.append("limit")
                elif selected_agent_name == "ImageGenerationAgent":
                    optional_keys.append("i")
                    optional_keys.append("image_inputs") # image_inputs is optional for ImageGenerationAgent

                missing_keys = []
                invalid_paths = []
                invalid_values = []

                # Check required keys
                required_keys = agent_key_map.get(selected_agent_name, [])
                for key in required_keys:
                    value = prepared_inputs.get(key)
                    
                    if key == "is_future_scenario" and selected_agent_name == "ModelRenderAgent":
                        if value is None: # Must be explicitly true or false by the LLM
                            missing_keys.append(key)
                        elif not isinstance(value, bool):
                            invalid_values.append(f"{key}: Value '{value}' is not a boolean (true/false).")
                    elif key == "image_inputs" and selected_agent_name == "ModelRenderAgent":
                        if value is None or not isinstance(value, list) or not value: # Must be a non-empty list
                            missing_keys.append(key)
                            if value is not None and isinstance(value,list) and not value :
                                invalid_values.append(f"{key}: List cannot be empty for ModelRenderAgent.")
                        elif not all(isinstance(item, str) for item in value):
                             invalid_values.append(f"{key}: All items in list must be string paths.")
                    elif value is None or (isinstance(value, str) and not value): # Check for empty strings for other required keys
                        missing_keys.append(key)
                    # If it's a list type other than image_inputs for ModelRenderAgent, and it's empty, it's missing
                    elif isinstance(value, list) and not value and key not in ["image_inputs"]: # e.g. image_paths for ImageRecognitionAgent
                        missing_keys.append(key)


                # Check optional keys and validate if present
                for key in optional_keys:
                    value = prepared_inputs.get(key)
                    if value is not None: # Only validate if present
                        if key == "initial_image_path":
                            if not isinstance(value, str) or not os.path.exists(value):
                                invalid_paths.append(f"{key}: '{value}' (Optional path provided but not found or invalid type)")
                        elif key == "limit":
                            try: int(value)
                            except (ValueError, TypeError): invalid_values.append(f"{key}: '{value}' (Must be integer)")
                        elif key == "i" and selected_agent_name == "ImageGenerationAgent":
                            try: prepared_inputs[key] = int(value)
                            except (ValueError, TypeError): invalid_values.append(f"{key}: '{value}' (Must be int for ImageGenerationAgent)")
                        elif key == "image_inputs" and selected_agent_name == "ImageGenerationAgent": # Optional image_inputs for ImageGen
                             if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                                 invalid_values.append(f"{key}: Optional input must be a list of string paths.")
                             else: # Validate paths if list is provided
                                 for item_idx, item_path in enumerate(value):
                                     if not os.path.exists(item_path):
                                        invalid_paths.append(f"{key}[{item_idx}]: Optional path '{item_path}' not found.")


                # Perform general validation on present keys (that are not None)
                for key, value in prepared_inputs.items():
                    if value is None: continue

                    if key in ["image_paths", "video_paths"] and isinstance(value, list): # For ImageRecognitionAgent, VideoRecognitionAgent
                         if not value: # Empty list for a required path list is an error unless handled above
                             if key in required_keys and key not in missing_keys: # If it wasn't caught as missing
                                invalid_values.append(f"{key}: List is empty. Must provide at least one path.")
                         for path_idx, path_val in enumerate(value):
                             if not isinstance(path_val, str) or not os.path.exists(path_val):
                                 invalid_paths.append(f"{key}[{path_idx}]: '{path_val}' (Path not found or invalid type)")
                    elif key == "image_path" and isinstance(value, str) and key not in optional_keys: # For Generate3DAgent
                         if not os.path.exists(value): invalid_paths.append(f"{key}: '{value}' (Path not found)")
                    
                    # Validation for ModelRenderAgent's image_inputs (paths inside the list)
                    elif key == "image_inputs" and selected_agent_name == "ModelRenderAgent" and isinstance(value, list) and value: # Already checked for non-list or empty list
                        for item_idx, item_path in enumerate(value):
                            # Path existence check (can be more robust, checking cache/output dirs like before)
                            # For now, a simple os.path.exists(), assuming LLM provides resolvable paths
                            if not isinstance(item_path, str): # Should have been caught by LLM prompt, but good to double check
                                if f"{key}[{item_idx}]" not in invalid_values : invalid_values.append(f"{key}[{item_idx}]: Item '{item_path}' is not a string path.")
                                continue
                            
                            # Re-check path validation logic here, ensuring it matches the previous detailed one.
                            # The current simple os.path.exists(item_path) might be too restrictive if paths are relative to cache.
                            # Let's use the more comprehensive check:
                            full_path_cache_item = os.path.join(RENDER_CACHE_DIR, os.path.basename(item_path))
                            full_path_output_item = os.path.join(OUTPUT_DIR, os.path.basename(item_path))
                            if not os.path.exists(item_path) and \
                               not os.path.exists(full_path_cache_item) and \
                               not os.path.exists(full_path_output_item):
                                invalid_paths.append(f"{key}[{item_idx}]: '{item_path}' (Not found as absolute, in cache, or in output dir)")

                    elif key == "is_future_scenario" and selected_agent_name == "ModelRenderAgent":
                        if not isinstance(value, bool): # Already caught None, this catches non-bool
                             if f"{key}: Value '{value}' is not a boolean (true/false)." not in invalid_values: # Avoid duplicates
                                invalid_values.append(f"{key}: Value '{value}' is not a boolean (true/false).")
                                
                    elif key == "i" and selected_agent_name == "ImageGenerationAgent":
                         pass # Already validated if present in optional_keys
                    elif key in ["prompt", "outer_prompt", "user_request", "keyword"] and not isinstance(value, str): # outer_prompt for ModelRender
                         invalid_values.append(f"{key}: Value must be a string.")

                # Consolidate errors
                unique_invalid_paths = list(set(invalid_paths))
                unique_invalid_values = list(set(invalid_values))

                if missing_keys or unique_invalid_paths or unique_invalid_values:
                    error_parts = []
                    if missing_keys: error_parts.append(f"Missing/empty required keys: {', '.join(missing_keys)}")
                    if unique_invalid_paths: error_parts.append(f"Invalid/missing paths: {', '.join(unique_invalid_paths)}")
                    if unique_invalid_values: error_parts.append(f"Invalid values: {', '.join(unique_invalid_values)}")
                    validation_err_msg = f"Input Validation Failed for {selected_agent_name}: {'. '.join(error_parts)}. LLM Output: {json.dumps(prepared_inputs, ensure_ascii=False)}"
                    _set_task_failed(current_task, validation_err_msg, node_name)
                else:
                    print(f"{node_name}: Inputs prepared and validated successfully: {list(prepared_inputs.keys())}")
                    current_task["task_inputs"] = prepared_inputs
                    current_task["error_log"] = None; current_task["feedback_log"] = None
                    current_task["status"] = "in_progress"

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


# # --- Tool Subgraph Router ---
# def determine_tool_route(state: WorkflowState) -> str: # 返回值改為 str
#     """Determines the next tool node based on the current task's selected agent."""
#     print(f"--- Tool Subgraph Router Node ---")
#     current_idx = state.get("current_task_index", -1)
#     tasks = state.get("tasks", [])

#     # 檢查索引有效性
#     if current_idx < 0 or current_idx >= len(tasks):
#         print("Tool Subgraph Router Error: Invalid task index."); return "finished" # 直接返回 END

#     current_task = tasks[current_idx]
#     agent_name = current_task.get("selected_agent")

#     # 檢查 agent_name 是否存在
#     if not agent_name:
#         print(f"Tool Subgraph Router Error: No 'selected_agent'. Routing to END."); return "finished" # 直接返回 END

#     print(f"--- Tool Subgraph Router: Routing for agent '{agent_name}' ---")

#     node_mapping = {
#         "ArchRAGAgent": "rag_agent", "ImageGenerationAgent": "image_gen_agent",
#         "WebSearchAgent": "web_search_agent", "CaseRenderAgent": "case_render_agent",
#         "Generate3DAgent": "generate_3d_agent", "SimulateFutureAgent": "simulate_future_agent",
#         "VideoRecognitionAgent": "video_recognition_agent", "ImageRecognitionAgent": "image_recognition_agent",
#         "LLMTaskAgent": "llm_task_agent",
#         # <<< NEW: Add mapping for Rhino MCP Coordinator >>>
#         "RhinoMCPCoordinator": "rhino_mcp_node" # 使用新節點名稱
#     }
#     target_node = node_mapping.get(agent_name)

#     if target_node:
#         print(f"--- Tool Subgraph Router: Target node: '{target_node}' ---"); return target_node # 返回目標節點名稱
#     else:
#         print(f"Tool Subgraph Router Error: Unknown agent name '{agent_name}'. Routing to END."); return "finished" # 直接返回 END

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
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

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
    """Executes the Image Generation tool based on task inputs."""
    node_name = "Image Gen Node" # Shorter name for clarity
    print(f"--- Running {node_name} (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}

    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs'")

        prompt = inputs.get("prompt")
        if not prompt:
            raise ValueError("Missing 'prompt' in task_inputs")

        image_inputs = inputs.get("image_inputs") # Optional

        # --- MODIFIED: Read 'i' for image count, default to 1 ---
        image_count = inputs.get("i", 1)
        if not isinstance(image_count, int) or image_count < 1:
             print(f"  - Warning: Invalid 'i' value ({image_count}) in inputs. Defaulting to 1.")
             image_count = 1
        # --- END MODIFIED ---

        print(f"{node_name}: Using prompt='{prompt[:50]}...', image_inputs: {'Provided' if image_inputs else 'None'}. Requesting i={image_count} image(s).") # Updated log

        # --- Call the tool, passing 'i' ---
        # Assuming generate_gemini_image is synchronous based on previous code
        tool_result = generate_gemini_image({
            "prompt": prompt,
            "image_inputs": image_inputs or [],
            "i": image_count # Pass 'i' instead of 'n'
        })
        # --- End Tool Call ---

        # --- Log tool result structure (based on expected simpler return from snippet) ---
        print(f"DEBUG: {node_name} tool_result type: {type(tool_result)}")
        if isinstance(tool_result, dict):
            print(f"DEBUG: {node_name} tool_result keys: {list(tool_result.keys())}")
            # Log expected keys based on tool snippet's likely return
            print(f"  - text_response: '{str(tool_result.get('text_response', 'N/A'))[:100]}...'")
            generated_files = tool_result.get('generated_files', [])
            print(f"  - generated_files count: {len(generated_files)}")
            print(f"  - First few generated_files: {generated_files[:2]}")
            if "error" in tool_result:
                 print(f"  - error key present: {tool_result['error']}")
        else:
            print(f"DEBUG: {node_name} tool_result is not a dictionary.")
        # --- End logging ---

        # --- MODIFIED: Simplified Error Handling based on user's tool snippet ---
        # 1. Check for explicit error key returned by the tool
        if isinstance(tool_result, dict) and tool_result.get("error"):
            # Tool itself reported a blocking error (e.g., client init failed)
            raise ValueError(f"Tool Error: {tool_result['error']}")
        elif not isinstance(tool_result, dict):
             # Tool returned something completely unexpected
             raise TypeError(f"Tool returned unexpected result type: {type(tool_result)}")
        # --- END MODIFIED ---

        # Process successful results (or partially successful if tool handles internal errors)
        text_response = tool_result.get("text_response", "") # Default to empty string
        tool_generated_files = tool_result.get("generated_files", []) # Default to empty list

        if tool_generated_files and isinstance(tool_generated_files, list):
            print(f"{node_name}: Tool reported {len(tool_generated_files)} generated files. Processing...")
            # --- File processing loop remains the same ---
            for file_idx, file_info in enumerate(tool_generated_files):
                if not isinstance(file_info, dict):
                     print(f"  - Warning: Item {file_idx} in generated_files is not a dictionary: {file_info}. Skipping.")
                     continue
                filename = file_info.get("filename")
                expected_path = file_info.get("path") # Assuming tool returns absolute path
                file_type = file_info.get("type", "image/png") # Default type
                # Add description indicating which image this is, using 'i' from input
                description = file_info.get("description", f"Generated image {file_idx+1} of {image_count} for: {prompt[:30]}...")

                if not filename or not expected_path:
                     print(f"  - Warning: Invalid or missing 'filename' or 'path': {file_info}. Skipping.")
                     continue

                print(f"  - Checking path from tool: {expected_path}")
                if os.path.exists(expected_path):
                     print(f"  - File FOUND at path: {expected_path}")
                     processed_file_info = {"filename": filename, "path": expected_path, "type": file_type, "description": description}
                     # Add base64 encoding (remains the same)
                     if file_type.startswith("image/"):
                         try:
                             with open(expected_path, "rb") as image_file: encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                             processed_file_info["base64_data"] = f"data:{file_type};base64,{encoded_string}"
                             print(f"    - Added base64 data URI for image: {filename}")
                         except Exception as e: print(f"    - Warning: Could not read/encode image {expected_path}: {e}")
                     output_files_list.append(processed_file_info)
                else:
                     print(f"  - Warning: File '{filename}' reported by tool not found at path '{expected_path}'. Skipping.")
            # --- End File processing loop ---

        elif not tool_generated_files:
             # Only log warning here, error check comes later
             print(f"{node_name}: Tool did not report any generated files in the result.")
        else:
             # Log warning if generated_files is not a list
             print(f"Warning: 'generated_files' from tool is not a list: {tool_generated_files}")

        # Store text response if available
        if text_response: final_outputs["text_response"] = text_response
        # Note: We are not expecting an 'errors' list from the tool based on the provided snippet

    except Exception as e:
        # Catch errors from this node's logic or the ValueError raised from tool's "error" key
        error_to_report = e
        final_outputs = {} # Clear outputs on error
        output_files_list = [] # Clear files on error
        traceback.print_exc()

    # --- MODIFIED: Final check: Fail ONLY if no files were generated when requested AND no other major error occurred ---
    if image_count > 0 and not output_files_list:
        no_files_msg = f"Image Generation Tool was requested to generate i={image_count} image(s) but produced no processable output files."
        print(f"{node_name} Warning: {no_files_msg}")
        # Only flag this as the primary error if no other exception was caught
        if error_to_report is None:
            error_to_report = ValueError(no_files_msg)
        else:
             # Append this warning to the existing error message
             print(f"  (Appending this warning to existing error: {error_to_report})")
             error_to_report = type(error_to_report)(f"{str(error_to_report)}; {no_files_msg}")
    # --- END MODIFIED ---

    # Update task state using the helper function
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=final_outputs,
        output_files=output_files_list or [], # Ensure it's a list
        error=error_to_report
    )
    # Return the updated tasks list and the current task copy
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_web_search_tool_node ---
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

# --- MODIFIED: run_model_render_tool_node (now handles both photorealism and future simulation) ---
def run_model_render_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    node_name = "Model Render Node"
    print(f"--- Running {node_name} (Handles Photorealism & Future Simulation) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    inputs = current_task.get("task_inputs")
    error_to_report = None
    aggregated_output_files_list = []
    aggregated_generated_filenames = []
    final_outputs = {}

    try:
        if not inputs or not isinstance(inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs'")

        initial_outer_prompt_context = inputs.get("outer_prompt") # This is the contextual prompt from prepare_tool_inputs
        image_inputs_paths = inputs.get("image_inputs") # List of image paths, RENAMED from render_image_input
        is_future_scenario = inputs.get("is_future_scenario", False) # Boolean flag

        if not initial_outer_prompt_context:
            raise ValueError("Missing required input 'outer_prompt' (contextual prompt)")
        if not image_inputs_paths or not isinstance(image_inputs_paths, list) or not image_inputs_paths or not all(isinstance(p, str) for p in image_inputs_paths): # Added check for empty list
            raise ValueError("Missing or invalid 'image_inputs' (must be a non-empty list of string paths)")
        if not isinstance(is_future_scenario, bool):
            raise ValueError("'is_future_scenario' must be a boolean")

        print(f"{node_name}: Task type: {'Future Simulation' if is_future_scenario else 'Photorealistic Render'}")
        print(f"  Initial outer_prompt='{initial_outer_prompt_context[:70]}...', processing {len(image_inputs_paths)} image(s).")

        for idx, image_full_path in enumerate(image_inputs_paths):
            print(f"  Processing image {idx+1}/{len(image_inputs_paths)}: {image_full_path}")
            
            refined_prompt_text = initial_outer_prompt_context # Default to initial prompt
            try:
                if is_future_scenario:
                    # Prompt for ImageRecognition: Ask it to generate a final English ComfyUI prompt for future simulation
                    recognition_prompt_ir = f"Based on the following context for a future architectural scenario: '{initial_outer_prompt_context}', and by analyzing the provided image, generate a detailed and specific, final, ENGLISH ComfyUI prompt. This prompt should be directly usable by an image generation tool to visually simulate the described future scenario on the given image. Focus on elements that would change or appear in the future based on the context and image content."
                else: # Photorealistic rendering
                    # Prompt for ImageRecognition: Ask it to generate a final English ComfyUI prompt for photorealism
                    recognition_prompt_ir = f"For the architectural scheme described as: '{initial_outer_prompt_context}', and by analyzing the provided image (paying close attention to its perspective, view, and existing elements), generate a detailed and specific, final, ENGLISH ComfyUI prompt. This prompt should be directly usable by an image generation tool to create a high-quality photorealistic rendering of the image, enhancing realism and matching the described scheme."
                
                print(f"    Invoking ImageRecognition for: {image_full_path} with IR prompt: '{recognition_prompt_ir[:100]}...'")
                
                refined_prompt_dict_or_str = img_recognition({
                    "image_paths": [image_full_path],
                    "prompt": recognition_prompt_ir
                })

                if isinstance(refined_prompt_dict_or_str, str):
                    refined_prompt_text = refined_prompt_dict_or_str
                elif isinstance(refined_prompt_dict_or_str, dict) and "description" in refined_prompt_dict_or_str:
                    refined_prompt_text = refined_prompt_dict_or_str["description"]
                else:
                    print(f"    Warning: ImageRecognition returned unexpected format for {image_full_path}. Using initial_outer_prompt. Output: {refined_prompt_dict_or_str}")
                    # Fallback already handled by refined_prompt_text = initial_outer_prompt_context
                
                if not refined_prompt_text or (isinstance(refined_prompt_text, str) and any(err_str in refined_prompt_text for err_str in ["錯誤", "Error", "找不到"])):
                     print(f"    Warning: ImageRecognition failed or returned error for {image_full_path}: '{refined_prompt_text}'. Using initial_outer_prompt for this image.")
                     refined_prompt_text = initial_outer_prompt_context # Ensure fallback

                print(f"    Refined prompt for {image_full_path}: '{refined_prompt_text[:70]}...'")
            except Exception as recog_err:
                print(f"    Error during ImageRecognition for {image_full_path}: {recog_err}. Using initial_outer_prompt.")
                # refined_prompt_text is already initial_outer_prompt_context by default

            print(f"    Calling model_render_image with outer_prompt='{refined_prompt_text[:50]}...', render_image='{os.path.basename(image_full_path)}'")
            # The model_render_image tool itself should handle the type of rendering based on the refined_prompt
            # or potentially different ComfyUI workflows if the tool is made more complex.
            # For now, we assume the refined_prompt is sufficient to guide the *single* model_render_image tool.
            result = model_render_image({
                "outer_prompt": refined_prompt_text,
                "render_image": os.path.basename(image_full_path)
            })

            if isinstance(result, str) and result.startswith("Error:"):
                print(f"    Tool Error for {image_full_path}: {result}")
                current_task["feedback_log"] = (current_task.get("feedback_log","") + f"\nFailed to render {image_full_path}: {result}").strip()
                continue 

            if not isinstance(result, str) or not result:
                print(f"    Tool returned unexpected/empty result for {image_full_path}: {result}")
                current_task["feedback_log"] = (current_task.get("feedback_log","") + f"\nUnexpected result for {image_full_path}: {result}").strip()
                continue 

            output_filename_from_tool = result.strip()
            if output_filename_from_tool:
                file_info = _save_tool_output_file(
                    output_filename_from_tool,
                    RENDER_CACHE_DIR, 
                    "image/png",
                    f"{'Future simulation' if is_future_scenario else 'Model render'} for {os.path.basename(image_full_path)}: {refined_prompt_text[:30]}..."
                )
                if file_info:
                    aggregated_output_files_list.append(file_info)
                    aggregated_generated_filenames.append(output_filename_from_tool)
                    print(f"    Successfully processed and saved: {output_filename_from_tool}")
                else:
                    print(f"    Could not find/process tool output file: {output_filename_from_tool} for {image_full_path}")
                    current_task["feedback_log"] = (current_task.get("feedback_log","") + f"\nOutput file not found/processed for {image_full_path}: {output_filename_from_tool}").strip()
            else:
                print(f"    Tool did not return a filename for {image_full_path}.")
                current_task["feedback_log"] = (current_task.get("feedback_log","") + f"\nNo output filename for {image_full_path}.").strip()


        if not aggregated_output_files_list:
            if not current_task.get("feedback_log"):
                 current_task["feedback_log"] = "ModelRenderAgent: Tool ran but failed to produce/locate any output files for the given render_image(s)."
            raise ValueError(current_task["feedback_log"] or "Tool ran but failed to produce/locate any output files.")

        final_outputs = {"generated_filenames": aggregated_generated_filenames}
        print(f"{node_name}: Completed. Generated {len(aggregated_generated_filenames)} file(s).")

    except Exception as e:
        error_to_report = e
        final_outputs = {} 
        traceback.print_exc()


    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=final_outputs,
        output_files=aggregated_output_files_list, 
        error=error_to_report
    )
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

        # --- <<< MODIFICATION START >>> ---
        # Determine the correct directory where the generate_3D tool actually saves files.
        # Based on execution logs, it saves directly to 'output/model_cache' relative to project root.
        # Construct this path robustly.
        # Assuming the script runs from the project root (e.g., /d:/MA system/LangGraph/).
        actual_3d_save_dir = os.path.abspath(os.path.join("output", "model_cache"))
        print(f"  Verifying 3D files in expected directory: {actual_3d_save_dir}")
        # Create the directory if it doesn't exist, just in case.
        os.makedirs(actual_3d_save_dir, exist_ok=True)
        # --- <<< MODIFICATION END >>> ---


        # Use imported tool
        result = generate_3D({"image_path": image_path})
        if isinstance(result, dict) and "error" in result: raise ValueError(f"Tool Error: {result['error']}")
        if not isinstance(result, dict): raise ValueError(f"Tool returned unexpected result type: {type(result)}")

        model_filename = result.get("model")
        video_filename = result.get("video")
        # --- <<< MODIFICATION START >>> ---
        # Use the *actual_3d_save_dir* when calling the helper function, NOT MODEL_CACHE_DIR
        if model_filename:
            # Pass the determined actual save directory
            file_info = _save_tool_output_file(model_filename, actual_3d_save_dir, "model/gltf-binary", f"3D model from {os.path.basename(image_path)}")
            if file_info: output_files_list.append(file_info)
        if video_filename:
            # Pass the determined actual save directory
            file_info = _save_tool_output_file(video_filename, actual_3d_save_dir, "video/mp4", f"Preview video from {os.path.basename(image_path)}")
            if file_info: output_files_list.append(file_info)
        # --- <<< MODIFICATION END >>> ---

        if not output_files_list: raise ValueError("Tool ran but produced no model or video file.")
        final_outputs = {"model_filename": model_filename, "video_filename": video_filename}

    except Exception as e:
        error_to_report = e
        final_outputs = None
        output_files_list = None
        traceback.print_exc() # Print traceback for debugging

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

# --- run_rhino_mcp_node ---
async def run_rhino_mcp_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes a Rhino task using the MCP agent logic within this node.
    Uses TaskState inputs and updates TaskState outputs/status.
    Returns MCP message history (raw objects) and necessary file info.
    """
    node_name = "Rhino MCP Node"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]

    if current_idx < 0 or current_idx >= len(tasks):
        print(f"{node_name} Error: Invalid task index {current_idx}.")
        return {"tasks": [t.copy() for t in state.get("tasks", [])], "current_task": state.get("current_task").copy() if state.get("current_task") else None}

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs")
    outer_error_to_report = None
    final_mcp_outcome = None
    mcp_loop_messages: List[BaseMessage] = [] # Store raw messages from the loop

    try:
        # 1. Extract inputs (remains the same)
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs' in the current task.")
        user_request = task_inputs.get("user_request")
        initial_image_path = task_inputs.get("initial_image_path") # Optional

        if not user_request:
            raise ValueError("Missing 'user_request' in task_inputs for RhinoMCPCoordinator")

        print(f"{node_name}: Starting MCP task for request: '{user_request[:100]}...'")
        if initial_image_path: print(f"  with initial image: {initial_image_path}")

        # 2. Construct Initial *Local* MCP State (remains the same)
        local_mcp_state: Dict[str, Any] = { # Use Dict for flexibility
            "messages": [],
            "initial_request": user_request,
            "initial_image_path": initial_image_path,
            "target_mcp": "rhino",
            "task_complete": False,
            "saved_image_path": None,
            "saved_image_data_uri": None
        }
        initial_human_content = [{"type": "text", "text": user_request}]
        if initial_image_path and os.path.exists(initial_image_path):
            try:
                with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                mime_type = "image/png" # Default
                ext = os.path.splitext(initial_image_path)[1].lower()
                if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                elif ext == ".gif": mime_type = "image/gif"
                elif ext == ".webp": mime_type = "image/webp"
                initial_human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                })
                print(f"  Successfully encoded initial image ({mime_type}) for local MCP state.")
            except Exception as img_err:
                print(f"  Warning: Failed to read/encode initial image {initial_image_path} for local MCP state: {img_err}")
        local_mcp_state["messages"] = [HumanMessage(content=initial_human_content)]

        # 3. Run Internal MCP Loop (logic remains mostly the same)
        max_mcp_steps = 1000
        step_count = 0
        mcp_loop_error = None
        while step_count < max_mcp_steps:
            step_count += 1
            print(f"\n{node_name}: --- MCP Internal Loop Step {step_count} ---")
            # Add raw messages to history *before* the next step
            mcp_loop_messages.extend(local_mcp_state.get("messages", [])) # Extend history with raw objects
            local_mcp_state["messages"] = [] # Clear messages for the next step's output

            # --- Routing logic (remains the same, including passing target_mcp) ---
            if step_count == 1:
                next_step = "rhino_agent"
                print(f"  First step: Directly routing to {next_step}")
            else:
                temp_state_for_should_continue = {
                    "messages": mcp_loop_messages,
                    "target_mcp": "rhino"
                }
                next_step = mcp_should_continue(temp_state_for_should_continue)
                print(f"  mcp_should_continue result: {next_step}")


            if next_step == END:
                print(f"{node_name}: MCP loop finished based on should_continue.")
                break
            elif next_step == "rhino_agent":
                print(f"  Calling call_rhino_agent...")
                agent_output = await call_rhino_agent({"messages": mcp_loop_messages}, config) # Pass raw history
                local_mcp_state["messages"].extend(agent_output.get("messages", [])) # Add raw new messages
                if "saved_image_path" in agent_output: local_mcp_state["saved_image_path"] = agent_output["saved_image_path"]
                if "saved_image_data_uri" in agent_output: local_mcp_state["saved_image_data_uri"] = agent_output["saved_image_data_uri"]
            elif next_step == "agent_tool_executor":
                print(f"  Calling agent_tool_executor...")
                last_ai_message = next( (msg for msg in reversed(mcp_loop_messages) if isinstance(msg, AIMessage)), None )
                if last_ai_message and last_ai_message.tool_calls:
                     executor_input_state = {"messages": [last_ai_message], "target_mcp": "rhino"}
                     executor_output = await agent_tool_executor(executor_input_state, config)
                     local_mcp_state["messages"].extend(executor_output.get("messages", [])) # Add raw tool results
                     # --- Update local state with path/uri from tool result ---
                     last_tool_msg = local_mcp_state["messages"][-1] if local_mcp_state["messages"] and isinstance(local_mcp_state["messages"][-1], ToolMessage) else None
                     if last_tool_msg and last_tool_msg.name == "capture_viewport":
                          content = last_tool_msg.content
                          prefix = "[IMAGE_FILE_PATH]:"
                          error_prefix = "[Error:"
                          if content.startswith(prefix):
                              path_from_tool = content[len(prefix):]
                              local_mcp_state["saved_image_path"] = path_from_tool
                              print(f"  Tool executor updated saved_image_path: {path_from_tool}")
                              # Attempt to generate URI here if path exists
                              if path_from_tool and os.path.exists(path_from_tool):
                                  try:
                                      with open(path_from_tool, "rb") as img_file: img_bytes = img_file.read()
                                      base64_data = base64.b64encode(img_bytes).decode('utf-8')
                                      mime_type = "image/png"
                                      ext = os.path.splitext(path_from_tool)[1].lower()
                                      if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                                      local_mcp_state["saved_image_data_uri"] = f"data:{mime_type};base64,{base64_data}"
                                      print(f"  Generated data URI for screenshot: {local_mcp_state['saved_image_data_uri'][:60]}...")
                                  except Exception as uri_gen_err:
                                      print(f"  Warning: Could not generate data URI for screenshot {path_from_tool}: {uri_gen_err}")
                                      local_mcp_state["saved_image_data_uri"] = None # Reset if generation failed
                          elif content.startswith(error_prefix):
                               print(f"  Tool executor received error from capture_viewport: {content}")
                               if not mcp_loop_error: mcp_loop_error = ValueError(f"MCP Screenshot Error: {content}")
                else:
                     print(f"  Warning: Expected AIMessage with tool_calls before agent_tool_executor, but not found.")
                     local_mcp_state["messages"].append(ToolMessage(content="[Error: No tool calls found in previous AI message]", tool_call_id="internal_error"))
            else:
                mcp_loop_error = ValueError(f"MCP loop error: Unknown step '{next_step}'")
                print(f"{node_name} Error: {mcp_loop_error}")
                break

            await asyncio.sleep(1)

        # Add the very last raw messages from the final step/break
        mcp_loop_messages.extend(local_mcp_state.get("messages", []))

        if step_count >= max_mcp_steps:
            print(f"{node_name} Warning: Reached max MCP steps ({max_mcp_steps}). Ending loop.")
            mcp_loop_error = TimeoutError("MCP task exceeded maximum steps.")

        # --- <<< MODIFIED: Create final outcome dict (No 'message', raw history) >>> ---
        final_mcp_outcome = {
            # "message": ..., # REMOVED - Let history speak for itself
            "saved_image_path": local_mcp_state.get("saved_image_path"),
            "saved_image_data_uri": local_mcp_state.get("saved_image_data_uri"),
            "mcp_message_history": mcp_loop_messages # Store raw BaseMessage list
        }
        print(f"  Stored {len(final_mcp_outcome['mcp_message_history'])} raw MCP messages for history.")

        # Store loop error if it occurred
        if mcp_loop_error:
            outer_error_to_report = mcp_loop_error
            # Optionally add error info to history if needed, though it should be in the messages already
            print(f"  MCP loop terminated with error: {mcp_loop_error}")
        # --- <<< END MODIFIED >>> ---

    except Exception as e:
        error_msg = f"Error during {node_name} execution: {e}"
        print(error_msg)
        traceback.print_exc()
        outer_error_to_report = e
        # Ensure history is still included (raw) in case of outer error
        final_mcp_outcome = {
             # "message": f"MCP coordination failed: {e}", # REMOVED
             "saved_image_path": None,
             "saved_image_data_uri": None,
             "mcp_message_history": mcp_loop_messages # Include whatever history was gathered
         }

    # 5. Update the main WorkflowState Task using the helper
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=None,
        output_files=None,
        error=outer_error_to_report,
        mcp_result=final_mcp_outcome # Pass dict with history and file info
    )

    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_pinterest_mcp_node ---
async def run_pinterest_mcp_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes a Pinterest task using the MCP agent logic within this node.
    Uses TaskState inputs and updates TaskState outputs/status.
    Returns MCP message history (raw objects) and necessary file info.
    """
    node_name = "Pinterest MCP Node"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]

    if current_idx < 0 or current_idx >= len(tasks):
        print(f"{node_name} Error: Invalid task index {current_idx}.")
        return {"tasks": [t.copy() for t in state.get("tasks", [])], "current_task": state.get("current_task").copy() if state.get("current_task") else None}

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs")
    outer_error_to_report = None
    final_mcp_outcome = None
    mcp_loop_messages: List[BaseMessage] = [] # Store raw messages from the loop

    try:
        # 1. Extract inputs
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs' in the current task.")
        user_request = task_inputs.get("keyword") # Pinterest uses 'keyword'
        limit = task_inputs.get("limit") # Optional

        if not user_request:
            raise ValueError("Missing 'keyword' in task_inputs for PinterestMCPCoordinator")

        print(f"{node_name}: Starting MCP task for keyword: '{user_request[:100]}...' (Limit: {limit or 'Default'})")

        # 2. Construct Initial *Local* MCP State
        local_mcp_state: Dict[str, Any] = { # Use Dict for flexibility
            "messages": [HumanMessage(content=user_request)], # Pinterest starts simple
            "initial_request": user_request,
            "initial_image_path": None, # Pinterest doesn't take initial image
            "target_mcp": "pinterest",  # Target is Pinterest
            "task_complete": False,
            "saved_image_path": None,
            "saved_image_data_uri": None,
            "saved_image_paths": None, # <<< ADDED: Initialize plural path list >>>
            # Add limit to the initial state if provided, so agent can use it
            "pinterest_limit": limit
        }

        # 3. Run Internal MCP Loop (Similar structure to Rhino)
        max_mcp_steps = 1000
        step_count = 0
        mcp_loop_error = None
        while step_count < max_mcp_steps:
            step_count += 1
            print(f"\n{node_name}: --- MCP Internal Loop Step {step_count} ---")
            # Add raw messages to history *before* the next step
            mcp_loop_messages.extend(local_mcp_state.get("messages", [])) # Extend history with raw objects
            local_mcp_state["messages"] = [] # Clear messages for the next step's output

            # --- Routing logic (using imported mcp_should_continue) ---
            if step_count == 1:
                next_step = "pinterest_agent"
                print(f"  First step: Directly routing to {next_step}")
            else:
                temp_state_for_should_continue = {
                    "messages": mcp_loop_messages,
                    "target_mcp": "pinterest"
                }
                next_step = mcp_should_continue(temp_state_for_should_continue)
                print(f"  mcp_should_continue result: {next_step}")

            if next_step == END:
                print(f"{node_name}: MCP loop finished based on should_continue.")
                break
            elif next_step == "pinterest_agent":
                print(f"  Calling call_pinterest_agent...")
                # Pass limit if available in state
                agent_input_state = {"messages": mcp_loop_messages}
                if "pinterest_limit" in local_mcp_state and local_mcp_state["pinterest_limit"] is not None:
                     # Need to check how the agent expects the limit. Assume it reads from message context for now.
                     # Or we might need to modify call_pinterest_agent to accept limit directly if possible.
                     # For simplicity, let's assume the agent can handle it via prompt/messages.
                     print(f"  (Limit {local_mcp_state['pinterest_limit']} should be handled by agent prompt)")
                     pass
                agent_output = await call_pinterest_agent(agent_input_state, config) # Pass raw history
                local_mcp_state["messages"].extend(agent_output.get("messages", [])) # Add raw new messages

                # --- <<< MODIFIED: Handle plural and singular paths from agent output >>> ---
                if "saved_image_paths" in agent_output and isinstance(agent_output["saved_image_paths"], list):
                    local_mcp_state["saved_image_paths"] = agent_output["saved_image_paths"] # Store the LIST
                    print(f"  Pinterest Agent returned {len(agent_output['saved_image_paths'])} paths (plural key).")
                elif "saved_image_path" in agent_output: # Fallback/compatibility check for single path key
                    single_path = agent_output["saved_image_path"]
                    if single_path and isinstance(single_path, str):
                         # If only single path key exists, wrap it in a list for consistency
                         local_mcp_state["saved_image_paths"] = [single_path]
                         print(f"  Pinterest Agent returned single path key: {single_path}. Stored as list.")
                         # Also store in singular key for potential compatibility if needed elsewhere (less likely now)
                         local_mcp_state["saved_image_path"] = single_path
                    else:
                         print(f"  Warning: Pinterest Agent returned 'saved_image_path' key but value is invalid: {single_path}")

                # Handle data URI if present (likely corresponds to the last image in list or the single image)
                if "saved_image_data_uri" in agent_output:
                    local_mcp_state["saved_image_data_uri"] = agent_output["saved_image_data_uri"]
                # --- <<< END MODIFIED >>> ---

            elif next_step == "agent_tool_executor":
                print(f"  Calling agent_tool_executor...")
                last_ai_message = next( (msg for msg in reversed(mcp_loop_messages) if isinstance(msg, AIMessage)), None )
                if last_ai_message and last_ai_message.tool_calls:
                     executor_input_state = {"messages": [last_ai_message], "target_mcp": "pinterest"}
                     executor_output = await agent_tool_executor(executor_input_state, config)
                     local_mcp_state["messages"].extend(executor_output.get("messages", []))
                     # Path/URI handling is done by the agent node after executor returns
                else:
                     print(f"  Warning: Expected AIMessage with tool_calls before agent_tool_executor, but not found.")
                     local_mcp_state["messages"].append(ToolMessage(content="[Error: No tool calls found in previous AI message]", tool_call_id="internal_error"))
            else:
                mcp_loop_error = ValueError(f"MCP loop error: Unknown step '{next_step}'")
                print(f"{node_name} Error: {mcp_loop_error}")
                break

            await asyncio.sleep(1)

        # Add the very last raw messages from the final step/break
        mcp_loop_messages.extend(local_mcp_state.get("messages", []))

        if step_count >= max_mcp_steps:
            print(f"{node_name} Warning: Reached max MCP steps ({max_mcp_steps}). Ending loop.")
            mcp_loop_error = TimeoutError("MCP task exceeded maximum steps.")

        # --- <<< MODIFIED: Include 'saved_image_paths' in final outcome >>> ---
        final_mcp_outcome = {
            "saved_image_path": local_mcp_state.get("saved_image_path"), # Keep last/single for reference if needed
            "saved_image_data_uri": local_mcp_state.get("saved_image_data_uri"), # Keep last/single for reference
            "saved_image_paths": local_mcp_state.get("saved_image_paths"), # Pass the full list
            "mcp_message_history": mcp_loop_messages
        }
        # --- <<< END MODIFIED >>> ---
        print(f"  Stored {len(final_mcp_outcome['mcp_message_history'])} raw MCP messages for history.")
        if final_mcp_outcome.get("saved_image_paths"):
             print(f"  Final MCP outcome includes {len(final_mcp_outcome['saved_image_paths'])} paths in 'saved_image_paths'.")
        else:
             print("  Final MCP outcome does not include 'saved_image_paths'.")


        if mcp_loop_error:
            outer_error_to_report = mcp_loop_error
            print(f"  MCP loop terminated with error: {mcp_loop_error}")

    except Exception as e:
        error_msg = f"Error during {node_name} execution: {e}"
        print(error_msg)
        traceback.print_exc()
        outer_error_to_report = e
        # --- <<< MODIFIED: Ensure consistency in error case >>> ---
        final_mcp_outcome = {
             "saved_image_path": None,
             "saved_image_data_uri": None,
             "saved_image_paths": None, # Include None here too
             "mcp_message_history": mcp_loop_messages
         }
        # --- <<< END MODIFIED >>> ---

    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=None, output_files=None,
        error=outer_error_to_report,
        mcp_result=final_mcp_outcome # Pass the potentially populated final_mcp_outcome
    )

    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- <<< ADDED: OSM MCP Coordinator Node >>> ---
async def run_osm_mcp_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes an OpenStreetMap task using the MCP agent logic within this node.
    Uses TaskState inputs (address) and updates TaskState outputs/status.
    Returns MCP message history (raw objects) and necessary file info.
    """
    node_name = "OSM MCP Node"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]

    if current_idx < 0 or current_idx >= len(tasks):
        print(f"{node_name} Error: Invalid task index {current_idx}.")
        return {"tasks": [t.copy() for t in state.get("tasks", [])], "current_task": state.get("current_task").copy() if state.get("current_task") else None}

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs")
    outer_error_to_report = None
    final_mcp_outcome = None
    mcp_loop_messages: List[BaseMessage] = [] # Store raw messages from the loop

    try:
        # 1. Extract inputs
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs' in the current task.")
        user_request = task_inputs.get("user_request") # Should contain the address

        if not user_request:
            raise ValueError("Missing 'user_request' (address) in task_inputs for OSMMCPCoordinator")

        print(f"{node_name}: Starting MCP task for address: '{user_request[:100]}...'")

        # 2. Construct Initial *Local* MCP State
        local_mcp_state: Dict[str, Any] = { # Use Dict for flexibility
            "messages": [HumanMessage(content=user_request)], # OSM starts simple
            "initial_request": user_request,
            "initial_image_path": None, # OSM doesn't take initial image
            "target_mcp": "osm",        # <<< Target is OSM >>>
            "task_complete": False,
            "saved_image_path": None,
            "saved_image_data_uri": None
        }

        # 3. Run Internal MCP Loop (Similar structure to Rhino/Pinterest)
        max_mcp_steps = 1000 # OSM usually finishes in fewer steps (agent -> executor -> agent -> END)
        step_count = 0
        mcp_loop_error = None
        while step_count < max_mcp_steps:
            step_count += 1
            print(f"\n{node_name}: --- MCP Internal Loop Step {step_count} ---")
            # Add raw messages to history *before* the next step
            mcp_loop_messages.extend(local_mcp_state.get("messages", [])) # Extend history with raw objects
            local_mcp_state["messages"] = [] # Clear messages for the next step's output

            # --- Routing logic (using imported mcp_should_continue) ---
            if step_count == 1:
                next_step = "osm_agent" # <<< Use OSM Agent >>>
                print(f"  First step: Directly routing to {next_step}")
            else:
                # Need to pass the *correct* target_mcp for should_continue
                temp_state_for_should_continue = {
                    "messages": mcp_loop_messages, # Pass full history for context
                    "target_mcp": "osm"            # <<< Important: Tell should_continue it's OSM >>>
                }
                next_step = mcp_should_continue(temp_state_for_should_continue)
                print(f"  mcp_should_continue result: {next_step}")

            if next_step == END:
                print(f"{node_name}: MCP loop finished based on should_continue.")
                break
            elif next_step == "osm_agent": # <<< Use OSM Agent >>>
                print(f"  Calling call_osm_agent...")
                agent_output = await call_osm_agent({"messages": mcp_loop_messages}, config) # Pass raw history
                local_mcp_state["messages"].extend(agent_output.get("messages", [])) # Add raw new messages
                # Update state with path/URI if agent node returns them (it does after successful geocode+screenshot)
                if "saved_image_path" in agent_output: local_mcp_state["saved_image_path"] = agent_output["saved_image_path"]
                if "saved_image_data_uri" in agent_output: local_mcp_state["saved_image_data_uri"] = agent_output["saved_image_data_uri"]
            elif next_step == "agent_tool_executor":
                print(f"  Calling agent_tool_executor...")
                last_ai_message = next( (msg for msg in reversed(mcp_loop_messages) if isinstance(msg, AIMessage)), None )
                if last_ai_message and last_ai_message.tool_calls:
                     # Ensure target_mcp is passed correctly
                     executor_input_state = {"messages": [last_ai_message], "target_mcp": "osm"} # <<< Target OSM tools >>>
                     executor_output = await agent_tool_executor(executor_input_state, config)
                     local_mcp_state["messages"].extend(executor_output.get("messages", [])) # Add raw tool results
                     # Path/URI handling is done by the agent node after executor returns
                else:
                     print(f"  Warning: Expected AIMessage with tool_calls before agent_tool_executor, but not found.")
                     local_mcp_state["messages"].append(ToolMessage(content="[Error: No tool calls found in previous AI message]", tool_call_id="internal_error"))
            else:
                mcp_loop_error = ValueError(f"MCP loop error: Unknown step '{next_step}'")
                print(f"{node_name} Error: {mcp_loop_error}")
                break

            await asyncio.sleep(1) # Small delay between steps

        # Add the very last raw messages from the final step/break
        mcp_loop_messages.extend(local_mcp_state.get("messages", []))

        if step_count >= max_mcp_steps:
            print(f"{node_name} Warning: Reached max MCP steps ({max_mcp_steps}). Ending loop.")
            mcp_loop_error = TimeoutError("MCP task exceeded maximum steps.")

        # Create final outcome dict (includes raw history, path, uri)
        final_mcp_outcome = {
            "saved_image_path": local_mcp_state.get("saved_image_path"),
            "saved_image_data_uri": local_mcp_state.get("saved_image_data_uri"),
            "mcp_message_history": mcp_loop_messages # Store raw BaseMessage list
        }
        print(f"  Stored {len(final_mcp_outcome['mcp_message_history'])} raw MCP messages for history.")

        if mcp_loop_error:
            outer_error_to_report = mcp_loop_error
            print(f"  MCP loop terminated with error: {mcp_loop_error}")

    except Exception as e:
        error_msg = f"Error during {node_name} execution: {e}"
        print(error_msg)
        traceback.print_exc()
        outer_error_to_report = e
        # Ensure history is still included (raw) in case of outer error
        final_mcp_outcome = {
             "saved_image_path": None,
             "saved_image_data_uri": None,
             "mcp_message_history": mcp_loop_messages # Include whatever history was gathered
         }

    # 5. Update the main WorkflowState Task using the helper
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=None, # Tool outputs are handled via mcp_result path/uri/history
        output_files=None, # Tool outputs are handled via mcp_result path/uri/history
        error=outer_error_to_report,
        mcp_result=final_mcp_outcome # Pass dict with history and file info
    )

    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}
# --- <<< END ADDED >>> ---

# =============================================================================
# 7. Subgraph Definition & Compilation (Updated)
# =============================================================================
tool_subgraph_builder = StateGraph(WorkflowState) # Use imported WorkflowState

# --- <<< MODIFIED: Add Nodes (including OSM) >>> ---
tool_subgraph_builder.add_node("prepare_tool_inputs", prepare_tool_inputs_node)
# Standard tool nodes
tool_subgraph_builder.add_node("rag_agent", run_rag_tool_node)
tool_subgraph_builder.add_node("image_gen_agent", run_image_gen_tool_node)
tool_subgraph_builder.add_node("web_search_agent", run_web_search_tool_node)
tool_subgraph_builder.add_node("model_render_agent", run_model_render_tool_node)
tool_subgraph_builder.add_node("generate_3d_agent", run_generate_3d_tool_node)
# --- MODIFIED: Remove SimulateFutureAgent node ---
# tool_subgraph_builder.add_node("simulate_future_agent", run_simulate_future_tool_node) # Removed
# --- END MODIFIED ---
tool_subgraph_builder.add_node("video_recognition_agent", run_video_recognition_tool_node)
tool_subgraph_builder.add_node("image_recognition_agent", run_image_recognition_tool_node)
tool_subgraph_builder.add_node("llm_task_agent", run_llm_task_node)
# MCP coordinator nodes
tool_subgraph_builder.add_node("rhino_mcp_node", run_rhino_mcp_node)
tool_subgraph_builder.add_node("pinterest_mcp_node", run_pinterest_mcp_node)
# --- <<< ADDED: OSM Node >>> ---
tool_subgraph_builder.add_node("osm_mcp_node", run_osm_mcp_node)
# --- <<< END ADDED >>> ---
# --- <<< END MODIFIED >>> ---

# --- <<< Routing Function (Updated Mapping including OSM) >>> ---
def route_after_prepare_inputs(state: WorkflowState) -> str:
    """
    Determines the next node after prepare_tool_inputs.
    Routes to the correct tool node if preparation succeeded, otherwise ENDs.
    Returns the name of the next node or "finished".
    """
    print("--- Routing decision @ route_after_prepare_inputs ---")
    current_idx = state.get("current_task_index", -1)
    tasks = state.get("tasks", [])

    if current_idx < 0 or current_idx >= len(tasks):
        print("  Routing Error: Invalid task index. Routing to END.")
        return "finished"

    current_task = tasks[current_idx]
    status = current_task.get("status", "unknown")

    if status == "failed":
        print(f"  Input preparation failed for task {current_task.get('task_id')}. Routing to END.")
        return "finished"
    elif status == "in_progress":
        agent_name = current_task.get("selected_agent")
        if not agent_name:
            print(f"  Routing Error: Prep succeeded but no 'selected_agent' found for task {current_task.get('task_id')}. Routing to END.")
            return "finished"

        print(f"  Input prep successful for '{agent_name}'. Determining target tool node...")
        node_mapping = {
            "ArchRAGAgent": "rag_agent", "ImageGenerationAgent": "image_gen_agent",
            "WebSearchAgent": "web_search_agent", "ModelRenderAgent": "model_render_agent",
            "Generate3DAgent": "generate_3d_agent", 
            # --- MODIFIED: Removed SimulateFutureAgent ---
            # "SimulateFutureAgent": "simulate_future_agent",
            # --- END MODIFIED ---
            "VideoRecognitionAgent": "video_recognition_agent", "ImageRecognitionAgent": "image_recognition_agent",
            "LLMTaskAgent": "llm_task_agent",
            "RhinoMCPCoordinator": "rhino_mcp_node",
            "PinterestMCPCoordinator": "pinterest_mcp_node",
            # --- <<< ADDED: OSM Mapping >>> ---
            "OSMMCPCoordinator": "osm_mcp_node"
            # --- <<< END ADDED >>> ---
        }
        target_node = node_mapping.get(agent_name)

        if target_node:
            print(f"  Routing -> {target_node}")
            return target_node
        else:
            print(f"  Routing Error: Unknown agent name '{agent_name}' after successful prep. Routing to END.")
            return "finished"
    else:
        print(f"  Routing Warning: Unexpected status '{status}' after input prep for task {current_task.get('task_id')}. Routing to END.")
        return "finished"
# --- <<< END ROUTING FUNCTION >>> ---

# --- <<< MODIFIED: Set Entry Point and Edges (Including OSM) >>> ---
# Set Entry Point
tool_subgraph_builder.set_entry_point("prepare_tool_inputs")

# Conditional Edge directly from Input Prep to Tool Nodes or END
tool_subgraph_builder.add_conditional_edges(
    "prepare_tool_inputs",
    route_after_prepare_inputs,
    {
        "rag_agent": "rag_agent",
        "image_gen_agent": "image_gen_agent",
        "web_search_agent": "web_search_agent",
        "model_render_agent": "model_render_agent",
        "generate_3d_agent": "generate_3d_agent",
        # --- MODIFIED: Removed SimulateFutureAgent ---
        # "simulate_future_agent": "simulate_future_agent",
        # --- END MODIFIED ---
        "video_recognition_agent": "video_recognition_agent",
        "image_recognition_agent": "image_recognition_agent",
        "llm_task_agent": "llm_task_agent",
        "rhino_mcp_node": "rhino_mcp_node",
        "pinterest_mcp_node": "pinterest_mcp_node",
        # --- <<< ADDED: OSM Route >>> ---
        "osm_mcp_node": "osm_mcp_node",
        # --- <<< END ADDED >>> ---
        "finished": END
    }
)

# Edges from ALL Tool Nodes (including OSM) to END
tool_subgraph_builder.add_edge("rag_agent", END)
tool_subgraph_builder.add_edge("image_gen_agent", END)
tool_subgraph_builder.add_edge("web_search_agent", END)
tool_subgraph_builder.add_edge("model_render_agent", END)
tool_subgraph_builder.add_edge("generate_3d_agent", END)
# --- MODIFIED: Remove SimulateFutureAgent edge ---
# tool_subgraph_builder.add_edge("simulate_future_agent", END) # Removed
# --- END MODIFIED ---
tool_subgraph_builder.add_edge("video_recognition_agent", END)
tool_subgraph_builder.add_edge("image_recognition_agent", END)
tool_subgraph_builder.add_edge("llm_task_agent", END)
tool_subgraph_builder.add_edge("rhino_mcp_node", END)
tool_subgraph_builder.add_edge("pinterest_mcp_node", END)
# --- <<< ADDED: OSM Edge to END >>> ---
tool_subgraph_builder.add_edge("osm_mcp_node", END)
# --- <<< END ADDED >>> ---
# --- <<< END MODIFIED >>> ---

# Compile the subgraph and assign it to the original variable name
AssignTeamsSubgraph_Compiled = tool_subgraph_builder.compile()
AssignTeamsSubgraph_Compiled.name = "AssignTeamsSubgraph" # Updated name
assign_teams = AssignTeamsSubgraph_Compiled # Assign back to assign_teams for consistent import
print("Unified Tool Subgraph ('assign_teams') compiled successfully with Rhino, Pinterest, and OSM MCP Coordinators.")

