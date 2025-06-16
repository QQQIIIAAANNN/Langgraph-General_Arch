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
# --- ADDED: Import PIL for image processing ---
try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow not installed. Image processing (like cropping for model_render_image) will be skipped. Please install with 'pip install Pillow'.")
    Image = None # Set to None if not available
# --- END ADDED ---

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
GENERATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "render_cache")
RENDER_CACHE_DIR = os.path.join(OUTPUT_DIR, "cache/render_cache")
MODEL_CACHE_DIR = os.path.join(OUTPUT_DIR, "cache/model_cache")
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

def _append_feedback(task: TaskState, feedback: str, node_name: str):
    """Appends feedback to the task's feedback_log."""
    current_log = task.get("feedback_log") or ""
    prefix = f"[{node_name} Feedback]:"
    # Append new feedback block
    task["feedback_log"] = (current_log + f"\n{prefix}\n{feedback}").strip()

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
                # --- MODIFIED: Truncate mcp_history to first 2 messages ---
                truncated_history = mcp_history[:2]
                print(f"  Original MCP history length: {len(mcp_history)}, Truncated to: {len(truncated_history)} messages for storage.")
                # --- END MODIFICATION ---

                # --- Make MCP History serializable before storing ---
                serializable_history = []
                for msg in truncated_history: # Iterate over truncated_history
                     if isinstance(msg, BaseMessage):
                         # Convert BaseMessage to a serializable dict representation
                         try:
                              # Use LangChain's recommended way or a custom representation
                              # Example simple representation:
                              msg_dict = {
                                   "type": msg.type,
                                   # --- MODIFIED: Adjust truncation, especially for AIMessage content ---
                                   "additional_kwargs": msg.additional_kwargs, # Keep kwargs first
                              }
                              
                              # Handle content display
                              content_to_display = "[No Content]"
                              if msg.content:
                                  # --- MODIFICATION: No truncation for [目標階段計劃] ---
                                  if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.startswith("[目標階段計劃]:"):
                                      content_to_display = msg.content # No truncation
                                  # --- END MODIFICATION ---
                                  else: # Apply existing truncation logic for other messages
                                      content_repr = repr(msg.content)
                                      if len(content_repr) > 500: # Increased general truncation limit
                                          content_to_display = content_repr[:500] + "..."
                                      else:
                                          content_to_display = content_repr
                              msg_dict["content"] = content_to_display
                              # --- END MODIFICATION for content truncation ---

                              if hasattr(msg, 'name'): msg_dict['name'] = msg.name
                              if hasattr(msg, 'tool_call_id'): msg_dict['tool_call_id'] = msg.tool_call_id
                              if hasattr(msg, 'tool_calls') and msg.tool_calls: # Check if tool_calls is not None or empty
                                  # Also truncate tool_calls representation if it's too verbose
                                  tool_calls_repr = repr(msg.tool_calls)
                                  # Increased truncation limit for tool_calls as well
                                  msg_dict['tool_calls'] = tool_calls_repr[:500] + "..." if len(tool_calls_repr) > 500 else tool_calls_repr
                              serializable_history.append(msg_dict)
                         except Exception as msg_ser_err:
                              print(f"  Warning: Could not serialize message {type(msg)}: {msg_ser_err}")
                              serializable_history.append({"type": "serialization_error", "original_type": type(msg).__name__})
                     else:
                          serializable_history.append(msg) # Append if already serializable
                # --- END MODIFICATION for content truncation ---

                current_task["outputs"]["mcp_internal_messages"] = serializable_history
                print(f"  Added {len(serializable_history)} messages from MCP internal history (serialized and truncated).")
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
        filename = f"{prefix}_{uuid.uuid4().hex[:2]}.{ext}"
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
    根據當前任務的需求，準備適合給 LLM (Agent) 思考或給工具使用的輸入。
    這可能涉及從狀態中提取、格式化資訊，或生成給 LLM 的提示。
    """
    node_name = "Prepare Tool Inputs"
    # Access tasks list from state directly to modify in place for current_task
    tasks = [t.copy() for t in state['tasks']] # Create a copy to avoid modifying state directly before returning
    current_idx = state['current_task_index']
    if not (0 <= current_idx < len(tasks)):
        print(f"Assign Subgraph Error ({node_name}): Invalid current_task_index {current_idx}")
        # Set task as failed if index is valid range but still problematic
        if 0 <= current_idx < len(tasks):
             _set_task_failed(tasks[current_idx], f"Invalid current_task_index {current_idx}", node_name)
        return {"tasks": tasks}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')
    print(f"--- Running Node: {node_name} for Task {current_idx} (Agent: {selected_agent}, Objective: {current_task.get('description')}) ---")

    # Retrieve runtime configuration
    runtime_config = config.get("configurable", {})

    # --- MODIFIED: Properly construct LLM config dict from runtime_config for AssignAgent ---
    # Determine which LLM to use (specific agent LLM or default)
    aa_llm_config_params = {
        "model_name": runtime_config.get("aa_model_name"),
        "temperature": runtime_config.get("aa_temperature"),
        "max_tokens": runtime_config.get("aa_max_tokens"),
        # Note: provider is inferred by initialize_llm from model_name
    }
    # Filter out None values if not present in runtime_config
    aa_llm_config_params = {k: v for k, v in aa_llm_config_params.items() if v is not None}

    llm = initialize_llm(aa_llm_config_params, agent_name_for_default_lookup="assign_agent") # Use assign_agent config
    # --- END MODIFIED ---

    prompt_template = None
    prompt_template_name = "prepare_tool_inputs_prompt" # This is the name in config

    # Get the prompt template string, prioritize runtime config
    prompt_template_str = runtime_config.get("aa_prepare_tool_inputs_prompt") # Check runtime config first
    if not prompt_template_str: # Fallback to static config
         agent_cfg = config_manager.get_agent_config("assign_agent")
         if agent_cfg and agent_cfg.prompts:
             prompt_config_obj = agent_cfg.prompts.get(prompt_template_name) # Now prompt_template_name is "prepare_tool_inputs_prompt"
             prompt_template = prompt_config_obj.template if prompt_config_obj else None
         else:
            prompt_template = None # Agent or prompts dict not found
    else: # Use the string from runtime config
         prompt_template = prompt_template_str


    if not prompt_template:
        err_msg = f"Missing required prompt template. Searched runtime key 'aa_prepare_tool_inputs_prompt' and static config key '{prompt_template_name}' for AssignAgent."
        print(f"Assign Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks} # Return updated tasks list

    # Gather information for the prompt
    user_input = state.get("user_input", "N/A")
    # workflow_history = state.get("workflow_history", []) # Not directly used, aggregated_summary is built
    previous_task_results = state.get("tasks", []) # Get the *actual* list from state to examine previous completed tasks

    # Summarize workflow history and previous task results (excluding the current task being prepared)
    history_summary_parts = ["Workflow and Task History Summary (Aggregated Summary):"]
    # Iterate through *all* tasks up to the current index to provide context
    for i, task in enumerate(previous_task_results[:current_idx]):
         task_id = task.get("task_id", f"task_{i}")
         desc = task.get("description", "N/A")
         agent = task.get("selected_agent", "N/A")
         status = task.get("status", "N/A")
         # error_log_summary = task.get("error_log", "N/A") # Not directly part of this summary text, but available for specific error_feedback
         # Get file info without base64
         output_files_filtered_summary = _filter_base64_from_files(task.get("output_files", []))

         history_summary_parts.append(
             f"  - Task {i} (ID: {task_id}): '{desc}' | Agent: {agent} | Status: {status}"
         )
         # Add outputs and files if task was completed
         if status == "completed":
              # Limit output size for history summary
              outputs_summary_str = json.dumps(task.get("outputs", {}), ensure_ascii=False, default=str)[:200] + "..." if len(json.dumps(task.get("outputs", {}))) > 200 else json.dumps(task.get("outputs", {}))
              files_summary_str = json.dumps(output_files_filtered_summary, ensure_ascii=False, default=str)[:200] + "..." if len(json.dumps(output_files_filtered_summary)) > 200 else json.dumps(output_files_filtered_summary)
              history_summary_parts.append(f"    Outputs: {outputs_summary_str}")
              history_summary_parts.append(f"    Output Files ({len(output_files_filtered_summary)}): {files_summary_str}")
         elif status == "failed":
              error_log_summary_str = (task.get("error_log", "N/A")[:200] + "...") if len(task.get("error_log", "N/A")) > 200 else task.get("error_log", "N/A")
              history_summary_parts.append(f"    Error Log: {error_log_summary_str}")

    prepared_history_summary = "\n".join(history_summary_parts)

    # Get task-specific context
    task_objective = current_task.get("task_objective", "N/A")
    task_description = current_task.get("description", "N/A")
    # task_parameters = current_task.get("parameters", {}) # Not directly used in default prompt
    # task_input_files = current_task.get("input_files", []) # Not directly used in default prompt
    # filtered_input_files_for_prompt = _filter_base64_from_files(task_input_files)


    # Prepare filtered evaluation summary from the current task
    evaluation_data = current_task.get("evaluation", {})
    keys_to_keep_from_evaluation = [
        "selected_option_identifier",
        "assessment",
        "feedback_llm_overall", # This was in your original code, but not in prompt's input_vars
        "detailed_assessment", # This was in your original code, but not in prompt's input_vars
        # Keys more aligned with typical evaluation output structure:
        "assessment_type", "feedback", "improvement_suggestions", "detailed_option_scores"
    ]
    filtered_evaluation_data = {
        key: evaluation_data.get(key)
        for key in keys_to_keep_from_evaluation
        if evaluation_data.get(key) is not None # Only include keys that are present and not None
    }
    latest_evaluation_results_json = json.dumps(filtered_evaluation_data, ensure_ascii=False, indent=2, default=str)

    # Prepare aggregated outputs and files from PREVIOUSLY COMPLETED tasks for the prompt
    aggregated_outputs_list_for_prompt = []
    aggregated_files_list_for_prompt = []
    for task in previous_task_results[:current_idx]: # Only tasks *before* the current one
        if task.get("status") == "completed":
            task_output_entry_for_prompt = {
                "task_id": task.get("task_id"),
                "task_description": task.get("description"),
                "selected_agent": task.get("selected_agent"),
                "outputs": task.get("outputs", {}),
                "output_files_summary": _filter_base64_from_files(task.get("output_files", []))
            }
            aggregated_outputs_list_for_prompt.append(task_output_entry_for_prompt)
            
            for file_item in task.get("output_files", []):
                file_copy = _filter_base64_from_files([file_item])[0] 
                file_copy["source_task_id"] = task.get("task_id")
                aggregated_files_list_for_prompt.append(file_copy)

    aggregated_outputs_json_str = json.dumps(aggregated_outputs_list_for_prompt, ensure_ascii=False, default=str, indent=2)
    aggregated_files_json_str = json.dumps(aggregated_files_list_for_prompt, ensure_ascii=False, default=str, indent=2)
    
    # error_feedback for the current task if it's a retry or has previous error context.
    # The prompt variable "error_feedback" refers to "Context from Previous Attempt".
    # This would typically be the error_log from the current task if it's being retried.
    error_feedback_str = current_task.get("error_log", "N/A")


    # Prepare prompt inputs for the LLM, matching input_variables for 'prepare_tool_inputs_prompt'
    prompt_inputs = {
        "selected_agent_name": selected_agent if selected_agent else "N/A",
        "agent_description": agent_descriptions.get(selected_agent, "N/A") if selected_agent else "N/A",
        "user_input": user_input,
        "task_objective": task_objective,
        "task_description": task_description,
        "aggregated_summary": prepared_history_summary, # This is the text summary of workflow history
        "aggregated_outputs_json": aggregated_outputs_json_str, # JSON string of previous outputs
        "aggregated_files_json": aggregated_files_json_str, # JSON string of previous files
        "error_feedback": error_feedback_str, # Error from previous attempt of this task
        "llm_output_language": runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT),
        "user_budget_limit": state.get("user_budget_limit", None), # Pass None if not set
        "latest_evaluation_results_json": latest_evaluation_results_json,
    }

    # Catch KeyError during format
    try:
        prep_prompt = prompt_template.format(**prompt_inputs)
    except KeyError as ke:
        err_msg = f"Formatting error (KeyError: {ke}). Check prompt template '{prompt_template_name}' and its input_variables. Missing key in prompt_inputs: '{ke}'."
        print(f"Assign Subgraph Error ({node_name}): {err_msg}")
        print(f"--- Problematic Prompt Template ('{prompt_template_name}') Content ---")
        print(repr(prompt_template)) # Print the actual template string
        print(f"--- Provided Inputs for .format() ---")
        print(json.dumps(prompt_inputs, indent=2, ensure_ascii=False, default=str))
        print(f"---------------------------------")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks} # Return updated tasks list


    print(f"  - Invoking LLM for tool input preparation...")
    try:
        prep_response = await llm.ainvoke(prep_prompt)
        prep_content = prep_response.content.strip()

        # Clean potential markdown code block
        if prep_content.startswith("```json"):
            prep_content = prep_content[7:-3].strip()
        elif prep_content.startswith("```"):
             # Handle generic code blocks too, assuming JSON within
            prep_content = prep_content[3:-3].strip()


        # Attempt to parse the LLM's response as JSON
        try:
            prepared_tool_inputs_dict = json.loads(prep_content) # Rename variable for clarity
            # Validate the expected structure (at least check if it's a dictionary)
            if isinstance(prepared_tool_inputs_dict, dict):
                # Store the prepared inputs in the current task state UNDER 'task_inputs'
                current_task["task_inputs"] = prepared_tool_inputs_dict # MODIFIED: Store under "task_inputs"
                # Optionally log success
                print(f"  - Tool inputs prepared successfully by LLM and stored in 'task_inputs'.")
                # Add feedback to the task log
                _append_feedback(current_task, "Tool inputs prepared and stored.", node_name)

                # Update status to "in_progress" if it wasn't failed, to allow routing
                if current_task.get("status") != "failed":
                    current_task["status"] = "in_progress"

            else:
                err_msg = f"LLM returned valid JSON, but not a dictionary (type: {type(prepared_tool_inputs_dict)}). Raw content: '{prep_content}'" # Use new variable name
                print(f"Assign Subgraph Error ({node_name}): {err_msg}")
                _set_task_failed(current_task, err_msg, node_name)
                _append_feedback(current_task, f"Failed to parse LLM response as dictionary: {err_msg}", node_name)

        except json.JSONDecodeError as json_e:
            err_msg = f"Could not parse LLM JSON response: {json_e}. Raw content: '{prep_content}'"
            print(f"Assign Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)
            _append_feedback(current_task, f"Failed to parse LLM response as JSON: {err_msg}", node_name)
        except Exception as e_parse:
             err_msg = f"Unexpected error parsing LLM response: {e_parse}. Raw content: '{prep_content}'"
             print(f"Assign Subgraph Error ({node_name}): {err_msg}")
             traceback.print_exc()
             _set_task_failed(current_task, err_msg, node_name)
             _append_feedback(current_task, f"Unexpected error parsing LLM response: {err_msg}", node_name)


    except Exception as e_llm:
        err_msg = f"LLM call failed during tool input preparation: {e_llm}"
        print(f"Assign Subgraph Error ({node_name}): {err_msg}")
        traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)
        _append_feedback(current_task, f"LLM call failed: {err_msg}", node_name)


    # Update the task in the state dictionary
    tasks[current_idx] = current_task
    # Return the updated state parts
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
        # --- MODIFIED: Properly construct LLM config dict from runtime_config for ToolAgent (used by LLMTaskAgent) ---
        # Using 'ta_llm' config for LLMTaskAgent execution
        llm_config_dict = {
            "model_name": runtime_config.get("ta_model_name"),
            "temperature": runtime_config.get("ta_temperature"),
            "max_tokens": runtime_config.get("ta_max_tokens"),
            # Note: provider is inferred by initialize_llm from model_name
        }
        # Filter out None values if not present in runtime_config
        llm_config_dict = {k: v for k, v in llm_config_dict.items() if v is not None}

        llm = initialize_llm(llm_config_dict, agent_name_for_default_lookup="tool_agent")
        # --- END MODIFIED ---
        print(f"LLM Task Node: Invoking LLM ({llm.__class__.__name__})...") # Log LLM class
        response = await llm.ainvoke(prompt_for_llm)
        result_content = response.content.strip()
        print(f"LLM Task Node: Received response: {result_content[:100]}...") # Handle non-string results if necessary
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
    node_name = "Image Generation Tool"
    print(f"--- Running {node_name} (Unified Subgraph) ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs", {})
    prompt = task_inputs.get("prompt", "Null")
    image_count = task_inputs.get("i", 1)
    error_to_report = None
    output_files_list = []
    final_outputs = {}

    try:
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs'")

        print(f"{node_name}: Using prompt='{prompt[:50]}...', image_inputs: {'Provided' if task_inputs.get('image_inputs') else 'None'}. Requesting i={image_count} image(s).") # Updated log

        # --- Call the tool, passing 'i' ---
        # Assuming generate_gemini_image is synchronous based on previous code
        tool_result = generate_gemini_image({
            "prompt": prompt,
            "image_inputs": task_inputs.get("image_inputs") or [],
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
            # --- File processing loop ---
            for idx, file_info in enumerate(tool_generated_files):
                filename = file_info.get("filename")
                file_type = file_info.get("file_type", "image/png") # Default to png
                # original_description_from_tool = file_info.get("description", "Generated by tool") # Optional: if you need to keep tool's own description

                if not filename:
                    print(f"  - Warning: Tool output missing filename. Skipping file.")
                    continue

                # Ensure the filename is just the name, not a path
                filename = os.path.basename(filename)
                expected_path = os.path.join(RENDER_CACHE_DIR, filename) # All tool outputs go to RENDER_CACHE_DIR

                if os.path.exists(expected_path):
                    print(f"  - File FOUND at path: {expected_path}")
                    
                    # --- MODIFIED: Construct structured description ---
                    structured_description = (
                        f"SourceAgent: {current_task.get('selected_agent', 'ImageGenerationAgent')}; "
                        f"TaskDesc: {current_task.get('description', 'N/A')}; "
                        f"ImageNum: {idx + 1}/{image_count}"

                    )
                    processed_file_info = {
                        "filename": filename,
                        "path": expected_path,
                        "type": file_type,
                        "description": structured_description # Use new structured description
                    }
                    # --- END MODIFICATION ---

                    # Add base64 encoding (remains the same)
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

# --- run_model_render_tool_node (MODIFIED to handle odd dimensions) ---
def run_model_render_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    node_name = "Model Render Tool"
    print(f"--- Running {node_name} (Handles Photorealistic Rendering) ---") # Modified description
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs", {})
    error_to_report = None
    aggregated_output_files_list = []
    aggregated_generated_filenames = []
    final_outputs = {}

    try:
        if not task_inputs or not isinstance(task_inputs, dict):
            raise ValueError("Invalid or missing 'task_inputs'")

        initial_outer_prompt_context = task_inputs.get("outer_prompt") # This is the contextual prompt from prepare_tool_inputs
        image_inputs_paths = task_inputs.get("image_inputs") # List of image paths

        if not initial_outer_prompt_context:
            raise ValueError("Missing required input 'outer_prompt' (contextual prompt)")
        if not image_inputs_paths or not isinstance(image_inputs_paths, list) or not image_inputs_paths or not all(isinstance(p, str) for p in image_inputs_paths):
            raise ValueError("Missing or invalid 'image_inputs' (must be a non-empty list of string paths)")

        print(f"{node_name}: Task type: Photorealistic Render") # Simplified task type
        print(f"  Initial outer_prompt='{initial_outer_prompt_context[:70]}...', processing {len(image_inputs_paths)} image(s).")

        for idx, image_full_path in enumerate(image_inputs_paths):
            print(f"  Processing image {idx+1}/{len(image_inputs_paths)}: {image_full_path}")
            
            # The prompt for model_render_image will be the initial_outer_prompt_context
            # as img_recognition based refinement is removed.
            current_prompt_for_tool = initial_outer_prompt_context
            processed_image_path_for_tool = image_full_path # Default to original path

            # --- Image Dimension Check and Cropping (Remains) ---
            if Image: 
                try:
                    with Image.open(image_full_path) as img:
                        original_width, original_height = img.size
                        needs_cropping = False
                        new_width, new_height = original_width, original_height

                        if original_width % 2 != 0:
                            new_width -= 1
                            needs_cropping = True
                            print(f"    Warning: Image width is odd ({original_width}). Will crop to {new_width}.")
                        if original_height % 2 != 0:
                            new_height -= 1
                            needs_cropping = True
                            print(f"    Warning: Image height is odd ({original_height}). Will crop to {new_height}.")

                        if needs_cropping:
                            cropped_img = img.crop((0, 0, new_width, new_height))
                            temp_filename = f"cropped_{uuid.uuid4().hex[:8]}_{os.path.basename(image_full_path)}"
                            temp_filepath = os.path.join(RENDER_CACHE_DIR, temp_filename)
                            save_format = img.format if img.format in ['JPEG', 'PNG', 'BMP', 'GIF'] else 'PNG'
                            if save_format == 'JPEG' and cropped_img.mode == 'RGBA':
                                cropped_img = cropped_img.convert('RGB')
                                print(f"    Converted cropped image from RGBA to RGB for JPEG saving.")
                            cropped_img.save(temp_filepath, format=save_format)
                            processed_image_path_for_tool = temp_filepath 
                            print(f"    Cropped image saved to temporary path: {temp_filepath}")
                        else:
                            print("    Image dimensions are already even. No cropping needed.")

                except FileNotFoundError:
                     print(f"    Error: Input image file not found at {image_full_path}. Skipping processing.")
                     _append_feedback(current_task, f"Input image not found: {image_full_path}", node_name)
                     continue 
                except Exception as img_e:
                    print(f"    Error processing image file {image_full_path} for dimension check/cropping: {img_e}")
                    traceback.print_exc()
                    _append_feedback(current_task, f"Image processing error for {os.path.basename(image_full_path)}: {img_e}", node_name)
                    continue 
            else:
                 print("    Pillow not available. Skipping image dimension check and cropping.")
            # --- END Image Dimension Check and Cropping ---

            # --- REMOVED: Image Recognition Prompt Refinement ---
            # The refined_prompt_text is now directly initial_outer_prompt_context

            # --- Call Model Render Tool (Use the *processed* image path and initial prompt) ---
            print(f"    Calling model_render_image with outer_prompt='{current_prompt_for_tool[:50]}...', image_input_path='{processed_image_path_for_tool}'")
            result = model_render_image({
                "outer_prompt": current_prompt_for_tool, # Use the initial prompt directly
                "image_inputs": processed_image_path_for_tool
            })

            if isinstance(result, str) and result.startswith("Error:"):
                print(f"    Tool Error for {image_full_path}: {result}")
                _append_feedback(current_task, f"Failed to render {os.path.basename(image_full_path)}: {result}", node_name)
                continue 

            if not isinstance(result, str) or not result:
                print(f"    Tool returned unexpected/empty result for {image_full_path}: {result}")
                _append_feedback(current_task, f"Unexpected result for {os.path.basename(image_full_path)}: {result}", node_name)
                continue 

            # --- Process Tool Output ---
            output_filename_from_tool = result.strip()
            if output_filename_from_tool:
                file_info = _save_tool_output_file( 
                    output_filename_from_tool,
                    GENERATION_OUTPUT_DIR,
                    "image/png", 
                    f"SourceAgent: {current_task.get('selected_agent', 'ModelRenderAgent')}; " # Corrected agent name
                    f"TaskDesc: {current_task.get('description', 'N/A')}; "
                    f"RenderedView: {idx + 1}/{len(image_inputs_paths)}; "
                    f"InputImage: {os.path.basename(image_full_path)}"
                )
                if file_info:
                    aggregated_output_files_list.append(file_info)
                    aggregated_generated_filenames.append(output_filename_from_tool)
                    print(f"    Successfully processed and saved tool output: {output_filename_from_tool}")
                else:
                    print(f"    Could not find/process tool output file: {output_filename_from_tool} for input {os.path.basename(image_full_path)}")
                    _append_feedback(current_task, f"Tool output file not found/processed for input {os.path.basename(image_full_path)}: {output_filename_from_tool}", node_name)
            else:
                print(f"    Tool did not return a filename for input {os.path.basename(image_full_path)}.")
                _append_feedback(current_task, f"Tool returned no output filename for input {os.path.basename(image_full_path)}.", node_name)

        # --- Final Check for Generated Files ---
        if not aggregated_output_files_list and image_inputs_paths:
            # If there were inputs but no successful outputs
            feedback = current_task.get("feedback_log") or "ModelRenderAgent: Tool ran but failed to produce/locate any output files for the given render_image(s)."
            print(f"{node_name} Warning: {feedback}")
            error_to_report = ValueError(feedback)


        final_outputs = {"generated_filenames": aggregated_generated_filenames}
        if aggregated_generated_filenames:
             print(f"{node_name}: Completed. Generated {len(aggregated_generated_filenames)} file(s).")
        else:
             print(f"{node_name}: Completed, but no files were successfully generated/processed.")


    except Exception as e: # Catch errors in the node's setup or general logic
        error_to_report = e
        final_outputs = {"error_message": str(e)}
        traceback.print_exc()

    # Update task state using the helper function
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=final_outputs,
        output_files=aggregated_output_files_list, # Contains all successfully processed files
        error=error_to_report
    )
    # Return the updated tasks list and the current task copy
    return {"tasks": tasks, "current_task": tasks[current_idx].copy()}

# --- run_generate_3d_tool_node ---
def run_generate_3d_tool_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    node_name = "3D Model Generation Tool"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state.get("current_task_index", -1)
    tasks = [t.copy() for t in state.get("tasks", [])]
    if current_idx < 0 or current_idx >= len(tasks): return {"tasks": state.get("tasks", [])}
    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs", {}) # Get task_inputs

    error_to_report = None
    output_files_list = []
    final_outputs = {} # Store overall results, e.g., a list of generated file names

    processed_files_summary = [] # To store a summary of what was generated for final_outputs

    try:
        # --- MODIFIED: 直接從 task_inputs 獲取 image_paths 或 image_path ---
        input_image_paths = []
        if "image_paths" in task_inputs and isinstance(task_inputs["image_paths"], list):
            input_image_paths = task_inputs["image_paths"]
            print(f"{node_name}: Found {len(input_image_paths)} image paths directly in 'task_inputs.image_paths'.")
        elif "image_path" in task_inputs and isinstance(task_inputs["image_path"], str):
            input_image_paths = [task_inputs["image_path"]]
            print(f"{node_name}: Found single image path directly in 'task_inputs.image_path'.")

        # --- Removed the check for tool_parameters dictionary ---
        # if not tool_parameters or not isinstance(tool_parameters, dict):
        #     raise ValueError("Invalid or missing 'tool_parameters' in task_inputs")
        # --- END MODIFIED ---

        if not input_image_paths:
            raise ValueError("Missing 'image_path' or 'image_paths' in task_inputs for 3D model generation.")

        actual_3d_save_dir = os.path.abspath(os.path.join("output", "model_cache"))
        os.makedirs(actual_3d_save_dir, exist_ok=True)
        print(f"  Ensuring 3D files will be saved/checked in directory: {actual_3d_save_dir}")

        for idx, current_image_path in enumerate(input_image_paths):
            if not current_image_path or not isinstance(current_image_path, str):
                print(f"  - Warning: Invalid image path at index {idx}: {current_image_path}. Skipping.")
                processed_files_summary.append({
                    "input_image": current_image_path or "N/A", "status": "skipped_invalid_path",
                    "model": None, "video": None
                })
                continue

            # Check if path is relative and make it absolute if needed (assuming relative to OUTPUT_DIR)
            # This part depends on how image_path is typically provided from previous tasks.
            # For now, let's assume it's either absolute or relative to a common base like OUTPUT_DIR.
            if not os.path.isabs(current_image_path):
                resolved_image_path = os.path.join(OUTPUT_DIR, current_image_path) # Check against OUTPUT_DIR
                if not os.path.exists(resolved_image_path): # Fallback to check if it's already in a cache dir
                    resolved_image_path = os.path.join(RENDER_CACHE_DIR, current_image_path)
                    if not os.path.exists(resolved_image_path):
                         resolved_image_path = os.path.join(MODEL_CACHE_DIR, current_image_path)
                         if not os.path.exists(resolved_image_path):
                            resolved_image_path = current_image_path # Use as is if not found in common locations
                current_image_path = os.path.abspath(resolved_image_path)


            if not os.path.exists(current_image_path):
                print(f"  - Warning: Input image path does not exist: {current_image_path}. Skipping for 3D gen.")
                processed_files_summary.append({
                    "input_image": current_image_path, "status": "skipped_not_found",
                    "model": None, "video": None
                })
                continue

            print(f"\n  Processing input image {idx + 1}/{len(input_image_paths)}: '{current_image_path}'")

            try:
                # --- MODIFIED: Ensure the tool call uses the absolute path ---
                result = generate_3D({"image_path": current_image_path}) # Call tool for current image with resolved path
                # --- END MODIFIED ---

                if isinstance(result, dict) and "error" in result:
                    print(f"    - Tool Error for image {current_image_path}: {result['error']}")
                    processed_files_summary.append({
                        "input_image": current_image_path, "status": "tool_error",
                        "error_detail": result['error'], "model": None, "video": None
                    })
                    # Continue to next image, don't let one failure stop all
                    if error_to_report is None: # Store first error
                        error_to_report = ValueError(f"Tool error on {os.path.basename(current_image_path)}: {result['error']}")
                    else: # Append subsequent errors
                        error_to_report = ValueError(f"{str(error_to_report)}; Tool error on {os.path.basename(current_image_path)}: {result['error']}")
                    continue
                if not isinstance(result, dict):
                    print(f"    - Tool returned unexpected result type for image {current_image_path}: {type(result)}")
                    processed_files_summary.append({
                        "input_image": current_image_path, "status": "unexpected_tool_result",
                        "model": None, "video": None
                    })
                    if error_to_report is None:
                        error_to_report = TypeError(f"Unexpected tool result on {os.path.basename(current_image_path)}")
                    else:
                        error_to_report = TypeError(f"{str(error_to_report)}; Unexpected tool result on {os.path.basename(current_image_path)}")
                    continue

                model_filename = result.get("model")
                video_filename = result.get("video")
                current_input_summary = {"input_image": current_image_path, "status": "processed", "model": None, "video": None}

                if model_filename:
                    structured_description = (
                        f"SourceAgent: {current_task.get('selected_agent', '3DGenerationAgent')}; "
                        f"TaskDesc: {current_task.get('description', 'N/A')}; "
                        f"InputNum: {idx + 1}/{len(input_image_paths)}; "
                        f"FileType: 3DModel"
                    )
                    # Ensure we pass the correct expected save directory
                    file_info = _save_tool_output_file(model_filename, actual_3d_save_dir, "model/gltf-binary", structured_description)
                    if file_info:
                        output_files_list.append(file_info)
                        current_input_summary["model"] = model_filename
                        print(f"    - Saved 3D Model: {model_filename}")
                    else:
                        print(f"    - Warning: Could not save/process model file: {model_filename}")

                if video_filename:
                    structured_description = (
                        f"SourceAgent: {current_task.get('selected_agent', '3DGenerationAgent')}; "
                        f"TaskDesc: {current_task.get('description', 'N/A')}; "
                        f"InputNum: {idx + 1}/{len(input_image_paths)}; "
                        f"FileType: AnimationVideo"
                    )
                     # Ensure we pass the correct expected save directory
                    file_info = _save_tool_output_file(video_filename, actual_3d_save_dir, "video/mp4", structured_description)
                    if file_info:
                        output_files_list.append(file_info)
                        current_input_summary["video"] = video_filename
                        print(f"    - Saved Animation Video: {video_filename}")
                    else:
                        print(f"    - Warning: Could not save/process video file: {video_filename}")
                processed_files_summary.append(current_input_summary)

            except Exception as e_inner: # Catch error during a single image processing
                print(f"    - Error processing image {current_image_path}: {e_inner}")
                traceback.print_exc()
                processed_files_summary.append({
                    "input_image": current_image_path, "status": "processing_exception",
                    "error_detail": str(e_inner), "model": None, "video": None
                })
                if error_to_report is None:
                    error_to_report = e_inner
                else: # Append to existing error
                    error_to_report = type(e_inner)(f"{str(error_to_report)}; Error on {os.path.basename(current_image_path)}: {str(e_inner)}")


        # If no files generated AND there were inputs to process, AND no specific error yet
        if not output_files_list and not error_to_report and input_image_paths:
             error_to_report = ValueError(f"Tool processed {len(input_image_paths)} input(s) but produced no valid output files (models or videos). Check logs for individual image processing details.")
        elif not output_files_list and not error_to_report and not input_image_paths:
            # This case should be caught by the "Missing image_path" check earlier, but good fallback
             error_to_report = ValueError("No input images were provided to the 3D generation tool.")


        final_outputs["processed_images_summary"] = processed_files_summary
        if output_files_list:
             final_outputs["generated_file_count"] = len(output_files_list)


    except Exception as e: # Catch errors in the node's setup or general logic
        error_to_report = e
        final_outputs = {"error_message": str(e)}
        # output_files_list remains as is (might be partially populated if error occurred mid-loop)
        traceback.print_exc()

    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs=final_outputs,
        output_files=output_files_list, # Contains all successfully processed files
        error=error_to_report
    )
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
    node_name = "Rhino MCP Tool"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state['tasks']]
    current_idx = state['current_task_index']
    if not (0 <= current_idx < len(tasks)):
        print(f"{node_name} Error: Invalid current_task_index {current_idx}")
        # No task update here as current_task is not well-defined
        return {"tasks": tasks} # Return original tasks

    current_task = tasks[current_idx]
    task_inputs = current_task.get("task_inputs", {})
    if not isinstance(task_inputs, dict): # Ensure task_inputs is a dict
        task_inputs = {}
        print(f"{node_name} Warning: task_inputs was not a dict, defaulting to empty dict.")

    outer_error_to_report = None
    output_files_list = []
    mcp_errors = [] # Collect errors from MCP steps

    # --- MODIFIED: Initialize screenshot counter for renaming ---
    rhino_screenshot_counter = 0
    # --- END MODIFICATION ---

    # This local_mcp_state is for the internal loop of this node.
    # It's distinct from the MCPAgentState used by mcp_test.py's graph.
    local_mcp_state: Dict[str, Any] = {
        "messages": [], # Will be populated with initial HumanMessage
        "initial_request": task_inputs.get("user_request"),
        "initial_image_path": task_inputs.get("initial_image_path"),
        "target_mcp": "rhino", # Explicitly set for this node's purpose
        # Fields to store results from the MCP loop relevant to file outputs
        "saved_image_path": None, # Path of the LATEST image saved in an MCP step
        "saved_image_data_uri": None,
        "rhino_screenshot_counter": 0 # Counter specifically for screenshots processed by this node
    }

    # --- Build initial HumanMessage for the MCP graph ---
    initial_human_content_parts = []
    if local_mcp_state["initial_request"]:
        initial_human_content_parts.append({"type": "text", "text": local_mcp_state["initial_request"]})
    if local_mcp_state["initial_image_path"] and os.path.exists(local_mcp_state["initial_image_path"]):
        try:
            with open(local_mcp_state["initial_image_path"], "rb") as img_file:
                img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            mime_type = "image/png" # Default or detect
            ext = os.path.splitext(local_mcp_state["initial_image_path"])[1].lower()
            if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            initial_human_content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
            })
        except Exception as e_img:
            print(f"  Warning: Could not encode initial image for Rhino MCP: {e_img}")
    if not initial_human_content_parts: # Must have some content
        err_msg = "Rhino MCP node requires 'user_request' in task_inputs."
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks, "current_task": current_task.copy()}
    local_mcp_state["messages"] = [HumanMessage(content=initial_human_content_parts)]
    # --- End initial HumanMessage build ---

    # --- MCP Graph (from mcp_test) Invocation ---
    # The MCP graph from mcp_test.py is now compiled as `graph`
    # We need to simulate its execution loop here, passing the `local_mcp_state`
    # and getting back updates.

    max_mcp_steps = 1000 # Max iterations for the MCP agent
    mcp_final_outcome_for_task_update = {} # Store the final result from the MCP loop

    try:
        # --- Get the compiled MCP graph (assuming it's globally available or passed) ---
        # This assumes `graph` is the compiled StateGraph from mcp_test.py
        # For direct use here, you'd typically import `graph` from `mcp_test`
        # from src.mcp_test import mcp_graph # Or whatever it's named
        # For this example, let's assume `mcp_graph_instance` is available
        # If mcp_test.py exposes `graph` directly:
        from src.mcp_test import graph as rhino_mcp_instance, MCPAgentState as InternalMCPAgentState

        # The state for the internal MCP graph needs to be compatible with MCPAgentState
        internal_mcp_graph_state: InternalMCPAgentState = {
            "messages": local_mcp_state["messages"],
            "initial_request": local_mcp_state["initial_request"],
            "initial_image_path": local_mcp_state["initial_image_path"],
            "target_mcp": "rhino", # Fixed for this context
            "task_complete": False,
            "saved_image_path": None,
            "saved_image_data_uri": None,
            "consecutive_llm_text_responses": 0,
            "rhino_screenshot_counter": 0, # Initialize counter for the internal graph
            "last_executed_node": None

        }
        
        print(f"  Starting Rhino MCP internal graph execution... Max steps: {max_mcp_steps}")
        final_internal_mcp_state = None

        # --- Main MCP Loop using the imported graph ---

        # NEW: Prepare a config with an increased recursion limit for the internal MCP graph
        mcp_invocation_config = config.copy() # Start with the config passed to the node
        if "configurable" not in mcp_invocation_config:
            mcp_invocation_config["configurable"] = {}
        
        # Increase the recursion limit for the MCP graph instance
        # The default is 25. Let's set it higher, e.g., 1000.
        # 您可以根據需要調整這個值
        new_recursion_limit = 1000
        mcp_invocation_config["configurable"]["recursion_limit"] = new_recursion_limit
        print(f"  Invoking Rhino MCP internal graph with recursion_limit: {new_recursion_limit}")

        async for step_config in rhino_mcp_instance.astream(
            internal_mcp_graph_state, 
            config=mcp_invocation_config, # Pass the modified config
            stream_mode="values"
        ):
            # step_config is a dictionary where keys are node names and values are their outputs
            # We are interested in the full state after each step to extract output_definitions
            # The 'values' stream_mode gives us the full state after each node that has executed.
            
            print(f"  Rhino MCP Internal Step Output Keys: {list(step_config.keys())}")
            
            # The complete state of the internal MCP graph is usually the value of the last executed node
            # or a specific key if the graph is structured to output its full state under one key.
            # For a standard StateGraph, the output of stream_mode="values" is a dict where each key
            # is a node that ran in that step, and its value is that node's output.
            # We need the *final state* of the internal graph after a node runs.
            # LangGraph typically returns the full state if the stream outputs are iterated.
            # The `step_config` here IS the full state if `stream_mode="values"` is used on the compiled graph.
            
            final_internal_mcp_state = step_config # The 'step_config' is the entire state dict
            
            # Extract messages for history
            # local_mcp_state["messages"].extend(final_internal_mcp_state.get("messages", [])) # This appends to outer history
            
            # --- Process output definitions from current MCP step ---
            # Output definitions are usually part of an AIMessage's tool_calls or a specific field
            # in the agent's response. This needs to be consistent with how `call_rhino_agent` structures its output.
            # Assuming output_definitions are in the AIMessage content or additional_kwargs:
            last_ai_message = next((m for m in reversed(final_internal_mcp_state.get("messages",[])) if isinstance(m, AIMessage)), None)
            mcp_step_config_from_ai = {}
            if last_ai_message:
                if isinstance(last_ai_message.content, str) and last_ai_message.content.startswith("{"): # JSON in content
                    try: mcp_step_config_from_ai = json.loads(last_ai_message.content)
                    except: pass
                if not mcp_step_config_from_ai and last_ai_message.additional_kwargs: # Check kwargs
                    mcp_step_config_from_ai = last_ai_message.additional_kwargs.get("configuration", {}) # Example key

            current_step_output_defs = mcp_step_config_from_ai.get("output_definitions", [])
            
            # --- MODIFICATION: Renaming and processing output files ---
            processed_image_from_state_this_step = False # Flag to avoid double processing
            if current_step_output_defs:
                print(f"  - Rhino MCP Step Output Definitions: {len(current_step_output_defs)}")
                for idx, output_def in enumerate(current_step_output_defs):
                    is_image_output = output_def.get("mime_type", "").startswith("image/")
                    
                    # Path handling:
                    # For images from capture_focused_view, the actual path comes from internal_mcp_graph_state["saved_image_path"]
                    # which would have been renamed by mcp_test.py's agent_node_logic.
                    # For other files, it's from output_def.get("file_path")
                    
                    path_for_entry = None
                    filename_for_entry = None
                    current_file_description_parts = [
                        f"SourceAgent: {current_task.get('selected_agent', 'RhinoMCPAgent')}",
                        f"TaskDesc: {current_task.get('description', 'N/A')}"
                    ]
                    planned_name_from_def = output_def.get('name', f"Output{idx+1}")

                    if is_image_output:
                        # This image was presumably saved by a tool like capture_focused_view,
                        # and its path (already renamed with a counter from mcp_test.py) is in final_internal_mcp_state.
                        img_path_from_internal_state = final_internal_mcp_state.get("saved_image_path")
                        internal_counter_val = final_internal_mcp_state.get("rhino_screenshot_counter", 0)

                        if img_path_from_internal_state and os.path.exists(img_path_from_internal_state):
                            path_for_entry = img_path_from_internal_state
                            filename_for_entry = os.path.basename(img_path_from_internal_state)
                            # The filename already includes the counter from mcp_test.py
                            current_file_description_parts.append(f"ScreenshotNumFromTool: {internal_counter_val}")
                            current_file_description_parts.append(f"FileName: {filename_for_entry}")
                            current_file_description_parts.append(f"PlannedName: {planned_name_from_def}")
                            print(f"    - Processing Screenshot (from internal state via output_def): {filename_for_entry}, Path: {path_for_entry}")
                            processed_image_from_state_this_step = True # Mark as processed
                        else:
                            # Fallback if saved_image_path from internal state is missing/invalid
                            mcp_errors.append(f"Rhino Agent planned image output '{planned_name_from_def}' but actual file from internal state ('{img_path_from_internal_state}') not found.")
                            print(f"    - Warning: Planned image output '{planned_name_from_def}' but file from internal state ('{img_path_from_internal_state}') not found.")
                            # Optionally, try to use planned_file_output_path if it exists, but it's unlikely for screenshots
                            planned_file_output_path = output_def.get("file_path")
                            if planned_file_output_path:
                                 abs_planned_path = os.path.join(OUTPUT_DIR, planned_file_output_path) # Adjust base dir as needed
                                 if os.path.exists(abs_planned_path):
                                     path_for_entry = abs_planned_path
                                     filename_for_entry = os.path.basename(abs_planned_path)
                                     current_file_description_parts.append(f"File: {filename_for_entry} (from planned path)")
                                 else:
                                     continue # Skip if no actual file found
                            else:
                                continue # Skip if no actual file found
                    else: # Non-image file definition
                        file_output_path = output_def.get("file_path")
                        if file_output_path:
                            abs_file_output_path = os.path.join(OUTPUT_DIR, file_output_path) # Adjust base dir
                            if not os.path.isabs(file_output_path):
                                if not os.path.exists(abs_file_output_path) and os.path.exists(os.path.join(RENDER_CACHE_DIR, file_output_path)):
                                    abs_file_output_path = os.path.join(RENDER_CACHE_DIR, file_output_path)
                                elif not os.path.exists(abs_file_output_path) and os.path.exists(os.path.join(MODEL_CACHE_DIR, file_output_path)):
                                    abs_file_output_path = os.path.join(MODEL_CACHE_DIR, file_output_path)
                            else:
                                abs_file_output_path = file_output_path
                            
                            if os.path.exists(abs_file_output_path):
                                path_for_entry = abs_file_output_path
                                filename_for_entry = os.path.basename(abs_file_output_path)
                                file_type_simple = output_def.get('type', 'File').capitalize()
                                current_file_description_parts.append(f"OutputType: {file_type_simple}_{planned_name_from_def}")
                                print(f"    - Processing Other File: {filename_for_entry}, Path: {path_for_entry}")
                            else:
                                mcp_errors.append(f"Rhino Agent planned output file '{file_output_path}' not found at '{abs_file_output_path}'.")
                                print(f"    - Warning: Planned output file '{file_output_path}' not found at '{abs_file_output_path}'.")
                                continue # Skip if no actual file found
                        else:
                            print(f"    - Info: Output definition {idx+1} did not specify a 'file_path'.")
                            continue # Skip if no path

                    if path_for_entry and filename_for_entry:
                        file_type = output_def.get("mime_type", "application/octet-stream")
                        add_base64 = output_def.get("encode_base64", False)
                        
                        structured_description = "; ".join(current_file_description_parts)
                        file_entry = {
                            "filename": filename_for_entry,
                            "path": path_for_entry,
                            "type": file_type,
                            "description": structured_description
                        }
                        if add_base64:
                            try:
                                with open(path_for_entry, "rb") as f_content:
                                    encoded_string = base64.b64encode(f_content.read()).decode('utf-8')
                                file_entry["base64_data"] = f"data:{file_type};base64,{encoded_string}"
                            except Exception as e_b64:
                                print(f"      - Warning: Could not read/encode Rhino output file {path_for_entry} for base64: {e_b64}")
                        output_files_list.append(file_entry)
            
            # --- ADDED: Check for saved_image_path directly from internal state if not processed via output_definitions ---
            # This handles cases where capture_focused_view ran, saved an image, but it wasn't in output_definitions
            # (e.g., if the AI message after the tool run didn't include it in output_definitions)
            if not processed_image_from_state_this_step:
                img_path_from_internal_state = final_internal_mcp_state.get("saved_image_path")
                internal_counter_val = final_internal_mcp_state.get("rhino_screenshot_counter", 0) # Get counter for description
                
                if img_path_from_internal_state and os.path.exists(img_path_from_internal_state):
                    # Check if this path was already added to output_files_list to prevent duplicates
                    # This is a simple check; a more robust one might involve comparing more than just path.
                    if not any(f.get("path") == img_path_from_internal_state for f in output_files_list):
                        print(f"  - Processing Screenshot (directly from internal state 'saved_image_path'): {img_path_from_internal_state}")
                        filename_for_entry = os.path.basename(img_path_from_internal_state)
                        
                        # Determine mime type
                        mime_type = "image/png" # Default
                        ext = os.path.splitext(filename_for_entry)[1].lower()
                        if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                        elif ext == ".gif": mime_type = "image/gif"
                        elif ext == ".webp": mime_type = "image/webp"

                        current_file_description_parts_direct = [
                            f"SourceAgent: {current_task.get('selected_agent', 'RhinoMCPAgent')}",
                            f"TaskDesc: {current_task.get('description', 'N/A')}",
                            f"ScreenshotNumFromTool: {internal_counter_val}",
                            f"FileName: {filename_for_entry}",
                            f"Origin: DirectFromInternalState" # Indicate source
                        ]
                        structured_description_direct = "; ".join(current_file_description_parts_direct)

                        file_entry_direct = {
                            "filename": filename_for_entry,
                            "path": img_path_from_internal_state,
                            "type": mime_type,
                            "description": structured_description_direct
                        }
                        # Add base64
                        try:
                            with open(img_path_from_internal_state, "rb") as f_content:
                                encoded_string = base64.b64encode(f_content.read()).decode('utf-8')
                            file_entry_direct["base64_data"] = f"data:{mime_type};base64,{encoded_string}"
                        except Exception as e_b64:
                            print(f"      - Warning: Could not read/encode Rhino output file {img_path_from_internal_state} for base64: {e_b64}")
                        
                        output_files_list.append(file_entry_direct)
                        print(f"    - Added {filename_for_entry} to output_files_list.")
                    else:
                        print(f"  - Info: Screenshot '{img_path_from_internal_state}' from internal state already processed, skipping direct add.")
            # --- END ADDED ---

            # --- MODIFIED Loop Termination Condition ---
            if final_internal_mcp_state.get("task_complete") or \
               (isinstance(final_internal_mcp_state.get("messages", [])[-1] if final_internal_mcp_state.get("messages") else None, AIMessage) and \
                "[fallback_confirmed_completion]" in str(final_internal_mcp_state.get("messages", [])[-1].content).lower()):
                print(f"  Rhino MCP internal graph indicated task completion (via task_complete flag or fallback_confirmed_completion).")
                break
            # --- END MODIFIED Loop Termination Condition ---
            
            # Check for max steps for the *outer* loop controlling the internal graph
            if len(local_mcp_state["messages"]) // 2 > max_mcp_steps : # Approximation of steps
                 print(f"  {node_name} Warning: Reached max MCP orchestrating steps ({max_mcp_steps}). Ending loop.")
                 mcp_errors.append("MCP task exceeded maximum orchestration steps.")
                 break
        
        # After loop, populate mcp_final_outcome_for_task_update
        if final_internal_mcp_state:
            mcp_final_outcome_for_task_update = {
                "mcp_message_history": final_internal_mcp_state.get("messages", []),
                # saved_image_path and data_uri are for the *overall task output if a primary one exists*
                # For Rhino, multiple files are handled via output_files_list
            }

            # --- ADDED: Check for saved_csv_path from internal state ---
            # This ensures that if the planning summary CSV was created, it's captured.
            saved_csv_path = final_internal_mcp_state.get("saved_csv_path")
            if saved_csv_path and os.path.exists(saved_csv_path):
                if not any(f.get("path") == saved_csv_path for f in output_files_list):
                    print(f"  - Processing CSV Report (from internal state 'saved_csv_path'): {saved_csv_path}")
                    csv_filename = os.path.basename(saved_csv_path)
                    csv_description_parts = [
                        f"SourceAgent: {current_task.get('selected_agent', 'RhinoMCPAgent')}",
                        f"TaskDesc: {current_task.get('description', 'N/A')}",
                        f"FileName: {csv_filename}",
                        f"Origin: DirectFromInternalState_CSVReport"
                    ]
                    structured_csv_description = "; ".join(csv_description_parts)

                    csv_file_entry = {
                        "filename": csv_filename,
                        "path": saved_csv_path,
                        "type": "text/csv",
                        "description": structured_csv_description
                    }
                    output_files_list.append(csv_file_entry)
                    print(f"    - Added {csv_filename} to output_files_list.")
                else:
                    print(f"  - Info: CSV Report '{saved_csv_path}' from internal state already processed, skipping direct add.")
            # --- END ADDED ---

            if mcp_errors:
                 mcp_final_outcome_for_task_update["mcp_execution_errors"] = mcp_errors
        else:
            mcp_errors.append("Rhino MCP internal graph did not yield a final state.")

    except ImportError:
        err_msg = "Failed to import Rhino MCP graph instance. Ensure mcp_test.py is correctly structured."
        print(f"{node_name} Error: {err_msg}")
        traceback.print_exc()
        outer_error_to_report = ImportError(err_msg)
    except Exception as mcp_e:
        err_msg = f"Error during Rhino MCP graph execution: {mcp_e}"
        print(f"{node_name} Error: {err_msg}")
        traceback.print_exc()
        outer_error_to_report = mcp_e
        if not mcp_final_outcome_for_task_update.get("mcp_message_history"): # Ensure history is included
            mcp_final_outcome_for_task_update["mcp_message_history"] = local_mcp_state["messages"] # Fallback to outer history

    # --- Consolidate MCP errors into outer_error_to_report if not already set ---
    if mcp_errors and not outer_error_to_report:
        outer_error_to_report = ValueError("Rhino MCP processing encountered errors: " + "; ".join(mcp_errors))
    elif mcp_errors and outer_error_to_report: # Append if outer error already exists
        outer_error_to_report = type(outer_error_to_report)(str(outer_error_to_report) + "; Rhino MCP internal errors: " + "; ".join(mcp_errors))


    # Update the main WorkflowState Task using the helper
    tasks[current_idx] = _update_task_state_after_tool(
        current_task,
        outputs={}, # Rhino outputs are primarily files
        output_files=output_files_list,
        error=outer_error_to_report,
        mcp_result=mcp_final_outcome_for_task_update
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

