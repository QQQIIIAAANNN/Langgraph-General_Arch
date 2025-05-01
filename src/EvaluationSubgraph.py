# =============================================================================
# 1. Imports
# =============================================================================
import os
import uuid
import json
import base64
from dotenv import load_dotenv
from typing import Dict, List, Any, Literal, Union, Optional, Tuple
import traceback

# LangChain/LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# --- Tool Imports ---
from src.tools.ARCH_rag_tool import ARCH_rag_tool
from src.tools.img_recognition import img_recognition
from src.tools.video_recognition import video_recognition
from src.tools.gemini_search_tool import perform_grounded_search

# --- Configuration & LLM Initialization ---
from src.configuration import ConfigManager, ModelConfig, MemoryConfig, initialize_llm, ConfigSchema

# --- <<< 修改：從 state.py 和 configuration.py 導入 >>> ---
from src.state import WorkflowState, TaskState # 從 state.py 導入狀態

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
# =============================================================================
# <<< NEW SECTION: Evaluation Subgraph Definition (Refactored) >>>
# =============================================================================
# --- Uses imported WorkflowState, TaskState, config_manager, initialize_llm, constants ---

# Helper function to remove base64 data from file list
def _filter_base64_from_files(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Creates a new list of file dictionaries without the 'base64_data' key."""
    if not files:
        return []
    filtered_files = []
    for file_info in files:
        # Create a copy to avoid modifying the original state
        filtered_info = file_info.copy()
        filtered_info.pop('base64_data', None) # Remove the key if it exists
        filtered_files.append(filtered_info)
    return filtered_files

async def prepare_evaluation_inputs_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Prepares inputs for the evaluation agent (EvaAgent)."""
    node_name = "Prepare Eval Inputs"
    tasks = state['tasks']
    current_idx = state['current_task_index']
    current_task = tasks[current_idx] # This is the EvaAgent task itself
    print(f"--- Running Node: {node_name} for Task {current_idx}: {current_task.get('description')} ---")

    # Load necessary configurations
    runtime_config = config.get("configurable", {})
    # Use the global config_manager for static defaults
    llm_output_language = runtime_config.get("global_llm_output_language", config_manager.get_workflow_config().llm_output_language)
    # --- MODIFIED: Call _get_llm_config_for_node correctly ---
    # _get_llm_config_for_node is not defined, directly get config
    # Assuming we use the 'eva_agent' llm config defined in runtime_config or static config
    llm_config_dict_runtime = runtime_config.get("ea_llm", {})
    if not llm_config_dict_runtime: # Fallback to static config if not in runtime
        static_agent_config = config_manager.get_agent_config("eva_agent")
        if static_agent_config and static_agent_config.llm:
            llm_config_dict_static = static_agent_config.llm.model_dump()
            print(f"Prepare Eval Inputs: Using static LLM config for eva_agent.")
            llm = initialize_llm(llm_config_dict_static)
        else:
            print("Prepare Eval Inputs: WARNING - No LLM config found for eva_agent in runtime or static. Using default.")
            llm = initialize_llm({}) # Use default
    else:
        print(f"Prepare Eval Inputs: Using runtime LLM config for eva_agent.")
        llm = initialize_llm(llm_config_dict_runtime)
    # --- END MODIFIED ---

    # Determine if this is the final evaluation task
    is_final_eval = current_task.get('task_objective') == 'final_evaluation'
    print(f"Eval Subgraph ({node_name}): Final Evaluation Mode = {is_final_eval}")

    # --- Prepare Prompt Inputs ---
    prompt_inputs = {}
    prompt_template_name = ""

    if is_final_eval:
        prompt_template_name = "prepare_final_evaluation_inputs"
        # Aggregate outputs and files from ALL completed tasks
        aggregated_outputs = {}
        aggregated_files_raw = []
        full_task_summary_parts = ["Workflow Summary:"]
        for i, task in enumerate(tasks):
             # Only include completed tasks BEFORE the final eval task
             if i < current_idx and task.get("status") == "completed":
                 task_id = task.get("task_id", f"task_{i}")
                 full_task_summary_parts.append(f"  Task {i} (ID: {task_id}): {task.get('description', 'N/A')}")
                 task_outputs = task.get("outputs")
                 if task_outputs:
                     aggregated_outputs[task_id] = task_outputs
                     full_task_summary_parts.append(f"    Outputs: {json.dumps(task_outputs, ensure_ascii=False, indent=2)}") # Indent for summary clarity
                 task_files = task.get("output_files")
                 if task_files:
                     aggregated_files_raw.extend(task_files) # Collect raw files
                     full_task_summary_parts.append(f"    Files: {[f.get('filename', 'N/A') for f in task_files]}")

        # Filter base64 BEFORE serializing aggregated files
        filtered_aggregated_files = _filter_base64_from_files(aggregated_files_raw)

        prompt_inputs = {
            "user_input": state.get("user_input", "N/A"),
            "full_task_summary": "\n".join(full_task_summary_parts),
            "aggregated_outputs_json": json.dumps(aggregated_outputs, ensure_ascii=False),
            "aggregated_files_json": json.dumps(filtered_aggregated_files, ensure_ascii=False), # Use filtered list
            "ltm_context": "N/A", # LTM is not used for final eval prep source
            "llm_output_language": llm_output_language,
        }

    else: # Regular evaluation
        prompt_template_name = "prepare_evaluation_inputs"
        # Target the PREVIOUS task for evaluation
        target_task_idx = current_idx - 1
        if target_task_idx < 0:
            err_msg = "Cannot perform regular evaluation, no previous task exists."
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks}

        target_task = tasks[target_task_idx]
        print(f"Eval Subgraph ({node_name}): Preparing inputs to evaluate Task {target_task_idx}: {target_task.get('description')}")

        # Filter base64 BEFORE serializing target task files
        target_task_files_raw = target_task.get("output_files", [])
        filtered_target_files = _filter_base64_from_files(target_task_files_raw)

        prompt_inputs = {
            "task_description": target_task.get("description", "N/A"),
            "task_objective": target_task.get("task_objective", "N/A"),
            "evaluated_task_outputs_json": json.dumps(target_task.get("outputs", {}), ensure_ascii=False),
            "evaluated_task_output_files_json": json.dumps(filtered_target_files, ensure_ascii=False), # Use filtered list
            "llm_output_language": llm_output_language,
        }

    # --- Load Prompt Template ---
    # Use the global config_manager instance
    prompt_template = runtime_config.get(f"ea_{prompt_template_name}_prompt") or \
                      config_manager.get_prompt_template("eva_agent", prompt_template_name)

    # --- Logging for template loading ---
    print(f"Eval Subgraph ({node_name}): Attempting to load prompt template '{prompt_template_name}'...")
    print(f"  - Runtime key 'ea_{prompt_template_name}_prompt' value: {runtime_config.get(f'ea_{prompt_template_name}_prompt') is not None}")
    # Use the global config_manager instance
    print(f"  - Static key '{prompt_template_name}' value: {config_manager.get_prompt_template('eva_agent', prompt_template_name) is not None}")
    # --- End logging ---

    if not prompt_template:
        err_msg = f"Missing required prompt template '{prompt_template_name}' for evaluation input preparation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    # --- Format Prompt & Invoke LLM ---
    try:
        print(f"Eval Subgraph ({node_name}): Formatting prompt with inputs: {list(prompt_inputs.keys())}")
        # <<<<<<<<<<<<<<<< DEBUGGING ADDED PREVIOUSLY >>>>>>>>>>>>>>>>
        print(f"--- DEBUG: Prompt Template BEFORE .format() ---")
        print(repr(prompt_template)) # Use repr() to see hidden characters like \n
        print(f"--- DEBUG: Prompt Inputs Dictionary (showing truncated strings) ---")
        # Print inputs carefully, avoid overly long outputs
        print({k: (v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) for k, v in prompt_inputs.items()})
        print(f"---------------------------------------------")
        # <<<<<<<<<<<<<<<< END DEBUGGING >>>>>>>>>>>>>>>>

        prep_prompt = prompt_template.format(**prompt_inputs) # This is the line causing the error

        print(f"Eval Subgraph ({node_name}): Invoking LLM ({'Final' if is_final_eval else 'Regular'})...")
        prep_response = await llm.ainvoke(prep_prompt)
        prep_content = prep_response.content.strip()
        # ... (json cleaning logic) ...
        if prep_content.startswith("```json"): prep_content = prep_content[7:-3].strip()
        elif prep_content.startswith("```"): prep_content = prep_content[3:-3].strip()

        # --- Parse and Update Task Inputs ---
        try:
            prepared_eval_inputs = json.loads(prep_content)
            if isinstance(prepared_eval_inputs, dict) and "error" in prepared_eval_inputs:
                llm_error_msg = prepared_eval_inputs['error']
                err_msg = f"LLM indicated error during eval input prep: {llm_error_msg}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                if "Missing expected output" in llm_error_msg:
                     err_msg += f". This might indicate an issue parsing the provided JSONs or the LLM misinterpreting the objective relative to the parsed content."
                _set_task_failed(current_task, err_msg, node_name)
            elif not isinstance(prepared_eval_inputs, dict):
                err_msg = f"LLM returned invalid format (expected dict). Content: {prep_content}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                _set_task_failed(current_task, err_msg, node_name)
            else:
                print(f"Eval Subgraph ({node_name}): Evaluation inputs prepared: {list(prepared_eval_inputs.keys())}") # Log keys instead of full dict
                current_task["task_inputs"] = prepared_eval_inputs
                current_task["error_log"] = None

        except json.JSONDecodeError:
            err_msg = f"Could not parse LLM JSON response. Raw content: '{prep_content}'"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)

    except IndexError as ie:
        # Catch the specific error
        err_msg = f"String formatting error (IndexError): {ie}. This likely means the prompt template '{prompt_template_name}' contains an unexpected positional placeholder like {{}} or {{0}}."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        # Print template again for context
        print(f"--- Problematic Prompt Template ({prompt_template_name}) ---")
        print(repr(prompt_template))
        print(f"---------------------------------")
        import traceback; traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)
    except KeyError as ke:
        err_msg = f"Unexpected error during LLM call or processing: Formatting failed, missing key {ke}. Check prompt template '{prompt_template_name}' and input keys."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        import traceback; traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)
    except Exception as prep_e:
        err_msg = f"Unexpected error during LLM call or processing: {prep_e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        import traceback; traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)

    # --- Return Updated State ---
    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}

async def gather_criteria_sources_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Concurrently fetches context from RAG/Search for criteria generation and updates TaskState."""
    print("--- Eval Subgraph (TaskState): Gathering Criteria Sources ---")
    # ... (rest of the function remains largely the same, ensure it reads from current_task["task_inputs"] if needed) ...
    # It primarily uses task_description and overall_goal, which are fine.
    # It updates current_task["evaluation"]["criteria_sources"]
    current_idx = state["current_task_index"]
    tasks = [t.copy() for t in state["tasks"]]
    current_task = tasks[current_idx]

    # Check if previous step failed
    if current_task.get("status") == "failed":
         print("  - Skipping criteria gathering, previous step failed.")
         # --- MODIFIED: Return full state on early exit ---
         return {"tasks": tasks, "current_task": current_task.copy()}

    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") # Carry over errors
    overall_goal = state.get('user_input', "N/A")
    # Read description from prepared inputs if available, fallback to task's description
    task_desc = current_task.get("task_inputs", {}).get("evaluation_target_description", current_task.get("description", ""))

    runtime_config = config["configurable"]
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)
    retriever_k = runtime_config.get("retriever_k", 5)

    # RAG Query based on task description
    rag_query = f"Find evaluation standards relevant to: {task_desc}" # Simplified query
    rag_context = "RAG context not retrieved."
    try:
        print(f"Eval Subgraph: Calling RAG with k={retriever_k} for query: {rag_query}")
        # Assume async invoke exists
        rag_result = await ARCH_rag_tool.ainvoke({"query": rag_query, "top_k": retriever_k})
        if rag_result and not (isinstance(rag_result, str) and rag_result.startswith("Error")):
            rag_context = f"Retrieved from Knowledge Base:\n{str(rag_result)}\n"
        elif isinstance(rag_result, str) and rag_result.startswith("Error"):
            rag_context = f"RAG Error: {rag_result}"
            if not subgraph_error: subgraph_error = rag_context # Log RAG error
        elif not rag_result:
             rag_context = "No relevant documents found by RAG."
    except Exception as e:
        print(f"Eval Subgraph: RAG call failed: {e}")
        rag_context = f"RAG call error: {e}"
        if not subgraph_error: subgraph_error = f"RAG Error: {e}"

    # Web Search (Optional, keep logic)
    search_query = f"Search for evaluation methods and standards for: {task_desc}"
    search_context = "Web search not performed or failed."
    try:
        print(f"Eval Subgraph: Calling Web Search for query: {search_query}")
        # 改用正確的參數名稱，這裡注意 perform_grounded_search 使用的參數格式
        search_result = perform_grounded_search({"query": search_query})
        if isinstance(search_result, dict) and search_result.get("text_content"):
            search_context = f"Web Search Results:\nText: {search_result['text_content']}\n"
            if search_result.get("grounding_sources"):
                sources = [f"- {src.get('title', 'N/A')}: {src.get('web_uri', 'N/A')}" 
                          for src in search_result['grounding_sources']]
                search_context += "Sources:\n" + "\n".join(sources) + "\n"
        elif isinstance(search_result, dict) and search_result.get("error"):
            search_context = f"Web search error: {search_result['error']}"
            if not subgraph_error:
                subgraph_error = f"Web Search Error: {search_result['error']}"
    except Exception as e:
        print(f"Eval Subgraph: Web Search call failed: {e}")
        search_context = f"Web Search call error: {e}"
        if not subgraph_error:
            subgraph_error = f"Web Search Error: {e}"

    # Update evaluation dictionary
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    current_task["evaluation"]["criteria_sources"] = {"rag": rag_context, "search": search_context}
    if subgraph_error:
        current_task["evaluation"]["subgraph_error"] = subgraph_error
        print(f"  - Error during criteria source gathering: {subgraph_error}")

    return {"tasks": tasks, "current_task": current_task.copy()}

def generate_specific_criteria_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Generates specific evaluation criteria and updates TaskState."""
    print("--- Eval Subgraph (TaskState): Generating Specific Criteria ---")
    node_name = "Generate Criteria"
    current_idx = state["current_task_index"]
    tasks = [t.copy() for t in state["tasks"]]
    current_task = tasks[current_idx]

    if current_task.get("status") == "failed":
        print("  - Skipping criteria generation, previous step failed.")
        # --- MODIFIED: Return full state on early exit ---
        return {"tasks": tasks, "current_task": current_task.copy()}

    if "evaluation" not in current_task: current_task["evaluation"] = {}
    is_final_eval = current_task.get("evaluation", {}).get("is_final_evaluation", False)
    prepared_inputs = current_task.get("task_inputs", {}) # Inputs from previous node
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")

    print(f"--- Generating Criteria (Final Eval: {is_final_eval}) ---")

    runtime_config = config["configurable"]
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)

    prompt_template = None
    prompt_inputs = {}
    generated_criteria = "Error: Could not generate criteria."

    if is_final_eval:
        prompt_template = runtime_config.get("ea_generate_final_criteria_prompt") or \
                          config_manager.get_prompt_template("eva_agent", "generate_final_criteria")
        prompt_inputs = {
            "user_input": state.get("user_input", "N/A"),
            "full_task_summary": prepared_inputs.get("evaluation_target_full_summary", "Workflow summary not available."),
            "final_eval_inputs_json": json.dumps(prepared_inputs, ensure_ascii=False, indent=2), # Pass all prepared inputs
            "llm_output_language": llm_output_language
        }
    else:
        prompt_template = runtime_config.get("ea_generate_criteria_prompt") or \
                          config_manager.get_prompt_template("eva_agent", "generate_criteria")
        criteria_sources = current_task.get("evaluation", {}).get("criteria_sources", {})
        prompt_inputs = {
            "task_description": prepared_inputs.get("evaluation_target_description", "N/A"),
            "rag_context": criteria_sources.get("rag", "Not available."),
            "search_context": criteria_sources.get("search", "Not available."),
            "llm_output_language": llm_output_language
        }

    if not prompt_template:
        err_msg = f"Missing required prompt template for {'final' if is_final_eval else 'regular'} criteria generation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
        # Don't fail task, allow eval nodes to use default criteria if needed
    else:
        try:
            prompt = prompt_template.format(**prompt_inputs)
            print(f"Eval Subgraph ({node_name}): Invoking LLM for criteria ({'Final' if is_final_eval else 'Regular'})...")
            response = llm.invoke(prompt)
            generated_criteria = response.content.strip()
            print(f"Eval Subgraph ({node_name}): Generated Criteria:\n{generated_criteria}")
        except Exception as e:
            err_msg = f"Criteria generation LLM error: {e}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
            generated_criteria = f"Error during generation: {e}" # Store error

    # Update evaluation dictionary
    current_task["evaluation"]["specific_criteria"] = generated_criteria
    if subgraph_error:
        current_task["evaluation"]["subgraph_error"] = subgraph_error

    # Update the task in the list
    tasks[current_idx] = current_task
    # --- MODIFIED Return ---
    return {"tasks": tasks, "current_task": current_task.copy()}

# --- MODIFIED Routing Function (Checks for final eval flag first) ---
def route_to_evaluation_tool_node(state: WorkflowState) -> str:
    """
    Routes to the correct first evaluation tool node.
    Checks if it's a final evaluation task first, otherwise routes based on outputs
    prepared in task_inputs.
    """
    current_idx = state["current_task_index"]
    tasks = state["tasks"]
    if not (0 <= current_idx < len(tasks)):
        print("Eval Subgraph Router Error: Invalid task index.")
        return "evaluate_with_llm_agent"

    current_task = tasks[current_idx]
    # Check status first - if prep/criteria failed, maybe go straight to end/fail?
    # For now, let the eval nodes handle potential missing inputs.
    if current_task.get("status") == "failed":
        print("  - Routing Decision: Task failed in previous step, routing to END.")
        return END # Route directly to end if prep/criteria failed

    is_final_eval = current_task.get("evaluation", {}).get("is_final_evaluation", False)

    if is_final_eval:
        print(f"--- Eval Subgraph Router: Task {current_task.get('task_id')} is FINAL evaluation. Routing to LLM Evaluation. ---")
        return "evaluate_with_llm_agent"
    else:
        # --- Regular Intermediate Evaluation Routing ---
        print(f"--- Eval Subgraph Router: Task {current_task.get('task_id')} is REGULAR evaluation. Routing based on prepared inputs. ---")
        prepared_inputs = current_task.get("task_inputs", {})
        # Check the prepared inputs for file paths
        has_image_output = bool(prepared_inputs.get("evaluation_target_image_paths"))
        has_video_output = bool(prepared_inputs.get("evaluation_target_video_paths"))
        # Assume text output exists if structured outputs were present
        has_text_output = bool(prepared_inputs.get("evaluation_target_outputs_json") != '{}')

        print(f"  - Prepared Inputs Check:")
        print(f"    - Has Text Output: {has_text_output}")
        print(f"    - Has Image Output: {has_image_output}")
        print(f"    - Has Video Output: {has_video_output}")

        # --- Routing Logic based on combinations ---
        if has_image_output and has_video_output:
            print("  - Routing Decision: Image & Video found -> Start with Image Evaluation")
            return "evaluate_with_image_agent"
        elif has_image_output: # Handles (Image only) and (Image + Text)
            print("  - Routing Decision: Image found -> Route to Image Evaluation")
            return "evaluate_with_image_agent"
        elif has_video_output: # Handles (Video only) and (Video + Text)
            print("  - Routing Decision: Video found -> Route to Video Evaluation")
            return "evaluate_with_video_agent"
        elif has_text_output: # Only text
            print("  - Routing Decision: Only Text found -> Route to LLM Evaluation")
            return "evaluate_with_llm_agent"
        else:
            # If no specific outputs found in prepared inputs, default to LLM
            print("  - Routing Decision: No specific outputs found in prepared inputs -> Defaulting to LLM Evaluation")
            return "evaluate_with_llm_agent"

# --- MODIFIED Evaluation Tool Nodes ---
# They now read inputs from task_inputs prepared by prepare_evaluation_inputs_node
# Logic for setting assessment vs status needs refinement.

# --- NEW Helper Function: Set Task Failed ---
def _set_task_failed(task: TaskState, error_message: str, node_name: str):
    """Sets task status to 'failed' and logs the error message."""
    print(f"--- Task Failure in Node '{node_name}' ---")
    print(f"Error: {error_message}")
    task["status"] = "failed"
    # Directly set error_log, avoiding potential None + str issues
    task["error_log"] = f"[{node_name}] {error_message}"
    # Clear outputs if any were partially populated before failure
    task["outputs"] = {}
    task["output_files"] = []
    # Optionally append to feedback log as well or clear it
    # task["feedback_log"] = (task.get("feedback_log") or "") + f"; Error in {node_name}" # Example

# --- Helper to append feedback consistently ---
def _append_feedback(task: TaskState, feedback: str, node_name: str):
     """Appends feedback to the task's feedback_log."""
     # Ensure current_log is always a string, handling None explicitly
     current_log = task.get("feedback_log") or ""
     # Avoid adding duplicate prefixes if called multiple times by same node logic path
     prefix = f"[{node_name} Feedback]:"
     # Now it's safe to call split on current_log because it's guaranteed to be a string
     if prefix not in current_log.split('\n')[-2:]: # Check last few lines
         task["feedback_log"] = (current_log + f"\n{prefix}\n{feedback}").strip()
     else: # Append to existing feedback from this node
         task["feedback_log"] = (current_log + f"\n{feedback}").strip()

def _update_eval_status_at_end(task: TaskState, node_name: str):
    """Sets final status based on assessment ONLY if this is the end of the eval path."""
    assessment = task.get("evaluation", {}).get("assessment", "Fail") # Default to Fail
    # If status is already failed (due to internal error), don't override
    if task.get("status") == "failed":
        print(f"  - [{node_name}] Task already failed internally, skipping final status update.")
        return

    if assessment == "Pass":
        task["status"] = "completed"
        print(f"  - [{node_name}] Assessment is Pass. Setting final status to COMPLETED.")
    else: # Assessment is Fail
        task["status"] = "failed"
        print(f"  - [{node_name}] Assessment is Fail. Setting final status to FAILED.")
        # Log the logical failure reason if not already logged as internal error
        if f"Assessment: {assessment}" not in task.get("feedback_log", ""):
             _append_feedback(task, f"Evaluation resulted in '{assessment}'.", node_name)

def evaluate_with_llm_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Performs evaluation using LLM and updates TaskState."""
    print("--- Eval Subgraph (TaskState): Evaluating with LLM ---")
    node_name = "LLM Evaluation"
    current_idx = state["current_task_index"]
    tasks = [t.copy() for t in state["tasks"]]
    current_task = tasks[current_idx]

    if current_task.get("status") == "failed":
        print("  - Skipping LLM evaluation, previous step failed.")
        return {"tasks": tasks}

    # --- Read Prepared Inputs ---
    prepared_inputs = current_task.get("task_inputs", {})
    task_description = prepared_inputs.get("evaluation_target_description", "N/A") # Includes "Final Workflow Review" if final
    task_objective = prepared_inputs.get("evaluation_target_objective", "N/A")   # Includes "final_evaluation" if final
    task_outputs_json = prepared_inputs.get("evaluation_target_outputs_json", "{}") # Regular eval outputs
    # Use the specific keys prepared for final eval artifacts if present
    image_paths = prepared_inputs.get("evaluation_target_key_image_paths", [])
    video_paths = prepared_inputs.get("evaluation_target_key_video_paths", [])
    other_files_summary = prepared_inputs.get("evaluation_target_other_artifacts_summary", "None")
    full_task_summary = prepared_inputs.get("evaluation_target_full_summary", "N/A") # Available if final
    ltm_context = prepared_inputs.get("evaluation_target_ltm_context", "N/A")       # Available if final

    # Format file info strings for the prompt
    image_paths_str = ", ".join(image_paths) if image_paths else "None"
    video_paths_str = ", ".join(video_paths) if video_paths else "None"
    # Note: other_files_str might need adjustment if final eval prep changes format
    other_files_str = other_files_summary # Use the summary prepared for final eval

    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply.") # Will be final criteria if final eval
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")

    # REMOVED: is_final_eval check here

    # --- Initialize LLM, Get Template ---
    runtime_config = config["configurable"]
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    evaluation_template = runtime_config.get("ea_evaluation_prompt") or \
                           config_manager.get_prompt_template("eva_agent", "evaluation")

    assessment = "Fail"; feedback = "Evaluation failed to run."; suggestions = "N/A"

    if not evaluation_template:
        # ... (handle missing template error - unchanged) ...
        err_msg = "Missing 'evaluation' prompt template."
        print(f"Eval Subgraph Error: {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks}

    # --- Format Prompt ---
    # The prompt template itself handles the difference based on description/objective and provided context
    prompt = evaluation_template.format(
        evaluation_target_description=task_description,
        evaluation_target_objective=task_objective,
        evaluation_target_outputs_json=task_outputs_json, # For regular eval part of prompt
        evaluation_target_image_paths_str=image_paths_str, # For regular eval part of prompt
        evaluation_target_video_paths_str=video_paths_str, # For regular eval part of prompt
        evaluation_target_other_files_str=other_files_str, # For regular eval part of prompt
        specific_criteria=specific_criteria,
        llm_output_language=llm_output_language,
        # Context for final eval part of prompt
        full_task_summary=full_task_summary,
        ltm_context=ltm_context
    )

    # --- Invoke LLM & Parse Result ---
    try:
        print(f"Eval Subgraph ({node_name}): Invoking LLM...")
        response = llm.invoke(prompt)
        # --- >>> ADDED: Extract content from response <<< ---
        content = response.content.strip() if response and hasattr(response, 'content') else ""
        if not content:
            # Handle empty response content if necessary
            print(f"Eval Subgraph Warning ({node_name}): LLM returned empty content.")
            # Decide if this should be an error or proceed with default 'Fail' assessment
            # For now, let it fall through to the parsing attempt, which might fail or result in Fail assessment
        # --- >>> END ADDED <<< ---

        # --- Parse the extracted content ---
        print(f"Eval Subgraph ({node_name}): Raw LLM response content:\n{content[:500]}...") # Log raw content

        # Clean JSON content (optional but recommended)
        if content.startswith("```json"): content = content[7:-3].strip()
        elif content.startswith("```"): content = content[3:-3].strip()

        try: # Nested try for JSON parsing
            parsed_json = json.loads(content) # Now uses the defined 'content' variable
            if isinstance(parsed_json, dict):
                assessment_val = parsed_json.get("assessment")
                # ... (rest of the parsing logic for assessment_val - unchanged) ...
                if isinstance(assessment_val, str) and assessment_val.lower() in ["pass", "fail"]:
                    assessment = "Pass" if assessment_val.lower() == "pass" else "Fail"
                elif isinstance(assessment_val, (str, int, float)):
                    try:
                        score_str = str(assessment_val).split("(")[0].strip()
                        score = float(score_str)
                        assessment = "Pass" if score >= 7 else "Fail" # Example scoring threshold
                        print(f"   - Final Eval Score Parsed: {score} -> Assessment: {assessment}")
                    except ValueError:
                        print(f"   - Warning: Could not parse final eval score from '{assessment_val}'. Defaulting to Fail.")
                        assessment = "Fail"
                else:
                    assessment = "Fail"

                feedback = parsed_json.get("feedback", "No feedback.")
                suggestions = parsed_json.get("improvement_suggestions", "No suggestions.")
            else:
                 # Handle case where LLM returns valid JSON but not a dictionary
                 print(f"Eval Subgraph Warning ({node_name}): LLM returned valid JSON, but not a dictionary: {type(parsed_json)}")
                 assessment = "Fail" # Default to fail if format is wrong
                 feedback = f"Unexpected JSON format received: {type(parsed_json)}"
                 suggestions = "Check LLM output format compliance."

        except json.JSONDecodeError as json_e:
            # Handle JSON parsing error specifically
            print(f"Eval Subgraph Error ({node_name}): Failed to parse LLM JSON response: {json_e}")
            print(f"Problematic Content:\n{content}") # Print the content that failed parsing
            assessment = "Fail" # Default to fail if parsing fails
            feedback = f"LLM output was not valid JSON. Raw output:\n{content}"
            suggestions = "Review LLM output or prompt instructions for JSON format."
            # Optionally log this as a subgraph error as well
            subgraph_error = (subgraph_error + f"; LLM response JSON parse error: {json_e}").strip("; ")


    except Exception as e: # Catch errors from llm.invoke() itself
        err_msg = f"LLM evaluation call error: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks} # Return early on LLM call failure

    # --- Update TaskState ---
    current_task["evaluation"]["assessment"] = assessment
    feedback_details = f"Assessment: {assessment}\nFeedback: {feedback}\nSuggestions: {suggestions}"
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Set Final Status ---
    _update_eval_status_at_end(current_task, node_name)

    return {"tasks": tasks}

def evaluate_with_image_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Performs evaluation using Image Recognition tool and updates TaskState."""
    print("--- Eval Subgraph (TaskState): Evaluating with Image Recognition ---")
    node_name = "Image Evaluation"
    current_idx = state["current_task_index"]
    tasks = [t.copy() for t in state["tasks"]]
    current_task = tasks[current_idx]

    if current_task.get("status") == "failed": # Check if previous step failed
        print("  - Skipping Image evaluation, previous step failed.")
        return {"tasks": tasks}

    # --- Read Prepared Inputs ---
    prepared_inputs = current_task.get("task_inputs", {})
    image_paths = prepared_inputs.get("evaluation_target_image_paths", []) # Use prepared key
    image_related_text = prepared_inputs.get("evaluation_target_outputs_json", "{}")
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply.")
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")

    # REMOVED: is_final_eval check

    assessment = "Fail"; feedback = "Evaluation failed to run."; suggestions = "N/A"

    if not image_paths:
        err_msg = "Image evaluation required but no valid image paths found in prepared inputs."
        # ... (handle error, _set_task_failed, update subgraph_error) ...
        _set_task_failed(current_task, err_msg, node_name)
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks}
    else:
        # --- Prepare Prompt (No final eval context needed here) ---
        eval_prompt_for_img_tool = f"""請根據以下具體標準，評估提供的圖片和相關文字輸出：
**評估標準:**
{specific_criteria} # Criteria generated by previous node (could be final or regular)

**相關文字輸出 (JSON):**
```json
{image_related_text}
```

**圖片路徑:** {', '.join(image_paths)}

**任務:**
請提供詳細的評估，說明圖片和文字輸出是否符合上述標準。
你的評估應包含：
1.  **整體評估:** 在回應的 **最開頭** 明確指出 "整體評估：通過" 或 "整體評估：失敗"。
2.  **詳細回饋:** 針對每個標準進行分析，解釋評估結果的原因。
3.  **改進建議:** 如果評估為 "失敗"，請提供具體的改進建議。

Respond in {LLM_OUTPUT_LANGUAGE_DEFAULT}.
"""
    # --- Invoke Image Tool & Parse Result ---
    try:
        print(f"Eval Subgraph ({node_name}): Calling Image Recognition tool...")
        result = img_recognition.run({
            "image_paths": image_paths,
            "prompt": eval_prompt_for_img_tool
        })

        if isinstance(result, str):
            # Parse assessment (prioritize explicit marker)
            if result.strip().startswith("整體評估：通過") or "overall assessment: pass" in result.lower(): assessment = "Pass"
            elif result.strip().startswith("整體評估：失敗") or "overall assessment: fail" in result.lower(): assessment = "Fail"
            else: assessment = "Pass" if "pass" in result.lower() else "Fail"; print(f"Warning: Inferring image eval assessment: {assessment}")
            feedback = result
            # Extract suggestions
            if "建議：" in result: suggestions = result.split("建議：", 1)[1].strip()
            elif "suggestions:" in result.lower(): suggestions = result.split("suggestions:", 1)[1].strip()
            print(f"Eval Subgraph ({node_name}): Assessment: {assessment}")
        else: # Tool returned unexpected type
            err_msg = f"Image tool returned unexpected type: {type(result)}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            assessment = "Fail"; feedback = err_msg; suggestions = "Tool output format error."
            subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")

    except Exception as e: # Tool call error
        err_msg = f"Error calling image tool: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks}

    # --- Update TaskState (Assessment and Feedback) ---
    current_task["evaluation"]["assessment"] = assessment # Store assessment
    feedback_details = f"Assessment: {assessment}\nTool Raw Output:\n{feedback}\nSuggestions: {suggestions}"
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Status Update (Provisional based on assessment) ---
    # Let routing decide the final status based on whether more steps follow.
    if assessment == "Fail":
        if current_task.get("status") != "failed":
             print(f"  - [{node_name}] Assessment is Fail. Setting provisional status to FAILED.")
             current_task["status"] = "failed"

    return {"tasks": tasks}

def evaluate_with_video_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """Performs evaluation using Video Recognition tool and updates TaskState."""
    print("--- Eval Subgraph (TaskState): Evaluating with Video Recognition ---")
    node_name = "Video Evaluation"
    current_idx = state["current_task_index"]
    tasks = [t.copy() for t in state["tasks"]]
    current_task = tasks[current_idx]

    if current_task.get("status") == "failed": # Check if previous step failed
        print("  - Skipping Video evaluation, previous step failed.")
        return {"tasks": tasks}

    # --- Read Prepared Inputs ---
    prepared_inputs = current_task.get("task_inputs", {})
    video_paths = prepared_inputs.get("evaluation_target_video_paths", []) # Use prepared key
    video_related_text = prepared_inputs.get("evaluation_target_outputs_json", "{}")
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply.")
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")

    # REMOVED: is_final_eval check

    assessment = "Fail"; feedback = "Evaluation failed to run."; suggestions = "N/A"

    if not video_paths:
        err_msg = "Video evaluation required but no valid video paths found in prepared inputs."
        # ... (handle error, _set_task_failed, update subgraph_error) ...
        _set_task_failed(current_task, err_msg, node_name)
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks}
    else:
        # --- Prepare Prompt (No final eval context needed here) ---
        eval_prompt_for_vid_tool = f"""請根據以下具體標準，評估提供的影片和相關文字輸出：
**評估標準:**
{specific_criteria} # Criteria generated by previous node (could be final or regular)

**相關文字輸出 (JSON):**
```json
{video_related_text}
```

**影片路徑:** {', '.join(video_paths)}

**任務:**
請提供詳細的評估，說明影片和文字輸出是否符合上述標準。
你的評估應包含：
1.  **整體評估:** 在回應的 **最開頭** 明確指出 "整體評估：通過" 或 "整體評估：失敗"。
2.  **詳細回饋:** 針對每個標準進行分析，解釋評估結果的原因。
3.  **改進建議:** (可選) 如果評估為 "失敗"，請提供具體的改進建議。

Respond in {LLM_OUTPUT_LANGUAGE_DEFAULT}.
"""
    # --- Invoke Video Tool & Parse Result ---
    try:
        print(f"Eval Subgraph ({node_name}): Calling Video Recognition tool...")
        result = video_recognition.run({"video_paths": video_paths, "prompt": eval_prompt_for_vid_tool})

        if isinstance(result, str):
            if any(err_str in result for err_str in ["不支援", "錯誤", "Error"]): # Check tool error
                err_msg = f"Video tool failed: {result}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                assessment = "Fail"; feedback = err_msg
                subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
            else: # Parse assessment
                if result.strip().startswith("整體評估：通過") or "overall assessment: pass" in result.lower(): assessment = "Pass"
                elif result.strip().startswith("整體評估：失敗") or "overall assessment: fail" in result.lower(): assessment = "Fail"
                else: assessment = "Pass" if "pass" in result.lower() else "Fail"; print(f"Warning: Inferring video eval assessment: {assessment}")
                feedback = result
                print(f"Eval Subgraph ({node_name}): Assessment: {assessment}")
        else: # Tool type error
            err_msg = f"Video tool returned unexpected type: {type(result)}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            assessment = "Fail"; feedback = err_msg
            subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")

    except Exception as e: # Tool call error
        err_msg = f"Error calling video tool: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        return {"tasks": tasks}

    # --- Update TaskState (Assessment and Feedback) ---
    current_task["evaluation"]["assessment"] = assessment
    feedback_details = f"Assessment: {assessment}\nTool Raw Output:\n{feedback}"
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Set Final Status ---
    # This node IS the end of its branch. Set final status based on assessment.
    _update_eval_status_at_end(current_task, node_name)

    return {"tasks": tasks}

# --- Condition Function for routing after Image Eval ---
def route_after_image_eval(state: WorkflowState) -> Literal["evaluate_with_video_agent", "finished"]: 
    """Checks if video evaluation is also needed after image evaluation."""
    current_idx = state["current_task_index"]
    tasks = state["tasks"]
    if not (0 <= current_idx < len(tasks)):
        print("Eval Subgraph Router (After Image) Error: Invalid task index.")
        return "finished"

    current_task = tasks[current_idx]
    # Check if image evaluation step itself failed internally or logically
    # The status reflects the outcome of evaluate_with_image_node
    if current_task.get("status") == "failed":
        print("--- Eval Subgraph Router (After Image): Image evaluation failed. Ending evaluation branch. ---")
        return "finished" # End the subgraph, status is already 'failed'

    # Image eval passed logically, check if video exists in prepared inputs
    prepared_inputs = current_task.get("task_inputs", {})
    has_video_output = bool(prepared_inputs.get("evaluation_target_video_paths"))

    if has_video_output:
        print("--- Eval Subgraph Router (After Image): Image eval passed, Video output exists. Routing to Video Evaluation. ---")
        return "evaluate_with_video_agent"
    else:
        print("--- Eval Subgraph Router (After Image): Image eval passed, No video output found. Ending evaluation branch. ---")
        # Since only image eval was needed and it passed logically, set final status to completed
        _update_eval_status_at_end(current_task, "Image Eval (Final Step)")
        # The helper function sets status='completed' based on assessment='Pass'
        state['tasks'][current_idx] = current_task # Update state directly
        return "finished"


# --- Build and Compile Evaluation Subgraph ---
evaluation_subgraph_builder = StateGraph(WorkflowState)

# --- NEW Entry Point ---
evaluation_subgraph_builder.set_entry_point("prepare_evaluation_inputs") # Use new name

# --- >>> MODIFIED: Add Nodes with new names <<< ---
evaluation_subgraph_builder.add_node("prepare_evaluation_inputs", prepare_evaluation_inputs_node)
evaluation_subgraph_builder.add_node("gather_criteria_sources", gather_criteria_sources_node)
evaluation_subgraph_builder.add_node("generate_specific_criteria", generate_specific_criteria_node)
evaluation_subgraph_builder.add_node("evaluate_with_llm_agent", evaluate_with_llm_node)
evaluation_subgraph_builder.add_node("evaluate_with_image_agent", evaluate_with_image_node)
evaluation_subgraph_builder.add_node("evaluate_with_video_agent", evaluate_with_video_node)
# --- >>> END MODIFIED <<< ---

# --- Conditional edge after Prep ---
def route_after_eval_prep(state: WorkflowState) -> Literal["gather_criteria_sources", "finished"]: # Modified Literal hint
    """Routes to criteria gathering or ends if prep failed."""
    current_idx = state["current_task_index"]
    tasks = state["tasks"]
    if not (0 <= current_idx < len(tasks)) or tasks[current_idx].get("status") == "failed":
        print("--- Eval Subgraph: Routing to END due to evaluation input preparation failure ---")
        return "finished"
    else:
        return "gather_criteria_sources"

evaluation_subgraph_builder.add_conditional_edges(
    "prepare_evaluation_inputs", # Use new name
    route_after_eval_prep,
    {
        # --- >>> MODIFIED MAPPING <<< ---
        "gather_criteria_sources": "gather_criteria_sources",
        "finished": END
    }
)

# --- Edges for criteria flow ---
evaluation_subgraph_builder.add_edge("gather_criteria_sources", "generate_specific_criteria") # Use new names

# --- Conditional Edge from Criteria Generation to First Eval Tool ---
evaluation_subgraph_builder.add_conditional_edges(
    "generate_specific_criteria", # Use new name
    route_to_evaluation_tool_node, # Uses updated route_to_evaluation_tool_node
    {
        # --- >>> MODIFIED MAPPING <<< ---
        "evaluate_with_llm_agent": "evaluate_with_llm_agent",
        "evaluate_with_image_agent": "evaluate_with_image_agent",
        "evaluate_with_video_agent": "evaluate_with_video_agent",
        "finished": END
    }
)

# --- Edges from Terminal Eval Nodes ---
# --- >>> MODIFIED NODE NAMES <<< ---
evaluation_subgraph_builder.add_edge("evaluate_with_llm_agent", END)
evaluation_subgraph_builder.add_edge("evaluate_with_video_agent", END)
# --- >>> END MODIFIED <<< ---

# --- Conditional Edge after Image Eval ---
evaluation_subgraph_builder.add_conditional_edges(
    "evaluate_with_image_agent", # Use new name
    route_after_image_eval, # Uses updated route_after_image_eval
    {
        # --- >>> MODIFIED MAPPING <<< ---
        "evaluate_with_video_agent": "evaluate_with_video_agent",
        "finished": END
    }
)

# Compile
evaluation_teams = evaluation_subgraph_builder.compile()
evaluation_teams.name = "EvaluationSubgraph" # Set name on the correct variable
print("Evaluation Subgraph compiled successfully and named 'evaluation_teams'.")
