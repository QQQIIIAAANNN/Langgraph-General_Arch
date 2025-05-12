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

# --- State Imports ---
from src.state import WorkflowState, TaskState # 從 state.py 導入狀態

# =============================================================================
# 2. 環境變數與組態載入
# =============================================================================
load_dotenv()
config_manager = ConfigManager("config.json")
_full_static_config = config_manager.get_full_config() # Load once for static parts
workflow_config_static = _full_static_config.workflow

# =============================================================================
# 3. 常數設定
# =============================================================================
OUTPUT_DIR = workflow_config_static.output_directory
RENDER_CACHE_DIR = os.path.join(OUTPUT_DIR, "render_cache")
MODEL_CACHE_DIR = os.path.join(OUTPUT_DIR, "model_cache")
os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
MEMORY_DIR = os.path.join("knowledge", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
LLM_OUTPUT_LANGUAGE_DEFAULT = workflow_config_static.llm_output_language

# =============================================================================
# 4. LLM 與記憶體初始化 (LTM部分)
# =============================================================================
ltm_embedding_config = _full_static_config.memory.long_term_memory
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
print(f"Long-term memory vector store initialized. Storing in: {MEMORY_DIR}")
print(f"LTM Retriever 'k' value will be determined at runtime from configuration.")

# =============================================================================
# 5. Helper Functions
# =============================================================================
def _filter_base64_from_files(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Creates a new list of file dictionaries without the 'base64_data' key."""
    if not files:
        return []
    filtered_files = []
    for file_info in files:
        filtered_info = file_info.copy()
        filtered_info.pop('base64_data', None)
        filtered_files.append(filtered_info)
    return filtered_files

def _set_task_failed(task: TaskState, error_message: str, node_name: str):
    """Sets task status to 'failed' and logs the error message."""
    print(f"--- Task Failure in Node '{node_name}' ---")
    print(f"Error: {error_message}")
    task["status"] = "failed"
    task["error_log"] = f"[{node_name}] {error_message}"
    task["outputs"] = {}
    task["output_files"] = []
    # Clear evaluation specific fields on failure
    if "evaluation" in task:
        task["evaluation"]["assessment"] = "Fail"
        task["evaluation"]["specific_criteria"] = "N/A due to failure"
        task["evaluation"]["subgraph_error"] = (task["evaluation"].get("subgraph_error", "") + f"; {error_message}").strip("; ")

def _append_feedback(task: TaskState, feedback: str, node_name: str):
    """Appends feedback to the task's feedback_log."""
    current_log = task.get("feedback_log") or ""
    prefix = f"[{node_name} Feedback]:"
    # Append new feedback block
    task["feedback_log"] = (current_log + f"\n{prefix}\n{feedback}").strip()

def _update_eval_status_at_end(task: TaskState, node_name: str):
    """Sets final task status based on evaluation assessment.
       Handles different assessment types (Pass/Fail vs Score).
       Always sets Special/Final Agents to completed unless internal error occurred.
    """
    # 如果已經因內部錯誤設置為failed，不要覆蓋
    if task.get("status") == "failed" and task.get("error_log"):
        print(f"  - [{node_name}] Task already failed internally ({task.get('error_log')}), skipping final status update based on assessment.")
        return

    selected_agent = task.get("selected_agent", "")  # 獲取代理名稱
    assessment = task.get("evaluation", {}).get("assessment", "Fail") # 使用主要的 assessment

    # --- <<< 關鍵修改：優先處理 Special/Final Agent >>> ---
    if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        task["status"] = "completed"
        print(f"  - [{node_name}] Agent is {selected_agent}. ENFORCING final status to COMPLETED.")
        return # 直接返回，不再執行後續的 Pass/Fail 判斷
    # --- <<< 結束修改 >>> ---

    # 標準評估代理(EvaAgent)的原始邏輯
    # Note: assessment here refers to the *main* assessment, likely from LLM or combined visual.
    is_pass_fail_eval = not isinstance(assessment, str) or assessment.lower() in ["pass", "fail"]
    is_score_eval = isinstance(assessment, str) and assessment.lower().startswith("score") # Should not happen for EvaAgent

    if is_pass_fail_eval:
        if assessment == "Pass":
            task["status"] = "completed"
            print(f"  - [{node_name}] Standard Assessment is Pass. Setting final status to COMPLETED.")
        else: # Assessment is Fail
            task["status"] = "failed" # Standard Fail means workflow failure
            print(f"  - [{node_name}] Standard Assessment is Fail. Setting final status to FAILED.")
            # Log the logical failure reason if not already logged
            if task.get("error_log") is None: # Only log if no internal error happened first
                 failure_reason = f"Evaluation resulted in '{assessment}'."
                 task["error_log"] = f"[{node_name}] {failure_reason}" # Use error_log for logical fail
                 _append_feedback(task, failure_reason, node_name)
    # --- (Removed score handling here as it's covered by Special/Final logic above) ---
    # elif is_score_eval: ...
    else: # Unexpected assessment value for EvaAgent
        task["status"] = "failed"
        err_msg = f"Unexpected assessment value '{assessment}' for EvaAgent. Setting status to FAILED."
        print(f"  - [{node_name}] Note: {err_msg}")
        if task.get("error_log") is None:
            task["error_log"] = f"[{node_name}] {err_msg}"
        _append_feedback(task, err_msg, node_name)


# =============================================================================
# <<< Evaluation Subgraph Nodes (Refactored) >>>
# =============================================================================

async def prepare_evaluation_inputs_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Prepares inputs for the evaluation agent based on its type (EvaAgent, SpecialEvaAgent, FinalEvaAgent).
    - EvaAgent: Focuses on the immediately preceding task's outputs.
    - SpecialEvaAgent / FinalEvaAgent: Focuses on aggregated workflow outputs/summary.
    Updates `current_task['task_inputs']` including a 'needs_detailed_criteria' flag.
    """
    node_name = "Prepare Eval Inputs"
    tasks = [t.copy() for t in state['tasks']]
    current_idx = state['current_task_index']
    if not (0 <= current_idx < len(tasks)):
         print(f"Eval Subgraph Error ({node_name}): Invalid current_task_index {current_idx}")
         return {"tasks": tasks}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')
    print(f"--- Running Node: {node_name} for Task {current_idx} (Agent: {selected_agent}, Objective: {current_task.get('description')}) ---")

    runtime_config = config.get("configurable", {})
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)

    current_task["task_inputs"] = {}
    prompt_inputs = {}
    prompt_template_name = ""
    is_standard_eval = selected_agent == "EvaAgent"
    is_special_or_final_eval = selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]

    if is_standard_eval:
        prompt_template_name = "prepare_evaluation_inputs"
        target_task_idx = current_idx - 1
        if target_task_idx < 0:
            err_msg = f"Cannot perform standard evaluation ({selected_agent}), no previous task exists."
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks}
        target_task = tasks[target_task_idx]
        print(f"Eval Subgraph ({node_name}): Preparing inputs to evaluate Task {target_task_idx} (Agent: {selected_agent}): {target_task.get('description')}")
        target_task_files_raw = target_task.get("output_files", [])
        filtered_target_files = _filter_base64_from_files(target_task_files_raw)
        prompt_inputs = {
            "current_task_objective": current_task.get("task_objective", "N/A"),
            "task_description": target_task.get("description", "N/A"),
            "task_objective": target_task.get("task_objective", "N/A"),
            "evaluated_task_outputs_json": json.dumps(target_task.get("outputs", {}), ensure_ascii=False),
            "evaluated_task_output_files_json": json.dumps(filtered_target_files, ensure_ascii=False),
            "llm_output_language": llm_output_language,
        }
    elif is_special_or_final_eval:
        prompt_template_name = "prepare_final_evaluation_inputs"
        print(f"Eval Subgraph ({node_name}): Preparing inputs for {selected_agent} evaluation.")
        aggregated_outputs = {}
        aggregated_files_raw = []
        full_task_summary_parts = ["Workflow Summary:"]
        for i, task in enumerate(tasks):
             if i < current_idx and task.get("status") == "completed":
                 task_id = task.get("task_id", f"task_{i}")
                 full_task_summary_parts.append(f"  Task {i} (ID: {task_id}): {task.get('description', 'N/A')}")
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
                         summary_outputs_json = json.dumps(outputs_for_summary, ensure_ascii=False, indent=2, default=str) # 添加 default=str
                         full_task_summary_parts.append(f"    Outputs: {summary_outputs_json[:500]}{'...' if len(summary_outputs_json)>500 else ''}") # 限制摘要長度
                     except Exception as e:
                          print(f"    Warning: Could not JSON dump filtered outputs for task {task_id} summary: {e}. Skipping outputs in summary string.")
                          full_task_summary_parts.append(f"    Outputs: [Could not serialize - check logs for task {task_id}]")
                 task_files = task.get("output_files")
                 if task_files:
                     for f in task_files: f['source_task_id'] = task_id
                     aggregated_files_raw.extend(task_files)
                     full_task_summary_parts.append(f"    Files: {[f.get('filename', 'N/A') for f in task_files]}")
        filtered_aggregated_files = _filter_base64_from_files(aggregated_files_raw)
        prompt_inputs = {
            "selected_agent": selected_agent,
            "current_task_objective": current_task.get("task_objective", "N/A"),
            "user_input": state.get("user_input", "N/A"),
            "full_task_summary": "\n".join(full_task_summary_parts),
            "aggregated_outputs_json": json.dumps(aggregated_outputs, ensure_ascii=False),
            "aggregated_files_json": json.dumps(filtered_aggregated_files, ensure_ascii=False),
            "llm_output_language": llm_output_language,
        }
    else:
        err_msg = f"Invalid or missing agent type '{selected_agent}' for evaluation task."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    prompt_template = runtime_config.get(f"ea_{prompt_template_name}_prompt") or \
                      config_manager.get_prompt_template("eva_agent", prompt_template_name)
    if not prompt_template:
        err_msg = f"Missing required prompt template '{prompt_template_name}' for {selected_agent} input preparation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    try:
        print(f"Eval Subgraph ({node_name}): Formatting prompt '{prompt_template_name}' with inputs: {list(prompt_inputs.keys())}")
        prep_prompt = prompt_template.format(**prompt_inputs)
        print(f"Eval Subgraph ({node_name}): Invoking LLM for {selected_agent} input prep...")
        prep_response = await llm.ainvoke(prep_prompt)
        prep_content = prep_response.content.strip()

        if prep_content.startswith("```json"): prep_content = prep_content[7:-3].strip()
        elif prep_content.startswith("```"): prep_content = prep_content[3:-3].strip()

        try:
            prepared_eval_inputs = json.loads(prep_content)
            if isinstance(prepared_eval_inputs, dict) and "error" in prepared_eval_inputs:
                llm_error_msg = prepared_eval_inputs['error']
                err_msg = f"LLM indicated error during eval input prep for {selected_agent}: {llm_error_msg}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                _set_task_failed(current_task, err_msg, node_name)
            elif not isinstance(prepared_eval_inputs, dict):
                err_msg = f"LLM returned invalid format (expected dict) for {selected_agent}. Content: {prep_content}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                _set_task_failed(current_task, err_msg, node_name)
            else:
                needs_detailed_criteria = prepared_eval_inputs.get("needs_detailed_criteria", False)
                print(f"Eval Subgraph ({node_name}): Evaluation inputs prepared successfully for {selected_agent}. Needs Detailed Criteria: {needs_detailed_criteria}")
                current_task["task_inputs"] = prepared_eval_inputs
                if "needs_detailed_criteria" not in current_task["task_inputs"]:
                    current_task["task_inputs"]["needs_detailed_criteria"] = False
                if "evaluation" not in current_task: current_task["evaluation"] = {}
                current_task["evaluation"]["subgraph_error"] = None

        except json.JSONDecodeError:
            err_msg = f"Could not parse LLM JSON response for {selected_agent}. Raw content: '{prep_content}'"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)

    except KeyError as ke:
        err_msg = f"Formatting error (KeyError: {ke}). Check prompt template '{prompt_template_name}' and inputs for {selected_agent}."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        print(f"--- Problematic Prompt Template ({prompt_template_name}) ---")
        print(repr(prompt_template))
        print(f"--- Provided Inputs ---")
        print(prompt_inputs)
        print(f"---------------------------------")
        import traceback; traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)
    except Exception as e:
        err_msg = f"Unexpected error during LLM call or processing for {selected_agent}: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        import traceback; traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)

    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}


async def gather_criteria_sources_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Gathers context from RAG/Search to potentially inform criteria generation.
    Updates `current_task["evaluation"]["criteria_sources"]`.
    (Currently runs for all evaluation types).
    """
    node_name = "Gather Criteria Sources"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): return {"tasks": tasks} # Safeguard
    current_task = tasks[current_idx]

    # Skip if previous step failed
    if current_task.get("status") == "failed":
         print(f"  - Skipping node {node_name}, previous step failed.")
         return {"tasks": tasks, "current_task": current_task.copy()} # Pass through state

    # Initialize evaluation dict if needed
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    subgraph_error = current_task["evaluation"].get("subgraph_error", "") # Carry over errors

    # Use description from prepared inputs if available, fallback to task description
    # This covers standard eval ("evaluate task X") and final/special ("Final Review", "Compare Options")
    task_desc_for_query = current_task.get("task_inputs", {}).get("evaluation_target_description", current_task.get("description", ""))

    runtime_config = config["configurable"]
    retriever_k = runtime_config.get("retriever_k", 5) # Use runtime config for K

    # RAG Query
    rag_query = f"Find evaluation standards or relevant context for: {task_desc_for_query}"
    rag_context = "RAG context not retrieved."
    try:
        print(f"  - Calling RAG with k={retriever_k} for query: {rag_query}")
        # Create retriever dynamically based on runtime config K
        retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        # Assume async invoke exists on retriever (if not, adapt)
        docs = await retriever.ainvoke(rag_query) # Example call
        if docs:
             rag_context = f"Retrieved from Knowledge Base:\n" + "\n".join([f"- {doc.page_content}" for doc in docs]) + "\n"
        else:
             rag_context = "No relevant documents found by RAG."
    except Exception as e:
        print(f"  - RAG call failed: {e}")
        rag_context = f"RAG call error: {e}"
        subgraph_error = (subgraph_error + f"; RAG Error: {e}").strip("; ")

    # Web Search Query
    search_query = f"Search for evaluation methods, standards, or context for: {task_desc_for_query}"
    search_context = "Web search not performed or failed."
    try:
        print(f"  - Calling Web Search for query: {search_query}")
        search_result = perform_grounded_search({"query": search_query}) # Assumes sync call is okay
        if isinstance(search_result, dict) and search_result.get("text_content"):
            # --- <<< 修改：過濾搜索結果 >>> ---
            text_content = search_result.get('text_content', '')
            search_context = f"Web Search Results:\nText: {text_content}\n"
            # 不再包含 sources 或 suggestions
            # --- <<< 結束修改 >>> ---
        elif isinstance(search_result, dict) and search_result.get("error"):
            search_context = f"Web search error: {search_result['error']}"
            subgraph_error = (subgraph_error + f"; Web Search Error: {search_result['error']}").strip("; ")
        else:
             search_context = "No relevant results from Web Search."
    except Exception as e:
        print(f"  - Web Search call failed: {e}")
        search_context = f"Web Search call error: {e}"
        subgraph_error = (subgraph_error + f"; Web Search Error: {e}").strip("; ")

    # Update evaluation dictionary
    current_task["evaluation"]["criteria_sources"] = {"rag": rag_context, "search": search_context}
    if subgraph_error:
        current_task["evaluation"]["subgraph_error"] = subgraph_error
        print(f"  - Error during criteria source gathering: {subgraph_error}")
    else:
        print(f"  - Criteria sources gathered successfully.")

    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}


def generate_specific_criteria_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates specific evaluation criteria or a rubric based on the agent type.
    - EvaAgent: Simple criteria list.
    - SpecialEvaAgent / FinalEvaAgent: Detailed rubric.
    Updates `current_task["evaluation"]["specific_criteria"]`.
    """
    node_name = "Generate Criteria/Rubric"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): return {"tasks": tasks} # Safeguard
    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # Skip if previous step failed
    if current_task.get("status") == "failed":
        print(f"  - Skipping node {node_name}, previous step failed.")
        return {"tasks": tasks, "current_task": current_task.copy()}

    # Ensure necessary dictionaries exist
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")
    criteria_sources = current_task.get("evaluation", {}).get("criteria_sources", {}) # Get sources if gathered

    print(f"  - Generating criteria/rubric for Agent: {selected_agent}")

    runtime_config = config["configurable"]
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)

    prompt_template = None
    prompt_inputs = {}
    prompt_template_name = ""
    generated_output = "Error: Could not generate criteria/rubric." # Default

    # Determine which prompt to use based on agent type
    if selected_agent == "EvaAgent":
        prompt_template_name = "generate_criteria"
        prompt_template = runtime_config.get("ea_generate_criteria_prompt") or \
                          config_manager.get_prompt_template("eva_agent", prompt_template_name)
        prompt_inputs = {
            "task_description": current_task.get("task_inputs", {}).get("evaluation_target_description", "N/A"),
            "task_objective": current_task.get("task_inputs", {}).get("evaluation_target_objective", "N/A"),
            "overall_goal": state.get("user_input", "N/A"),
            "rag_context": criteria_sources.get("rag", "Not gathered or available."),
            "search_context": criteria_sources.get("search", "Not gathered or available."),
            "llm_output_language": llm_output_language
        }
    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        prompt_template_name = "generate_final_criteria"
        prompt_template = runtime_config.get("ea_generate_final_criteria_prompt") or \
                          config_manager.get_prompt_template("eva_agent", prompt_template_name)
        prompt_inputs = {
            "selected_agent": selected_agent,
            "current_task_objective": current_task.get("task_objective", "N/A"),
            "user_input": state.get("user_input", "N/A"),
            "full_task_summary": current_task.get("task_inputs", {}).get("evaluation_target_full_summary", "Workflow summary not available."),
            "final_eval_inputs_json": json.dumps(current_task.get("task_inputs", {}), ensure_ascii=False, indent=2),
            "rag_context": criteria_sources.get("rag", "Not gathered or available."),
            "search_context": criteria_sources.get("search", "Not gathered or available."),
            "llm_output_language": llm_output_language
        }
    else:
        err_msg = f"Invalid agent type '{selected_agent}' encountered in criteria generation."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
        generated_output = err_msg
        # Update evaluation dictionary and return
        current_task["evaluation"]["specific_criteria"] = generated_output
        if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error
        tasks[current_idx] = current_task
        return {"tasks": tasks, "current_task": current_task.copy()}

    if not prompt_template:
        err_msg = f"Missing required prompt template '{prompt_template_name}' for {selected_agent} criteria/rubric generation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
        generated_output = f"Error: Prompt template '{prompt_template_name}' not found."
        # Allow eval nodes to potentially use default criteria if generation fails
    else:
        try:
            prompt = prompt_template.format(**prompt_inputs)
            print(f"  - Invoking LLM for {selected_agent} criteria/rubric...")
            response = llm.invoke(prompt)
            generated_output = response.content.strip()
            print(f"  - Generated Criteria/Rubric:\n{generated_output[:500]}...") # Log preview
        except KeyError as ke:
            err_msg = f"Formatting error (KeyError: {ke}). Check prompt template '{prompt_template_name}' and inputs for {selected_agent}."
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            print(f"--- Problematic Prompt Template ({prompt_template_name}) ---")
            print(repr(prompt_template))
            print(f"--- Provided Inputs ---")
            print(prompt_inputs)
            print(f"---------------------------------")
            subgraph_error = (subgraph_error + f"; Criteria Gen Formatting Error: {ke}").strip("; ")
            generated_output = f"Error during generation (Formatting): {ke}"
        except Exception as e:
            err_msg = f"Criteria/Rubric generation LLM error for {selected_agent}: {e}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
            generated_output = f"Error during generation: {e}" # Store error

    # Update evaluation dictionary
    current_task["evaluation"]["specific_criteria"] = generated_output # Store the generated text
    if subgraph_error:
        current_task["evaluation"]["subgraph_error"] = subgraph_error

    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}


def route_to_evaluation_tool_node(state: WorkflowState) -> str:
    """
    Routes to the correct evaluation tool node based on prepared inputs.
    Checks `task_inputs` for image/video paths. Defaults to LLM evaluation.
    """
    node_name = "Route to Eval Tool"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    tasks = state["tasks"]
    if not (0 <= current_idx < len(tasks)):
        print(f"  - Error: Invalid task index {current_idx}. Routing to END.")
        return END # Cannot proceed

    current_task = tasks[current_idx]

    # Check if previous steps (prep/criteria) failed
    if current_task.get("status") == "failed":
        print(f"  - Routing Decision: Task failed in previous step ({current_task.get('error_log')}). Routing to END.")
        return END

    # Check the inputs prepared by `prepare_evaluation_inputs_node`
    prepared_inputs = current_task.get("task_inputs", {})
    # Use the keys consistent with the output of prepare_evaluation_inputs
    # Standard eval keys:
    has_image_output = bool(prepared_inputs.get("evaluation_target_image_paths"))
    has_video_output = bool(prepared_inputs.get("evaluation_target_video_paths"))
    # Final/Special eval keys:
    has_key_image_output = bool(prepared_inputs.get("evaluation_target_key_image_paths"))
    has_key_video_output = bool(prepared_inputs.get("evaluation_target_key_video_paths"))
    # Check for any text/structured output (covers all cases)
    has_text_output = (
        prepared_inputs.get("evaluation_target_outputs_json") != '{}' or # Standard
        bool(prepared_inputs.get("evaluation_target_full_summary")) # Final/Special implies text context
    )

    # Combine checks: does *any* relevant image/video path exist?
    needs_image_eval = has_image_output or has_key_image_output
    needs_video_eval = has_video_output or has_key_video_output

    print(f"  - Agent: {current_task.get('selected_agent')}")
    print(f"  - Prepared Inputs Check:")
    print(f"    - Needs Image Eval: {needs_image_eval}")
    print(f"    - Needs Video Eval: {needs_video_eval}")
    print(f"    - Has Text/Structured Output: {has_text_output}")

    # Routing Logic: Prioritize visual tools if applicable
    if needs_image_eval: # Handles (Image only) or (Image + Video) or (Image + Text) etc.
        print("  - Routing Decision: Image found -> Route to Image Evaluation")
        return "evaluate_with_image_agent"
    elif needs_video_eval: # Handles (Video only) or (Video + Text) etc.
        print("  - Routing Decision: Video found (no image) -> Route to Video Evaluation")
        return "evaluate_with_video_agent"
    elif has_text_output: # Only text/structured outputs or summary
        print("  - Routing Decision: Only Text/Summary found -> Route to LLM Evaluation")
        return "evaluate_with_llm_agent"
    else:
        # If no outputs were prepared at all (error in prep?), default to LLM
        print("  - Routing Decision: No specific outputs found in prepared inputs -> Defaulting to LLM Evaluation")
        return "evaluate_with_llm_agent"


def evaluate_with_llm_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs evaluation using LLM based on prepared inputs and criteria/rubric.
    Handles Standard, Special, and Final evaluation types via the core 'evaluation' prompt.
    Updates `current_task['evaluation']` and final status.
    """
    node_name = "LLM Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): return {"tasks": tasks} # Safeguard
    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # Skip if previous step failed
    if current_task.get("status") == "failed":
        print(f"  - Skipping node {node_name}, previous step failed.")
        # Return state with failed task. The END connection handles this.
        return {"tasks": tasks}

    # Ensure necessary dictionaries exist
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "")

    print(f"  - Performing LLM Evaluation for Agent: {selected_agent}")

    # --- Read Prepared Inputs and Criteria/Rubric ---
    prepared_inputs = current_task.get("task_inputs", {})
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")

    # --- Initialize LLM, Get Unified Template ---
    runtime_config = config["configurable"]
    ea_llm_config = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config)
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    # Use the single unified evaluation prompt
    evaluation_template = runtime_config.get("ea_evaluation_prompt") or \
                           config_manager.get_prompt_template("eva_agent", "evaluation")

    # Default results
    assessment = "Fail"; feedback = "Evaluation failed to run."; improvement_suggestions = "N/A"; selected_option = None; assessment_type = None

    if not evaluation_template:
        err_msg = "Missing unified 'evaluation' prompt template for eva_agent."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        if "evaluation" in current_task: # Ensure dict exists before adding error
            current_task["evaluation"]["subgraph_error"] = (subgraph_error + f"; {err_msg}").strip("; ")
        tasks[current_idx] = current_task
        return {"tasks": tasks} # Return updated state

    # --- Prepare Inputs for the Unified Prompt ---
    # The prompt expects inputs based on evaluation type, use .get() with defaults
    prompt_inputs = {
        "selected_agent": selected_agent,
        "evaluation_target_description": prepared_inputs.get("evaluation_target_description", "N/A"),
        "evaluation_target_objective": prepared_inputs.get("evaluation_target_objective", "N/A"),
        "specific_criteria": specific_criteria,
        "llm_output_language": llm_output_language,
        # Standard Eval Inputs (provide even if empty for template)
        "evaluation_target_outputs_json": prepared_inputs.get("evaluation_target_outputs_json", "{}"),
        "evaluation_target_image_paths_str": ", ".join(prepared_inputs.get("evaluation_target_image_paths", [])) or "None",
        "evaluation_target_video_paths_str": ", ".join(prepared_inputs.get("evaluation_target_video_paths", [])) or "None",
        "evaluation_target_other_files_str": str(prepared_inputs.get("evaluation_target_other_files", []) or "None"), # Simple string representation
        # Final/Special Eval Inputs (provide even if empty for template)
        "full_task_summary": prepared_inputs.get("evaluation_target_full_summary", "N/A"),
        "evaluation_target_key_image_paths_str": ", ".join(prepared_inputs.get("evaluation_target_key_image_paths", [])) or "None",
        "evaluation_target_key_video_paths_str": ", ".join(prepared_inputs.get("evaluation_target_key_video_paths", [])) or "None",
        "evaluation_target_other_artifacts_summary_str": prepared_inputs.get("evaluation_target_other_artifacts_summary", "N/A")
    }

    # --- Invoke LLM & Parse Result ---
    try:
        print(f"  - Formatting unified evaluation prompt for {selected_agent}...")
        prompt = evaluation_template.format(**prompt_inputs)
        print(f"  - Invoking LLM for {selected_agent} evaluation...")
        response = llm.invoke(prompt) # Sync call
        content = response.content.strip() if response and hasattr(response, 'content') else ""

        if not content:
            print(f"Eval Subgraph Warning ({node_name}): LLM returned empty content for {selected_agent}. Defaulting to Fail.")
            # Keep default assessment='Fail'
            feedback = "LLM returned empty response."
        else:
            print(f"  - Raw LLM response content received.") # Avoid logging potentially large content

            # Clean JSON content
            if content.startswith("```json"): content = content[7:-3].strip()
            elif content.startswith("```"): content = content[3:-3].strip()

            try: # Nested try for JSON parsing
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    assessment = parsed_json.get("assessment", "Fail") # Default to Fail if key missing
                    feedback = parsed_json.get("feedback", "No feedback provided.")
                    improvement_suggestions = parsed_json.get("improvement_suggestions", "N/A")
                    selected_option = parsed_json.get("selected_option_identifier") # For SpecialEvaAgent
                    assessment_type = parsed_json.get("assessment_type") # Standard/Special/Final
                    print(f"  - Parsed Assessment: {assessment}")
                    print(f"  - Parsed Type: {assessment_type}")
                    print(f"  - Parsed Selected Option: {selected_option}")
                else:
                    print(f"Eval Subgraph Warning ({node_name}): LLM returned valid JSON, but not a dictionary: {type(parsed_json)}. Defaulting to Fail.")
                    assessment = "Fail"; feedback = f"Unexpected JSON format received: {type(parsed_json)}"

            except json.JSONDecodeError as json_e:
                print(f"Eval Subgraph Error ({node_name}): Failed to parse LLM JSON response for {selected_agent}: {json_e}")
                # print(f"Problematic Content:\n{content}") # Avoid logging potentially sensitive PII in content
                assessment = "Fail"; feedback = f"LLM output was not valid JSON. Check prompt instructions for JSON format."
                subgraph_error = (subgraph_error + f"; LLM JSON parse error: {json_e}").strip("; ")

    except KeyError as ke:
        err_msg = f"Formatting error (KeyError: {ke}). Check unified 'evaluation' prompt and inputs for {selected_agent}."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        # print prompt details if needed for debugging (omitted for brevity)
        _set_task_failed(current_task, err_msg, node_name)
        subgraph_error = (subgraph_error + f"; LLM Formatting Error: {ke}").strip("; ")
        tasks[current_idx] = current_task
        if "evaluation" in current_task: current_task["evaluation"]["subgraph_error"] = subgraph_error
        return {"tasks": tasks} # Return early

    except Exception as e: # Catch errors from llm.invoke() itself
        err_msg = f"LLM evaluation call error for {selected_agent}: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name) # Internal failure
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
        tasks[current_idx] = current_task
        if "evaluation" in current_task: current_task["evaluation"]["subgraph_error"] = subgraph_error
        return {"tasks": tasks} # Return early on LLM call failure

    # --- Update TaskState ---
    current_task["evaluation"]["assessment"] = assessment
    current_task["evaluation"]["assessment_type"] = assessment_type # Store type
    # Consolidate feedback
    feedback_details = f"Assessment Type: {assessment_type}\nAssessment: {assessment}\nFeedback: {feedback}\nSuggestions: {improvement_suggestions}"
    if selected_option:
        feedback_details += f"\nSelected Option: {selected_option}"
        current_task["evaluation"]["selected_option_identifier"] = selected_option # Store selected option if applicable
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Set Final Status ---
    # This node is always terminal for the LLM evaluation path
    _update_eval_status_at_end(current_task, node_name)

    tasks[current_idx] = current_task
    return {"tasks": tasks}


def evaluate_with_image_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs evaluation using Image Recognition tool based on prepared inputs and criteria.
    Updates `current_task['evaluation']` and potentially sets provisional 'failed' status.
    """
    node_name = "Image Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): return {"tasks": tasks} # Safeguard
    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # Skip if previous step failed
    if current_task.get("status") == "failed":
        print(f"  - Skipping node {node_name}, previous step failed.")
        return {"tasks": tasks}

    # Ensure necessary dictionaries exist
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or "" # Ensure string

    print(f"  - Performing Image Evaluation (related to Agent: {selected_agent})")
    prepared_inputs = current_task.get("task_inputs", {})
    image_paths = prepared_inputs.get("evaluation_target_image_paths", []) or \
                  prepared_inputs.get("evaluation_target_key_image_paths", [])
    image_related_text = prepared_inputs.get("evaluation_target_outputs_json", "{}")
    if image_related_text == '{}': image_related_text = prepared_inputs.get("evaluation_target_full_summary", "N/A")
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")

    # Default results
    image_tool_assessment = "N/A"; feedback = "Image evaluation failed to run."; # Use neutral default
    
    # --- <<< 修改：用於收集多個圖像評估結果 >>> ---
    individual_image_results = []
    has_multiple_images = len(image_paths) > 1
    # --- <<< 結束修改 >>> ---

    if not image_paths:
        err_msg = "Image evaluation required but no valid image paths found in prepared inputs."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        feedback = err_msg
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
    else:
        # --- Prepare Prompt for Image Tool ---
        eval_prompt_for_img_tool = f"""請根據以下具體標準/評分準則，評估提供的圖片和相關文字輸出：
**評估標準/準則:**
{specific_criteria}

**相關文字輸出/摘要:**
```
{image_related_text}
```

**圖片路徑:** {', '.join(image_paths)}

**任務:**
請提供詳細的評估，說明圖片是否符合上述標準/準則。
{'' if not has_multiple_images else '''
由於有多張圖片，請針對每張圖片分別評估，並在評估後清楚標示各張圖片的優缺點及排名。
請明確指出哪張圖片是最佳選擇，並解釋選擇理由。
'''} 
你的評估應包含：
1.  **整體評估:** 在回應的 **最開頭** 明確指出 "整體評估：通過" 或 "整體評估：失敗"。 (這是工具層面的評估)
2.  **詳細回饋:** 針對標準/準則進行分析，解釋評估結果的原因。
3.  **改進建議:** (可選) 如果評估為 "失敗"，請提供具體的改進建議。
{'' if not has_multiple_images else '''
4.  **圖片排名:** 請將多張圖片從最好到最差排序，並指明最佳圖片的檔案名。
'''} 

Respond in {LLM_OUTPUT_LANGUAGE_DEFAULT}.
"""
        # --- Invoke Image Tool & Parse Result ---
        try:
            print(f"  - Calling Image Recognition tool for {len(image_paths)} image(s)...")
            # --- <<< 修改：正確的工具調用方式 >>> ---
            tool_input_dict = {
                "image_paths": image_paths,
                "prompt": eval_prompt_for_img_tool
            }
            result = img_recognition.run(tool_input=tool_input_dict)
            # --- <<< 結束修改 >>> ---

            if isinstance(result, str):
                feedback = result # Store the full tool output
                
                # --- 原有的解析 Pass/Fail 邏輯繼續保持不變 ---
                if selected_agent == "EvaAgent":
                    if result.strip().lower().startswith("整體評估：通過") or "overall assessment: pass" in result.lower():
                         image_tool_assessment = "Pass"
                    elif result.strip().lower().startswith("整體評估：失敗") or "overall assessment: fail" in result.lower():
                         image_tool_assessment = "Fail"
                    else:
                         image_tool_assessment = "Pass" if "pass" in result.lower() else "Fail"
                         print(f"  - Warning (EvaAgent): Could not explicitly parse Pass/Fail from image tool output. Inferred: {image_tool_assessment}")
                    print(f"  - Image Tool Assessment (EvaAgent): {image_tool_assessment}")
                else: # Special/Final Agent
                    image_tool_assessment = "Not Applicable" # Indicate Pass/Fail is not relevant here
                    print(f"  - Image Tool Raw Output Stored (Special/Final Agent).")
                
                # --- <<< 添加：如果是 SpecialEvaAgent 且有多圖，嘗試直接解析排名結果 >>> ---
                if has_multiple_images and selected_agent in ["SpecialEvaAgent"]:
                    print(f"  - Analyzing multi-image result for {selected_agent}...")
                    
                    # 嘗試從結果中提取最佳圖像信息
                    best_image = None
                    ranking = []
                    
                    # 查找包含「最佳圖像」或「最好的圖像」等關鍵詞的行
                    result_lines = result.split('\n')
                    for line in result_lines:
                        line = line.strip().lower()
                        if any(term in line for term in ["最佳圖像", "最好的圖像", "最適合的圖像", "排名第一", "最佳選擇"]):
                            # 尋找這行中的檔案名稱
                            for img_path in image_paths:
                                img_name = os.path.basename(img_path)
                                if img_name.lower() in line.lower():
                                    best_image = img_name
                                    print(f"  - Found potential best image: {best_image}")
                                    break
                        
                        # 查找包含「排名」的行來提取完整排名
                        if "排名" in line and ":" in line:
                            rank_text = line.split(":", 1)[1].strip()
                            # 從排名文本中提取文件名
                            for img_path in image_paths:
                                img_name = os.path.basename(img_path)
                                if img_name.lower() in rank_text.lower() and img_name not in ranking:
                                    ranking.append(img_name)
                    
                    # 如果找到了最佳圖像，更新評估結果
                    if best_image:
                        print(f"  - Setting best image as selected option: {best_image}")
                        current_task["evaluation"]["selected_option_identifier"] = best_image
                        
                        # 如果有排名，也將其存儲
                        if ranking:
                            current_task["evaluation"]["image_ranking"] = ranking
                            print(f"  - Extracted ranking: {ranking}")
                    
                    # 如果無法從輸出中提取最佳圖像，考慮進行附加的LLM分析
                    if not best_image and not ranking:
                        print(f"  - Could not extract best image or ranking. May need follow-up LLM analysis.")
                # --- <<< 結束添加 >>> ---
            else: # Tool returned unexpected type
                err_msg = f"Image tool returned unexpected type: {type(result)}"
                print(f"Eval Subgraph Error ({node_name}): {err_msg}")
                image_tool_assessment = "Fail"; feedback = err_msg;
                subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")

        except Exception as e: # Tool call error
            err_msg = f"Error calling image tool: {e}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            # --- <<< 修改：處理 TypeError >>> ---
            # 將錯誤消息添加到 feedback_log
            _append_feedback(current_task, f"Video tool call failed: {e}", node_name)
            # 設置任務失敗
            _set_task_failed(current_task, err_msg, node_name) # Pass the specific error message
            subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
            tasks[current_idx] = current_task
            if "evaluation" in current_task: current_task["evaluation"]["subgraph_error"] = subgraph_error
            # 打印 traceback 以便調試
            traceback.print_exc()
            return {"tasks": tasks} # 工具錯誤時提前返回
            # --- <<< 結束修改 >>> ---

    # --- <<< 修改：仅当无法从工具输出直接提取排名时才调用LLM >>> ---
    if has_multiple_images and selected_agent in ["SpecialEvaAgent"] and not current_task["evaluation"].get("selected_option_identifier"):
        print(f"  - Multiple images detected ({len(image_paths)}) for {selected_agent}. Running additional LLM analysis...")
        
        # 獲取LLM配置
        runtime_config = config.get("configurable", {})
        ea_llm_config = runtime_config.get("ea_llm", {})
        llm = initialize_llm(ea_llm_config)
        llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
        
        # 構建比較提示
        comparison_prompt = f"""
您需要從以下圖像評估結果中提取關鍵信息，作為特殊評估（SpecialEvaAgent）的一部分。

**評估標準/準則:**
{specific_criteria}

**圖像路徑:**
{chr(10).join([f"{i+1}. {os.path.basename(path)}" for i, path in enumerate(image_paths)])}

**圖像評估結果:**
{feedback}

**任務:**
1. 分析上述評估結果
2. 確定哪張圖像是最佳選擇
3. 如可能，對所有圖像進行排序（從最好到最差）
4. 提供最佳圖像的檔案名

請以JSON格式回覆，包含以下欄位：
- ranking: 圖像排名（從最佳到最差的檔案名稱列表）
- best_image: 最佳圖像的檔案名
- comparison_reasoning: 排名理由
- improvement_suggestions: 對最佳圖像的改進建議（如果有）

回應語言: {llm_output_language}
"""
        
        try:
            # 調用LLM進行比較
            comparison_response = llm.invoke(comparison_prompt)
            comparison_text = comparison_response.content.strip()
            
            # 嘗試解析JSON
            comparison_result = {}
            try:
                if comparison_text.startswith("```json"):
                    comparison_text = comparison_text[7:-3].strip()
                elif comparison_text.startswith("```"):
                    comparison_text = comparison_text[3:-3].strip()
                    
                comparison_result = json.loads(comparison_text)
                print(f"  - Successfully parsed comparison result: {list(comparison_result.keys())}")
                
                # 更新評估結果
                current_task["evaluation"]["image_comparison"] = comparison_result
                current_task["evaluation"]["selected_option_identifier"] = comparison_result.get("best_image", "")
                
                # 加入比較結果到feedback
                additional_feedback = f"\n\n--- 多圖像比較結果 ---\n排名: {comparison_result.get('ranking', [])}\n最佳圖像: {comparison_result.get('best_image', 'N/A')}\n分析理由: {comparison_result.get('comparison_reasoning', 'N/A')}"
                feedback += additional_feedback
                
            except json.JSONDecodeError:
                print(f"  - Warning: Could not parse LLM comparison result as JSON. Using raw text.")
                current_task["evaluation"]["image_comparison_text"] = comparison_text
                feedback += f"\n\n--- 多圖像比較結果 ---\n{comparison_text}"
                
        except Exception as comp_e:
            print(f"  - Error during image comparison: {comp_e}")
            feedback += f"\n\n--- 多圖像比較錯誤 ---\n{comp_e}"
    # --- <<< 結束修改 >>> ---

    # --- Update TaskState (Assessment and Feedback) ---
    current_task["evaluation"]["image_tool_assessment"] = image_tool_assessment # Store tool specific result
    feedback_details = f"Image Tool Assessment: {image_tool_assessment}\nTool Raw Output:\n{feedback}"
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Check if video eval is needed (logic remains) ---
    prepared_inputs = current_task.get("task_inputs", {})
    needs_video_eval = bool(prepared_inputs.get("evaluation_target_video_paths")) or \
                       bool(prepared_inputs.get("evaluation_target_key_video_paths"))

    # If this is the last visual eval step, call the final status updater
    if not needs_video_eval:
        # --- <<< 修改：調用 _update_eval_status_at_end >>> ---
        # The assessment used by _update_eval_status_at_end will be the main 'assessment' field,
        # which is not yet set here (only image_tool_assessment is).
        # For Special/Final, _update_eval_status_at_end forces 'completed'.
        # For EvaAgent, we need to set the main 'assessment' based on image_tool_assessment now.
        if selected_agent == "EvaAgent":
            current_task["evaluation"]["assessment"] = image_tool_assessment # Set main assessment for EvaAgent
        else:
            # For Special/Final, the main assessment comes from the LLM later.
            # We don't need to set a main assessment here.
            pass

        print(f"  - No video evaluation needed. Calling final status update...")
        # The helper function will now correctly handle the agent type.
        _update_eval_status_at_end(current_task, f"{node_name} (Image Only)")
        print(f"  - Final status set to: {current_task['status']}")
        # --- <<< 結束修改 >>> ---

    tasks[current_idx] = current_task
    return {"tasks": tasks}


def evaluate_with_video_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    執行影片評估，循環處理多個影片檔案。
    """
    node_name = "Video Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): return {"tasks": tasks} # Safeguard
    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # Skip if previous step failed
    if current_task.get("status") == "failed":
        print(f"  - Skipping node {node_name}, task status already 'failed'.")
        return {"tasks": tasks}

    # Ensure necessary dictionaries exist
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or "" # Ensure string
    prepared_inputs = current_task.get("task_inputs", {})
    video_paths = prepared_inputs.get("evaluation_target_video_paths", []) or \
                  prepared_inputs.get("evaluation_target_key_video_paths", [])
    video_related_text = prepared_inputs.get("evaluation_target_outputs_json", "{}")
    if video_related_text == '{}': video_related_text = prepared_inputs.get("evaluation_target_full_summary", "N/A")
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")

    # --- <<< 修改：初始化用於循環的變數 >>> ---
    accumulated_feedback: List[str] = [] # 儲存每個影片的反饋
    overall_video_assessment = "Pass" if selected_agent == "EvaAgent" else "Not Applicable"
    any_video_failed = False # 追蹤是否有影片分析出錯或評估失敗
    
    # --- 添加結構化儲存每個影片的結果 ---
    individual_video_results = []
    has_multiple_videos = len(video_paths) > 1
    # --- <<< 結束修改 >>> ---

    if not video_paths:
        err_msg = "Video evaluation required but no valid video paths found in prepared inputs."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        accumulated_feedback.append(f"錯誤: {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
        overall_video_assessment = current_task.get("evaluation", {}).get("image_tool_assessment", "Fail") # Fallback based on image
    else:
        # --- 循環處理每個影片路徑 ---
        for idx, video_path in enumerate(video_paths):
            print(f"\n  --- Analyzing Video {idx+1}/{len(video_paths)}: {os.path.basename(video_path)} ---")
            # --- 準備單個影片的提示 ---
            eval_prompt_for_single_vid = f"""請根據以下具體標準/評分準則，評估**這個影片**和相關文字輸出：
**評估標準/準則:**
{specific_criteria}

**相關文字輸出/摘要 (供參考):**
```
{video_related_text}
```

**當前影片路徑:** {video_path}

**任務:**
請提供對**這個影片**的詳細評估，說明其是否符合上述標準/準則。
你的評估應包含：
1.  **對此影片的整體評估:** 在回應的 **最開頭** 明確指出 "整體評估：通過" 或 "整體評估：失敗"。 (工具層面評估)
2.  **詳細回饋:** 針對標準/準則進行分析，解釋評估結果的原因。
3.  **改進建議:** (可選) 如果評估為 "失敗"，請提供具體的改進建議。
4.  **時間戳記分析:** 指出影片中關鍵時刻的時間戳記，格式為 MM:SS。

請以繁體中文回應。
"""
            single_video_feedback = f"影片 '{os.path.basename(video_path)}' 分析失敗。" # 預設失敗反饋
            single_video_assessment = "Fail" # 預設失敗

            try:
                print(f"    - 調用 video_recognition 工具...")
                # --- <<< 關鍵修改：將參數打包成字典傳遞給 .run() >>> ---
                tool_input_dict = {
                    "video_path": video_path,
                    "prompt": eval_prompt_for_single_vid
                }
                result = video_recognition.run(tool_input=tool_input_dict)
                # --- <<< 結束修改 >>> ---

                if not result or not isinstance(result, str):
                    err_msg = f"影片工具對 '{os.path.basename(video_path)}' 返回無效結果: {result}"
                    print(f"    Eval Subgraph Error ({node_name}): {err_msg}")
                    single_video_feedback = f"錯誤: {err_msg}"
                    subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ")
                    any_video_failed = True # 標記有影片處理失敗
                elif result.startswith("錯誤："): # 檢查工具返回的錯誤訊息
                    err_msg = f"影片工具對 '{os.path.basename(video_path)}' 返回錯誤: {result}"
                    print(f"    Eval Subgraph Error ({node_name}): {err_msg}")
                    single_video_feedback = result # 使用工具返回的錯誤信息
                    subgraph_error = (subgraph_error + f"; {result}").strip("; ")
                    any_video_failed = True # 標記有影片處理失敗
                else:
                    single_video_feedback = result # 儲存完整的工具輸出
                    
                    # --- <<< 添加：儲存結構化結果 >>> ---
                    individual_video_results.append({
                        "path": video_path,
                        "filename": os.path.basename(video_path),
                        "assessment": single_video_assessment,  # 單個影片的通過/失敗狀態
                        "raw_output": result  # 完整工具輸出
                    })
                    # --- <<< 結束添加 >>> ---
                    
                    # 僅為 EvaAgent 解析 Pass/Fail
                    if selected_agent == "EvaAgent":
                        if "整體評估：通過" in result or "overall assessment: pass" in result.lower():
                            single_video_assessment = "Pass"
                        elif "整體評估：失敗" in result or "overall assessment: fail" in result.lower():
                            single_video_assessment = "Fail"
                        else:
                            single_video_assessment = "Fail" # 嚴格模式：未明確說通過則視為失敗
                            print(f"    - Warning (EvaAgent): 無法從影片工具輸出中明確解析 Pass/Fail。設為 Fail。")
                        print(f"    - 單個影片評估 (EvaAgent): {single_video_assessment}")
                        if single_video_assessment == "Fail":
                            any_video_failed = True # 標記有影片評估失敗
                    else: # Special/Final Agent
                        single_video_assessment = "Not Applicable"
                        print(f"    - 單個影片原始輸出已儲存 (Special/Final Agent)")

            except Exception as e: # Tool call error for single video
                # --- <<< 修改：更清晰地打印錯誤來源 >>> ---
                err_msg = f"調用 video_recognition.run() 處理 '{os.path.basename(video_path)}' 時捕獲到異常: {e}"
                print(f"    Eval Subgraph Error ({node_name}): {err_msg}")
                single_video_feedback = f"錯誤: {err_msg}"
                subgraph_error = (subgraph_error + f"; Tool Exception: {err_msg}").strip("; ") # Add marker
                any_video_failed = True # 標記有影片處理失敗
                traceback.print_exc() # 打印錯誤追蹤
                # --- <<< 結束修改 >>> ---

            # 累積每個影片的反饋
            accumulated_feedback.append(f"--- 影片: {os.path.basename(video_path)} (評估: {single_video_assessment}) ---\n{single_video_feedback}\n")
            # --- 結束單個影片的 try-except ---
        # --- <<< 結束循環 >>> ---

        # --- 根據循環結果設置最終評估值 ---
        if selected_agent == "EvaAgent":
            overall_video_assessment = "Fail" if any_video_failed else "Pass"
        else:
            overall_video_assessment = "Not Applicable" # Special/Final Agent 不基於 Pass/Fail

    # --- <<< 添加：當有多個影片時進行LLM比較 >>> ---
    if has_multiple_videos and selected_agent in ["SpecialEvaAgent"] and len(individual_video_results) > 1:
        print(f"  - Multiple videos detected ({len(video_paths)}) for {selected_agent}. Running comparison LLM...")
        
        # 獲取LLM配置
        runtime_config = config.get("configurable", {})
        ea_llm_config = runtime_config.get("ea_llm", {})
        llm = initialize_llm(ea_llm_config)
        llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
        
        # 構建比較提示
        comparison_prompt = f"""
您需要比較並排序以下多個影片的評估結果，作為特殊評估（SpecialEvaAgent）的一部分。

**評估標準/準則:**
{specific_criteria}

**每個影片的評估結果:**
{chr(10).join([f"--- 影片 {i+1} ({res['filename']}) ---{chr(10)}{res['raw_output']}{chr(10)}" 
               for i, res in enumerate(individual_video_results)])}

**任務:**
1. 仔細分析每個影片的評估結果
2. 根據上述評估標準/準則，對所有影片進行排序（從最好到最差）
3. 清楚識別哪個影片是最佳選擇，並解釋原因
4. 提供最佳影片的檔案名

請以JSON格式回覆，包含以下欄位：
- ranking: 影片排名（從最佳到最差的檔案名稱列表）
- best_video: 最佳影片的檔案名
- comparison_reasoning: 詳細的比較分析和排名理由
- improvement_suggestions: 對最佳影片的改進建議（如果有）

回應語言: {llm_output_language}
"""
        
        try:
            # 調用LLM進行比較
            comparison_response = llm.invoke(comparison_prompt)
            comparison_text = comparison_response.content.strip()
            
            # 嘗試解析JSON
            comparison_result = {}
            try:
                if comparison_text.startswith("```json"):
                    comparison_text = comparison_text[7:-3].strip()
                elif comparison_text.startswith("```"):
                    comparison_text = comparison_text[3:-3].strip()
                    
                comparison_result = json.loads(comparison_text)
                print(f"  - Successfully parsed comparison result: {list(comparison_result.keys())}")
                
                # 更新評估結果
                current_task["evaluation"]["video_comparison"] = comparison_result
                current_task["evaluation"]["selected_option_identifier"] = comparison_result.get("best_video", "")
                
                # 加入比較結果到累積反饋
                additional_feedback = f"\n\n--- 多影片比較結果 ---\n排名: {comparison_result.get('ranking', [])}\n最佳影片: {comparison_result.get('best_video', 'N/A')}\n分析理由: {comparison_result.get('comparison_reasoning', 'N/A')}"
                accumulated_feedback.append(additional_feedback)
                
            except json.JSONDecodeError:
                print(f"  - Warning: Could not parse LLM comparison result as JSON. Using raw text.")
                current_task["evaluation"]["video_comparison_text"] = comparison_text
                accumulated_feedback.append(f"\n\n--- 多影片比較結果 ---\n{comparison_text}")
                
        except Exception as comp_e:
            print(f"  - Error during video comparison: {comp_e}")
            accumulated_feedback.append(f"\n\n--- 多影片比較錯誤 ---\n{comp_e}")
    # --- <<< 結束添加 >>> ---

    # --- Update TaskState ---
    final_feedback_str = "\n".join(accumulated_feedback) if accumulated_feedback else "沒有生成影片評估反饋。"
    current_task["evaluation"]["video_tool_assessment"] = overall_video_assessment
    feedback_details = f"影片工具總體評估 (基於所有影片): {overall_video_assessment}\n{final_feedback_str}"
    _append_feedback(current_task, feedback_details, node_name)
    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error

    # --- Set Final Status for this branch ---
    final_assessment_for_path = overall_video_assessment
    # 如果整體影片評估通過（僅對 EvaAgent），檢查圖像評估
    if overall_video_assessment == "Pass" and selected_agent == "EvaAgent":
        image_assessment = current_task.get("evaluation", {}).get("image_tool_assessment")
        if image_assessment == "Fail": # Image failed previously
            final_assessment_for_path = "Fail" # 整體視為失敗
    elif selected_agent == "EvaAgent" and overall_video_assessment != "Pass":
        final_assessment_for_path = "Fail" # 如果 EvaAgent 的影片評估不是明確 Pass，則整體失敗

    current_task["evaluation"]["combined_visual_assessment"] = final_assessment_for_path
    current_task["evaluation"]["assessment"] = final_assessment_for_path # Update main assessment

    print(f"  - Combined Visual Assessment: {final_assessment_for_path}")
    print(f"  - Calling final status update...")
    _update_eval_status_at_end(current_task, f"{node_name} (Combined Visual)")
    print(f"  - Final status set to: {current_task['status']}")

    tasks[current_idx] = current_task
    return {"tasks": tasks}


def route_after_image_eval(state: WorkflowState) -> Literal["evaluate_with_video_agent", "finished"]:
    """
    Checks if video evaluation is needed after image evaluation.
    Also handles setting final status if image evaluation was the last step and passed.
    """
    node_name = "Route After Image Eval"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    tasks = state["tasks"] # Read directly from state for latest status
    if not (0 <= current_idx < len(tasks)):
        print(f"  - Error: Invalid task index {current_idx}. Routing to finished.")
        return "finished"

    current_task = tasks[current_idx] # Get the task potentially modified by image_eval_node

    # Check if image evaluation step itself failed (either internally or tool assessment fail)
    if current_task.get("status") == "failed":
        print(f"  - Routing Decision: Image evaluation failed or previous error. Ending evaluation branch.")
        # The status is already 'failed', END connection handles it.
        return "finished"

    # Image eval step completed without setting status to failed. Check if video exists.
    prepared_inputs = current_task.get("task_inputs", {})
    # Check the *original* prepared inputs for video need
    needs_video_eval = bool(prepared_inputs.get("evaluation_target_video_paths")) or \
                       bool(prepared_inputs.get("evaluation_target_key_video_paths"))

    if needs_video_eval:
        print(f"  - Routing Decision: Image evaluation succeeded, Video output exists. Routing to Video Evaluation.")
        return "evaluate_with_video_agent"
    else:
        print(f"  - Routing Decision: Image evaluation succeeded, No video output found. Ending evaluation branch.")
        # Since only image eval was needed and it didn't fail, set final status based on its assessment.
        # The assessment logic resides within the helper function.
        # Get the image tool assessment to determine final state.
        image_assessment = current_task.get("evaluation", {}).get("image_tool_assessment", "Fail")
        # Create a temporary dict to pass to the status update helper
        temp_eval_state_for_status = {"assessment": image_assessment}
        temp_task_for_status = current_task.copy()
        temp_task_for_status["evaluation"] = temp_eval_state_for_status
        # Call helper based *only* on the image assessment outcome
        _update_eval_status_at_end(temp_task_for_status, f"{node_name} (Image Only)")
        # Apply the determined status back to the state
        # IMPORTANT: Modify state directly here as this is the final step for this path
        state['tasks'][current_idx]['status'] = temp_task_for_status['status']
        state['tasks'][current_idx]['error_log'] = temp_task_for_status.get('error_log', state['tasks'][current_idx].get('error_log'))
        print(f"  - Final status set to: {state['tasks'][current_idx]['status']}")

        # tasks[current_idx] = current_task # State is modified directly above, no need to reassign to local var
        # return {"tasks": tasks} # <<< REMOVE THIS LINE >>>
        return "finished" # <<< ADD THIS LINE >>>

# --- MODIFIED: Condition Function after Prep ---
def route_after_eval_prep(state: WorkflowState) -> Literal["gather_criteria_sources", "generate_specific_criteria", "finished"]:
    """
    Routes to criteria gathering or directly to generation based on the 'needs_detailed_criteria' flag,
    or ends if prep failed.
    """
    node_name = "Route After Prep"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    tasks = state["tasks"]

    # Check index validity and if the current task (prep task) failed
    if not (0 <= current_idx < len(tasks)) or tasks[current_idx].get("status") == "failed":
        print(f"  - Routing Decision: Evaluation input preparation failed or invalid index. Routing to finished.")
        return "finished"

    # Check the flag set by prepare_evaluation_inputs_node
    current_task = tasks[current_idx]
    needs_detailed = current_task.get("task_inputs", {}).get("needs_detailed_criteria", False) # Default to False

    if needs_detailed:
        print(f"  - Routing Decision: 'needs_detailed_criteria' is True. Proceeding to Gather Criteria Sources.")
        return "gather_criteria_sources"
    else:
        print(f"  - Routing Decision: 'needs_detailed_criteria' is False. Skipping gathering and proceeding to Generate Specific Criteria.")
        # Skip gathering, directly provide default context to generate_specific_criteria node if needed later
        if "evaluation" not in current_task: current_task["evaluation"] = {} # Ensure eval dict exists
        current_task["evaluation"]["criteria_sources"] = {"rag": "Not gathered.", "search": "Not gathered."} # Indicate skipping
        # Update the task state directly here before returning the routing decision
        state["tasks"][current_idx] = current_task
        return "generate_specific_criteria"

# =============================================================================
# Build and Compile Evaluation Subgraph
# =============================================================================
evaluation_subgraph_builder = StateGraph(WorkflowState)

# --- Define Nodes ---
evaluation_subgraph_builder.add_node("prepare_evaluation_inputs", prepare_evaluation_inputs_node)
evaluation_subgraph_builder.add_node("gather_criteria_sources", gather_criteria_sources_node)
evaluation_subgraph_builder.add_node("generate_specific_criteria", generate_specific_criteria_node)
evaluation_subgraph_builder.add_node("evaluate_with_llm_agent", evaluate_with_llm_node)
evaluation_subgraph_builder.add_node("evaluate_with_image_agent", evaluate_with_image_node)
evaluation_subgraph_builder.add_node("evaluate_with_video_agent", evaluate_with_video_node)

# --- Set Entry Point ---
evaluation_subgraph_builder.set_entry_point("prepare_evaluation_inputs")

# --- Define Edges ---

# --- MODIFIED: Conditional Edge after Preparation Step ---
evaluation_subgraph_builder.add_conditional_edges(
    "prepare_evaluation_inputs",
    route_after_eval_prep, # Use the updated routing function
    {
        "gather_criteria_sources": "gather_criteria_sources",
        "generate_specific_criteria": "generate_specific_criteria", # New direct route
        "finished": END # End subgraph if preparation failed
    }
)

# Linear Flow for Criteria (remains the same, gather leads to generate)
evaluation_subgraph_builder.add_edge("gather_criteria_sources", "generate_specific_criteria")

# Conditional Edge from Criteria Generation to Appropriate Evaluation Tool (remains the same)
evaluation_subgraph_builder.add_conditional_edges(
    "generate_specific_criteria",
    route_to_evaluation_tool_node,
    {
        "evaluate_with_llm_agent": "evaluate_with_llm_agent",
        "evaluate_with_image_agent": "evaluate_with_image_agent",
        "evaluate_with_video_agent": "evaluate_with_video_agent",
        "finished": END
    }
)

# Terminal Edges for LLM and Video Evaluation Nodes (remains the same)
evaluation_subgraph_builder.add_edge("evaluate_with_llm_agent", END)
evaluation_subgraph_builder.add_edge("evaluate_with_video_agent", END)

# Conditional Edge after Image Evaluation (to Video or End) (remains the same)
evaluation_subgraph_builder.add_conditional_edges(
    "evaluate_with_image_agent",
    route_after_image_eval,
    {
        "evaluate_with_video_agent": "evaluate_with_video_agent",
        "finished": END
    }
)

# Compile the subgraph (remains the same)
evaluation_teams = evaluation_subgraph_builder.compile()
evaluation_teams.name = "EvaluationSubgraph"
print("Evaluation Subgraph refactored and compiled successfully as 'evaluation_teams'.")

# =============================================================================
# End of File
# =============================================================================
