# =============================================================================
# 1. Imports
# =============================================================================
import os
import uuid
import json
import base64
import shutil # NEW IMPORT for file copying
import cv2 # 新增或確保 cv2 在頂部被引入
from typing import Dict, List, Any, Annotated, Literal, Union, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
import traceback # Added for error printing
from jinja2 import Environment, FileSystemLoader # NEW IMPORT
from docx import Document as DocxDocument # NEW IMPORT for Word document creation
from docx.shared import Inches, Pt # NEW IMPORT for sizing
from docx.enum.text import WD_ALIGN_PARAGRAPH # NEW IMPORT for alignment

# LangChain/LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import necessary prompt classes

# Custom Imports
# --- Core Tools ---
from src.tools.ARCH_rag_tool import ARCH_rag_tool
from src.tools.img_recognition import img_recognition
from src.tools.video_recognition import video_recognition
from src.tools.gemini_image_generation_tool import generate_gemini_image
from src.tools.gemini_search_tool import perform_grounded_search
# --- ComfyUI Tools ---
from src.tools.model_render_image import model_render_image # Ensure this is the one being imported
from src.tools.generate_3D import generate_3D
# --- MODIFIED: Remove simulate_future_image import ---
# from src.tools.simulate_future_image import simulate_future_image

# --- MODIFIED ---
# Import ConfigSchema, but remove direct dependency on FullConfig, AgentConfig etc. for runtime
from src.configuration import ConfigManager, ModelConfig, MemoryConfig, initialize_llm, ConfigSchema
# --- <<< 新增：從 state 導入 >>> ---
from src.state import WorkflowState, TaskState
# --- <<< 結束新增 >>> ---

# --- <<< 新增：從子圖檔案導入 >>> ---
from src.AssignTeamsSubgraph import assign_teams
from src.EvaluationSubgraph import evaluation_teams 
# --- <<< 結束新增 >>> ---

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
# 4. LLM 與記憶體初始化
# =============================================================================
# --- MODIFIED: Use MEMORY_DIR and _full_static_config from common_definitions ---
ltm_embedding_config = _full_static_config.memory.long_term_memory
if ltm_embedding_config.provider == "openai":
     embeddings = OpenAIEmbeddings(model=ltm_embedding_config.model_name)
else:
     print(f"Warning: Unsupported LTM embedding provider '{ltm_embedding_config.provider}'. Defaulting to OpenAI.")
     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=MEMORY_DIR, # Use imported MEMORY_DIR
    embedding_function=embeddings,
    collection_name="workflow_memory"
)
ltm_memory_key = "ltm_context"
ltm_input_key = "input"

print(f"Long-term memory vector store initialized. Storing in: {MEMORY_DIR}")
print(f"LTM Retriever 'k' value will be determined at runtime from configuration.")

# =============================================================================
# 5. 狀態定義
# =============================================================================
# --- REMOVED: Definitions moved to src/state.py ---

# =============================================================================
# 6. Agent 類別定義 (PM, Eva, QA)
# =============================================================================
class ProcessManagement:
    # --- MODIFIED __init__ ---
    def __init__(self, long_term_memory_vectorstore=vectorstore): # Assuming 'vectorstore' is globally available or passed
        # Remove LLM and config args if not needed directly for saving
        # self.llm = llm
        # self.config = config
        # self.prompts = config.get("prompts", {}) # Keep prompts if failure analysis still uses them
        # self.max_retries = config.get("parameters", {}).get("max_retries", 3)
        # self.llm_output_language = global_config.get("llm_output_language", "繁體中文")

        # --- Store the vectorstore ---
        if long_term_memory_vectorstore is None:
            raise ValueError("ProcessManagement requires a valid vectorstore instance.")
        self.vectorstore = long_term_memory_vectorstore
        print("ProcessManagement initialized with vectorstore.")

        # Load prompts if needed for failure analysis/interrupt commands
        # You might load the config manager here if necessary
        config_manager = ConfigManager() # Or get instance if singleton
        pm_config = config_manager.get_agent_config("process_management")
        self.prompts = {name: cfg.template for name, cfg in pm_config.prompts.items()} if pm_config else {}
        self.llm_output_language = config_manager.get_workflow_config().llm_output_language

        # Get LLM for potential failure analysis / non-QA interrupts
        self.llm = initialize_llm(pm_config.llm.model_dump()) if pm_config else None # Initialize LLM if needed

    def _create_task_summary(self, task: TaskState) -> str:
        """Creates a textual summary of a task for LTM."""
        # ...(Existing summary logic - seems fine)...
        summary_lines = [
            f"Task ID: {task.get('task_id', 'N/A')}",
            f"Status: {task.get('status', 'N/A')}",
            f"Objective: {task.get('task_objective', 'N/A')}",
            f"Description: {task.get('description', 'N/A')}",
            f"Agent: {task.get('selected_agent', 'N/A')}",
            f"Requires Eval: {task.get('requires_evaluation', 'N/A')}",
            f"Retry Count: {task.get('retry_count', 'N/A')}",
        ]
        # Include summarized inputs if they exist and are simple enough
        if task.get("task_inputs"):
             try:
                  # Limit size/depth of inputs in summary
                  inputs_str = json.dumps(task["task_inputs"], ensure_ascii=False, default=lambda o: "<non-serializable>", indent=None)[:200]
                  summary_lines.append(f"Inputs Summary: {inputs_str}")
             except Exception:
                  summary_lines.append("Inputs Summary: <error serializing>")

        # Include summarized outputs if they exist
        if task.get("outputs"):
             try:
                  outputs_str = json.dumps(task["outputs"], ensure_ascii=False, default=lambda o: "<non-serializable>", indent=None)[:300]
                  summary_lines.append(f"Outputs Summary: {outputs_str}")
             except Exception:
                  summary_lines.append("Outputs Summary: <error serializing>")

        # Include file info
        if task.get("output_files"):
            file_info = [f"{f.get('filename', 'N/A')} ({f.get('type', 'N/A')})" for f in task["output_files"]]
            summary_lines.append(f"Output Files: {', '.join(file_info)}")

        # Include evaluation summary if present
        if task.get("evaluation") and task["evaluation"].get("assessment"):
            summary_lines.append(f"Evaluation: {task['evaluation']['assessment']} - {task['evaluation'].get('feedback', '')[:100]}...")

        # Include error log if present
        if task.get("error_log"):
            summary_lines.append(f"Error Log: {task['error_log'][:150]}...") # Truncate error log

        # Include feedback log if present
        if task.get("feedback_log"):
             summary_lines.append(f"Feedback Log: {task['feedback_log'][:150]}...") # Truncate feedback log

        return "\n".join(summary_lines)

    # --- Method to save task summary to LTM ---
    async def _save_task_to_ltm(self, task: TaskState):
        """Generates a summary and saves a task state to the vectorstore."""
        task_id = task.get('task_id', 'N/A')
        status = task.get('status', 'N/A')
        agent_name = task.get('selected_agent', 'N/A') # Get agent name for logging

        if task_id == 'N/A':
            print("PM LTM Warning: Cannot save task without ID.")
            return

        print(f"PM LTM: Preparing to save task {task_id} (Agent: {agent_name}, Status: '{status}')...")

        # --- <<< ADDED LOGGING for outputs >>> ---
        task_outputs_to_log = task.get('outputs')
        if task_outputs_to_log:
            try:
                # Log a preview, be careful with large outputs
                outputs_preview = json.dumps(task_outputs_to_log, ensure_ascii=False, default=str)[:500]
                print(f"  - Task {task_id} Outputs being saved to LTM (Preview): {outputs_preview}")
                # Specifically check Pinterest output structure if agent matches
                if agent_name == "PinterestMCPCoordinator":
                     pinterest_paths = task_outputs_to_log.get("saved_image_paths")
                     if isinstance(pinterest_paths, list):
                         print(f"    - Pinterest paths found in outputs: Count={len(pinterest_paths)}, First few: {pinterest_paths[:3]}")
                     else:
                         print(f"    - Pinterest paths ('saved_image_paths') key not found or not a list in outputs.")
            except Exception as log_e:
                print(f"  - Warning: Could not serialize/preview outputs for LTM logging: {log_e}")
        else:
            print(f"  - Task {task_id} has no 'outputs' field to save to LTM.")
        # --- <<< END ADDED LOGGING >>> ---

        summary = self._create_task_summary(task)
        # print(f"PM LTM Summary for {task_id}:\n{summary}\n--------------------") # Optional: Debug print summary

        # Create LangChain Document object
        # Metadata helps retrieval filtering later if needed
        metadata = {
            "task_id": task_id,
            "status": status,
            "agent": agent_name,
            "requires_evaluation": task.get("requires_evaluation", False),
            "timestamp": datetime.now().isoformat() # Add timestamp
        }
        doc = Document(page_content=summary, metadata=metadata)

        try:
            await self.vectorstore.aadd_documents([doc]) # Use async add
            print(f"PM LTM: Successfully saved task {task_id} (Status: {status}) to LTM.")
        except Exception as e:
            print(f"PM LTM Error: Failed to save task {task_id} to vectorstore: {e}")

    # --- MODIFIED run method ---
    async def run(self, state: WorkflowState, config: RunnableConfig) -> WorkflowState:
        print("\n=== Process Management Running ===")
        output_state = state.copy()
        output_state["interrupt_result"] = None # Ensure cleared initially

        # ... (打印 initial state 的日誌 - 不變) ...
        initial_tasks = state.get("tasks", [])
        initial_idx = state.get("current_task_index", 0)
        print(f"  PM Start: Received initial current_task_index: {initial_idx}")
        print(f"  PM Start: Received initial task count: {len(initial_tasks)}")
        if 0 <= initial_idx < len(initial_tasks):
             print(f"  PM Start: Initial task at index {initial_idx} ID: '{initial_tasks[initial_idx].get('task_id')}', Status: '{initial_tasks[initial_idx].get('status')}'")
        else:
             print(f"  PM Start: Initial index {initial_idx} is out of bounds.")


        tasks = state.get("tasks", [])
        current_idx = state.get("current_task_index", 0)
        interrupt_input = state.get("interrupt_input") # Use original state's input

        # --- 1. Interrupt Processing ---
        if interrupt_input:
            print(f"PM: Interrupt detected: '{interrupt_input[:100]}...'. Invoking LLM for analysis.")
            # ... (LLM interrupt analysis logic - remains the same) ...
            summarized_tasks = [{"task_id": t.get("task_id", "N/A"), "objective": t.get("task_objective", "N/A")} for t in tasks]
            tasks_json_for_interrupt = json.dumps(summarized_tasks, ensure_ascii=False)
            current_task_for_interrupt = tasks[current_idx] if 0 <= current_idx < len(tasks) else None
            
            # --- <<< 修改：過濾當前任務以生成 JSON >>> ---
            current_task_filtered = {}
            if current_task_for_interrupt:
                current_task_filtered = current_task_for_interrupt.copy()
                # 從 task['outputs'] 中移除不需要的鍵
                if "outputs" in current_task_filtered and isinstance(current_task_filtered["outputs"], dict):
                    current_task_outputs_filtered = current_task_filtered["outputs"].copy()
                    current_task_outputs_filtered.pop("mcp_internal_messages", None)
                    current_task_outputs_filtered.pop("grounding_sources", None)
                    current_task_outputs_filtered.pop("search_suggestions", None) 
                    current_task_filtered["outputs"] = current_task_outputs_filtered 
                
                # --- NEW: Filter base64_data from output_files for interrupt prompt ---
                if "output_files" in current_task_filtered and isinstance(current_task_filtered["output_files"], list):
                    filtered_output_files = []
                    for file_dict in current_task_filtered["output_files"]:
                        if isinstance(file_dict, dict):
                            file_copy = file_dict.copy()
                            file_copy.pop("base64_data", None)
                            file_copy.pop("fileName", None)
                            filtered_output_files.append(file_copy)
                        else:
                            filtered_output_files.append(file_dict) # Keep non-dict items as is
                    current_task_filtered["output_files"] = filtered_output_files
                # --- END NEW ---

                # 可以考慮也過濾 task_inputs 如果需要
                # if "task_inputs" in current_task_filtered and isinstance(current_task_filtered["task_inputs"], dict): ...

            current_task_json_for_interrupt = json.dumps(current_task_filtered, ensure_ascii=False, default=lambda o: '<not serializable>') if current_task_filtered else "{}"
            # --- <<< 結束修改 >>> ---

            interrupt_prompt_input = {
                "user_input": state.get("user_input", ""), "interrupt_input": interrupt_input,
                "current_task_index": current_idx, "tasks_json": tasks_json_for_interrupt,
                "current_task_json": current_task_json_for_interrupt, "llm_output_language": self.llm_output_language,
            }
            try:
                if not self.prompts.get("process_interrupt"): raise ValueError("process_interrupt prompt missing")
                if not self.llm: raise ValueError("LLM for PM not initialized.")

                interrupt_chain = self.llm | StrOutputParser()
                interrupt_content = await interrupt_chain.ainvoke(self.prompts["process_interrupt"].format(**interrupt_prompt_input))

                if interrupt_content.startswith("```json"): interrupt_content = interrupt_content[7:-3].strip()
                elif interrupt_content.startswith("```"): interrupt_content = interrupt_content[3:-3].strip()
                interrupt_result = json.loads(interrupt_content)
                action = interrupt_result.get("action", "PROCEED")
                print(f"PM: Parsed LLM interrupt analysis action: {action}, Result: {interrupt_result}") # Log parsed result

                # --- <<< 修改：處理 Command Actions 並控制返回邏輯 >>> ---
                processed_interrupt = False # Flag to track if interrupt caused an immediate return

                if action == "CONVERSATION":
                    print("PM: LLM analysis resulted in CONVERSATION. Setting phase to 'qa'.")
                    output_state["current_phase"] = "qa"
                    output_state["interrupt_result"] = interrupt_result # Keep result for QA entry check
                    # Keep interrupt_input for QA Agent? No, QA Agent reads from context. Clear it.
                    output_state["interrupt_input"] = None
                    print(f"PM (CONVERSATION): Returning state for QA. Index remains {current_idx}.")
                    processed_interrupt = True
                    return output_state # CONVERSATION requires immediate return to route to QA

                elif action == "REPLACE_TASKS":
                    print("PM: Processing REPLACE_TASKS command.")
                    new_tasks_data = interrupt_result.get("new_tasks_list")
                    if isinstance(new_tasks_data, list):
                        created_tasks = []
                        for i, task_data in enumerate(new_tasks_data):
                            if isinstance(task_data, dict):
                                task_id = str(uuid.uuid4())
                                task = TaskState(
                                    task_id=task_id, status="pending",
                                    task_objective=task_data.get("task_objective", f"Replaced Objective {i+1}"),
                                    description=task_data.get("description", f"Replaced Task {i+1}"),
                                    selected_agent=task_data.get("selected_agent", "LLMTaskAgent"),
                                    task_inputs=task_data.get("inputs", {}),
                                    outputs={}, output_files=[], evaluation={}, error_log=None, feedback_log=None,
                                    requires_evaluation=task_data.get("requires_evaluation", False), retry_count=0
                                )
                                if not task["selected_agent"]:
                                    print(f"PM Warning (REPLACE): Task {i} lacks 'selected_agent'. Skipping.")
                                    continue
                                created_tasks.append(task)
                            else: print(f"PM Warning (REPLACE): Invalid item in new_tasks_list at index {i}.")

                        if created_tasks:
                            original_tasks = state.get("tasks", [])
                            interrupt_idx = state.get("current_task_index", 0)
                            completed_prefix = [t for idx, t in enumerate(original_tasks) if idx < interrupt_idx and t.get("status") == "completed"]
                            print(f"PM (REPLACE): Preserving {len(completed_prefix)} completed tasks before index {interrupt_idx}.")
                            output_state["tasks"] = completed_prefix + created_tasks
                            output_state["current_task_index"] = len(completed_prefix)
                            output_state["current_task"] = created_tasks[0].copy() if created_tasks else None
                            print(f"PM (REPLACE): State updated. New index: {output_state['current_task_index']}. New task count: {len(output_state['tasks'])}.")
                        else:
                            print("PM Warning (REPLACE): No valid tasks created. Proceeding.")
                            # No need to explicitly set action = "PROCEED", will fall through
                    else:
                        print("PM Warning (REPLACE): 'new_tasks_list' missing or invalid. Proceeding.")
                        # No need to explicitly set action = "PROCEED", will fall through

                    # --- Common logic for REPLACE/INSERT ---
                    output_state["interrupt_input"] = None # Clear after processing
                    output_state["interrupt_result"] = None
                    processed_interrupt = True
                    print(f"PM (REPLACE): Returning updated state to router after replacing tasks.")
                    return output_state # REPLACE requires immediate return

                elif action == "INSERT_TASKS":
                    print("PM: Processing INSERT_TASKS command.")
                    insert_tasks_data = interrupt_result.get("insert_tasks_list")
                    if isinstance(insert_tasks_data, list):
                        inserted_tasks = []
                        for i, task_data in enumerate(insert_tasks_data):
                            if isinstance(task_data, dict):
                                task_id = str(uuid.uuid4())
                                task = TaskState(
                                    task_id=task_id, status="pending", # Ensure status is pending
                                    task_objective=task_data.get("task_objective", f"Inserted Objective {i+1}"),
                                    description=task_data.get("description", f"Inserted Task {i+1}"),
                                    selected_agent=task_data.get("selected_agent", "LLMTaskAgent"),
                                    task_inputs=task_data.get("inputs", {}),
                                    outputs={}, output_files=[], evaluation={}, error_log=None, feedback_log=None,
                                    requires_evaluation=task_data.get("requires_evaluation", False), retry_count=0
                                )
                                if not task["selected_agent"]:
                                    print(f"PM Warning (INSERT): Task {i} lacks 'selected_agent'. Skipping.")
                                    continue
                                inserted_tasks.append(task)
                                print(f"  - Created inserted task {i+1} (ID: {task_id})")
                            else: print(f"PM Warning (INSERT): Invalid item in insert_tasks_list at index {i}.")

                        if inserted_tasks:
                            original_tasks_before_insert = state.get("tasks", [])
                            insert_after_idx = state.get("current_task_index", 0)
                            valid_insert_point = insert_after_idx + 1
                            print(f"PM (INSERT): Original index before insert: {insert_after_idx}")
                            print(f"PM (INSERT): Inserting {len(inserted_tasks)} tasks at index {valid_insert_point}.")

                            current_tasks_in_output_state = output_state.get("tasks", original_tasks_before_insert)
                            new_sequence = current_tasks_in_output_state[:valid_insert_point] + inserted_tasks + current_tasks_in_output_state[valid_insert_point:]

                            output_state["tasks"] = new_sequence
                            output_state["current_task_index"] = valid_insert_point
                            output_state["current_task"] = inserted_tasks[0].copy()
                            print(f"PM (INSERT): State updated. New index: {output_state['current_task_index']}. New task count: {len(new_sequence)}.")
                            print(f"PM (INSERT): Current task set to ID: {output_state['current_task']['task_id']}")
                        else:
                            print("PM Warning (INSERT): No valid tasks created from LLM response. Proceeding.")
                            # No need to explicitly set action = "PROCEED", will fall through
                    else:
                        print("PM Warning (INSERT): 'insert_tasks_list' missing or invalid in LLM response. Proceeding.")
                        # No need to explicitly set action = "PROCEED", will fall through

                    # --- Common logic for REPLACE/INSERT ---
                    output_state["interrupt_input"] = None # Clear after processing
                    output_state["interrupt_result"] = None
                    processed_interrupt = True
                    print(f"PM (INSERT): Returning updated state to router after inserting tasks.")
                    return output_state # INSERT requires immediate return

                # --- Handle PROCEED (or fallback): Only clear flags, DO NOT return ---
                elif action == "PROCEED":
                    print("PM: PROCEED command received or fallback. Clearing interrupt flags and continuing.")
                    output_state["interrupt_input"] = None
                    output_state["interrupt_result"] = None
                    # --- DO NOT RETURN HERE ---
                    processed_interrupt = False # Indicate we should continue to task processing loop

                # --- Cleanup after handling non-returning actions (like PROCEED) ---
                # This part is now only reached if action was PROCEED
                if not processed_interrupt:
                    print(f"PM: Processed non-returning interrupt command '{action}'. Proceeding to task check loop.")
                    # Flags already cleared above for PROCEED case.

                # --- Catch potential errors during interrupt processing ---
                # Removed the explicit `except Exception as e:` block that was wrapping the action handling,
                # as individual actions returning should handle their own errors or the main function try-except will catch it.
                # Ensure the try-except around the LLM call itself remains.

            except Exception as e:
                print(f"PM Error processing interrupt command: {e}")
                traceback.print_exc()
                output_state["interrupt_input"] = None # Clear on error too
                output_state["interrupt_result"] = None
                # Consider returning state here too, or let it flow to task processing?
                # Let's allow it to flow to task processing for now, which might handle failure.
                print(f"PM: Error during interrupt processing. Allowing flow to task status check.")


        # --- 2. Task Processing Logic ---
        # This part is now reached if:
        # a) There was no interrupt_input initially.
        # b) The interrupt_input resulted in a 'PROCEED' action.
        tasks = output_state.get("tasks", []) # Use potentially updated tasks (though PROCEED doesn't modify them)
        current_idx = output_state.get("current_task_index", 0) # Use potentially updated index (though PROCEED doesn't modify it)

        # --- Initial Workflow Creation (if tasks is empty and no interrupt happened) ---
        if not tasks and not state.get("interrupt_input"): # Only create if no interrupt was processed
            # ... (Existing workflow creation logic) ...
            print("PM: Task list is empty. Generating initial workflow...")
            try:
                if not self.prompts.get("create_workflow"): raise ValueError("create_workflow prompt missing")
                if not self.llm: raise ValueError("LLM for PM not initialized.") # Check LLM
                create_chain = self.llm | StrOutputParser()
                create_input = {"user_input": state.get("user_input", ""), "llm_output_language": self.llm_output_language}
                plan_content = await create_chain.ainvoke(self.prompts["create_workflow"].format(**create_input))

                if plan_content.startswith("```json"): plan_content = plan_content[7:-3].strip()
                elif plan_content.startswith("```"): plan_content = plan_content[3:-3].strip()
                plan_data = json.loads(plan_content)

                created_tasks = []
                if isinstance(plan_data, list):
                    for i, task_data in enumerate(plan_data):
                        if isinstance(task_data, dict):
                            task_id = str(uuid.uuid4())
                            task = TaskState(
                                task_id=task_id, status="pending",
                                task_objective=task_data.get("task_objective", f"Planned Objective {i+1}"),
                                description=task_data.get("description", f"Planned Task {i+1}"),
                                selected_agent=task_data.get("selected_agent"),
                                task_inputs=task_data.get("inputs", {}),
                                outputs={}, output_files=[], evaluation={}, error_log=None, feedback_log=None,
                                requires_evaluation=task_data.get("requires_evaluation", False), retry_count=0
                            )
                            if not task["selected_agent"]:
                                print(f"PM Warning (Create): Task {i} lacks 'selected_agent'. Skipping.")
                                continue
                            created_tasks.append(task)
                        else: print(f"PM Warning (Create): Invalid item in plan at index {i}.")
                else:
                    print(f"PM Error (Create): LLM did not return a list. Response: {plan_content[:200]}...")

                if created_tasks:
                    print(f"PM: Initial workflow created with {len(created_tasks)} tasks.")
                    output_state["tasks"] = created_tasks
                    output_state["current_task_index"] = 0
                    output_state["current_task"] = created_tasks[0].copy()
                    tasks = created_tasks # Update local variable
                    current_idx = 0     # Update local variable
                else:
                    print("PM Error (Create): Failed to create valid tasks from initial plan.")
                    output_state["tasks"] = []
                    output_state["current_task_index"] = 0
                    output_state["current_task"] = None
                    return output_state # Let router handle empty tasks
            except Exception as e:
                print(f"PM Error during initial workflow creation: {e}")
                traceback.print_exc()
                output_state["tasks"] = []
                output_state["current_task_index"] = 0
                output_state["current_task"] = None
                return output_state

        # --- Loop to Check Status, Save to LTM, and Handle Failure ---
        print(f"\nPM: --- Entering status check loop ---") # Log entry clearly
        tasks = output_state.get("tasks", []) # Get final task list for the loop
        current_idx = output_state.get("current_task_index", 0) # Get final index for the loop
        print(f"PM: Starting loop with current_idx = {current_idx}, task count = {len(tasks)}")

        while 0 <= current_idx < len(tasks):
            task_to_check = tasks[current_idx]
            status = task_to_check.get("status")
            task_id = task_to_check.get("task_id", "N/A")
            print(f"PM Loop: Checking Task {task_id} (Idx: {current_idx}), Status: '{status}'")

            if status in ["failed", "max_retries_reached"]:
                print(f"PM Loop: Handling '{status}' for task {task_id}")
                await self._save_task_to_ltm(task_to_check) # Save failed task

                # --- MODIFIED RETRY HANDLING ---
                max_retries = _full_static_config.agents.get("process_management", {}).parameters.get("max_retries", 3)
                current_retry_count = task_to_check.get("retry_count", 0)
                fail_action = "SKIP" # Default action if errors occur or retries exceeded

                if current_retry_count >= max_retries:
                    print(f"PM (Fail): Max retries ({max_retries}) already reached for task {task_id}. Forcing SKIP.")
                    fail_action = "SKIP"
                    # Ensure status reflects max retries if it wasn't already
                    if status != "max_retries_reached":
                        tasks[current_idx]["status"] = "max_retries_reached"
                        tasks[current_idx]["feedback_log"] = (tasks[current_idx].get("feedback_log", "") + "\n[PM Note: Marked as max_retries_reached during check.]").strip()
                else:
                    # Retries are available, increment count BEFORE calling LLM
                    current_retry_count += 1
                    tasks[current_idx]["retry_count"] = current_retry_count
                    print(f"PM (Fail): Incremented retry count for task {task_id} to {current_retry_count}/{max_retries}.")

                    # Determine failure type (existing logic)
                    failure_type = "unknown"
                    # ... (existing failure_type determination logic) ...
                    eval_assessment = task_to_check.get("evaluation", {}).get("assessment", "").lower()
                    if eval_assessment == "fail": # Check if eval failed
                        failure_type = "evaluation"
                    elif task_to_check.get("error_log"): # Check for execution error
                        failure_type = "execution"
                    else: # Default if no specific indicators
                        failure_type = "execution" # Or maybe 'unknown'? Let's stick with execution for now.


                    print(f"PM Failure Analysis: Determined Failure Type = '{failure_type}' for Task {task_id}")

                    # Calculate new is_max_retries status AFTER incrementing
                    is_max_retries = current_retry_count >= max_retries

                    failure_context = f"Task failed at index {current_idx}. Status: {status}. Failure Type: {failure_type}." # 加入 type

                    fail_prompt_input = {
                        "failure_context": failure_context,
                        "is_max_retries": is_max_retries, # Pass the updated status
                        "max_retries": max_retries, # Pass the limit itself
                        "selected_agent_name": task_to_check.get("selected_agent", "N/A"),
                        "task_description": task_to_check.get("description", "N/A"),
                        "task_objective": task_to_check.get("task_objective", "N/A"),
                        "inputs_json": json.dumps(task_to_check.get("task_inputs", {}), ensure_ascii=False),
                        "execution_error_log": task_to_check.get("error_log", "None"),
                        "feedback_log": task_to_check.get("feedback_log", "None (Includes Eval Results if any)"),
                        "llm_output_language": self.llm_output_language,
                        # "alternative_agent" is removed based on previous discussion
                        "original_requires_evaluation": task_to_check.get("requires_evaluation", False)
                    }

                    print("--- PM Failure Analysis: Inputs to LLM ---")
                    # ... (log inputs) ...
                    print(f"  Failure Type: {failure_type}")
                    print(f"  Is Max Retries (Passed to LLM): {is_max_retries}") # Log what LLM sees
                    print(f"  Current Retry Count (Passed to LLM): {current_retry_count}") # Log count LLM sees
                    # print(f"  Feedback Log (Input): {fail_prompt_input['feedback_log'][:500]}...")
                    print(f"  Error Log (Input): {fail_prompt_input['execution_error_log']}")
                    print("-----------------------------------------")

                    try:
                        if not self.prompts.get("failure_analysis"): raise ValueError("failure_analysis prompt missing")
                        if not self.llm: raise ValueError("LLM for PM not initialized.")
                        fail_chain = self.llm | StrOutputParser()
                        # Correctly format the prompt with the dictionary
                        fail_content = await fail_chain.ainvoke(self.prompts["failure_analysis"].format(**fail_prompt_input))


                        # Parse LLM response (existing logic)
                        if fail_content.startswith("```json"): fail_content = fail_content[7:-3].strip()
                        elif fail_content.startswith("```"): fail_content = fail_content[3:-3].strip()
                        fail_result = {}
                        try: fail_result = json.loads(fail_content)
                        except json.JSONDecodeError as json_err:
                            print(f"PM Failure Analysis Error: Failed to parse LLM JSON: {json_err}")
                            fail_result = {"action": "SKIP"} # Default to SKIP on parse error

                        # Get action, default to SKIP if missing or invalid
                        fail_action = fail_result.get("action", "SKIP")
                        if fail_action not in ["FALLBACK_GENERAL", "MODIFY", "SKIP"]:
                            print(f"PM Warning (Fail): LLM returned invalid action '{fail_action}'. Defaulting to SKIP.")
                            fail_action = "SKIP"

                        # Log parsed action
                        print(f"--- PM Failure Analysis: Parsed Result ---")
                        print(f"  Parsed Action: {fail_action}")
                        print(f"  Parsed Result Dict: {fail_result}")
                        print("-----------------------------------------")

                        # --- Overwrite LLM action if max retries were reached by the increment ---
                        if is_max_retries and fail_action != "SKIP":
                            print(f"PM (Fail): Max retries reached after increment ({current_retry_count}/{max_retries}). Overriding LLM action '{fail_action}' to SKIP.")
                            fail_action = "SKIP"
                            # Update status if needed
                            if status != "max_retries_reached":
                                tasks[current_idx]["status"] = "max_retries_reached"
                                tasks[current_idx]["feedback_log"] = (tasks[current_idx].get("feedback_log", "") + "\n[PM Note: Marked as max_retries_reached after increment.]").strip()


                    except Exception as e:
                        print(f"PM Error during failure analysis LLM call: {e}")
                        traceback.print_exc()
                        print("PM (Fail): Defaulting to SKIP action due to error during LLM call.")
                        fail_action = "SKIP"
                        # Ensure status reflects failure if possible
                        if status != "max_retries_reached":
                            tasks[current_idx]["status"] = "failed" # Keep as failed if not max retries
                            tasks[current_idx]["error_log"] = (tasks[current_idx].get("error_log", "") + f"\n[PM Error: Failure analysis LLM call failed: {e}]").strip()

                # --- END MODIFIED RETRY HANDLING ---

                # --- Handle Actions (using the potentially modified fail_action) ---
                if fail_action == "FALLBACK_GENERAL":
                    # --- MODIFIED LOG MESSAGE ---
                    print(f"PM (Fail): Processing FALLBACK_GENERAL (ALL new tasks inherit retry count {current_retry_count}).")
                    # --- END MODIFIED LOG ---
                    new_tasks_data = fail_result.get("new_tasks_list")
                    new_task_data = fail_result.get("new_task")
                    created_fallback_tasks = []
                    tasks_source = new_tasks_data if isinstance(new_tasks_data, list) else ([new_task_data] if isinstance(new_task_data, dict) else [])
                    print(f"PM (Fail): FALLBACK_GENERAL expecting {len(tasks_source)} task(s).")

                    for i, task_data in enumerate(tasks_source):
                        if isinstance(task_data, dict):
                            task_id_fb = str(uuid.uuid4())
                            # --- >>> NEW MODIFICATION HERE <<< ---
                            # ALL fallback tasks inherit the same incremented retry count.
                            inherit_retry = current_retry_count
                            # --- >>> END NEW MODIFICATION <<< ---

                            task_fb = TaskState(
                                task_id=task_id_fb, status="pending",
                                task_objective=task_data.get("task_objective", f"Fallback Objective {i+1}"),
                                description=task_data.get("description", f"Fallback Task {i+1}"),
                                selected_agent=task_data.get("selected_agent", "LLMTaskAgent"),
                                task_inputs=task_data.get("inputs", {}),
                                outputs={}, output_files=[], evaluation={}, error_log=None, feedback_log=None,
                                requires_evaluation=task_data.get("requires_evaluation", False),
                                # --- APPLY THE MODIFICATION ---
                                retry_count=inherit_retry # All tasks get the same inherited count
                                # --- END APPLY ---
                            )
                            if not task_fb["selected_agent"]:
                                print(f"PM Warning (Fallback): Task {i} lacks 'selected_agent'. Skipping.")
                                continue
                            created_fallback_tasks.append(task_fb)
                        else: print(f"PM Warning (Fallback): Invalid item {i} in tasks source.")


                    if created_fallback_tasks:
                        # --- MODIFIED LOG MESSAGE ---
                        print(f"PM (Fail): Replacing failed task at index {current_idx} with {len(created_fallback_tasks)} fallback task(s), ALL inheriting retry count {inherit_retry}.")
                        # --- END MODIFIED LOG ---
                        tasks = tasks[:current_idx] + created_fallback_tasks + tasks[current_idx+1:]
                        output_state["tasks"] = tasks
                        output_state["current_task_index"] = current_idx # Index points to the first new task
                        output_state["current_task"] = created_fallback_tasks[0].copy()
                        print(f"PM (Fail): Fallback applied. Continuing loop to process new task at index {current_idx}.")
                        continue # Continue WHILE loop to process the new task
                    else:
                        print("PM Warning (Fail): FALLBACK_GENERAL action chosen, but no valid tasks created/found. Defaulting to SKIP.")
                        fail_action = "SKIP" # If LLM fails to provide tasks, force SKIP

                if fail_action == "MODIFY":
                    print(f"PM (Fail): Modifying task {task_id} based on action 'MODIFY' and resetting to 'pending' for retry.")
                    # Apply modifications suggested by LLM (optional)
                    new_desc = fail_result.get("modify_description")
                    if new_desc and isinstance(new_desc, str):
                        print(f"  - Applying modification: description='{new_desc[:50]}...'")
                        tasks[current_idx]["description"] = new_desc
                    new_obj = fail_result.get("modify_objective")
                    if new_obj and isinstance(new_obj, str):
                        print(f"  - Applying modification: task_objective='{new_obj[:50]}...'")
                        tasks[current_idx]["task_objective"] = new_obj
                    # Add more fields if needed (e.g., modifying task_inputs requires careful handling)

                    # Reset status for retry
                    tasks[current_idx]["status"] = "pending"
                    # Add note to feedback log
                    tasks[current_idx]["feedback_log"] = (tasks[current_idx].get("feedback_log", "") + "\n[PM Note: Modified task based on failure analysis, resetting to pending for retry.]").strip()
                    # Clear error log for the retry attempt
                    tasks[current_idx]["error_log"] = None # Clear error for retry

                    # Update state and break loop
                    output_state["tasks"] = tasks
                    output_state["current_task_index"] = current_idx
                    output_state["current_task"] = tasks[current_idx].copy()
                    print(f"PM (Fail): Task {task_id} modified and set to 'pending'. Breaking loop.")
                    break # Exit WHILE loop, let router handle 'pending'

                # --- Handle SKIP Action (handles forced skip and LLM skip) ---
                if fail_action == "SKIP":
                    print(f"PM (Fail): Skipping task {task_id} at index {current_idx} based on action 'SKIP'.")
                    # Ensure status is max_retries_reached if it was forced skip due to retries
                    if tasks[current_idx].get("retry_count", 0) >= max_retries:
                         tasks[current_idx]["status"] = "max_retries_reached"
                         # --- FIX: Save skipped task to LTM ---
                         print(f"PM Loop: Saving skipped/max_retries task {task_id} to LTM before skipping.")
                         await self._save_task_to_ltm(tasks[current_idx])
                         # --- END FIX ---

                    current_idx += 1
                    output_state["current_task_index"] = current_idx # Update index before continuing
                    # Ensure current_task reflects the next task or None
                    output_state["current_task"] = tasks[current_idx].copy() if 0 <= current_idx < len(tasks) else None
                    print(f"PM (Fail): Updated current_idx to {current_idx}. Continuing loop.")
                    continue # Continue WHILE loop to check the NEXT task

            # --- End of failed/max_retries_reached block ---

            elif status == "completed":
                # ... (原本的 completed handling 邏輯) ...
                # --- 確保增加 current_idx 並 continue ---
                 print(f"PM Loop: Handling 'completed' for task {task_id}. Saving to LTM.")
                 await self._save_task_to_ltm(task_to_check)
                 current_idx += 1 # Increment index AFTER processing the completed task
                 # --- Update state *after* incrementing ---
                 output_state["current_task_index"] = current_idx # Update index in state
                 # Update current_task based on new index before continuing loop check
                 output_state["current_task"] = tasks[current_idx].copy() if 0 <= current_idx < len(tasks) else None
                 print(f"PM Loop: Incremented index to {current_idx} after completing task {task_id}. Continuing loop check.")
                 continue # Go back to the start of the while loop check


            elif status in ["pending", "in_progress", None]:
                # 發現需要執行的任務，設置狀態並跳出循環
                print(f"PM Loop: Task {task_id} is '{status}'. Ready for routing. Breaking loop.")
                output_state["current_task_index"] = current_idx
                output_state["current_task"] = task_to_check.copy()
                break # 跳出 WHILE 循環

            else: # Unexpected status
                # ... (原本的 unexpected status handling 邏輯) ...
                # --- 確保增加 current_idx 並 continue ---
                print(f"PM Loop Warning: Unhandled task status '{status}' for Task {task_id}. Skipping.")
                await self._save_task_to_ltm(task_to_check)
                current_idx += 1
                output_state["current_task_index"] = current_idx
                output_state["current_task"] = tasks[current_idx].copy() if 0 <= current_idx < len(tasks) else None
                print(f"PM Loop (Warning): Updated current_idx to {current_idx}. Continuing loop.")
                continue


        # --- After the loop ---
        # Logging remains the same
        print(f"PM: --- Debug log: Exited status check loop ---")
        final_idx_after_loop = output_state.get("current_task_index", current_idx) # Get potentially updated index
        print(f"PM: Final index after loop processing: {final_idx_after_loop}")
        tasks_final_count = len(output_state.get("tasks", [])) # Get final task count
        if final_idx_after_loop >= tasks_final_count:
            print("PM: Reached end of task list after loop.")
            # State for current_task/index already set inside loop or at end
            output_state["current_task"] = None # Ensure current task is None if index is out of bounds
            output_state["current_task_index"] = final_idx_after_loop # Keep the final index
        else:
             # Task found, index/task already set inside loop's break condition
             # Ensure state reflects this correctly
             output_state["current_task_index"] = final_idx_after_loop
             if "current_task" not in output_state or output_state["current_task"] is None or output_state["current_task"].get("task_id") != tasks[final_idx_after_loop].get("task_id"):
                  output_state["current_task"] = tasks[final_idx_after_loop].copy()
             print(f"PM: Exiting with index {final_idx_after_loop} pointing to task {output_state['current_task'].get('task_id') if output_state.get('current_task') else 'N/A'}.")

        print(f"=== Process Management Finished. Routing Index: {output_state.get('current_task_index', 'N/A')}, Phase: {output_state.get('current_phase')} ===")
        return output_state # 返回最終狀態

class EvaAgent:
    """
    Parent node responsible for initiating evaluation by invoking the evaluation_subgraph.
    It now acts as a simple trigger based on routing.
    """
    def __init__(self):
        pass

    async def run(self, state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Checks if evaluation is needed (based on routing) and invokes subgraph.
        Evaluation type logic is handled within the subgraph based on selected_agent.
        """
        print("--- Running EvaAgent Node (Simple Trigger) ---") # Updated description
        current_idx = state["current_task_index"]
        tasks = [t.copy() for t in state["tasks"]]
        if current_idx >= len(tasks):
             print("EvaAgent Error: Invalid task index.")
             return {}
        current_task = tasks[current_idx]

        # Simplified check: If routed here, assume evaluation is needed and pending
        if not (current_task.get("requires_evaluation", False) and current_task.get("status") == "pending"):
            status = current_task.get('status')
            req_eval = current_task.get('requires_evaluation')
            print(f"EvaAgent Warning: Routed here but task {current_task['task_id']} might not be ready? (Requires Eval: {req_eval}, Status: {status}). Proceeding anyway.")
            # Continue, assuming routing logic is correct

        print(f"EvaAgent: Triggering evaluation for Task {current_task['task_id']} (Agent: {current_task.get('selected_agent')})")

        # --- REMOVED: Logic to check/set/infer boolean flags ---

        # --- Set status to in_progress ---
        print(f"EvaAgent: Setting task status to in_progress.")
        current_task["status"] = "in_progress"

        # Update the tasks list in the state *before* calling the subgraph
        tasks = tasks[:current_idx] + [current_task] + tasks[current_idx+1:]
        state["tasks"] = tasks

        # --- Invoke Subgraph ---
        try:
            print(f"EvaAgent: Invoking evaluation subgraph...")
            # --- MODIFICATION: Use Send to pass the *entire* state to the subgraph ---
            # The subgraph now operates on the main WorkflowState
            # return Send(to="evaluation_teams", inputs=state) # Send doesn't return dict directly
            # --- CORRECTION: EvaAgent node itself should return the output dict ---
            # We directly invoke the subgraph here.
            subgraph_output_state: WorkflowState = await evaluation_teams.ainvoke(state, config=config)
            print(f"EvaAgent: Evaluation subgraph finished.")

            # --- Process Subgraph Output (logic remains same) ---
            final_tasks = subgraph_output_state.get("tasks", tasks) # Get updated tasks
            if current_idx >= len(final_tasks):
                 print("EvaAgent Error: Task index out of bounds after subgraph execution.")
                 # Attempt to get the original task to update its status if possible
                 original_task_to_update = tasks[current_idx] if 0 <= current_idx < len(tasks) else {}
                 if original_task_to_update:
                     original_task_to_update["status"] = "failed"
                     original_task_to_update["error_log"] = (original_task_to_update.get("error_log", "") + "; EvaAgent: Task index error post-subgraph").strip("; ")
                     original_task_to_update["feedback_log"] = (original_task_to_update.get("feedback_log", "") + "; EvaAgent: Task index error post-subgraph").strip("; ")
                     return {"tasks": tasks} # Return original tasks list with updated error task
                 else:
                     return {"tasks": tasks} # Return original list if index was truly invalid initially

            final_task = final_tasks[current_idx] # Get the updated task
            print(f"Task {final_task['task_id']} finished evaluation phase with status: {final_task['status']}")
            if final_task["status"] == "failed":
                print(f"  - Feedback/Error Log: {final_task.get('feedback_log', 'N/A')} / {final_task.get('error_log', 'N/A')}")
            elif final_task["status"] == "completed":
                 print(f"  - Feedback Log: {final_task.get('feedback_log', 'N/A')}")

            # --- <<< ADD LOGGING HERE >>> ---
            if final_tasks and 0 <= current_idx < len(final_tasks):
                final_task_checked = final_tasks[current_idx]
                eval_result = final_task_checked.get("evaluation", {})
                assessment = eval_result.get("assessment", "N/A")
                feedback = eval_result.get("feedback", "N/A")
                final_status = final_task_checked.get("status", "N/A")
                final_feedback_log = final_task_checked.get("feedback_log", "N/A")
                print(f"--- EvaAgent Post-Subgraph Check (Task ID: {final_task_checked.get('task_id', 'N/A')}, Index: {current_idx}) ---")
                print(f"  Evaluation Assessment: {assessment}")
                print(f"  Evaluation Feedback: {feedback[:100]}...") # Log truncated feedback
                print(f"  Final Status Set on Task: {final_status}")
                print(f"  Final Feedback Log on Task: {final_feedback_log[:100]}...") # Log truncated feedback log
                print("-----------------------------------------------------")
            else:
                 print(f"EvaAgent Warning: Could not log final task status due to index ({current_idx}) or tasks list issues after subgraph.")
            # --- <<< END ADD LOGGING >>> ---
            final_tasks = subgraph_output_state.get("tasks", tasks)
            # Return the full updated tasks list from the subgraph output state
            return {"tasks": final_tasks}

        except Exception as e:
            print(f"EvaAgent: Error invoking evaluation subgraph: {e}")
            import traceback
            traceback.print_exc()
            # Attempt to get the original task to update its status
            original_task_to_update = tasks[current_idx] if 0 <= current_idx < len(tasks) else None
            if original_task_to_update:
            # Ensure evaluation dict exists
                if "evaluation" not in original_task_to_update or not isinstance(original_task_to_update["evaluation"], dict):
                     original_task_to_update["evaluation"] = {}
            # Set failure status in evaluation dict
                original_task_to_update["evaluation"]["assessment"] = "Fail"
                original_task_to_update["evaluation"]["subgraph_error"] = f"Subgraph invocation error: {e}"
            # Set task status to failed
                original_task_to_update["status"] = "failed"
            # Append error to feedback log, ensuring it's always a string
                existing_feedback = original_task_to_update.get("feedback_log") or "" # Use 'or ""' to handle None or empty string
                original_task_to_update["feedback_log"] = (existing_feedback + f"; Evaluation Subgraph Invocation Error: {e}").strip("; ")
                # --- <<< ADD LOGGING IN EXCEPTION TOO >>> ---
                print(f"--- EvaAgent Exception Handling ---")
                print(f"  Error during subgraph invocation: {e}")
                print(f"  Attempting to mark Task ID: {original_task_to_update.get('task_id', 'N/A')} as failed.")
                print(f"  Status set to: {original_task_to_update.get('status')}")
                print(f"  Error Log: {original_task_to_update.get('error_log')}")
                print("---------------------------------")
                # --- <<< END ADD LOGGING >>> ---
            return {"tasks": tasks} # Return original task list if we couldn't update


class QA_Agent:
    def __init__(self, long_term_memory_vectorstore=vectorstore):
        self.vectorstore = long_term_memory_vectorstore
        # --- 從配置加載模板字串 ---
        qa_prompt_config = _qa_prompts.get("qa_prompt") # 獲取 PromptConfig 對象
        self.qa_prompt_template_str = qa_prompt_config.template if qa_prompt_config else "" # 獲取模板字串
        self.qa_prompt_input_variables = qa_prompt_config.input_variables if qa_prompt_config else [] # 獲取變數列表

        if not self.qa_prompt_template_str:
             print("QA_Agent WARNING: qa_prompt template string not found in config. Using a basic default.")
             # 使用一個非常基礎的預設模板，以防萬一
             self.qa_prompt_template_str = """對話記錄:\n{chat_history}\n\n人類: {last_user_query}\nAI:"""
             self.qa_prompt_input_variables = ["chat_history", "last_user_query"] # 預設模板的變數

        # --- 驗證必要的變數是否存在 ---
        # 根據 configuration.py 的預期變數進行檢查
        expected_vars = {"task_summary", "retrieved_ltm_context", "chat_history", "last_user_query", "llm_output_language"}
        if not expected_vars.issubset(self.qa_prompt_input_variables):
            print(f"QA_Agent WARNING: Loaded qa_prompt template might be missing expected variables.")
            print(f"  Expected: {expected_vars}")
            print(f"  Found in config: {self.qa_prompt_input_variables}")
            # Consider raising an error or falling back more gracefully

        # --- STM setup ---
        self.memory_window_size = _full_static_config.agents.get("qa_agent", {}).parameters.get("memory_window_size", 10)
        # Note: ConversationBufferWindowMemory formats history into a single string key, matching 'chat_history'
        self.stm = ConversationBufferWindowMemory(
            k=self.memory_window_size, return_messages=False, # Set return_messages=False to get formatted string
            memory_key="chat_history", input_key="human_input" # Use different keys for clarity
        )
        # --- 創建 ChatPromptTemplate 實例 ---
        # 使用從配置中讀取的模板字串
        self.prompt_template = ChatPromptTemplate.from_template(self.qa_prompt_template_str)
        print(f"QA_Agent initialized using prompt template from configuration: 'qa_prompt'")


    async def run(self, state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        node_name = "QA Agent Node (chat_bot)"
        print(f"--- 運行節點: {node_name} ---")

        # 獲取配置和上下文
        runtime_config = config["configurable"]
        qa_llm_config = runtime_config.get("qa_llm", {})
        retriever_k = runtime_config.get("retriever_k", 5)
        llm = initialize_llm(qa_llm_config)
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=retriever_k))
        ltm = VectorStoreRetrieverMemory(retriever=retriever, memory_key=ltm_memory_key, input_key=ltm_input_key)
        llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)

        qa_context_list: List[BaseMessage] = state.get("qa_context", [])
        # qa_context is expected to have 0 or 1 message (the current turn's message)
        current_turn_message = qa_context_list[0] if qa_context_list else None

        # --- <<< 修改循環判斷邏輯 >>> ---
        if not current_turn_message:
            # Case 1: qa_context is empty. This means no new user input was provided via qa_loop_node.
            # This is the entry point for QA, or user didn't provide input after AI's last turn.
            # Send an initial/generic prompt.
            prompt_text = "您好，請問有什麼可以協助您的嗎？或是有其他問題嗎？ (您可以輸入 '結束對話' 或 '繼續任務')"
            print(f"{node_name}: Context is empty or no HumanMessage. Sending initial prompt.")
            return {
                "current_phase": "qa",
                "qa_context": [AIMessage(content=prompt_text)]
            }
        elif not isinstance(current_turn_message, HumanMessage):
            # Case 2: Last message in qa_context (current turn) is an AIMessage.
            # This implies AI spoke, graph paused, user provided NO input, and graph resumed.
            # AI should not speak again. It should wait for actual user input.
            print(f"{node_name}: Last message in context was AIMessage. AI waits for user input. No new response generated.")
            return {
                "current_phase": "qa",        # Stay in QA phase
                "qa_context": qa_context_list # Return the existing context (which has AIMessage)
            }
        # --- <<< 結束修改循環判斷邏輯 >>> ---

        # If we reach here, current_turn_message is a HumanMessage.
        last_user_query_content = current_turn_message.content
        print(f"{node_name}: Processing user query: '{last_user_query_content[:100]}...'")
        query_for_ltm = last_user_query_content # Use the same for LTM retrieval

        # 初始化返回狀態
        return_state_update = {
            "current_phase": "qa"  # 預設保持QA狀態
        }

        # 準備提示輸入
        try:
            # ... (從LTM獲取上下文、準備其他輸入的邏輯保持不變) ...
            retrieved_ltm_context = "LTM: N/A"
            if query_for_ltm:
                try:
                    ltm_loaded_vars = await ltm.aload_memory_variables({ltm_input_key: query_for_ltm})
                    context_from_ltm = ltm_loaded_vars.get(ltm.memory_key)
                    if context_from_ltm:
                        retrieved_ltm_context = context_from_ltm
                except Exception as e:
                    print(f"{node_name}: LTM錯誤: {e}")

            task_summary = "\n".join([f"- Task {i+1}: {t['description']} ({t['status']})" for i, t in enumerate(state.get("tasks", []))]) or "沒有執行任何任務。"
            current_stm_vars = self.stm.load_memory_variables({})
            formatted_chat_history = current_stm_vars.get(self.stm.memory_key, "沒有STM歷史。")

            chain_input = {
                "last_user_query": last_user_query_content,
                "retrieved_ltm_context": retrieved_ltm_context,
                "window_size": self.memory_window_size,
                "chat_history": formatted_chat_history,
                "llm_output_language": llm_output_language,
                "task_summary": task_summary,
            }

            # LLM調用
            qa_chain = self.prompt_template | llm
            response_message = await qa_chain.ainvoke(chain_input)
            response_content = response_message.content.strip()
            print(f"{node_name} 原始回應: {response_content[:150]}...")

            # ... (意圖檢測邏輯保持不變) ...
            next_phase = "qa"  # 預設階段
            terminate_indicators = ["TERMINATE", "結束對話", "再見", "謝謝再見", "不需要了"]
            is_terminate = any(ind in response_content for ind in terminate_indicators)
            resume_indicators = ["RESUME_TASK", "繼續任務", "返回任務", "回到任務", "繼續工作流程"]
            is_resume = any(ind in response_content for ind in resume_indicators)
            new_task_match = None
            if response_content.startswith("NEW_TASK:"):
                new_task_match = response_content[len("NEW_TASK:"):].strip()

            if is_terminate:
                response_content = "好的，對話結束。"
                next_phase = "finished"
                print(f"{node_name}: 檢測到終止對話意圖")
            elif is_resume:
                response_content = "好的，正在返回任務執行流程。"
                next_phase = "task_execution"
                print(f"{node_name}: 檢測到繼續任務意圖")
            elif new_task_match:
                response_content = f"收到新任務：'{new_task_match}'。正在返回任務規劃..."
                next_phase = "task_execution"
                return_state_update["user_input"] = new_task_match # Pass new task goal back
                print(f"{node_name}: 檢測到新任務請求")
            else:
                print(f"{node_name}: 普通回答，維持QA階段")

            # 更新STM
            # --- <<< 修改：確保使用正確的 key >>> ---
            self.stm.save_context({"human_input": last_user_query_content}, {"output": response_content})
            # --- <<< 結束修改 >>> ---

            # 準備返回狀態
            return_state_update["current_phase"] = next_phase
            # --- <<< 修改：返回 AI 回應，而不是提示 >>> ---
            return_state_update["qa_context"] = [AIMessage(content=response_content)]
            # --- <<< 結束修改 >>> ---

            print(f"{node_name}: 返回狀態: phase='{next_phase}', 消息='{response_content[:50]}...'")
            return return_state_update

        except Exception as e:
            print(f"{node_name} 錯誤: {e}")
            traceback.print_exc() # Print traceback for debugging
            error_message = f"處理您的問題時發生錯誤: {e}"
            # Return error message in qa_context, keep in QA phase
            return_state_update["current_phase"] = "qa"
            return_state_update["qa_context"] = [AIMessage(content=error_message)]
            return return_state_update


# --- Keep loading agent descriptions as they are still needed ---
agent_descriptions = _full_static_config.agents.get("assign_agent", {}).parameters.get("specialized_agents_description", {})
if not agent_descriptions:
     print("WARNING: specialized_agents_description not found in config for AssignAgent!")

# =============================================================================
# 7. Agent 實例化
# =============================================================================
process_management = ProcessManagement(long_term_memory_vectorstore=vectorstore)
eva_agent = EvaAgent() # Instantiate the refactored EvaAgent
qa_agent = QA_Agent(long_term_memory_vectorstore=vectorstore)

# =============================================================================
# 8. 輔助/節點函數定義
# =============================================================================
def save_final_summary(state: WorkflowState) -> WorkflowState:
    """
    Saves the final summary of the workflow, including all task details,
    outputs, and generated files into a Word document and a JSON file.
    The Word document includes a title, overall goal, task summaries, and images.
    The radar chart and its context are now placed under the Detailed Assessment section.
    """
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH # Ensure this is imported

    print("--- Saving Final Workflow Summary ---")
    if not state:
        print("Error: State is None, cannot save summary.")
        return state

    final_summary_output_dir = "D:/MA system/LangGraph/output/Report"
    os.makedirs(final_summary_output_dir, exist_ok=True)

    tasks = state.get("tasks", [])
    print(f"Save Summary Debug: Total tasks in state: {len(tasks)}")
    for i, task_debug in enumerate(tasks):
        print(f"Save Summary Debug: Task {i+1} ID: {task_debug.get('task_id', 'N/A')}, Agent: {task_debug.get('selected_agent', 'N/A')}, Status: {task_debug.get('status', 'N/A')}")
        if task_debug.get("selected_agent") in ["SpecialEvaAgent", "FinalEvaAgent"]:
            print(f"  Eval Task Debug: Outputs: {task_debug.get('outputs', {}).keys()}")
            print(f"  Eval Task Debug: Output Files: {task_debug.get('output_files')}")
            print(f"  Eval Task Debug: Evaluation Dict: {task_debug.get('evaluation')}")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_filename = f"Final_Workflow_Summary_{timestamp}"
    word_filename = f"{base_filename}.docx"
    json_filename = f"{base_filename}.json"
    word_filepath = os.path.join(final_summary_output_dir, word_filename)
    json_filepath = os.path.join(final_summary_output_dir, json_filename)

    doc = DocxDocument()
    # Report Title
    heading_paragraph = doc.add_heading('Workflow Final Summary', level=0)
    if heading_paragraph.runs:
        heading_paragraph.runs[0].bold = True

    doc.add_paragraph(f"User Goal: {state.get('user_input', 'N/A')}")
    doc.add_paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph()

    # --- Find the LATEST completed evaluation task for Detailed Assessment section ---
    # We need the data and files from the MOST RECENT completed evaluation task.
    latest_eval_task = None
    print(f"Save Summary: Searching for latest completed evaluation task for Detailed Assessment section.")

    for task_idx in range(len(tasks) - 1, -1, -1): # Iterate backwards
        task_eval_check = tasks[task_idx]
        task_id_check = task_eval_check.get('task_id', f'task_at_idx_{task_idx}')
        print(f"Save Summary: Checking task (ID: {task_id_check}, Agent: {task_eval_check.get('selected_agent')}, Status: {task_eval_check.get('status')}) for latest evaluation data.")

        if task_eval_check.get("selected_agent") in ["SpecialEvaAgent", "FinalEvaAgent"] and \
           task_eval_check.get("status") == "completed": # Ensure it's a completed evaluation task
            print(f"Save Summary: Task {task_id_check} is a relevant completed evaluation task. Using it for evaluation data and charts.")
            latest_eval_task = task_eval_check # Store the task
            break # Found the latest completed evaluation task, exit loop


    print(f"Save Summary: Found latest relevant eval task (if any): ID {latest_eval_task.get('task_id') if latest_eval_task else 'N/A'}")

    # --- Start of Workflow Details Section ---
    doc.add_heading("工作流程細節 (Workflow Details)", level=1)

    if not tasks:
        doc.add_paragraph("No tasks were executed in this workflow.")
    else:
        completed_tasks_count = sum(1 for t in tasks if t.get("status") == "completed")
        print(f"Save Summary Debug: Found {completed_tasks_count} completed tasks out of {len(tasks)} total for Workflow Details section.")

        for i, task in enumerate(tasks):
            task_status = task.get("status", "N/A")
            task_agent = task.get("selected_agent")
            print(f"Save Summary Debug: Processing task {i+1}/{len(tasks)} for details, Status: {task_status}, Agent: {task_agent}")

            # --- MODIFIED: Skip evaluation tasks from this main details list ---
            # Their detailed assessment and charts will be handled in the dedicated section later.
            if task_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
                 print(f"Save Summary Debug: Skipping evaluation task {task.get('task_id', 'N/A')} from Workflow Details section.")
                 continue # Skip to the next task


            # Process non-evaluation tasks that are completed, failed, or max_retries_reached
            if task_status in ["completed", "failed", "max_retries_reached"]:
                # Use original task index for heading number
                doc.add_heading(f"Task {i+1}: {task.get('description', 'N/A')}", level=2)

                p_objective = doc.add_paragraph()
                p_objective.add_run("Objective: ").bold = True
                p_objective.add_run(task.get('task_objective', 'N/A'))

                p_agent = doc.add_paragraph()
                p_agent.add_run("Agent: ").bold = True
                p_agent.add_run(task.get('selected_agent', 'N/A'))

                p_status = doc.add_paragraph()
                p_status.add_run("Status: ").bold = True
                p_status.add_run(task_status)

                doc.add_paragraph()

                # Evaluation Criteria/Rubric (Still include for non-eval tasks if somehow present)
                if task.get("evaluation") and isinstance(task["evaluation"], dict):
                    specific_criteria = task["evaluation"].get("specific_criteria")
                    is_specific_criteria_valid_and_not_default = (
                        specific_criteria and
                        isinstance(specific_criteria, str) and
                        specific_criteria.strip() and
                        specific_criteria.lower() not in [
                            "default criteria apply / rubric not generated.",
                            "default criteria apply / rubric not generated",
                            "default criteria apply"
                            ]
                    )

                    # Only add the heading if there's specific criteria or we will add the general explanation
                    if is_specific_criteria_valid_and_not_default or True: # Always include general explanation
                        doc.add_heading("Evaluation Criteria/Rubric:", level=3)

                        if is_specific_criteria_valid_and_not_default:
                            doc.add_paragraph(specific_criteria)
                            doc.add_paragraph() # Add a paragraph break
                            doc.add_paragraph().add_run("通用評估標準補充說明：").bold = True
                        else:
                            doc.add_paragraph("（以下為通用評估標準說明）").italic = True

                        # Always add the hardcoded explanations
                        # Cost-benefit explanation
                        p_cost_title = doc.add_paragraph()
                        p_cost_title.add_run("早期成本效益估算說明：").bold = True
                        doc.add_paragraph(
                            "有預算上限時：因為屬於前期成本概算，設定成本偏差閾值 ±50%計算得分。低於預算50%（即預算 * 0.5）因可能低於合理標的底價，視為1分；高於預算50%（即預算 * 1.5）因成本效益過低，亦視為1分。在預算 ±50%範圍內，成本越低（越接近預算 * 0.5），分數越高，呈線性關係。"
                        )
                        doc.add_paragraph(
                            "無預算上限時：基於成本效率分數計算，通常將觀察到的成本範圍（例如從最低成本到最高成本）進行線性映射給分，成本越低，分數越高。"
                        )
                        doc.add_paragraph() # Spacer

                        # Green building explanation
                        p_green_title = doc.add_paragraph()
                        p_green_title.add_run("綠建築永續潛力估算說明：").bold = True
                        doc.add_paragraph(
                            "大致基於綠建築標章之主要評估指標（如生態、健康、節能、減廢等四大項）進行計分。潛力分數的計算方式可能為各指標預期得分的加權總和，再轉換為0-10分制（例如，總達成率百分比除以10）。"
                        )
                        doc.add_paragraph(
                            "此潛力分數可對應至業界常見的綠建築評級潛力：3分以下約為合格級；3-6分約為銅級；6-8分約為銀級；8-9.5分約為黃金級；9.5分以上則具備鑽石級潛力。"
                        )
                        doc.add_paragraph()
                # --- END MODIFICATION (Adjusted check for heading) ---


                text_outputs_found = False
                if task.get("outputs"):
                    for key, value in task["outputs"].items():
                        # Exclude evaluation-specific keys from standard text outputs
                        if isinstance(value, str) and key not in ["mcp_internal_messages", "grounding_sources", "search_suggestions", "radar_chart_path", "detailed_option_scores", "detailed_assessment", "assessment", "feedback_llm_overall", "final_llm_feedback_overall"]:
                            if len(value) > 10: # Only include meaningful text outputs
                                if not text_outputs_found:
                                    doc.add_heading("關鍵文字輸出 (Key Text Outputs):", level=3) # Modified heading
                                    text_outputs_found = True

                                doc.add_paragraph(f"{key.replace('_', ' ').title()}:", style='Intense Quote')
                                doc.add_paragraph(value)
                                doc.add_paragraph()


                if task.get("output_files"):
                    # Modified heading text
                    doc.add_heading("關聯檔案 (Associated Files):", level=3)
                    files_processed = 0

                    for file_info in task.get("output_files", []):
                        if not isinstance(file_info, dict):
                            print(f"Save Summary Warning: Skipping non-dict file_info in Task {i+1}: {file_info}")
                            continue

                        original_path_str = file_info.get("path")
                        file_description = file_info.get("description", Path(original_path_str).name if original_path_str else "N/A")
                        file_type = file_info.get("type", "Unknown").lower()
                        file_filename = file_info.get("filename", Path(original_path_str).name if original_path_str else "N/A")

                        # --- MODIFIED: Skip specific evaluation charts (radar, bar) from *any* task's file list ---
                        # Use updated matching criteria
                        is_eval_chart = False
                        if original_path_str and Path(original_path_str).exists() and "image" in file_type:
                             filename_lower = file_filename.lower() if file_filename else ""
                             description_lower = file_description.lower() if file_description else ""
                             # --- UPDATED MATCHING CRITERIA ---
                             if "evaluation_radar" in filename_lower or "evaluation_stacked_bar" in filename_lower or \
                                "radar_chart" in description_lower or "bar_chart" in description_lower: # Keep description check as fallback
                                  is_eval_chart = True
                                  print(f"Save Summary Debug: File {file_filename} identified as an evaluation chart.")

                        if is_eval_chart:
                            print(f"Save Summary: Skipping evaluation chart file {file_filename} in task details as it will be handled in the Detailed Assessment section.")
                            continue # Skip this file in this section
                        # --- END MODIFIED ---


                        files_processed += 1
                        # --- MODIFIED: Change file info display format ---
                        # Create the first line: File: SourceAgent: ...; TaskDesc: ...; ImageNum: ...
                        p_info_line1 = doc.add_paragraph()
                        # Check if description contains agent/task info to avoid duplication
                        source_agent = file_info.get("SourceAgent", "N/A")
                        # Use TaskDescShort if available, otherwise fallback to full description
                        task_desc_short = file_info.get("TaskDescShort", file_description) 
                        image_num_info = file_info.get("ImageNum", "")
                        image_num_display = f"; ImageNum: {image_num_info}" if image_num_info else ""

                        # Construct the first line text
                        info_line1_text = f"File: SourceAgent: {source_agent}; TaskDesc: {task_desc_short}{image_num_display}"
                        p_info_line1.add_run(info_line1_text).bold = True

                        # Create the second line: "檔案位置: [檔案路徑]"
                        p_file_path = doc.add_paragraph()
                        p_file_path.add_run("檔案位置 (File Path): ").bold = True # Modified text
                        p_file_path.add_run(original_path_str if original_path_str else "N/A")

                        # Create the third line (optional): Filename and Type if needed, or just add space
                        p_file_details = doc.add_paragraph()
                        # --- MODIFIED: Get the Run object and set properties on it ---
                        run_file_details = p_file_details.add_run(f"(檔名: {file_filename}, 類型: {file_type.capitalize()})") # Get the Run object
                        run_file_details.italic = True
                        run_file_details.font.size = Pt(9) # Set font size on the Run object
                        # --- END MODIFIED ---

                        doc.add_paragraph() # Add space after the file info block
                        # --- END MODIFIED ---


                        if original_path_str:
                             original_file_path = Path(original_path_str)
                             if original_file_path.exists():
                                print(f"Save Summary: Found file {file_filename} at {original_path_str}. Type: {file_type}")
                                # Create task-specific asset directory *once* per task
                                asset_target_dir = Path(final_summary_output_dir) / "task_assets" / task.get('task_id', f"task_{i+1}")
                                os.makedirs(asset_target_dir, exist_ok=True)

                                try:
                                    destination_file = asset_target_dir / original_file_path.name
                                    print(f"Save Summary: Copying {original_file_path} to {destination_file}")
                                    # Ensure parent directory exists for destination
                                    destination_file.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(original_path_str, destination_file) # Use original_path_str for source

                                    # Removed the "Copied to" paragraph addition as requested previously
                                    # doc.add_paragraph(f"  - Copied to: .\\task_assets\\{task.get('task_id', f'task_{i+1}')}\\{original_file_path.name}")
                                    print(f"Save Summary: Successfully copied {file_filename}.")

                                    if "image" in file_type:
                                        try:
                                            print(f"Save Summary: Attempting to embed image {destination_file} directly for task file.")
                                            doc.add_picture(str(destination_file), width=Inches(5.0))
                                            print(f"Save Summary: Successfully embedded task image {file_filename} directly.")

                                            # Keep the caption for embedded images
                                            para_caption = doc.add_paragraph()
                                            # Use file description for caption
                                            run_caption = para_caption.add_run(f"圖: {file_description}") # Modified text
                                            run_caption.italic = True
                                            run_caption.font.size = Pt(9)
                                            para_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                            doc.add_paragraph()

                                        except Exception as direct_embed_err:
                                            print(f"Save Summary Error: Direct embedding of task image {destination_file} failed: {direct_embed_err}. Trying PIL.")
                                            traceback.print_exc()
                                            try:
                                                from PIL import Image # Ensure PIL is imported if not already at top
                                                img = Image.open(str(destination_file))
                                                if img.mode == 'RGBA':
                                                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                                    rgb_img.paste(img, mask=img.split()[3])
                                                    img = rgb_img
                                                
                                                temp_task_img_path = str(destination_file) + "_pil_temp.jpg"
                                                img.save(temp_task_img_path, format='JPEG')
                                                
                                                doc.add_picture(temp_task_img_path, width=Inches(5.0))
                                                print(f"Save Summary: Successfully embedded task image {file_filename} using PIL conversion.")

                                                para_caption_pil = doc.add_paragraph()
                                                run_caption_pil = para_caption_pil.add_run(f"圖: {file_description} (PIL 處理)") # Modified text
                                                run_caption_pil.italic = True
                                                run_caption_pil.font.size = Pt(9)
                                                run_caption_pil.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                                doc.add_paragraph()
                                            except Exception as pil_err_task:
                                                print(f"Save Summary Error: PIL embedding of task image {destination_file} also failed: {pil_err_task}")
                                                traceback.print_exc()
                                                doc.add_paragraph(f"  [無法自動嵌入圖片 '{file_filename}': 直接嵌入與 PIL 處理皆失敗。錯誤: {pil_err_task}]") # Modified text
                                                doc.add_paragraph()
                                        
                                    elif "video" in file_type:
                                        # --- MODIFIED: Add logic to extract and embed video specific frames ---
                                        print(f"Save Summary: File {file_filename} is a video. Attempting to extract frames at 12s and 18s.")
                                        video_path = original_path_str # Use the original path to open the video
                                        temp_frame_paths = [] # List to store temporary frame file paths

                                        try:
                                            # import cv2 # Moved import to top
                                            vidcap = cv2.VideoCapture(video_path)
                                            if not vidcap.isOpened():
                                                print(f"Save Summary Warning: Could not open video file {video_path}")
                                                doc.add_paragraph(f"  [無法打開影片檔案 '{file_filename}'] ([Could not open video file '{file_filename}'])")
                                                doc.add_paragraph()
                                            else:
                                                fps = vidcap.get(cv2.CAP_PROP_FPS)
                                                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                                                duration = frame_count / fps if fps > 0 else 0

                                                print(f"Save Summary Debug: Video {file_filename} - FPS: {fps}, Frame Count: {frame_count}, Duration: {duration:.2f}s")

                                                # Define timestamps in seconds
                                                timestamps_seconds = [12, 18]

                                                for ts in timestamps_seconds:
                                                    if duration > ts:
                                                        # Calculate target frame number
                                                        target_frame_number = int(ts * fps)
                                                        print(f"Save Summary Debug: Attempting to capture frame at {ts}s (Frame number: {target_frame_number})")

                                                        # Set video position to the target frame
                                                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

                                                        # Read the frame
                                                        success, image = vidcap.read()

                                                        if success:
                                                            # Save the frame as a temporary image file
                                                            frame_filename = f"{Path(video_path).stem}_{ts}s.jpg"
                                                            temp_dir = Path(final_summary_output_dir) / "temp"
                                                            temp_dir.mkdir(parents=True, exist_ok=True)
                                                            temp_frame_path = temp_dir / frame_filename
                                                            
                                                            # Ensure image is in BGR for saving (cv2.imwrite expects BGR)
                                                            cv2.imwrite(str(temp_frame_path), image)
                                                            temp_frame_paths.append(temp_frame_path) # Add to list for cleanup
                                                            print(f"Save Summary: Successfully extracted and saved frame at {ts}s to {temp_frame_path}")

                                                            # Embed the saved frame into the document
                                                            try:
                                                                print(f"Save Summary: Attempting to embed frame image {temp_frame_path}.")
                                                                doc.add_picture(str(temp_frame_path), width=Inches(5.0)) # Adjust width as needed
                                                                print(f"Save Summary: Successfully embedded frame image for {ts}s.")

                                                                # Add caption
                                                                para_caption = doc.add_paragraph()
                                                                run_caption = para_caption.add_run(f"圖: 影片 '{file_filename}' 在 {ts} 秒的畫面 (Frame at {ts}s of video '{file_filename}')") # Modified text
                                                                run_caption.italic = True
                                                                run_caption.font.size = Pt(9)
                                                                para_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                                                doc.add_paragraph() # Add space

                                                            except Exception as embed_frame_err:
                                                                 print(f"Save Summary Error: Failed to embed frame from {ts}s ({temp_frame_path}): {embed_frame_err}")
                                                                 traceback.print_exc()
                                                                 doc.add_paragraph(f"  [無法自動嵌入影片 '{file_filename}' 在 {ts} 秒的畫面。錯誤: {embed_frame_err}] ([Could not automatically embed frame at {ts}s of video '{file_filename}'. Error: {embed_frame_err}]") # Modified text
                                                                 doc.add_paragraph()

                                                        else:
                                                            print(f"Save Summary Warning: Could not read frame at {ts}s from video {file_filename}.")
                                                            doc.add_paragraph(f"  [無法讀取影片 '{file_filename}' 在 {ts} 秒的畫面] ([Could not read frame at {ts}s of video '{file_filename}'])") # Modified text
                                                            doc.add_paragraph()
                                                    else:
                                                         print(f"Save Summary Warning: Video {file_filename} is shorter than {ts}s ({duration:.2f}s). Skipping frame extraction at {ts}s.")
                                                         doc.add_paragraph(f"  [影片 '{file_filename}' 時長不足 {ts} 秒，無法擷取該時間點畫面] ([Video '{file_filename}' is shorter than {ts}s, cannot extract frame at this time])") # Modified text
                                                         doc.add_paragraph()


                                            vidcap.release() # Release the video capture object

                                        except Exception as frame_extract_err:
                                            print(f"Save Summary Error: Error extracting/embedding frames from video {file_filename}: {frame_extract_err}")
                                            traceback.print_exc()
                                            doc.add_paragraph(f"  [處理影片 '{file_filename}' 時發生錯誤：{frame_extract_err}] ([Error processing video '{file_filename}': {frame_extract_err}]") # Modified text
                                            doc.add_paragraph()
                                        finally:
                                            # Clean up the temporary frame files
                                            for temp_path in temp_frame_paths:
                                                if temp_path.exists():
                                                    try:
                                                        temp_path.unlink()
                                                        print(f"Save Summary: Cleaned up temporary frame file {temp_path}")
                                                    except Exception as cleanup_err:
                                                        print(f"Save Summary Warning: Failed to clean up temporary frame file {temp_path}: {cleanup_err}")
                                        # --- END MODIFIED ---

                                        # Add the note about the original video file being copied
                                        doc.add_paragraph("  [影片檔案已複製 - 請至 task_assets 資料夾觀看] (Video file copied - Please view externally from the task_assets folder)") # Modified text
                                        doc.add_paragraph()

                                    elif "model" in file_type or original_file_path.suffix.lower() in ['.glb', '.obj', '.fbx', '.stl']:
                                        # Modified text for external viewing note
                                        doc.add_paragraph("  [3D 模型檔案已複製 - 請至 task_assets 資料夾觀看] (3D Model file copied - Please view externally from the task_assets folder)") # Modified text
                                        doc.add_paragraph()
                                    else:
                                         print(f"Save Summary: File type '{file_type}' not automatically embedded.")
                                         # Modified text for file copied note
                                         doc.add_paragraph(f"  [檔案類型 '{file_type}' 已複製 - 路徑已提供] (File type '{file_type}' copied - Path provided)") # Modified text
                                         doc.add_paragraph()
                                except Exception as copy_e:
                                    print(f"Save Summary Error: Error copying asset {original_path_str} for Word report: {copy_e}")
                                    traceback.print_exc()
                                    # Modified error message
                                    doc.add_paragraph(f"\n  - 原始路徑: {original_path_str} (複製錯誤: {copy_e}) (Original path: {original_path_str} (Error copying: {copy_e}))") # Modified text
                                    doc.add_paragraph()
                             else:
                                 print(f"Save Summary Warning: File path does not exist: {original_path_str}")
                                 # Modified warning message
                                 doc.add_paragraph(f"\n  - 原始路徑 (檔案未找到): {original_path_str} (Original path (file not found): {original_path_str})") # Modified text
                                 doc.add_paragraph()
                        else:
                            print(f"Save Summary Warning: File info dictionary missing 'path': {file_info}")
                            # Modified warning message
                            doc.add_paragraph("\n  - 檔案資訊缺少 'path' 欄位。(Path not specified in file info.)") # Modified text
                            doc.add_paragraph()

                    print(f"Save Summary Debug: Processed {files_processed} files for task {i+1}")


                # Add page break after each non-evaluation task details section, unless the very last task is also non-eval.
                # Check if the next task exists and is *not* an evaluation task.
                # Also check if there are subsequent non-eval tasks
                has_subsequent_non_eval_task = False
                for j in range(i + 1, len(tasks)):
                    if not (tasks[j].get("selected_agent") in ["SpecialEvaAgent", "FinalEvaAgent"]):
                        has_subsequent_non_eval_task = True
                        break

                if has_subsequent_non_eval_task:
                    doc.add_page_break()
                 # No page break needed if it's the last non-eval task or the next task IS an eval task


            # --- End of completed/failed/max_retries_reached block for non-eval tasks ---

            else:
                 # This covers 'pending', 'in_progress', or other unexpected statuses for non-eval tasks
                 # or eval tasks that weren't completed. These aren't included in the summary details section.
                 print(f"Save Summary Debug: Skipping task {i+1} with status '{task_status}' from Workflow Details section (or it's an eval task not completed).")


    # --- Start of Detailed Assessment Results Section ---
    # Modified heading text
    doc.add_heading("詳細評估結果 (Detailed Assessment Results)", level=1)
    detailed_assessment = None
    eval_assessment_text = "N/A"
    eval_feedback_text = "N/A"
    eval_chart_files = [] # List to hold relevant chart file_info dicts

    # Use the latest_eval_task found earlier
    if latest_eval_task:
        print(f"Save Summary: Processing latest eval task (ID: {latest_eval_task.get('task_id')}) for Detailed Assessment section.")
        eval_data = latest_eval_task.get("evaluation", {})
        # Prefer detailed_assessment, fallback to detailed_option_scores
        detailed_assessment = eval_data.get("detailed_assessment") or eval_data.get("detailed_option_scores")
        eval_assessment_text = eval_data.get("assessment", "N/A")
        # Try multiple keys for overall feedback
        eval_feedback_text = eval_data.get("final_llm_feedback_overall", eval_data.get("feedback_llm_overall", "N/A"))

        # Find charts in this evaluation task's output_files
        output_files_eval = latest_eval_task.get("output_files", [])
        print(f"Save Summary: Checking {len(output_files_eval)} files in the latest eval task ({latest_eval_task.get('task_id')}) for charts.")
        for f_info in output_files_eval:
             if isinstance(f_info, dict):
                  original_path_str = f_info.get("path")
                  file_type = f_info.get("type", "Unknown").lower()
                  file_filename = f_info.get("filename", Path(original_path_str).name if original_path_str else "")
                  description_lower = f_info.get("description", "").lower()

                  # Check if path exists and is an image, and filename/description indicates chart
                  if original_path_str and Path(original_path_str).exists() and "image" in file_type:
                       filename_lower = file_filename.lower()
                       # --- UPDATED MATCHING CRITERIA ---
                       if "evaluation_radar" in filename_lower or "evaluation_stacked_bar" in filename_lower or \
                          "radar_chart" in description_lower or "bar_chart" in description_lower: # Keep description check as fallback
                            eval_chart_files.append(f_info) # Add the whole file_info dict
                            print(f"Save Summary Debug: Found an evaluation chart for Detailed Assessment section: {file_filename} (Path: {original_path_str}, Desc: {f_info.get('description')})")
                       else:
                            print(f"Save Summary Debug: File {file_filename} (Path: {original_path_str}, Type: {file_type}) is an image but not identified as an eval chart for this section.")

    # --- Only add content to this section if we found detailed data OR charts ---
    if detailed_assessment or eval_chart_files:
        print(f"Save Summary: Adding Detailed Assessment content. Detailed data found: {bool(detailed_assessment)}, Charts found: {len(eval_chart_files)}")

        # --- Add overall assessment and feedback before the detailed options/charts ---
        doc.add_heading("整體評估與回饋摘要 (Overall Assessment and Feedback Summary):", level=2) # New heading

        p_assessment = doc.add_paragraph()
        p_assessment.add_run("整體評估 (Overall Assessment): ").bold = True
        p_assessment.add_run(eval_assessment_text)

        p_feedback = doc.add_paragraph()
        p_feedback.add_run("整體 LLM 回饋 (Overall LLM Feedback): ").bold = True
        p_feedback.add_run(eval_feedback_text)
        doc.add_paragraph() # Spacer


        if eval_chart_files: # Only add chart section if charts were found
             # Add a heading for the charts section within Detailed Assessment
             if detailed_assessment: # If detailed data was present, add this as a sub-heading
                  doc.add_heading("評估圖表總覽 (Evaluation Charts Overview):", level=2)
             else: # If no detailed data, this acts as the main content after the Level 1 heading
                  # Keep level 2 heading for consistency within the section
                  doc.add_heading("評估圖表總覽 (Evaluation Charts Overview):", level=2)

             # Add explanatory text regardless of detailed data presence
             doc.add_paragraph("以下圖表視覺化呈現了設計方案在各評估指標上的表現。")

             # Sort charts if needed (e.g., radar first, then bars)
             # Simple sort: radar charts first
             eval_chart_files.sort(key=lambda x: 0 if "evaluation_radar" in x.get("filename", "").lower() else 1) # Use updated sorting key

             print(f"Save Summary: Proceeding to embed {len(eval_chart_files)} charts.")
             for chart_file_info in eval_chart_files:
                  chart_path_str = chart_file_info.get("path")
                  chart_filename = chart_file_info.get("filename", "N/A")
                  chart_description = chart_file_info.get("description", chart_filename)

                  # --- NEW: Add the three-line file info for the chart file itself ---
                  if chart_path_str: # Only add file info if path exists
                      print(f"Save Summary: Adding file info for chart {chart_filename}")
                      p_chart_info_line1 = doc.add_paragraph()
                      chart_source_agent = chart_file_info.get("SourceAgent", "N/A")
                      chart_task_desc_short = chart_file_info.get("TaskDescShort", chart_description)
                      chart_image_num_info = chart_file_info.get("ImageNum", "")
                      chart_image_num_display = f"; ImageNum: {chart_image_num_info}" if chart_image_num_info else ""
                      chart_info_line1_text = f"File: SourceAgent: {chart_source_agent}; TaskDesc: {chart_task_desc_short}{chart_image_num_display}"
                      p_chart_info_line1.add_run(chart_info_line1_text).bold = True

                      p_chart_file_path = doc.add_paragraph()
                      p_chart_file_path.add_run("檔案位置 (File Path): ").bold = True
                      p_chart_file_path.add_run(chart_path_str)

                      p_chart_file_details = doc.add_paragraph()
                      run_chart_file_details = p_chart_file_details.add_run(f"(檔名: {chart_filename}, 類型: Image/png)") # Assume charts are png
                      run_chart_file_details.italic = True
                      run_chart_file_details.font.size = Pt(9)
                      doc.add_paragraph() # Space after chart file info
                  # --- END NEW ---


                  if chart_path_str and Path(chart_path_str).exists():
                       print(f"Save Summary: Embedding evaluation chart: {chart_filename} from {chart_path_str}")
                       try:
                            # Copy chart to assets folder for consistency
                            asset_target_dir = Path(final_summary_output_dir) / "eval_charts"
                            os.makedirs(asset_target_dir, exist_ok=True)
                            destination_chart_file = asset_target_dir / Path(chart_path_str).name
                            # Ensure parent exists
                            destination_chart_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(chart_path_str, destination_chart_file)
                            print(f"Save Summary: Copied chart to {destination_chart_file}")


                            try:
                                # Attempt to embed the chart image
                                print(f"Save Summary: Attempting to embed chart image {destination_chart_file}.")
                                # --- MODIFIED: Create paragraph for chart image and set alignment ---
                                p_chart_image = doc.add_paragraph()
                                run_chart_image = p_chart_image.add_run()
                                run_chart_image.add_picture(str(destination_chart_file), width=Inches(6.0))
                                p_chart_image.alignment = WD_ALIGN_PARAGRAPH.CENTER # Center the image paragraph
                                # --- END MODIFIED ---
                                print(f"Save Summary: Successfully embedded chart image {chart_filename} directly and centered.")

                                # Add caption for the chart
                                para_caption = doc.add_paragraph()
                                # Use the file description for the caption
                                run_caption = para_caption.add_run(f"圖: {chart_description}")
                                run_caption.italic = True
                                run_caption.font.size = Pt(9)
                                para_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                doc.add_paragraph() # Add space after image/caption

                            except Exception as chart_embed_err:
                                print(f"Save Summary Error: Direct embedding of chart image {destination_chart_file} failed: {chart_embed_err}. Trying PIL.")
                                traceback.print_exc()
                                try:
                                    from PIL import Image # Ensure PIL is imported
                                    img = Image.open(str(destination_chart_file))
                                    if img.mode == 'RGBA':
                                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                        rgb_img.paste(img, mask=img.split()[3])
                                        img = rgb_img

                                    temp_chart_img_path = str(destination_chart_file) + "_pil_temp.jpg"
                                    img.save(temp_chart_img_path, format='JPEG')

                                    doc.add_picture(temp_chart_img_path, width=Inches(6.0))
                                    print(f"Save Summary: Successfully embedded chart image {chart_filename} using PIL conversion.")

                                    para_caption_pil = doc.add_paragraph()
                                    run_caption_pil = para_caption_pil.add_run(f"圖: {chart_description} (PIL 處理)")
                                    run_caption_pil.italic = True
                                    run_caption_pil.font.size = Pt(9)
                                    para_caption_pil.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                    doc.add_paragraph()

                                except Exception as pil_chart_err:
                                    print(f"Save Summary Error: PIL embedding of chart image {destination_chart_file} also failed: {pil_chart_err}")
                                    traceback.print_exc()
                                    doc.add_paragraph(f"  [無法自動嵌入圖表 '{chart_filename}': 直接嵌入與 PIL 處理皆失敗。錯誤: {pil_chart_err}] ([Could not automatically embed chart '{chart_filename}': Both direct and PIL failed. Error: {pil_chart_err}]")
                                    doc.add_paragraph()

                       except Exception as chart_copy_err:
                            print(f"Save Summary Error: Error copying chart asset {chart_path_str} for Word report: {chart_copy_err}")
                            traceback.print_exc()
                            doc.add_paragraph(f"\n  [圖表檔案複製錯誤: {chart_copy_err}] (Chart file copy error: {chart_copy_err})")
                            doc.add_paragraph()
                  else:
                       print(f"Save Summary Warning: Chart file path does not exist: {chart_path_str}")
                       doc.add_paragraph(f"[圖表檔案未找到: {chart_filename} (路徑無效或不存在: {chart_path_str})] (Chart file not found: {chart_filename} (Invalid or non-existent path: {chart_path_str}))")
                       doc.add_paragraph()

        doc.add_paragraph() # Spacer after charts section


        if detailed_assessment and isinstance(detailed_assessment, list): # Only add detailed scores if data exists
            # --- Add Detailed Option Scores Table ---
            # Add a page break before the detailed scores table IF charts were added before it
            if eval_chart_files:
                 doc.add_page_break()

            doc.add_heading("各方案詳細評估分數 (Detailed Scores per Option):", level=2) # New heading

            for option_data in detailed_assessment:
                if isinstance(option_data, dict):
                    option_id = option_data.get("option_id", "未知方案 (Unknown Option)") # Modified text
                    doc.add_heading(f"方案: {option_id}", level=3) # Changed level to 3

                    desc = option_data.get("description", "無可用描述 (No description available)") # Modified text
                    p_desc = doc.add_paragraph()
                    p_desc.add_run("描述 (Description): ").bold = True # Modified text
                    p_desc.add_run(desc)

                    doc.add_heading("分數 (Scores):", level=4) # Changed level to 4
                    table = doc.add_table(rows=1, cols=2)
                    table.style = 'Table Grid'
                    hdr_cells = table.rows[0].cells
                    hdr_cells[0].text = '評估標準 (Criteria)' # Modified text
                    hdr_cells[1].text = '分數 (0-10) (Score (0-10))' # Modified text

                    score_keys = [
                        ("user_goal_responsiveness_score_final", "使用者目標響應度 (User Goal Responsiveness)"), # Modified text
                        ("aesthetics_context_score_final", "美學與情境契合度 (Aesthetics & Context)"), # Modified text
                        ("functionality_flexibility_score_final", "功能性與彈性 (Functionality & Flexibility)"), # Modified text
                        ("durability_maintainability_score_final", "耐久性與可維護性 (Durability & Maintainability)"), # Modified text
                        ("cost_efficiency_score_final", "成本效益 (Cost Efficiency)"), # Modified text
                        ("green_building_score_final", "綠建築永續潛力 (Green Building Potential)") # Modified text
                    ]

                    for key, display_name in score_keys:
                        row_cells = table.add_row().cells
                        row_cells[0].text = display_name
                        score_val = option_data.get(key)
                        if isinstance(score_val, (int, float)):
                            row_cells[1].text = f"{score_val:.1f}" if isinstance(score_val, float) else str(score_val)
                        elif score_val is None:
                            row_cells[1].text = "N/A"
                        else:
                            row_cells[1].text = str(score_val)

                    rationale = option_data.get("scoring_rationale", "無詳細分數理由 (No detailed rationale available)") # Modified text
                    doc.add_heading("分數理由 (Scoring Rationale):", level=4) # Changed level to 4
                    doc.add_paragraph(rationale)
                    doc.add_paragraph() # Spacer after each option

        doc.add_page_break() # Add page break after the entire detailed assessment section if content was added


    else: # No detailed assessment data AND no chart files found in the latest eval task
        doc.add_paragraph("未在已完成的評估任務中找到詳細評估結果或相關圖表資料。(No detailed assessment data or related charts found in the completed evaluation tasks.)") # Modified text
        print(f"Save Summary: No detailed assessment data or charts found to include in the Detailed Assessment section.")
        doc.add_paragraph() # Add space even if empty
        doc.add_page_break() # Add page break even if empty

    # --- End of Detailed Assessment Results Section ---


    doc.add_paragraph()
    # Modified heading text
    doc.add_heading("報告結束 (End of Report)", level=1)

    try:
        doc.save(word_filepath)
        print(f"Final summary Word document saved to: {word_filepath}")
    except Exception as e:
        print(f"Error saving Word document: {e}")
        traceback.print_exc()
        try:
            fallback_word_path = os.path.join(final_summary_output_dir, "Fallback_Summary.docx")
            doc.save(fallback_word_path)
            print(f"Saved Word document with fallback name: {fallback_word_path}")
        except Exception as fe:
            print(f"Failed to save Word document with fallback name: {fe}")

    # ... (JSON saving logic remains the same) ...
    try:
        serializable_state = {}
        for key, value in state.items():
            if key == "config":
                serializable_state[key] = "Configuration object (not fully serialized)"
            elif isinstance(value, Path):
                serializable_state[key] = str(value)
            elif key == "tasks" and isinstance(value, list):
                 serializable_state[key] = []
                 for task_item in value:
                     if isinstance(task_item, dict):
                         s_task = {}
                         for t_key, t_val in task_item.items():
                             if t_key == "mcp_internal_messages" and isinstance(t_val, list):
                                 s_task[t_key] = t_val
                             elif t_key == "output_files" and isinstance(t_val, list): # Filter base64 from files before saving state
                                 s_task[t_key] = []
                                 for f_item in t_val:
                                     if isinstance(f_item, dict):
                                         f_copy = f_item.copy()
                                         f_copy.pop("base64_data", None)
                                         s_task[t_key].append(f_copy)
                                     else:
                                         s_task[t_key].append(f_item) # Should not happen
                             elif isinstance(t_val, (dict, list, str, int, float, bool, type(None))):
                                 s_task[t_key] = t_val
                             else:
                                 s_task[t_key] = f"<Non-serializable type: {type(t_val).__name__}>"
                         serializable_state[key].append(s_task)
                     else:
                         serializable_state[key].append(f"<Non-dict task item: {type(task_item).__name__}>")
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable_state[key] = value
            else:
                serializable_state[key] = f"<Non-serializable type: {type(value).__name__}>"


        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=4, ensure_ascii=False, default=str)
        print(f"Final summary JSON state saved to: {json_filepath}")
    except Exception as e:
        print(f"Error saving JSON state: {e}")
        try:
            fallback_json_path = os.path.join(final_summary_output_dir, "Fallback_State.json")
            with open(fallback_json_path, 'w', encoding='utf-8') as f:
                 json.dump({"error": "Failed to serialize full state, this is a minimal fallback.", "original_error": str(e)}, f, indent=4, ensure_ascii=False)
            print(f"Saved JSON state with fallback name: {fallback_json_path}")
        except Exception as fe:
            print(f"Failed to save JSON state with fallback name: {fe}")


    state["final_summary_word_path"] = word_filepath
    state["final_summary_json_path"] = json_filepath

    state["current_phase"] = "qa"
    print(f"Final summary saved. Set current_phase to 'qa' for routing to QA loop.")

    return state

def initialize_workflow(user_input: str) -> WorkflowState:
    """Initializes the workflow state for a new run."""
    print(f"Initializing workflow for input: {user_input[:100]}...")
    initialized_state = WorkflowState(
        user_input=user_input,
        tasks=[],
        current_task_index=0,
        current_phase="task_execution",
        qa_context=[],
        interrupt_input=None,
        current_task=None
    )
    print(f"Initialized workflow state: Phase='{initialized_state['current_phase']}', user_input='{user_input[:50]}...'")
    return initialized_state



# =============================================================================
# 9. 圖邊緣條件函數定義 (Update routing)
# =============================================================================
def route_from_process_management(state: WorkflowState) -> Literal[
    "assign_teams", "eva_teams", "save_final_summary", "qa_loop", "finished", "process_management"
]:
    """ Determines the next node based on task status and selected agent."""
    print("--- Routing decision @ route_from_process_management (v2) ---") # Version indicator
    current_phase = state.get("current_phase")
    interrupt_result = state.get("interrupt_result")
    interrupt_action = interrupt_result.get("action") if isinstance(interrupt_result, dict) else None

    print(f"Router received state: Phase='{current_phase}', Interrupt Action='{interrupt_action}'")

    # --- 1. Check QA / Finished ---
    if current_phase == "qa" or interrupt_action == "CONVERSATION":
         state["interrupt_result"] = None
         print(f"Routing -> qa_loop (Reason: Phase='{current_phase}' or Interrupt Action='CONVERSATION')")
         return "qa_loop"
    if current_phase == "finished":
        print("Routing -> finished (Reason: Phase is 'finished')")
        return "finished"

    # --- 2. Normal Task Execution Flow ---
    tasks = state.get("tasks", [])
    current_index = state.get("current_task_index", 0)

    if not tasks or current_index >= len(tasks):
        if current_phase != "finished":
             print(f"Routing -> save_final_summary (Reason: All tasks completed or list empty, index={current_index}, count={len(tasks)})")
             return "save_final_summary"
        else:
             print("Routing -> finished (Reason: Tasks done and phase 'finished')")
             return "finished"


    if 0 <= current_index < len(tasks):
        current_task = tasks[current_index]
        task_id = current_task.get('task_id', 'N/A')
        task_status = current_task.get("status")
        selected_agent = current_task.get("selected_agent")
        requires_eval = current_task.get("requires_evaluation", False)

        print(f"  Checking Task {task_id} (Idx: {current_index}), Status: '{task_status}', Agent: '{selected_agent}', RequiresEval: {requires_eval}")

        if task_status in ["failed", "max_retries_reached"]:
            print(f"Routing -> process_management (Reason: Task {task_id} status '{task_status}', returning for failure analysis)")
            return "process_management"

        # --- MODIFIED: Check for specific evaluation agents ---
        is_evaluation_agent = selected_agent in ["EvaAgent", "SpecialEvaAgent", "FinalEvaAgent"]

        if is_evaluation_agent:
            if requires_eval:
                if task_status in ["pending", "in_progress", None]: # Should be pending
                    print(f"Routing -> eva_teams (Reason: Task {task_id} is agent '{selected_agent}', requires evaluation, and status '{task_status}')")
                    # --- CORRECTION: The target node is still called "eva_teams" which triggers EvaAgent.run ---
                    return "eva_teams" # Route to the single evaluation entry point
                else:
                    print(f"Routing Warning: Evaluation Task {task_id} ({selected_agent}) has unexpected status '{task_status}'. Looping PM.")
                    return "process_management"
            else:
                 print(f"Routing Error: Evaluation Task {task_id} ({selected_agent}) has requires_evaluation=False. Looping PM.")
                 # This indicates an error in PM's plan generation
                 return "process_management"
        else: # Not an evaluation agent
            if task_status in ["pending", "in_progress", None]:
                print(f"Routing -> assign_teams (Reason: Task {task_id} agent '{selected_agent}' status '{task_status}' and ready for execution)")
                return "assign_teams"
            else: # Should already be completed or failed, handled above
                print(f"Routing Warning: Non-Eval Task {task_id} has unexpected status '{task_status}'. Looping PM.")
                return "process_management"
    else:
        print(f"Routing Warning: index ({current_index}) out of bounds (size {len(tasks)}). Fallback to save_final_summary.")
        return "save_final_summary"


# ... (route_after_assign_teams保持不變) ...
def route_after_assign_teams(state: WorkflowState) -> Literal["process_management", "finished"]:
    """
    Routes after the assign_teams subgraph. ALWAYS routes back to Process Management
    for status checking, LTM saving, and potential evaluation triggering.
    """
    # --- <<< 增加詳細日誌 >>> ---
    print(f"--- Debug log @ route_after_assign_teams ---")
    tasks = state.get("tasks", [])
    current_idx = state.get("current_task_index", -1)
    print(f"  Received current_task_index: {current_idx}")
    if 0 <= current_idx < len(tasks):
        task_in_state = tasks[current_idx]
        status_in_state = task_in_state.get('status')
        task_id_in_state = task_in_state.get('task_id')
        print(f"  Task {task_id_in_state} at index {current_idx} has status: '{status_in_state}' in the received state.")
    else:
        print(f"  Invalid current_task_index ({current_idx}) or empty tasks list.")
    # --- <<< 結束增加日誌 >>> ---

    # Basic index check
    if current_idx < 0 or current_idx >= len(tasks):
        print(f"Routing Error: Invalid task index {current_idx} after assign_teams. Routing to finished.")
        return "finished"

    print(f"Route after assign_teams: Routing decision -> process_management")
    return "process_management"

# def route_after_evaluation(state: WorkflowState) -> Literal["process_management", "assign_teams", "finished"]:
#     """Routes after EvaAgent (evaluation subgraph): Continue (PM) or retry (assign_teams)."""
#     current_phase = state.get("current_phase")
#     if current_phase != "task_execution":
#         print(f"Route after Eva Error: Unexpected phase '{current_phase}'. Routing to finished.")
#         return "finished"

#     tasks = state.get("tasks", [])
#     current_idx = state.get("current_task_index", -1)
#     if current_idx < 0 or current_idx >= len(tasks):
#         print("Routing Error: Invalid index after evaluation. Routing to finished.")
#         return "finished"

#     task = tasks[current_idx]
#     status = task.get("status")
#     retry_count = task.get("retry_count", 0)
#     max_retries = _full_static_config.agents.get("process_management", {}).parameters.get("max_retries", 3)

#     print(f"Route after Eva: Checking Task {current_idx}, Status: {status}, RetryCount: {retry_count}")

#     if status == "completed":
#         print("Route after Eva: Evaluation passed. Routing to PM to advance.")
#         return "process_management"
#     elif status == "failed":
#         print(f"Route after Eva: Evaluation failed. Routing to PM for assessment.")
#         return "process_management"
#     else:
#         print(f"Route after Eva Error: Unexpected status '{status}'. Routing to PM as failsafe.")
#         return "process_management"


# --- <<< 修改 route_after_chat_bot >>> ---
def route_after_chat_bot(state: WorkflowState) -> Literal["process_management", "qa_loop", "finished"]:
    """基於QA代理設置的階段路由到適當的節點"""
    current_phase = state.get("current_phase")
    print(f"--- 路由決策 @ route_after_chat_bot ---")
    print(f"  當前階段: '{current_phase}'")
    
    # 直接根據QA代理設置的階段進行路由，不再進行重複檢測
    if current_phase == "task_execution":
        print("路由 -> process_management (原因: QA代理設置階段為'task_execution')")
        return "process_management"
    elif current_phase == "finished":
        print("路由 -> finished (原因: QA代理設置階段為'finished')")
        return "finished"
    elif current_phase == "qa":
        print("路由 -> qa_loop (原因: QA代理設置階段為'qa')")
        return "qa_loop"
    else:
        print(f"路由警告: 未預期的階段 '{current_phase}'。預設路由到qa_loop。")
        return "qa_loop"


def qa_loop_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Manages the user input part of the QA loop.
    Processes interrupt_input ONLY if present.
    Returns update for qa_context (with HumanMessage) and clears interrupt_input.
    If no interrupt_input, returns minimal state update.
    """
    node_name = "QA Loop Manager"
    print(f"--- Running Node: {node_name} ---")

    interrupt_input = state.get("interrupt_input")
    update_dict = {} # Initialize update dictionary

    if interrupt_input:
        print(f"{node_name}: Processing query from interrupt_input: '{interrupt_input[:100]}...'")
        message_to_add = HumanMessage(content=interrupt_input)
        update_dict["qa_context"] = [message_to_add] # Add user message
        update_dict["interrupt_input"] = None # Clear interrupt input
    else:
        # No new interrupt input, just return empty update or clear interrupt if needed
        print(f"{node_name}: No new interrupt_input found. Returning minimal update.")
        # Ensure interrupt_input is None in the returned state
        update_dict["interrupt_input"] = None
        # Do NOT add any message to qa_context here

    return update_dict


# =============================================================================
# 10. 圖定義與節點/邊緣添加
# =============================================================================
workflow = StateGraph(WorkflowState, config_schema=ConfigSchema)

# Add Nodes (No changes here)
workflow.add_node("process_management", process_management.run)
workflow.add_node("assign_teams", assign_teams)
workflow.add_node("eva_teams", eva_agent.run) # Still points to the EvaAgent.run trigger
workflow.add_node("chat_bot", qa_agent.run)
workflow.add_node("save_final_summary", save_final_summary)
workflow.add_node("qa_loop", qa_loop_node)

# Set Entry Point
workflow.set_entry_point("process_management")

# Add Edges
workflow.add_conditional_edges(
    "process_management",
    route_from_process_management, # Uses updated logic
    {
        "assign_teams": "assign_teams",
        "eva_teams": "eva_teams", # Still route to the eva_teams node trigger
        "save_final_summary": "save_final_summary",
        "qa_loop": "qa_loop",
        "process_management": "process_management", # Loop back for failure/retries
        "finished": END,
    }
)

workflow.add_conditional_edges(
    "assign_teams",
    route_after_assign_teams, # Still goes back to PM
    {
        "process_management": "process_management",
        "finished": END # Should ideally not happen if PM handles empty tasks
    }
)

# --- REMOVED Conditional edge for eva_teams ---
# The main router now handles the state after evaluation completes.
# workflow.add_conditional_edges(
#     "eva_teams",
#     route_after_evaluation, # REMOVED
#     {       
#        "process_management": "process_management",
#        "finished": END
# )
# --- Instead, eva_teams implicitly returns to the main router ---
# The output of eva_teams (which is the state updated by the subgraph)
# flows back to the main graph logic implicitly after the node finishes.
# The next routing decision happens at the START of the next cycle, triggered
# by the process_management node again reading the updated state.
# THEREFORE, we need an edge FROM eva_teams BACK to process_management.
workflow.add_edge("eva_teams", "process_management")


# QA Loop Edges (Remain same)
workflow.add_edge("qa_loop", "chat_bot")
workflow.add_conditional_edges(
    "chat_bot",
    route_after_chat_bot,
    {
        "process_management": "process_management",
        "qa_loop": "qa_loop",
        "finished": END
    }
)

# Final Summary Edge (Remains same)
workflow.add_edge("save_final_summary", "process_management")


# =============================================================================
# 11. 圖編譯
# =============================================================================
# Need to adjust interrupt points if EvaAgent.run is just a trigger
# Interrupting before eva_teams might be less useful now.
# Interrupting *inside* the subgraph might be better if needed.
# For now, keep existing interrupts.
graph = workflow.compile(interrupt_before=["qa_loop"], interrupt_after=["chat_bot"])
# graph = workflow.compile() #沒有中斷點的編譯
graph.name = "General_Arch_graph_v20_AgentBasedEval" # Updated name
print("Main Graph compiled successfully (v20 - Agent-Based Eval with QA interrupts enabled).")

# =============================================================================
# 12. TODOLIST
# =============================================================================
# 1. QA test   V
# 2. from insert enter QA node test    V
# 3. Fauilure analysis test   V
# 4. Building life cycle tools add in  (maybe 3D model)  V
# 5. final configuration systematically...    V
# 6. 3D model input / output test(MCP)  V
# 7. Change node name to agent name    V
# 8. final evaluation 好像不太行 
# 9. 檢查save summary 與長期記憶有關   V

