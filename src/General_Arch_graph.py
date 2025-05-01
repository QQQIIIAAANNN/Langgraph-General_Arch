# =============================================================================
# 1. Imports
# =============================================================================
import os
import uuid
import json
import base64
from typing import Dict, List, Any, Annotated, Literal, Union, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
import traceback # Added for error printing

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
from src.tools.case_render_image import case_render_image
from src.tools.generate_3D import generate_3D
from src.tools.simulate_future_image import simulate_future_image

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
        if task_id == 'N/A':
            print("PM LTM Warning: Cannot save task without ID.")
            return

        print(f"PM LTM: Preparing to save task {task_id} with status '{status}'...")
        summary = self._create_task_summary(task)
        # print(f"PM LTM Summary for {task_id}:\n{summary}\n--------------------") # Optional: Debug print summary

        # Create LangChain Document object
        # Metadata helps retrieval filtering later if needed
        metadata = {
            "task_id": task_id,
            "status": status,
            "agent": task.get("selected_agent", "N/A"),
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
        # --- <<< 修改：移除 QA 階段快速通道檢查，讓路由處理 >>> ---
        # current_phase = state.get("current_phase")
        # if current_phase == "qa":
        #     print("PM: In QA phase. Skipping task processing. Routing based on phase.")
        #     return state # 直接返回

        output_state = state.copy()
        output_state["interrupt_result"] = None

        print(f"--- Debug log @ PM.run (Start) ---") # 日誌保持
        # ... (打印 initial state 的日誌 - 不變) ...
        initial_tasks = state.get("tasks", [])
        initial_idx = state.get("current_task_index", 0)
        print(f"  Received initial current_task_index: {initial_idx}")
        print(f"  Received initial task count: {len(initial_tasks)}")
        if 0 <= initial_idx < len(initial_tasks):
             print(f"  Initial task at index {initial_idx} status: '{initial_tasks[initial_idx].get('status')}'")


        tasks = state.get("tasks", [])
        current_idx = state.get("current_task_index", 0)
        interrupt_input = state.get("interrupt_input")

        # --- 1. Interrupt Processing ---
        if interrupt_input:
            print(f"PM: Interrupt detected: '{interrupt_input[:100]}...'. Invoking LLM for analysis.")
            # ... (LLM interrupt analysis logic - 保持不變) ...
            summarized_tasks = [{"task_id": t.get("task_id", "N/A"), "objective": t.get("task_objective", "N/A")} for t in tasks]
            tasks_json_for_interrupt = json.dumps(summarized_tasks, ensure_ascii=False)
            current_task_for_interrupt = tasks[current_idx] if 0 <= current_idx < len(tasks) else None
            current_task_json_for_interrupt = json.dumps(current_task_for_interrupt, ensure_ascii=False, default=lambda o: '<not serializable>') if current_task_for_interrupt else "{}"

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

                # --- <<< 修改：只存儲結果，路由函數決定是否進入 QA >>> ---
                output_state["interrupt_result"] = interrupt_result # Store full result
                print(f"PM: LLM analysis action: {action}")

                if action == "CONVERSATION":
                    print("PM: LLM analysis resulted in CONVERSATION. Setting phase to 'qa'.")
                    output_state["current_phase"] = "qa"
                    # --- 不清除 interrupt_input，留給 QA Agent ---
                    # --- 但清除 interrupt_result，因為 phase 已設定，路由會處理 ---
                    # --- 或者讓路由函數讀取 interrupt_result['action']? 保持 interrupt_result 供路由使用 ---
                    # output_state["interrupt_result"] = None # 清除可能更好? 路由直接看 phase
                    # <<< 決定：讓路由函數檢查 interrupt_result['action'] 和 current_phase >>>
                    return output_state # 返回，讓路由函數處理

                # --- 處理 Command Actions (REPLACE, INSERT, PROCEED) ---
                if action == "REPLACE_TASKS":
                    # ... (原本的 REPLACE_TASKS 邏輯) ...
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
                        else:
                            print("PM Warning (REPLACE): No valid tasks created. Proceeding.")
                            action = "PROCEED"
                    else:
                        print("PM Warning (REPLACE): 'new_tasks_list' missing or invalid. Proceeding.")
                        action = "PROCEED"

                elif action == "INSERT_TASKS":
                    # ... (原本的 INSERT_TASKS 邏輯) ...
                    print("PM: Processing INSERT_TASKS command.")
                    insert_tasks_data = interrupt_result.get("insert_tasks_list")
                    if isinstance(insert_tasks_data, list):
                        inserted_tasks = []
                        for i, task_data in enumerate(insert_tasks_data):
                            if isinstance(task_data, dict):
                                task_id = str(uuid.uuid4())
                                task = TaskState(
                                    task_id=task_id, status="pending",
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
                            else: print(f"PM Warning (INSERT): Invalid item in insert_tasks_list at index {i}.")

                        if inserted_tasks:
                            current_tasks_in_state = output_state.get("tasks", tasks)
                            insert_after_idx = state.get("current_task_index", 0) # Index of the task *before* insertion
                            valid_insert_point = insert_after_idx + 1 # Insert after this index
                            print(f"PM (INSERT): Inserting {len(inserted_tasks)} tasks at index {valid_insert_point}.")
                            new_sequence = current_tasks_in_state[:valid_insert_point] + inserted_tasks + current_tasks_in_state[valid_insert_point:]
                            output_state["tasks"] = new_sequence
                            # --- <<< 修改：插入後，index 應指向第一個插入的任務 >>> ---
                            output_state["current_task_index"] = valid_insert_point
                            output_state["current_task"] = inserted_tasks[0].copy()
                            # --- <<< 結束修改 >>> ---
                        else:
                            print("PM Warning (INSERT): No valid tasks created. Proceeding.")
                            action = "PROCEED"
                    else:
                        print("PM Warning (INSERT): 'insert_tasks_list' missing or invalid. Proceeding.")
                        action = "PROCEED"


                if action == "PROCEED":
                    print("PM: PROCEED command received or fallback.")
                    pass

                # --- 清除非 CONVERSATION 命令的中斷輸入和結果 ---
                print(f"PM: Processed command '{action}'. Clearing interrupt_input and result.")
                output_state["interrupt_input"] = None
                output_state["interrupt_result"] = None # 清除結果

            except Exception as e:
                print(f"PM Error processing interrupt command: {e}")
                traceback.print_exc()
                output_state["interrupt_input"] = None
                output_state["interrupt_result"] = None # 清除錯誤時的結果

        # --- 2. Task Processing Logic ---
        # (確保這部分邏輯只在非 QA 階段執行)
        tasks = output_state.get("tasks", []) # Use potentially updated tasks
        current_idx = output_state.get("current_task_index", 0)

        # --- Initial Workflow Creation (if tasks is empty) ---
        if not tasks:
            # ... (原本的 workflow creation 邏輯 - 保持不變) ...
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
        print(f"\nPM: --- Debug log: Entering status check loop ---")
        tasks = output_state.get("tasks", []) # Use potentially updated tasks from interrupt/creation
        current_idx = output_state.get("current_task_index", 0)
        print(f"PM: Starting loop with current_idx = {current_idx}, task count = {len(tasks)}")

        while 0 <= current_idx < len(tasks):
            task_to_check = tasks[current_idx]
            status = task_to_check.get("status")
            task_id = task_to_check.get("task_id", "N/A")
            # --- Log current task being checked ---
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
                    print(f"  Feedback Log (Input): {fail_prompt_input['feedback_log'][:500]}...")
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

                    current_idx += 1
                    output_state["current_task_index"] = current_idx # Update index before continuing
                    # Ensure current_task reflects the next task or None
                    output_state["current_task"] = tasks[current_idx].copy() if 0 <= current_idx < len(tasks) else None
                    print(f"PM (Fail): Updated current_idx to {current_idx}. Continuing loop.")
                    continue # Continue WHILE loop to check the NEXT task

            # --- End of failed/max_retries_reached block ---

            elif status == "completed":
                # Completed logic remains the same
                print(f"PM Loop: Handling 'completed' for task {task_id}. Saving to LTM.")
                await self._save_task_to_ltm(task_to_check)
                current_idx += 1
                output_state["current_task_index"] = current_idx
                print(f"PM Loop: Incremented index to {current_idx} after completing task {task_id}. Continuing loop.")
                continue # Continue WHILE loop

            elif status in ["pending", "in_progress", None]:
                # Pending/In Progress logic remains the same
                print(f"PM Loop: Task {task_id} is '{status}'. Ready for routing. Breaking loop.")
                output_state["current_task_index"] = current_idx
                output_state["current_task"] = task_to_check.copy()
                break # Break WHILE loop

            else:
                # Unexpected status logic remains the same
                print(f"PM Loop Warning: Unhandled task status '{status}' for Task {task_id}. Skipping.")
                current_idx += 1
                output_state["current_task_index"] = current_idx
                print(f"PM Loop (Warning): Updated current_idx to {current_idx}. Continuing loop.")
                continue

        # --- After the loop ---
        # Logging remains the same
        print(f"PM: --- Debug log: Exited status check loop ---")
        final_idx_after_loop = output_state.get("current_task_index", current_idx) # Get potentially updated index
        print(f"PM: Final index after loop processing: {final_idx_after_loop}")
        if final_idx_after_loop >= len(tasks):
            print("PM: Reached end of task list after loop.")
            # State for current_task/index already set inside loop or at end
            output_state["current_task"] = None # Ensure current task is None if index is out of bounds
        # else: Task found, index/task already set inside loop's break condition

        # Final state preparation remains the same
        if "tasks" not in output_state: output_state["tasks"] = state.get("tasks", [])
        print(f"=== Process Management Finished. Routing Index: {output_state.get('current_task_index', 'N/A')}, Phase: {output_state.get('current_phase')} ===")
        return output_state

class EvaAgent:
    """
    Parent node responsible for initiating evaluation by invoking the evaluation_subgraph.
    It now marks tasks intended for final/holistic evaluation.
    """
    def __init__(self):
        pass # No internal state needed

    async def run(self, state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Checks if evaluation is needed, marks final eval tasks, invokes subgraph,
        and updates Task status based on subgraph results.
        """
        print("--- Running EvaAgent Node (TaskState Version) ---")
        current_idx = state["current_task_index"]
        tasks = [t.copy() for t in state["tasks"]]
        if current_idx >= len(tasks):
             print("EvaAgent Error: Invalid task index.")
             return {} # Should not happen if routing is correct
        current_task = tasks[current_idx]

        # --- MODIFIED: Check if evaluation is needed ---
        # If the task requires evaluation AND is pending, EvaAgent should handle it,
        # regardless of whether it was assigned directly to EvaAgent or is an intermediate eval.
        needs_evaluation_now = (
            current_task.get("requires_evaluation", False) and
            current_task.get("status") == "pending"
        )
        # The is_final_evaluation_task check remains separate and correct
        # It also implies status is 'pending' and requires_evaluation is True.
        is_final_evaluation_task = (
            current_task.get("task_objective") == "final_evaluation" and
            needs_evaluation_now # Simplified: If it's final objective AND pending+requires_eval
        )

        # Combine the conditions: If it's pending and requires evaluation (either regular or final)
        should_run_evaluation = needs_evaluation_now # This now covers both regular and final pending evaluations

        if not should_run_evaluation:
            # Log reason more clearly
            status = current_task.get('status')
            req_eval = current_task.get('requires_evaluation')
            print(f"EvaAgent: Skipping task {current_task['task_id']} - Evaluation not triggered (Requires Eval: {req_eval}, Status: {status}).")
            return {"tasks": state.get("tasks", [])} # Return original tasks list

        # --- If should_run_evaluation is True, proceed with the rest of the logic ---
        print(f"EvaAgent: Preparing evaluation for Task {current_task['task_id']}")
        if "evaluation" not in current_task: # Ensure evaluation dict exists
            current_task["evaluation"] = {}

        # --- Set status to in_progress to signify evaluation start ---
        # This should happen for both regular and final evaluation tasks entering here
        # Use is_final_evaluation_task flag determined above for logging clarity
        print(f"EvaAgent: Task requires evaluation (Final: {is_final_evaluation_task}). Updating status to in_progress.")
        current_task["status"] = "in_progress" # Mark evaluation as started

        # --- Mark if this is a final evaluation task ---
        # is_final_evaluation_task is already determined above
        if is_final_evaluation_task:
             print("EvaAgent: Identified as FINAL evaluation task.")
             current_task["evaluation"]["is_final_evaluation"] = True
        else:
             print("EvaAgent: Identified as REGULAR intermediate evaluation task.")
             current_task["evaluation"]["is_final_evaluation"] = False # Ensure it's False for regular

        # Update the tasks list in the state *before* calling the subgraph
        # Create a new list with the updated task
        tasks = tasks[:current_idx] + [current_task] + tasks[current_idx+1:]
        state["tasks"] = tasks # Update the state that will be passed to the subgraph


        # --- Invoke Subgraph (logic remains same) ---
        try:
            print(f"EvaAgent: Invoking evaluation subgraph (operates on WorkflowState)...")
            # Pass the modified state (with updated task status) to the subgraph
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


            final_task = final_tasks[current_idx] # Get the potentially modified task
            subgraph_internal_error = final_task.get("evaluation", {}).get("subgraph_error")

            # Ensure status reflects internal errors (logic remains same)
            if subgraph_internal_error and final_task["status"] != "failed":
                print(f"EvaAgent: Evaluation subgraph reported an internal error: {subgraph_internal_error}. Ensuring status is 'failed'.")
                final_task["status"] = "failed"
                error_log = final_task.get("error_log", "")
                if f"Evaluation Subgraph Error: {subgraph_internal_error}" not in error_log:
                    final_task["error_log"] = f"{error_log}; Evaluation Subgraph Error: {subgraph_internal_error}".strip("; ")

            # Final status logging (logic remains same)
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

            # Return the full updated tasks list from the subgraph output state
            return {"tasks": final_tasks}

        except Exception as e:
            # Handle errors during the invocation of the subgraph itself (logic remains same)
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


    async def run(self, state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]: # 返回類型改為 Dict
        """Generates AI response based on context, determines next phase. Returns minimal state update."""
        node_name = "QA Agent Node (chat_bot)"
        print(f"--- Running Node: {node_name} (AI Response Turn) ---") # Updated log message

        # --- <<< 新增：檢查最後一條消息是否為提示 >>> ---
        qa_context_list: List[BaseMessage] = state.get("qa_context", [])
        last_message = qa_context_list[-1] if qa_context_list else None
        prompt_text_from_loop = "請問您想問什麼？請在Interrupt Input欄位中提供您的想法。"

        if isinstance(last_message, AIMessage) and last_message.content == prompt_text_from_loop:
            print(f"{node_name}: Last message was the prompt from qa_loop. Skipping LLM invocation and waiting for user input.")
            return {}
        # --- 結束檢查 ---


        runtime_config = config["configurable"]
        qa_llm_config = runtime_config.get("qa_llm", {})
        retriever_k = runtime_config.get("retriever_k", 5)
        llm = initialize_llm(qa_llm_config)
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=retriever_k))
        ltm = VectorStoreRetrieverMemory(retriever=retriever, memory_key=ltm_memory_key, input_key=ltm_input_key)
        llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)

        current_phase = state.get("current_phase")

        # --- 安全檢查 ---
        if not qa_context_list or not isinstance(qa_context_list[-1], (HumanMessage, AIMessage)):
             print(f"{node_name}: Skipping, invalid qa_context state (last message is not User/AI).")
             return {}

        # --- 獲取查詢 ---
        # We need the user's actual query for LTM and prompt input.
        # The last message *should* be a HumanMessage here because the prompt case was handled above.
        query_for_ltm = ""
        last_user_query_content = ""
        if isinstance(last_message, HumanMessage):
            query_for_ltm = last_message.content
            last_user_query_content = last_message.content
        else:
            print(f"{node_name}: Warning - Expected last message to be HumanMessage, but got {type(last_message)}. Trying previous message for query.")
            if len(qa_context_list) > 1 and isinstance(qa_context_list[-2], HumanMessage):
                 query_for_ltm = qa_context_list[-2].content
                 last_user_query_content = qa_context_list[-2].content
            else:
                 print(f"{node_name}: Could not determine a user query for LTM/prompt.")


        # --- 初始化返回字典 ---
        return_state_update = {
             "current_phase": "qa" # 預設保持 QA 狀態
        }

        # --- Prepare context for the prompt ---
        retrieved_ltm_context = "LTM: N/A"
        if query_for_ltm: # Only retrieve if we have a query
            try:
                ltm_loaded_vars = await ltm.aload_memory_variables({ltm_input_key: query_for_ltm})
                context_from_ltm = ltm_loaded_vars.get(ltm.memory_key)
                if context_from_ltm: 
                    retrieved_ltm_context = context_from_ltm
            except Exception as e: 
                print(f"{node_name}: LTM Error: {e}")
        else:
            retrieved_ltm_context = "無用戶查詢可供 LTM 檢索。"

        task_summary = "\n".join([f"- Task {i+1}: {t['description']} ({t['status']})" for i, t in enumerate(state.get("tasks", []))]) or "No tasks executed."
        current_stm_vars = self.stm.load_memory_variables({})
        formatted_chat_history = current_stm_vars.get(self.stm.memory_key, "No STM history.")

        # --- 準備 LLM 輸入 ---
        chain_input = {}
        possible_inputs = {
            "last_user_query": last_user_query_content,
            "retrieved_ltm_context": retrieved_ltm_context,
            "window_size": self.memory_window_size,
            "chat_history": formatted_chat_history,
            "llm_output_language": llm_output_language,
            "task_summary": task_summary,
        }
        missing_prompt_vars = []
        for var in self.qa_prompt_input_variables:
            if var in possible_inputs:
                chain_input[var] = possible_inputs[var]
            else:
                missing_prompt_vars.append(var)
                chain_input[var] = "N/A"
        if missing_prompt_vars:
            print(f"QA_Agent Warning: Prompt template requires variables not available: {missing_prompt_vars}. Using 'N/A'.")


        # --- LLM Invocation ---
        ai_message = None
        try:
            qa_chain = self.prompt_template | llm
            response_message: AIMessage = await qa_chain.ainvoke(chain_input)
            response_content = response_message.content.strip()
            print(f"{node_name} Response: {response_content}")
            ai_message = response_message
        except Exception as llm_e:
             print(f"{node_name} LLM Error: {llm_e}")
             ai_message = AIMessage(content=f"Error: {llm_e}")
             return_state_update["current_phase"] = "qa"
             return_state_update["qa_context"] = [ai_message]
             return return_state_update

        # --- <<< 修改：處理 LLM 回應，增加 RESUME_TASK 判斷 >>> ---
        response_content = ai_message.content
        next_phase = "qa" # Default phase
        if response_content == "TERMINATE":
            ai_message.content = "好的，對話結束。" # 可以給用戶一個友好的回覆
            next_phase = "finished" # 更新 phase 為 finished
            print(f"{node_name}: Termination detected.")
        elif response_content.startswith("NEW_TASK:"):
            new_task_desc = response_content[len("NEW_TASK:") :].strip()
            ai_message.content = f"收到新任務：'{new_task_desc}'。正在返回任務規劃..." # 友好的回覆
            next_phase = "task_execution" # 更新 phase 為 task_execution
            return_state_update["user_input"] = new_task_desc # 將新任務目標傳回 state
            print(f"{node_name}: New task requested.")
        elif response_content == "RESUME_TASK": # <<< 新增判斷 >>>
            ai_message.content = "好的，正在返回任務執行流程。" # 友好的回覆
            next_phase = "task_execution" # 更新 phase 為 task_execution
            print(f"{node_name}: Task resumption requested.")
            # 不需要清除 qa_context，因為下次進入 PM 會重新開始任務流程
        else:
            # 普通回答，保持在 QA 階段
            print(f"{node_name}: Provided answer. Staying in QA.")
            next_phase = "qa"
        # --- <<< 結束修改 >>> ---


        # --- 更新 STM ---
        if isinstance(last_message, HumanMessage):
            self.stm.save_context({"human_input": last_message.content}, {"output": ai_message.content})

        # --- 準備返回的狀態更新 ---
        return_state_update["current_phase"] = next_phase
        return_state_update["qa_context"] = [ai_message] # 返回包含 AI 回覆的列表

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
    """Saves a summary of the completed workflow tasks to a Markdown file."""
    print("--- Saving Final Workflow Summary ---")
    tasks = state.get("tasks", [])
    user_input = state.get("user_input", "N/A") # Use current user_input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use UUID for guaranteed uniqueness even if run multiple times per second
    summary_filename = f"workflow_summary_{timestamp}_{uuid.uuid4().hex[:8]}.md" # Shortened UUID
    # Ensure output directory exists relative to the script or use an absolute path
    output_dir = Path(_full_static_config.workflow.output_directory) # Use loaded static config
    # Save summary in the parent of the output/Cache directory
    output_parent_dir = output_dir.parent
    os.makedirs(output_parent_dir, exist_ok=True) # Create parent dir if needed
    summary_filepath = os.path.join(output_parent_dir, summary_filename)

    summary_content = f"# Workflow Summary ({timestamp})\n\n"
    summary_content += f"## User Goal\n\n```text\n{user_input}\n```\n\n" # Changed title
    summary_content += f"## Executed Tasks ({len(tasks)})\n\n"

    for i, task in enumerate(tasks):
        summary_content += f"### Task {i+1}: {task.get('description', 'N/A')} (ID: {task.get('task_id', 'N/A')})\n\n"
        summary_content += f"*   **Objective:** `{task.get('task_objective', 'N/A')}`\n" # Renamed from Type
        summary_content += f"*   **Agent:** `{task.get('selected_agent', 'N/A')}`\n" # Added Agent
        summary_content += f"*   **Status:** {task.get('status', 'N/A')}\n"

        # --- MODIFIED: Add more detail for failed/retried tasks ---
        if task.get('status') in ['failed', 'max_retries_reached']:
            summary_content += f"*   **Retry Attempts:** {task.get('retry_count', 0)}\n"
            if task.get('error_log'):
                 summary_content += f"*   **Last Error Log:**\n    ```text\n    {task['error_log']}\n    ```\n"
            if task.get('feedback_log'): # Feedback log might contain analysis or eval feedback contributing to failure
                 summary_content += f"*   **Last Feedback Log (Analysis/Eval):**\n    ```text\n    {task['feedback_log']}\n    ```\n"
        # --- End MODIFIED ---

        inputs_str = json.dumps(task.get('task_inputs', {}), ensure_ascii=False, indent=2) # Use task_inputs
        summary_content += f"*   **Final Inputs Used:**\n    ```json\n    {inputs_str[:1000]}{'...' if len(inputs_str) > 1000 else ''}\n    ```\n" # Renamed

        outputs = task.get('outputs', {}) # Use correct key
        outputs_str = json.dumps(outputs, ensure_ascii=False, indent=2)
        if outputs:
            # Distinguish between content output and error output if structure allows
            if outputs.get('content'):
                 summary_content += f"*   **Output Content:**\n    ```text\n    {str(outputs['content'])[:2000]}{'...' if len(str(outputs['content'])) > 2000 else ''}\n    ```\n"
            elif outputs.get('error'):
                 error_detail = outputs.get('analysis', outputs['error'])
                 summary_content += f"*   **Error Output:**\n    ```text\n    Error: {outputs['error']}\n    Analysis: {error_detail}\n    ```\n"
            else: # General outputs dictionary
                 summary_content += f"*   **Structured Outputs:**\n    ```json\n    {outputs_str[:2000]}{'...' if len(outputs_str) > 2000 else ''}\n    ```\n"

        # Use correct key 'output_files'
        files = task.get('output_files', [])
        if files:
            summary_content += "*   **Generated Files:**\n"
            for file_info in files:
                file_desc = file_info.get('description', 'N/A')
                # Prefer filename, fallback to basename of path
                file_name = file_info.get('filename', os.path.basename(file_info.get('path', 'N/A')))
                # Check if base64 was included (it shouldn't be based on filter, but check anyway)
                has_base64 = " (Base64 Included)" if "base64_data" in file_info else ""
                summary_content += f"    *   `{file_name}` ({file_info.get('type', 'N/A')}): {file_desc}{has_base64}\n" # Path removed for brevity

        evaluation = task.get('evaluation', {})
        if evaluation and ('assessment' in evaluation or 'specific_criteria' in evaluation): # Check if eval dict has content
            summary_content += "*   **Evaluation:**\n"
            if 'assessment' in evaluation:
                 summary_content += f"    *   Assessment: **{str(evaluation.get('assessment', 'N/A')).upper()}**\n"
            # Feedback log now captures eval feedback, so let's check there primarily
            # if evaluation.get('feedback'):
            #      summary_content += f"    *   Feedback: {evaluation.get('feedback', 'N/A')}\n"
            # if evaluation.get('improvement_suggestions'): summary_content += f"    *   Suggestions: {evaluation.get('improvement_suggestions')}\n"
            if evaluation.get('specific_criteria'): # Use correct key name
                 # Indent criteria properly
                 criteria_lines = evaluation['specific_criteria'].split('\n')
                 indented_criteria = "\n        ".join(criteria_lines)
                 summary_content += f"    *   Criteria Used:\n        ```text\n        {indented_criteria}\n        ```\n"

        # Remove Task Messages as they are less relevant for summary
        # messages = task.get('messages', [])
        # if messages:
        #     summary_content += "*   **Task Messages:**\n"
        #     for msg in messages:
        #         prefix = "Human Input" if isinstance(msg, HumanMessage) else "AI Response/Log"
        #         summary_content += f"    *   {prefix}: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}\n"

        summary_content += "\n---\n\n"
    try:
        with open(summary_filepath, "w", encoding="utf-8") as f: f.write(summary_content)
        print(f"Summary saved to: {summary_filepath}")
    except Exception as e: print(f"Error saving summary: {e}")

    # --- <<< 新增：設置 Phase 為 QA >>> ---
    print("Setting current_phase to 'qa' after saving summary.")
    state["current_phase"] = "qa"
    # --- <<< 結束新增 >>> ---

    return state # Return the state with updated phase

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

# Async function placeholder (if needed later for async tools/MCP)
async def initialize_mcp_client():
    # ... (MCP client initialization logic) ...
    pass

# =============================================================================
# 9. 圖邊緣條件函數定義 (Update routing)
# =============================================================================
def route_from_process_management(state: WorkflowState) -> Literal[
    "assign_teams", "eva_teams", "save_final_summary", "qa_loop", "finished", "process_management"
]:
    """ Determines the next node. Routes to QA loop if needed."""
    print("--- Routing decision @ route_from_process_management ---")
    current_phase = state.get("current_phase")
    interrupt_result = state.get("interrupt_result")
    interrupt_action = interrupt_result.get("action") if isinstance(interrupt_result, dict) else None

    print(f"Router received state: Phase='{current_phase}', Interrupt Action='{interrupt_action}'")

    # --- 1. Check if entering QA mode ---
    if current_phase == "qa" or interrupt_action == "CONVERSATION":
         # 清除 interrupt_result 避免影響後續步驟 (如果它是 CONVERSATION)
         state["interrupt_result"] = None # 直接修改狀態字典（因為路由函數不應返回它）
         print(f"Routing -> qa_loop (Reason: Phase='{current_phase}' or Interrupt Action='CONVERSATION')")
         return "qa_loop"

    # --- 2. Check Finished Phase ---
    if current_phase == "finished":
        print("Routing -> finished (Reason: Phase is 'finished')")
        return "finished"

    # --- 3. Normal Task Execution Flow ---
    # ... (這部分邏輯保持不變，檢查任務列表、狀態、是否需要評估等) ...
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

        if task_status in ["failed", "max_retries_reached"]:
            print(f"Routing -> process_management (Reason: Task {task_id} status '{task_status}', returning for failure analysis)")
            return "process_management"

        requires_eval = current_task.get("requires_evaluation", False)
        if requires_eval:
            if task_status in ["pending", "in_progress", None]:
                print(f"Routing -> eva_teams (Reason: Task {task_id} status '{task_status}' and requires evaluation)")
                return "eva_teams"
            else:
                print(f"Routing Warning: Task {task_id} needs eval but has unexpected status '{task_status}'. Looping PM.")
                return "process_management"
        else: # Doesn't require evaluation
            if task_status in ["pending", "in_progress", None]:
                print(f"Routing -> assign_teams (Reason: Task {task_id} status '{task_status}' and ready for execution)")
                return "assign_teams"
            else:
                print(f"Routing Warning: Task {task_id} ready but has unexpected status '{task_status}'. Looping PM.")
                return "process_management"
    else:
        print(f"Routing Warning: index ({current_index}) out of bounds (size {len(tasks)}). Fallback to save_final_summary.")
        return "save_final_summary"


# ... (route_after_assign_teams, route_after_evaluation 保持不變) ...
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

def route_after_evaluation(state: WorkflowState) -> Literal["process_management", "assign_teams", "finished"]:
    """Routes after EvaAgent (evaluation subgraph): Continue (PM) or retry (assign_teams)."""
    current_phase = state.get("current_phase")
    if current_phase != "task_execution":
        print(f"Route after Eva Error: Unexpected phase '{current_phase}'. Routing to finished.")
        return "finished"

    tasks = state.get("tasks", [])
    current_idx = state.get("current_task_index", -1)
    if current_idx < 0 or current_idx >= len(tasks):
        print("Routing Error: Invalid index after evaluation. Routing to finished.")
        return "finished"

    task = tasks[current_idx]
    status = task.get("status")
    retry_count = task.get("retry_count", 0)
    max_retries = _full_static_config.agents.get("process_management", {}).parameters.get("max_retries", 3)

    print(f"Route after Eva: Checking Task {current_idx}, Status: {status}, RetryCount: {retry_count}")

    if status == "completed":
        print("Route after Eva: Evaluation passed. Routing to PM to advance.")
        return "process_management"
    elif status == "failed":
        print(f"Route after Eva: Evaluation failed. Routing to PM for assessment.")
        return "process_management"
    else:
        print(f"Route after Eva Error: Unexpected status '{status}'. Routing to PM as failsafe.")
        return "process_management"


# --- <<< 修改 route_after_chat_bot >>> ---
def route_after_chat_bot(state: WorkflowState) -> Literal["process_management", "qa_loop", "finished"]:
    """Routes after QA agent. Loops back to qa_loop for QA, goes to PM for new tasks, or finishes."""
    # 讀取 QA Agent 返回的 state 更新中的 phase
    # 因為 QA Agent 只返回了部分狀態，我們需要從主 state 中讀取 current_phase
    current_phase = state.get("current_phase")
    print(f"--- Routing decision @ route_after_chat_bot ---")
    print(f"  Current phase after chat_bot execution: {current_phase}")

    if current_phase == "task_execution":
        print("Routing -> process_management (Reason: QA requested new task sequence)")
        return "process_management"
    elif current_phase == "finished":
        print("Routing -> finished (Reason: QA detected termination request)")
        return "finished"
    elif current_phase == "qa":
        print("Routing -> qa_loop (Reason: Continue QA conversation)")
        return "qa_loop" # <<< 修改：路由回新的循環節點
    else:
        print(f"Routing Warning: Unexpected phase '{current_phase}' after chat_bot. Routing to finished.")
        return "finished"


def qa_loop_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Manages the user input part of the QA loop.
    Processes interrupt_input or prompts user if empty upon resuming.
    Returns only the user message update for qa_context.
    """
    node_name = "QA Loop Manager"
    print(f"--- Running Node: {node_name} ---")

    interrupt_input = state.get("interrupt_input")
    qa_context_list = state.get("qa_context", []) # Get current context

    message_to_add = None
    new_interrupt_input = None # Default to clearing

    if interrupt_input:
        print(f"{node_name}: Processing query from interrupt_input: '{interrupt_input[:100]}...'")
        message_to_add = HumanMessage(content=interrupt_input)
        # Clear interrupt_input for the next step
        new_interrupt_input = None
    else:
        # Interrupt occurred, but no new input provided by external loop upon resume
        print(f"{node_name}: No interrupt_input found upon resuming. Prompting user.")
        # Use an AIMessage to signify a system prompt to the user
        prompt_text = "請問您想問什麼？請在Interrupt Input欄位中提供您的想法。"
        message_to_add = AIMessage(content=prompt_text)
        # Keep interrupt_input as None, wait for actual user input next time
        new_interrupt_input = None # Ensure it stays None

    # Prepare the minimal state update dictionary
    update_dict = {
        "interrupt_input": new_interrupt_input
    }
    if message_to_add:
        update_dict["qa_context"] = [message_to_add] # Wrap in list for add_messages

    return update_dict


# =============================================================================
# 10. 圖定義與節點/邊緣添加
# =============================================================================
workflow = StateGraph(WorkflowState, config_schema=ConfigSchema)

# 添加節點
workflow.add_node("process_management", process_management.run)
workflow.add_node("assign_teams", assign_teams)
workflow.add_node("eva_teams", eva_agent.run)
workflow.add_node("chat_bot", qa_agent.run)
workflow.add_node("save_final_summary", save_final_summary)
workflow.add_node("qa_loop", qa_loop_node) # <<< 修改節點名稱和函數名 >>>

# 設置入口點
workflow.set_entry_point("process_management")

# 添加邊緣和條件邊緣
workflow.add_conditional_edges(
    "process_management",
    route_from_process_management, # Uses updated logic
    {
        "assign_teams": "assign_teams",
        "eva_teams": "eva_teams",
        "save_final_summary": "save_final_summary",
        "qa_loop": "qa_loop", # <<< 修改：指向新的 QA 循環節點名稱 >>>
        "process_management": "process_management",
        "finished": END,
    }
)

# assign_teams always goes back to process_management (Unchanged)
workflow.add_conditional_edges(
    "assign_teams",
    route_after_assign_teams,
    {
        "process_management": "process_management",
        "finished": END
    }
)

# Edge after Evaluation - Always goes back to PM (Unchanged)
workflow.add_conditional_edges(
    "eva_teams",
    route_after_evaluation,
    {
        "process_management": "process_management",
        "finished": END
    }
)

# --- <<< 添加從 QA 循環管理器到 Chat Bot 的邊緣 >>> ---
workflow.add_edge("qa_loop", "chat_bot") # 從新的管理器節點到聊天機器人

# --- <<< 修改 Chat Bot 之後的條件邊緣 >>> ---
workflow.add_conditional_edges(
    "chat_bot",
    route_after_chat_bot, # Use the updated routing function
    {
        "process_management": "process_management", # Go back to PM if QA ends/new task
        "qa_loop": "qa_loop",           # <<< 修改：指向新的 QA 循環節點名稱 >>> Go back to loop for more input
        "finished": END                  # Fallback/error case
    }
)

# --- <<< 新增：從 save_final_summary 返回 PM >>> ---
workflow.add_edge("save_final_summary", "process_management")
# --- <<< 結束新增 >>> ---


# =============================================================================
# 11. 圖編譯
# =============================================================================
graph = workflow.compile(interrupt_before=["qa_loop"], interrupt_after=["chat_bot"])
graph.name = "General_Arch_graph_v19_DedicatedQA_Refactored"
print("Main Graph compiled successfully (v19 - Refactored Imports).")

# =============================================================================
# 12. TODOLIST
# =============================================================================
# 1. QA test   V
# 2. from insert enter QA node test    V
# 3. Fauilure analysis test   V
# 4. Building life cycle tools add in  (maybe 3D model)
# 5. final configuration systematically...
# 6. 3D model input / output test(MCP)
# 7. Change node name to agent name    V
# 8. final evaluation 好像不太行
# 9. 檢查save summary 與長期記憶有關   V
