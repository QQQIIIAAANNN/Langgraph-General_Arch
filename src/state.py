import os
from typing import Dict, List, Any, Literal, Union, Optional, Tuple
from typing_extensions import TypedDict

# --- Configuration Import ---
# Assuming ConfigManager can be initialized without dependency on General_Arch_graph
from src.configuration import ConfigManager

# Import BaseMessage types if WorkflowState uses them directly
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage # Add this import

# =============================================================================
# 狀態定義 (Moved from General_Arch_graph.py)
# =============================================================================
class TaskState(TypedDict):
    """State for managing tasks in the workflow."""
    task_id: str
    status: Literal["pending", "in_progress", "completed", "failed", "max_retries_reached"]
    task_objective: str
    description: str
    selected_agent: Optional[str]
    task_inputs: Optional[Dict[str, Any]]
    outputs: Dict[str, Any]
    output_files: List[Dict[str, str]]
    evaluation: Dict[str, Any]
    requires_evaluation: bool
    error_log: Optional[str]
    feedback_log: Optional[str]
    retry_count: int

class WorkflowState(TypedDict):
    """State for managing the overall workflow."""
    user_input: str
    interrupt_input: Optional[str]
    interrupt_result: Optional[str]
    current_task: Optional[TaskState]
    current_task_index: int
    tasks: List[TaskState]
    current_phase: Literal["task_execution", "qa", "finished"]
    qa_context: List[Union[HumanMessage, AIMessage]] # Keep BaseMessage types here? Or define dummy? Let's keep for now. Needs Langchain import
