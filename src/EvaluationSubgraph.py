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
import numpy as np # For Radar Chart
import matplotlib # Import the top-level matplotlib package
# --- NEW: Set a non-interactive backend for Matplotlib ---
# This helps prevent "main thread is not in main loop" errors in non-GUI environments
try:
    matplotlib.use('Agg')
    print("Matplotlib backend set to 'Agg'.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to 'Agg' - {e}")
import matplotlib.pyplot as plt # For Radar Chart
from math import pi # For Radar Chart
from matplotlib.lines import Line2D # For Radar Chart Legend
from matplotlib.font_manager import FontProperties # For font control
from pathlib import Path # Ensure Path is imported
# New import for path effects
import matplotlib.patheffects as PathEffects
import textwrap

# --- NEW: Add Matplotlib font configuration for Chinese characters ---
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS'] # Add fallback fonts
    plt.rcParams['axes.unicode_minus'] = False  # Resolve the minus sign display issue
except Exception as e:
    print(f"Warning: Could not set Chinese font for Matplotlib - {e}")
# --- END NEW ---

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
# --- NEW: Diagram Cache Directory ---
DIAGRAM_CACHE_DIR = os.path.join(OUTPUT_DIR, "diagram_cache") 
# --- END NEW ---
os.makedirs(RENDER_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
# --- NEW: Create Diagram Cache Directory ---
os.makedirs(DIAGRAM_CACHE_DIR, exist_ok=True)
# --- END NEW ---
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

    # --- NEW: Ensure feedback_log is also updated/initialized on failure ---
    failure_feedback = f"Task failed in node '{node_name}': {error_message}"
    current_feedback_log = task.get("feedback_log", "") # Get current or default to empty string
    if current_feedback_log:
        task["feedback_log"] = f"{current_feedback_log}\n{failure_feedback}"
    else:
        task["feedback_log"] = failure_feedback
    # --- END NEW ---

    # Clear evaluation specific fields on failure
    if "evaluation" in task:
        task["evaluation"]["assessment"] = "Fail"
        task["evaluation"]["specific_criteria"] = "N/A due to failure"
        # Ensure subgraph_error is string before appending
        current_subgraph_error = task["evaluation"].get("subgraph_error", "") or ""
        task["evaluation"]["subgraph_error"] = (current_subgraph_error + f"; NodeError ({node_name}): {error_message}").strip("; ")

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
    # --- MODIFICATION: Get assessment from detailed_assessment if available for Special/Final Agents ---
    # For Special/Final, the primary 'assessment' field might be just "Analysis Complete" or a score summary.
    # The detailed scores are in 'detailed_assessment'. The status update relies on selected_agent.
    # We will keep the original logic for `assessment` for EvaAgent.
    assessment_for_status_update = task.get("evaluation", {}).get("assessment", "Fail")
    # --- END MODIFICATION ---

    # --- <<< 關鍵修改：優先處理 Special/Final Agent >>> ---
    if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        task["status"] = "completed"
        print(f"  - [{node_name}] Agent is {selected_agent}. ENFORCING final status to COMPLETED.")
        return # 直接返回，不再執行後續的 Pass/Fail 判斷
    # --- <<< 結束修改 >>> ---

    # 標準評估代理(EvaAgent)的原始邏輯
    # Note: assessment here refers to the *main* assessment, likely from LLM or combined visual.
    is_pass_fail_eval = not isinstance(assessment_for_status_update, str) or assessment_for_status_update.lower() in ["pass", "fail"] # Use assessment_for_status_update
    is_score_eval = isinstance(assessment_for_status_update, str) and assessment_for_status_update.lower().startswith("score") # Should not happen for EvaAgent

    if is_pass_fail_eval:
        if assessment_for_status_update == "Pass": # Use assessment_for_status_update
            task["status"] = "completed"
            print(f"  - [{node_name}] Standard Assessment is Pass. Setting final status to COMPLETED.")
        else: # Assessment is Fail
            task["status"] = "failed" # Standard Fail means workflow failure
            print(f"  - [{node_name}] Standard Assessment is Fail. Setting final status to FAILED.")
            # Log the logical failure reason if not already logged
            if task.get("error_log") is None: # Only log if no internal error happened first
                 failure_reason = f"Evaluation resulted in '{assessment_for_status_update}'." # Use assessment_for_status_update
                 task["error_log"] = f"[{node_name}] {failure_reason}" # Use error_log for logical fail
                 _append_feedback(task, failure_reason, node_name)
    # --- (Removed score handling here as it's covered by Special/Final logic above) ---
    # elif is_score_eval: ...
    else: # Unexpected assessment value for EvaAgent
        task["status"] = "failed"
        err_msg = f"Unexpected assessment value '{assessment_for_status_update}' for EvaAgent. Setting status to FAILED." # Use assessment_for_status_update
        print(f"  - [{node_name}] Note: {err_msg}")
        if task.get("error_log") is None:
            task["error_log"] = f"[{node_name}] {err_msg}"
        _append_feedback(task, err_msg, node_name)

# --- <<< NEW HELPER: Calculate Scores for Special/Final EvaAgent >>> ---
def _calculate_final_scores_for_options(
    options_data_with_raw_scores: List[Dict[str, Any]], # Receive data containing collected/averaged scores from branches
    budget_limit: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Calculates final scores for radar chart dimensions (0-10 scale).
    This function now receives the collected/merged raw scores and percentages,
    calculates the final cost efficiency and green building scores,
    and ensures all final score keys are present, mapping others directly.
    Input keys are now expected to be prefixed with 'collected_'.
    Removes deprecated score dimensions.
    Also carries forward original 'image_paths' and 'video_paths'.
    """
    processed_options = []

    # Define the mapping from COLLECTED/MERGED score keys to FINAL score keys for the radar chart
    # Removed deprecated keys: collected_special_criteria_score, collected_aesthetics_score, collected_functionality_score
    score_mapping = {
        "collected_user_goal_responsiveness_score": "user_goal_responsiveness_score_final",
        "collected_aesthetics_context_score": "aesthetics_context_score_final",
        "collected_functionality_flexibility_score": "functionality_flexibility_score_final",
        "collected_durability_maintainability_score": "durability_maintainability_score_final",
        # Cost and Green are calculated specifically, but mapping collected score if calculation fails
        "collected_cost_efficiency_score": "cost_efficiency_score_final",
        "collected_green_building_score": "green_building_score_final",
    }

    # All keys expected in the final output for the detailed assessment list
    # This list defines the order of keys in the output dictionary for each option
    all_final_output_keys_ordered = [
        "option_id",
        "description",
        "architecture_type",
        # Final Scores (0-10 scale) - Order for radar chart dimensions
        "user_goal_responsiveness_score_final",
        "aesthetics_context_score_final",
        "functionality_flexibility_score_final",
        "durability_maintainability_score_final",
        "cost_efficiency_score_final",
        "green_building_score_final",
        # Other collected info
        "scoring_rationale", # Merged feedback text
        "errors_during_processing",
        "collected_estimated_cost",
        "collected_green_building_potential_percentage",
        # {{ EDIT START }}
        "image_paths", # Add original paths for tracking
        "video_paths", # Add original paths for tracking
        # {{ EDIT END }}
    ]

    # Create a set for quick lookup of final score keys (0-10 dimensions)
    final_score_dimension_keys = set(score_mapping.values())


    for option_raw_data in options_data_with_raw_scores:
        if not isinstance(option_raw_data, dict):
            print(f"  - Warning: Expected a dictionary for option_raw_data, got {type(option_raw_data)}. Skipping this option.")
            continue

        # Initialize the output dictionary for this option based on the desired order
        final_scores: Dict[str, Any] = {}
        for key in all_final_output_keys_ordered:
             final_scores[key] = None # Initialize all keys to None first

        # Ensure option_id is set first
        final_scores["option_id"] = option_raw_data.get("option_id", "Unknown")

        # Initialize final score dimensions (0-10) to 0.0
        for key in final_score_dimension_keys:
             final_scores[key] = 0.0

        # Transfer collected/merged scores using the mapping
        # This covers user_goal_responsiveness, aesthetics, functionality, durability.
        for collected_key, final_key in score_mapping.items():
             # Only map if the collected key exists in the raw data and is numeric
             collected_value = option_raw_data.get(collected_key)
             # Only map the core 4 dimensions directly here. Cost/Green are calculated below.
             if final_key not in ["cost_efficiency_score_final", "green_building_score_final"]:
                 if isinstance(collected_value, (int, float)):
                      final_scores[final_key] = float(collected_value)
                 # Note: If collected_value is None, it remains None initially, then overwritten by 0.0 initialization


        # --- Calculate Final Cost Efficiency Score (0-10) ---
        estimated_cost_raw = option_raw_data.get("collected_estimated_cost")
        cost_efficiency_calculated = 1.0 # Default to 1.0, as per user request for worst-case

        budget_calc_done = False
        if estimated_cost_raw is not None:
            try:
                estimated_cost_float = float(estimated_cost_raw)
                
                valid_budget_limit = None
                if budget_limit is not None: # budget_limit is from function args
                    try:
                        valid_budget_limit = float(budget_limit)
                        if valid_budget_limit <= 0: # Budget must be positive
                            print(f"  - Info: Invalid budget_limit (<=0): {budget_limit}. Treating as no budget for option {option_raw_data.get('option_id', 'Unknown')}.")
                            valid_budget_limit = None
                    except (ValueError, TypeError):
                        print(f"  - Info: Cannot parse budget_limit: {budget_limit}. Treating as no budget for option {option_raw_data.get('option_id', 'Unknown')}.")
                        valid_budget_limit = None

                if valid_budget_limit is not None and estimated_cost_float > 0:
                    ratio = estimated_cost_float / valid_budget_limit
                    if ratio < 0.5: # Significantly under budget
                        cost_efficiency_calculated = 10.0
                    elif ratio > 1.5: # Significantly over budget
                        cost_efficiency_calculated = 1.0
                    else: # Within 0.5x to 1.5x of budget
                        # Linear interpolation:
                        # Point 1: (ratio=0.5, score=10)
                        # Point 2: (ratio=1.5, score=1)
                        # Score = 10 + ( (1-10) / (1.5-0.5) ) * (ratio - 0.5)
                        # Score = 10 + (-9 / 1) * (ratio - 0.5)
                        # Score = 10 - 9 * ratio + 4.5
                        # Score = 14.5 - 9 * ratio
                        cost_efficiency_calculated = max(1.0, min(10.0, 14.5 - 9.0 * ratio))
                    budget_calc_done = True
                    print(f"  - CostCalc (Option {option_raw_data.get('option_id', 'Unknown')}): Budget {valid_budget_limit}, Cost {estimated_cost_float}, Ratio {ratio:.2f}, Score {cost_efficiency_calculated:.2f}")
            except (ValueError, TypeError):
                print(f"  - Warning: Could not parse collected_estimated_cost '{estimated_cost_raw}' for option {option_raw_data.get('option_id', 'Unknown')}. Cost efficiency will rely on collected score or default to 1.0.")
                # budget_calc_done remains False, will fall through to next block

        if not budget_calc_done:
            # Fallback to collected_cost_efficiency_score if budget calculation was not performed,
            # was not applicable (e.g. cost <= 0), or failed.
            # This also covers cases where estimated_cost_raw was None initially.
            collected_score_raw = option_raw_data.get("collected_cost_efficiency_score")
            if collected_score_raw is not None:
                try:
                    collected_score_float = float(collected_score_raw)
                    # Ensure the collected score is also within 1-10 range (minimum 1.0)
                    cost_efficiency_calculated = min(10.0, max(1.0, collected_score_float))
                    print(f"  - CostCalc (Option {option_raw_data.get('option_id', 'Unknown')}): No budget calc, using collected_score {collected_score_float}, Final Score {cost_efficiency_calculated:.2f}")
                except (ValueError, TypeError):
                    print(f"  - Warning: Could not parse collected_cost_efficiency_score '{collected_score_raw}' for option {option_raw_data.get('option_id', 'Unknown')}. Cost efficiency set to default 1.0.")
                    # cost_efficiency_calculated remains 1.0 (default)
            else:
                print(f"  - CostCalc (Option {option_raw_data.get('option_id', 'Unknown')}): No budget calc and no collected_score. Defaulting to 1.0.")
                # cost_efficiency_calculated remains 1.0 (default as no collected_score_raw)

        final_scores["cost_efficiency_score_final"] = cost_efficiency_calculated


        # --- Calculate Final Green Building Score (0-10) ---
        # Use the COLLECTED green_building_potential_percentage for this calculation
        green_percentage = option_raw_data.get("collected_green_building_potential_percentage")
        green_building_calculated = 0.0 # Default if calculation fails or not applicable

        if green_percentage is not None:
            try:
                green_percentage_float = float(green_percentage)
                # Scale 0-100% to 0-10 score
                green_building_calculated = max(0.0, min(10.0, green_percentage_float / 10.0))
            except (ValueError, TypeError):
                print(f"  - Warning: Could not parse collected_green_building_potential_percentage '{green_percentage}' for option {final_scores['option_id']}. Green score calculation skipped.")
                # Fallback to collected score if parsing fails and it's numeric
                if "collected_green_building_score" in option_raw_data and isinstance(option_raw_data["collected_green_building_score"], (int, float)):
                    green_building_calculated = float(option_raw_data["collected_green_building_score"])
                else:
                    green_building_calculated = 0.0 # Default if all else fails

        final_scores["green_building_score_final"] = green_building_calculated

        # Add collected texts/errors and original metadata
        # These are added according to the order defined in all_final_output_keys_ordered
        final_scores["description"] = option_raw_data.get("description", "N/A")
        final_scores["architecture_type"] = option_raw_data.get("architecture_type", "General")
        final_scores["scoring_rationale"] = option_raw_data.get("scoring_rationale", "No detailed feedback available.")
        final_scores["errors_during_processing"] = option_raw_data.get("errors_during_processing", "")
        # Collected raw values for reference (placed at the end as per request)
        final_scores["collected_estimated_cost"] = option_raw_data.get("collected_estimated_cost")
        final_scores["collected_green_building_potential_percentage"] = option_raw_data.get("collected_green_building_potential_percentage")
        
        # {{ EDIT START }}
        # Copy original paths from raw data
        final_scores["image_paths"] = option_raw_data.get("image_paths", [])
        final_scores["video_paths"] = option_raw_data.get("video_paths", [])
        # {{ EDIT END }}

        # --- Final check: Ensure all score_final keys are numeric (float), default to 0.0 if something went wrong ---
        for score_key in final_score_dimension_keys:
            if not isinstance(final_scores.get(score_key), (int, float)):
                 final_scores[score_key] = 0.0

        processed_options.append(final_scores)

    return processed_options
# --- <<< END NEW HELPER >>> ---

# --- Helper function to get a font that supports CJK characters ---
def get_cjk_font():
    """
    Attempts to find and return a FontProperties object for a CJK font.
    Returns FontProperties object on success, None on failure.
    """
    # Priority: SimHei, Microsoft JhengHei, Source Han Sans TC, Noto Sans CJK TC
    # These are common fonts that support CJK characters.
    # Users might need to install them.
    font_names = ['SimHei', 'Microsoft JhengHei', 'Source Han Sans TC', 'Noto Sans CJK TC', 'Arial Unicode MS']
    found_font_path = None

    # First, try to find font by name using Matplotlib's font manager
    # This might update Matplotlib's font cache
    from matplotlib.font_manager import findSystemFonts, FontProperties

    try:
        # findSystemFonts returns a list of font paths
        system_fonts = findSystemFonts(fontpaths=None, fontext='ttf') + findSystemFonts(fontpaths=None, fontext='ttc')
        for font_name_attempt in font_names:
            for font_path in system_fonts:
                 if font_name_attempt.lower() in os.path.basename(font_path).lower():
                     print(f"  - Found CJK font '{font_name_attempt}' at path: {font_path}")
                     found_font_path = font_path
                     break # Found a path for this name, break inner loop
            if found_font_path:
                break # Found a font, break outer loop

        if found_font_path:
             print(f"  - Using found CJK font path: {found_font_path}")
             return FontProperties(fname=found_font_path)

    except Exception as e:
         print(f"  - Error using Matplotlib's font manager: {e}")
         traceback.print_exc()


    # If font manager failed or didn't find it, try specific paths
    print("  - Matplotlib font manager search failed or found nothing. Trying hardcoded paths.")
    try:
        if os.name == 'nt': # Windows
            font_paths_win = [
                os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'msjh.ttc'), # Microsoft JhengHei
                os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'simhei.ttf') # SimHei
            ]
            for fp_win in font_paths_win:
                if Path(fp_win).exists():
                    print(f"  - Found CJK font at hardcoded Windows path: {fp_win}")
                    return FontProperties(fname=str(Path(fp_win)))
        elif os.name == 'posix': # Linux/macOS
             # On Linux, fc-list can find fonts. For macOS, check common paths.
             # This is a simplified check. A more robust solution might involve fc-list.
            font_paths_unix = [
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Example for Noto Sans CJK
                '/System/Library/Fonts/Supplemental/Microsoft JhengHei.ttf', # macOS JhengHei
                '/Library/Fonts/Microsoft SimHei.ttf', # macOS SimHei
                 os.path.expanduser('~/Library/Fonts/Microsoft SimHei.ttf'), # User fonts macOS
                 os.path.expanduser('~/.fonts/simhei.ttf'), # User fonts Linux
            ]
            for fp_unix in font_paths_unix:
                 if Path(fp_unix).exists():
                    print(f"  - Found CJK font at hardcoded Unix path: {fp_unix}")
                    return FontProperties(fname=str(Path(fp_unix)))
    except Exception as e:
         print(f"  - Error during hardcoded path search: {e}")
         traceback.print_exc()


    # If all attempts fail
    print("Warning: Could not find a preferred CJK font using any method. CJK characters might not display correctly (may show as squares).")
    print("Please ensure you have a CJK font like 'Microsoft JhengHei', 'SimHei', or 'Noto Sans CJK TC' installed on your system.")
    return None # Fallback to Matplotlib's default handling (likely no CJK support)

# --- Re-run font search after potentially updating font cache ---
# You might need to rebuild the font cache for Matplotlib to pick up newly found fonts.
# This can sometimes be done programmatically:
try:
    print("  - Attempting to rebuild Matplotlib font cache...")
    from matplotlib.font_manager import fontManager
    fontManager.findSystemFonts(fontpaths=None, fontext='ttf') # Trigger a search
    fontManager.findSystemFonts(fontpaths=None, fontext='ttc') # Trigger a search
    # fontManager.update() # This method might exist depending on version
    print("  - Matplotlib font cache rebuild attempted.")
except Exception as e:
    print(f"  - Warning: Could not rebuild Matplotlib font cache - {e}")


cjk_font_prop = get_cjk_font() # Call the improved function

def _generate_radar_chart_for_options(
    options_data: List[Dict[str, Any]],
    output_filename: str,
    title: str = "Design Options Comparison",
    cjk_font_prop: Optional[FontProperties] = None # <<< ADDED THIS PARAMETER <<<
) -> Optional[str]:
    """
    Generates a radar chart comparing multiple options across predefined score dimensions (6 dimensions).
    Uses the final scores with '_score_final' suffix.
    Returns the saved file path on success, or None on failure.
    Applies CJK font properties only if a valid font was found.
    Sets legend labels to use option 'description' or 'option_id' as fallback.
    Adds faint radial grid lines and adjusts category label positioning.
    """
    if not options_data:
        print("  - Radar Chart: No options data provided.")
        return None # Return None on purpose if no data

    # --- MODIFICATION: Updated score keys for radar chart (6 dimensions) ---
    score_keys = [
        "user_goal_responsiveness_score_final",
        "aesthetics_context_score_final",
        "functionality_flexibility_score_final",
        "durability_maintainability_score_final",
        "cost_efficiency_score_final",
        "green_building_score_final"
    ]
    categories_display = [
        "使用者目標回應性", "美學與場域關聯性", "機能性與適應彈性",
        "耐久性與維護性", "早期成本效益估算", "綠建築永續潛力"
    ]
    # --- END MODIFICATION ---

    if len(categories_display) != len(score_keys):
        print(f"  - Warning: Mismatch between score_keys ({len(score_keys)}) and categories_display ({len(categories_display)}). Using default names.")
        categories_display = [s.replace("_score_final", "").replace("_", " ").title() for s in score_keys]

    num_vars = len(score_keys)
    if num_vars == 0:
        print("  - Radar Chart: No score keys defined for plotting.")
        return None # Return None if no score keys


    try: # Add a try-except block around the main plotting logic
        # --- Style constants and Color Management ---
        BG_WHITE = "#fbf9f4"
        TITLE_COLOR = "#333333" # Dark grey for title
        # GREY70 = "#b3b3b3" # Original for major grid, axis labels - NOW BLACK
        AXIS_LABEL_COLOR = "#000000" # Black for axis labels (categories)
        RADIAL_LABEL_COLOR = "#000000" # Black for radial labels (0, 5, 10)
        GRID_LINE_COLOR = "#b3b3b3" # Grey for major grid lines (spokes)
        GREY_LIGHT = "#f2efe8"
        GREY_VERY_LIGHT = "#e0e0e0" # Original for minor grid lines - NOW BLUE for 0.5 line
        RADIAL_GRID_INNER_BLUE = "#007ACC" # A distinct blue for the 5 radial line
        RADIAL_GRID_OUTER_GREY = "#cccccc" # Lighter grey for 0 and 10 radial lines

        # Color management for data series
        user_primary_colors = ["#FF5A5F", "#FFB400", "#007A87"]
        morandi_colors = [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',
            '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82',
            '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
        ]

        num_options = len(options_data)
        plot_colors = []
        if num_options <= len(user_primary_colors):
            plot_colors = user_primary_colors[:num_options]
        else:
            plot_colors.extend(user_primary_colors)
            remaining_options = num_options - len(user_primary_colors)
            if remaining_options <= len(morandi_colors):
                plot_colors.extend(morandi_colors[:remaining_options])
            else:
                plot_colors.extend(morandi_colors)
                # Fallback to colormap if still more colors are needed
                cmap_needed = remaining_options - len(morandi_colors)
                if cmap_needed > 0:
                    # Using a perceptually uniform colormap like 'viridis' or 'cividis' can be good.
                    # 'tab10' is also a good categorical colormap.
                    cmap = plt.cm.get_cmap('tab10', cmap_needed)
                    for i in range(cmap_needed):
                        plot_colors.append(cmap(i))

        if not plot_colors: # Should not happen if num_options > 0
            plot_colors = ['#000000'] # Default to black if something went wrong

        # --- Angles for the radar chart ---
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + angles[:1] # Close the loop

        # --- Initialize layout ---
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True)) # Increased figsize a bit
        fig.patch.set_facecolor(BG_WHITE)
        ax.set_facecolor(BG_WHITE)

        # Rotate the 0 degrees on top.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Setting lower limit to negative value can help with 0-value overlap if data is normalized 0-1
        # Since our data is 0-10, let's set ylim appropriately.
        ax.set_ylim(-0.5, 10.5) # Adjusted for 0-10 scale, with slight padding

        # Plot lines and dots
        for idx, option_data in enumerate(options_data):
            # --- MODIFICATION: Safely get score values, ensure they are numeric ---
            values = []
            for key in score_keys:
                 value = option_data.get(key, 0.0) # Default to 0.0 if key is missing
                 if isinstance(value, (int, float)):
                      values.append(float(value))
                 else:
                      print(f"    - Warning (Radar Chart): Score for option {option_data.get('option_id','Unknown')} key '{key}' is not numeric ({value}). Using 0.0.")
                      values.append(0.0)

            if len(values) != num_vars:
                 print(f"    - Error (Radar Chart): Mismatch in expected ({num_vars}) vs actual ({len(values)}) scores for option {option_data.get('option_id','Unknown')}. Skipping plot for this option.")
                 continue # Skip plotting this option if scores list is malformed

            values_closed = values + values[:1] # Close the loop

            # Ensure we have a color for this option
            current_color = plot_colors[idx % len(plot_colors)]

            # Use description for label if available, otherwise use option_id
            # --- MODIFICATION: Define label using description first, fallback to option_id ---
            option_label = option_data.get("description")
            if not option_label or not isinstance(option_label, str):
                 option_label = option_data.get("option_id", f"Option {idx+1}")
                 print(f"    - Using option_id '{option_label}' for legend label as description is missing or invalid.")
            else:
                 print(f"    - Using description '{option_label}' for legend label.")
            # --- END MODIFICATION ---

            ax.plot(angles_closed, values_closed, color=current_color, linewidth=2.5, linestyle='solid', zorder=2, label=option_label)
            ax.fill(angles_closed, values_closed, color=current_color, alpha=0.2, zorder=1)
            ax.scatter(angles_closed, values_closed, color=current_color, s=80, zorder=3)


        # --- Customize guides and annotations ---
        # Set values for the angular axis (x)
        ax.set_xticks(angles)
        # Apply CJK font properties to tick labels if available
        # --- MODIFICATION: Apply font properties conditionally ---
        if cjk_font_prop:
             xtick_labels = ax.set_xticklabels(categories_display, color=AXIS_LABEL_COLOR, size=12, fontproperties=cjk_font_prop) # <<< Applied font property here <<<
        else:
             print("  - Radar Chart: CJK font property not available. Using default font for x-tick labels.")
             xtick_labels = ax.set_xticklabels(categories_display, color=AXIS_LABEL_COLOR, size=12) # Use default font


        # Adjust tick label positions to prevent overlap
        # --- MODIFICATION: Improved alignment for all labels ---
        for i, label in enumerate(xtick_labels):
            angle_rad = angles[i]
            # 增加偏移值以提供更多空間
            horizontal_offset = 0.2  # 增加水平偏移
            vertical_offset = 0.15   # 增加垂直偏移

            angle_deg = np.degrees(angle_rad)

            # 頂部標籤 (接近正上方的標籤)
            if -30 <= angle_deg <= 30 or angle_deg >= 330 or angle_deg <= -330:
                label.set_horizontalalignment('center')
                label.set_verticalalignment('bottom')  # 確保標籤在點的下方
                # 往上移動更多
                label.set_y(label.get_position()[1] + vertical_offset * 0.1)

            # 底部標籤 (接近正下方的標籤)
            elif 150 <= angle_deg <= 210 or -210 <= angle_deg <= -150:
                label.set_horizontalalignment('center')
                label.set_verticalalignment('top')  # 確保標籤在點的上方
                # 底部標籤也往上移動 (減少向下的偏移)
                label.set_y(label.get_position()[1] - vertical_offset * 0.05)

            # 右側標籤
            elif (30 < angle_deg < 150) or (-330 < angle_deg < -210):
                label.set_horizontalalignment('left')
                label.set_verticalalignment('center')
                # 根據角度調整偏移量
                distance_factor = 1 + abs(np.sin(angle_rad)) * 0.5  # 角度接近垂直時增加偏移
                label.set_x(label.get_position()[0] + horizontal_offset * distance_factor)

            # 左側標籤 (特別處理左上角標籤)
            else:
                label.set_horizontalalignment('right')
                label.set_verticalalignment('center')

                # 特別處理左上角標籤 (約210-270度或-150到-90度範圍)
                if (210 < angle_deg < 270) or (-150 > angle_deg > -90):
                    # 左上標籤特別處理 - 水平向左偏移更多，垂直略微上移
                    distance_factor = 1.3 + abs(np.sin(angle_rad)) * 0.3
                    label.set_x(label.get_position()[0] - horizontal_offset * distance_factor)
                    # 如果標籤太靠近上方，給予輕微的上移
                    if angle_deg > 240 or angle_deg < -120:
                        label.set_y(label.get_position()[1] + vertical_offset * 0.3)
                else:
                    # 其他左側標籤正常處理
                    distance_factor = 1 + abs(np.sin(angle_rad)) * 0.5
                    label.set_x(label.get_position()[0] - horizontal_offset * distance_factor)

        # --- END MODIFICATION ---


        # Remove default lines for radial axis (y) and spines
        ax.set_yticks([]) # Remove default y-axis ticks
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID_LINE_COLOR, linestyle='--', linewidth=0.8, alpha=0.7) # Keep spoke lines
        ax.spines["start"].set_color("none")
        ax.spines["polar"].set_color("none")

        # Add custom lines for radial axis (y) at 0, 5 and 10.
        # HANGLES for drawing circles
        hangles_smooth = np.linspace(0, 2 * np.pi, 100)
        ax.plot(hangles_smooth, np.zeros(len(hangles_smooth)) + 0, linestyle='--', color=RADIAL_GRID_OUTER_GREY, linewidth=1.2)
        ax.plot(hangles_smooth, np.zeros(len(hangles_smooth)) + 5, linestyle='--', color=RADIAL_GRID_INNER_BLUE, linewidth=1.2) # Blue inner circle for 5
        ax.plot(hangles_smooth, np.zeros(len(hangles_smooth)) + 10, linestyle='--', color=RADIAL_GRID_OUTER_GREY, linewidth=1.2)

        # --- MODIFICATION: Add faint radial grid lines at 1, 2, 3, 4, 6, 7, 8, 9 ---
        faint_grid_values = [1, 2, 3, 4, 6, 7, 8, 9]
        for val in faint_grid_values:
             ax.plot(hangles_smooth, np.zeros(len(hangles_smooth)) + val, linestyle='--', color=GREY_VERY_LIGHT, linewidth=0.8, alpha=0.5) # Use GRID_LINE_COLOR with some transparency
        # --- END MODIFICATION ---


        # Fill background (optional, if needed, current BG_WHITE is fine)
        # ax.fill(hangles_smooth, np.zeros(len(hangles_smooth)) + 10.5, GREY_LIGHT, zorder=0)


        # Custom radial axis labels (0, 5, 10)
        # Increased PAD for more offset
        RADIAL_LABEL_OFFSET = 0.15  # Increased padding from axis
        # Angle for placing labels (e.g., slightly to the left of the first axis)
        label_angle_rad = angles[0] - np.deg2rad(10) # Position labels slightly off the first category axis

        # --- MODIFICATION: Apply font properties conditionally for radial labels ---
        if cjk_font_prop:
            ax.text(label_angle_rad, 0 + RADIAL_LABEL_OFFSET, "0", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center', fontproperties=cjk_font_prop) # <<< Applied font property here <<<
            ax.text(label_angle_rad, 5 + RADIAL_LABEL_OFFSET, "5", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center', fontproperties=cjk_font_prop) # <<< Applied font property here <<<
            ax.text(label_angle_rad, 10 + RADIAL_LABEL_OFFSET, "10", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center', fontproperties=cjk_font_prop) # <<< Applied font property here <<<
        else:
            print("  - Radar Chart: CJK font property not available. Using default font for radial labels.")
            ax.text(label_angle_rad, 0 + RADIAL_LABEL_OFFSET, "0", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center') # Use default font
            ax.text(label_angle_rad, 5 + RADIAL_LABEL_OFFSET, "5", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center') # Use default font
            ax.text(label_angle_rad, 10 + RADIAL_LABEL_OFFSET, "10", color=RADIAL_LABEL_COLOR, size=10, ha='center', va='center') # Use default font
        # --- END MODIFICATION ---

        # --- Create and add legends ---
        # Use option_id for labels, ensure font supports CJK if option_ids are in Chinese
        handles = []
        for i, option in enumerate(options_data):
             # --- MODIFICATION: Define label using description first, fallback to option_id ---
             option_label = option.get("description")
             if not option_label or not isinstance(option_label, str):
                  option_label = option.get("option_id", f"Option {i+1}")
             # --- END MODIFICATION ---
             handles.append(
                 Line2D(
                     [], [],
                     color=plot_colors[i % len(plot_colors)],
                     lw=2.5,
                     marker="o", # Add marker to legend
                     markersize=8,
                     label=option_label # Use the determined label
                 )
             )

        # Legend at bottom-right
        # Adjust bbox_to_anchor for fine-tuning. (1, 0) is bottom-right OF THE AXES.
        # To place it relative to FIGURE, use fig.legend or adjust loc with bbox_to_anchor more carefully.
        # For placing outside the plot area at bottom right:
        # --- MODIFICATION: Apply font properties conditionally for legend ---
        if cjk_font_prop:
            legend = ax.legend(
                handles=handles,
                loc='lower right', # General location
                bbox_to_anchor=(1.1, -0.1), # x, y coordinates (can be outside axes)
                                             # (1,0) is bottom right of axes, (1.1, -0.1) pushes it further out and down
                labelspacing=1.2,
                frameon=False,
                fontsize=10,
                prop=cjk_font_prop # <<< Applied font property here <<<
            )
        else:
            print("  - Radar Chart: CJK font property not available. Using default font for legend.")
            legend = ax.legend(
                handles=handles,
                loc='lower right',
                bbox_to_anchor=(1.1, -0.1),
                labelspacing=1.2,
                frameon=False,
                fontsize=10
            ) # Use default font
        # --- END MODIFICATION ---
        # For legend text properties (redundant if prop set above, but can be used for more control)
        # for text_leg in legend.get_texts():
        #     text_leg.set_fontproperties(cjk_font_prop)
        #     text_leg.set_fontsize(10) # Ensure legend text is also black or desired color
        #     text_leg.set_color(AXIS_LABEL_COLOR)


        # --- Add title ---
        # Ensure title uses CJK font if it contains CJK characters
        # --- MODIFICATION: Apply font properties conditionally for title ---
        if cjk_font_prop:
            fig.suptitle(
                title,
                x=0.5, # Centered
                y=0.96, # Position near top
                ha="center",
                fontsize=18,
                fontweight="bold",
                color=TITLE_COLOR,
                fontproperties=cjk_font_prop # <<< Applied font property here <<<
            )
        else:
            print("  - Radar Chart: CJK font property not available. Using default font for title.")
            fig.suptitle(
                title,
                x=0.5, # Centered
                y=0.96, # Position near top
                ha="center",
                fontsize=18,
                fontweight="bold",
                color=TITLE_COLOR
            ) # Use default font
        # --- END MODIFICATION ---

        # --- Attempt to center the plot and prevent label cutoff ---
        # fig.tight_layout() # Often helps, but can sometimes interact strangely with suptitle or legend outside axes.
        # Adjust subplot parameters if tight_layout isn't enough or causes issues
        # Increased right boundary slightly to accommodate moved legend
        # Also increased left boundary slightly for symmetry with right labels
        plt.subplots_adjust(top=0.88, bottom=0.17, left=0.14, right=0.88) # Example: make space for legend and title


        # Save the figure
        try:
            # Ensure the output directory exists (it should be handled by the calling function)
            # output_dir = Path(output_filename).parent
            # output_dir.mkdir(parents=True, exist_ok=True) # This should already be done upstream
            print(f"  - Radar Chart: Attempting to save chart to {output_filename}")
            # Added facecolor to savefig to ensure background is correct in saved file
            plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=BG_WHITE)
            print(f"  - Radar chart saved successfully to: {output_filename}")
            plt.close(fig) # Close the figure to free memory
            return output_filename # Return path on success
        except Exception as e:
            print(f"  - Error saving radar chart: {e}")
            # --- MODIFICATION: Close figure on save failure ---
            try:
                 plt.close(fig)
            except Exception:
                 pass
            # --- END MODIFICATION ---
            # --- MODIFICATION: Return None on save failure, error message is printed ---
            return None
            # --- END MODIFICATION ---

    except Exception as e:
        print(f"  - Critical Error during radar chart generation logic: {e}")
        traceback.print_exc()
        # --- MODIFICATION: Close figure on critical error ---
        try:
            plt.close(fig)
        except Exception:
            pass # Ignore errors if fig is already closed or invalid
        # --- END MODIFICATION ---
        # --- MODIFICATION: Indicate a more critical failure within the plotting process ---
        return None # Still return None, the message is printed internally
        # --- END MODIFICATION ---

def _generate_stacked_bar_chart_for_options(
    options_input_data: List[Dict[str, Any]], # Master list of options (for IDs and descriptions)
    llm_assessments: List[Dict[str, Any]],
    image_assessments: List[Dict[str, Any]],
    video_assessments: List[Dict[str, Any]],
    output_filename: str,
    title: str = "各方案於各評估維度之詳細分數來源比較圖",
    cjk_font_prop: Optional[FontProperties] = None
) -> Optional[str]:
    """
    Generates a stacked bar chart.
    Converts raw green building percentages and estimated costs to 0-10 scores before plotting.
    Correctly maps input keys from assessment data.
    Ensures all expected plot keys have default values if source data is missing.
    """
    if not options_input_data:
        print("  - Stacked Bar Chart: No options_input_data provided.")
        return None

    num_options = len(options_input_data)
    if num_options == 0:
        print("  - Stacked Bar Chart: No options to plot.")
        return None

    def get_scores_by_option_id(assessments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapped_assessments = {}
        for item in assessments:
            if not isinstance(item, dict):
                print(f"  - Warning (get_scores_by_option_id): Item is not a dict, skipping: {item}")
                continue
            option_id_val = item.get("option_id", item.get("Option Id"))
            if option_id_val:
                mapped_assessments[option_id_val] = item
            else:
                print(f"  - Warning (get_scores_by_option_id): Item missing option_id, skipping: {str(item)[:100]}")
        return mapped_assessments

    processed_llm_assessments = []
    processed_image_assessments = []
    processed_video_assessments = []
    
    all_estimated_costs = []
    llm_assessments_list = llm_assessments if isinstance(llm_assessments, list) else []
    image_assessments_list = image_assessments if isinstance(image_assessments, list) else []
    video_assessments_list = video_assessments if isinstance(video_assessments, list) else []

    temp_assessments_for_cost_collection = []
    if llm_assessments_list: temp_assessments_for_cost_collection.extend(llm_assessments_list)
    if image_assessments_list: temp_assessments_for_cost_collection.extend(image_assessments_list)
    if video_assessments_list: temp_assessments_for_cost_collection.extend(video_assessments_list)

    for assessment in temp_assessments_for_cost_collection:
        if not isinstance(assessment, dict):
            continue
        cost = assessment.get("estimated_cost", assessment.get("estimated_cost_img", assessment.get("estimated_cost_vid")))
        if cost is not None:
            try:
                all_estimated_costs.append(float(cost))
            except (ValueError, TypeError):
                option_id_for_error = assessment.get("option_id", assessment.get("Option Id", "Unknown Option"))
                print(f"  - Warning (Cost Collection): Invalid cost value '{cost}' in assessment for option '{option_id_for_error}'.")
                pass 

    min_overall_cost, max_overall_cost = (min(all_estimated_costs), max(all_estimated_costs)) if all_estimated_costs else (0, 0)
    if not all_estimated_costs:
        print("  - Stacked Bar Chart: No valid estimated cost data found globally. Cost efficiency scores might be default.")

    plot_base_keys = [
        "user_goal_responsiveness_score", "aesthetics_context_score",
        "functionality_flexibility_score", "durability_maintainability_score",
        "cost_efficiency_score",  "green_building_score"
    ]

    def scale_assessment_item(item: Dict[str, Any], source_suffix_short: str) -> Dict[str, Any]:
        scaled_item_output = {} 
        option_id_val = item.get("option_id", item.get("Option Id"))
        if option_id_val:
             scaled_item_output["option_id"] = option_id_val
        else: 
            print(f"  - Warning (scale_assessment_item): Item from source '{source_suffix_short}' is missing option_id. Defaulting scores to 0. Item: {str(item)[:100]}")
            scaled_item_output["option_id"] = f"unknown_option_src_{source_suffix_short}"


        direct_score_internal_bases = [
            "user_goal_responsiveness_score", "aesthetics_context_score",
            "functionality_flexibility_score", "durability_maintainability_score"
        ]

        def find_score_value(item_dict: Dict[str, Any], base_key_underscore: str, current_source_suffix_short: str) -> Optional[float]:
            def format_title_case_key(base_key: str, suffix_for_format: str) -> str:
                 parts = base_key.split('_')
                 formatted_base = ' '.join(word.capitalize() for word in parts[:-1]) + " Score" if parts[-1] == "score" else ' '.join(word.capitalize() for word in parts)
                 capitalized_suffix_for_format = suffix_for_format.capitalize()
                 # Handle "Llm" specifically if suffix is "llm"
                 if suffix_for_format.lower() == "llm": capitalized_suffix_for_format = "Llm"
                 return f"{formatted_base} {capitalized_suffix_for_format}"

            possible_keys_to_try = []

            # Order of attempts:
            # 1. Title Case with specific suffix (Llm, Img, Vid)
            possible_keys_to_try.append(format_title_case_key(base_key_underscore, current_source_suffix_short))
            
            # 2. If current source is "img", specifically try "Llm" suffix as per user feedback
            if current_source_suffix_short == "img":
                possible_keys_to_try.append(format_title_case_key(base_key_underscore, "llm")) # Try "Llm"
                possible_keys_to_try.append(f"{base_key_underscore}_llm") # and "lowercase_llm"

            # 3. If current source is "llm", specifically try "Lim" suffix
            if current_source_suffix_short == "llm":
                possible_keys_to_try.append(format_title_case_key(base_key_underscore, "Lim"))


            # 4. lowercase_underscore_suffix (for the original source_suffix_short)
            possible_keys_to_try.append(f"{base_key_underscore}_{current_source_suffix_short}")
            
            # 5. Universal keys (no suffix) - less likely for direct scores but good fallback
            possible_keys_to_try.append(base_key_underscore) # e.g. "user_goal_responsiveness_score"
            parts = base_key_underscore.split('_') # For Title case no suffix
            possible_keys_to_try.append(' '.join(word.capitalize() for word in parts[:-1]) + " Score" if parts[-1] == "score" else ' '.join(word.capitalize() for word in parts))

            # Deduplicate keys while preserving order of first appearance
            final_possible_keys = list(dict.fromkeys(possible_keys_to_try))
            
            # print(f"  - Debug (find_score_value): Option '{option_id_val}', BaseKey '{base_key_underscore}', Source '{current_source_suffix_short}', Trying Keys: {final_possible_keys}")


            for key_attempt in final_possible_keys:
                raw_value = item_dict.get(key_attempt)
                if raw_value is not None:
                     try:
                         score = float(raw_value)
                         # if key_attempt not in [possible_keys_to_try[0], f"{base_key_underscore}_{current_source_suffix_short}"]: # Log if a less common key was used
                         #     print(f"  - Info (find_score_value): Found score for '{base_key_underscore}' in option '{option_id_val or 'unknown'}' using specific key '{key_attempt}' from source '{current_source_suffix_short}'.")
                         return score
                     except (ValueError, TypeError):
                         print(f"  - Warning (find_score_value): Found non-numeric value '{raw_value}' for key '{key_attempt}' for option '{option_id_val or 'unknown'}' from source '{current_source_suffix_short}'.")
            
            print(f"  - Info (find_score_value): Could not find valid numeric score for '{base_key_underscore}' using any expected key formats for option '{option_id_val or 'unknown'}' from source '{current_source_suffix_short}'.")
            return None

        for internal_base_key in direct_score_internal_bases:
            output_plot_key = f"{internal_base_key}_{source_suffix_short}"
            score = find_score_value(item, internal_base_key, source_suffix_short)
            scaled_item_output[output_plot_key] = max(0.0, min(10.0, score if score is not None else 0.0))
            
        cost_raw_value = item.get("estimated_cost")
        gb_perc_raw_value = item.get("green_building_potential_percentage")

        gb_score_key_scaled_output = f"green_building_score_{source_suffix_short}_scaled"
        gb_score = 0.0
        if gb_perc_raw_value is not None:
            try:
                gb_score = max(0.0, min(10.0, float(gb_perc_raw_value) / 10.0))
            except (ValueError, TypeError):
                print(f"  - Warning (Scale Item): Invalid green building percentage '{gb_perc_raw_value}' for source '{source_suffix_short}' in option '{option_id_val or 'unknown'}'. Setting scaled score to 0.")
        else:
             print(f"  - Info (Scale Item): 'green_building_potential_percentage' is None for option '{option_id_val or 'unknown'}' from source '{source_suffix_short}'. Scaled green score set to 0.")
        scaled_item_output[gb_score_key_scaled_output] = gb_score

        cost_eff_score_key_scaled_output = f"cost_efficiency_score_{source_suffix_short}_scaled"
        cost_score = 0.0
        if cost_raw_value is not None:
            try:
                current_cost = float(cost_raw_value)
                if not all_estimated_costs or min_overall_cost == max_overall_cost:
                    cost_score = 5.0 if all_estimated_costs else 0.0 
                else:
                    score_val_cost = 10.0 * (max_overall_cost - current_cost) / (max_overall_cost - min_overall_cost)
                    cost_score = max(0.0, min(10.0, score_val_cost)) # Renamed variable to avoid conflict
            except (ValueError, TypeError):
                print(f"  - Warning (Scale Item): Invalid estimated cost value '{cost_raw_value}' for source '{source_suffix_short}' in option '{option_id_val or 'unknown'}'. Setting scaled cost score to 0.")
        else:
             print(f"  - Info (Scale Item): 'estimated_cost' is None for option '{option_id_val or 'unknown'}' from source '{source_suffix_short}'. Scaled cost score set to 0.")
        scaled_item_output[cost_eff_score_key_scaled_output] = cost_score
        
        for key_base in plot_base_keys:
            expected_plot_key_for_source = f"{key_base}_{source_suffix_short}"
            if key_base == "cost_efficiency_score" or key_base == "green_building_score":
                expected_plot_key_for_source += "_scaled"
            
            if expected_plot_key_for_source not in scaled_item_output:
                 print(f"  - Debug (Scale Item): Fallback Initializing missing plot key '{expected_plot_key_for_source}' to 0.0 for option '{option_id_val or 'unknown'}' from source '{source_suffix_short}'.")
                 scaled_item_output[expected_plot_key_for_source] = 0.0
        
        return scaled_item_output

    for item in llm_assessments_list:
        if isinstance(item, dict): processed_llm_assessments.append(scale_assessment_item(item, "llm"))
    for item in image_assessments_list:
        if isinstance(item, dict): processed_image_assessments.append(scale_assessment_item(item, "img"))
    for item in video_assessments_list:
        if isinstance(item, dict): processed_video_assessments.append(scale_assessment_item(item, "vid"))
    
    llm_option_scores_map = get_scores_by_option_id(processed_llm_assessments)
    image_option_scores_map = get_scores_by_option_id(processed_image_assessments)
    video_option_scores_map = get_scores_by_option_id(processed_video_assessments)

    print(f"  - Debug: LLM Scores Map: {json.dumps(llm_option_scores_map, indent=2, ensure_ascii=False)}")
    print(f"  - Debug: Image Scores Map: {json.dumps(image_option_scores_map, indent=2, ensure_ascii=False)}")
    print(f"  - Debug: Video Scores Map: {json.dumps(video_option_scores_map, indent=2, ensure_ascii=False)}")

    dimension_display_names = [
        "使用者目標回應性", "美學與場域關聯性", "機能性與適應彈性",
        "耐久性與維護性", "早期成本效益估算", "綠建築永續潛力"
    ]
    num_dimensions = len(dimension_display_names)
    
    source_configs = [
        {"name": "LLM 評估", "data_map": llm_option_scores_map, "suffix_short": "llm", "color": "cornflowerblue"},
        {"name": "圖像工具評估", "data_map": image_option_scores_map, "suffix_short": "img", "color": "salmon"},
        {"name": "影片工具評估", "data_map": video_option_scores_map, "suffix_short": "vid", "color": "mediumseagreen"},
    ]
    
    total_distinct_bars = num_options * num_dimensions
    x_indices = np.arange(total_distinct_bars) 
    
    fig, ax = plt.subplots(figsize=(max(12, total_distinct_bars * 0.7), 10))

    legend_handles_from_bars = []
    x_tick_labels_major = [] 
    x_tick_positions_major = []
    max_y_value_observed = 0.0

    current_bar_idx = 0
    for opt_idx, option_info_original in enumerate(options_input_data):
        option_id = option_info_original.get("option_id", option_info_original.get("Option Id", f"Opt{opt_idx+1}"))
        option_desc_short = textwrap.fill(option_info_original.get("description", option_id), width=12)
        
        major_tick_pos = (opt_idx * num_dimensions) + (num_dimensions / 2.0) - 0.5
        x_tick_positions_major.append(major_tick_pos)
        x_tick_labels_major.append(option_desc_short)

        max_height_in_option_group = 0.0

        for dim_idx, dim_name in enumerate(dimension_display_names):
            base_score_key_internal = plot_base_keys[dim_idx] 
            current_bar_bottom = 0.0
            
            for src_cfg_idx, source_cfg in enumerate(source_configs):
                source_name = source_cfg["name"]
                # Get the pre-processed dictionary for this option_id from this source's map
                option_scores_dict_from_source = source_cfg["data_map"].get(option_id) 
                
                actual_score_key_to_fetch = f"{base_score_key_internal}_{source_cfg['suffix_short']}"
                if base_score_key_internal == "cost_efficiency_score" or base_score_key_internal == "green_building_score":
                    actual_score_key_to_fetch += "_scaled"
                
                score_value = 0.0
                if option_scores_dict_from_source and isinstance(option_scores_dict_from_source, dict):
                    # Now get the specific score using the constructed key
                    score_value = option_scores_dict_from_source.get(actual_score_key_to_fetch, 0.0)
                    if not isinstance(score_value, (int,float)): # Double check type
                        # print(f"  - Plot Warning: Score '{actual_score_key_to_fetch}' for option '{option_id}' from '{source_name}' not numeric ({score_value}). Using 0.")
                        score_value = 0.0
                else: 
                    # This case means the entire option_id was missing from this source's map
                    # (e.g., image_option_scores_map had no entry for this option_id)
                    # The scale_assessment_item should have ensured all keys exist with 0.0,
                    # so if option_scores_dict_from_source is None, it means no data AT ALL for this option from this source.
                    # print(f"  - Plot Debug: No scores dict for option '{option_id}' from source '{source_name}'. Scores will be 0.")
                    pass

                # Ensure score_value is float for plotting
                try:
                    score_value = float(score_value)
                except (ValueError, TypeError):
                    score_value = 0.0


                bar_plot = ax.bar(x_indices[current_bar_idx], score_value, 
                                  bottom=current_bar_bottom,
                                  color=source_cfg["color"], 
                                  edgecolor='grey', width=0.7,
                                  label=source_name if opt_idx == 0 and dim_idx == 0 and src_cfg_idx == 0 else "") # Simpler legend handle creation

                if score_value > 0.01:
                    ax.text(x_indices[current_bar_idx], current_bar_bottom + score_value / 2, f"{score_value:.1f}",
                            ha='center', va='center', color='black', fontsize=6,
                            fontproperties=cjk_font_prop,
                            path_effects=[PathEffects.withStroke(linewidth=0.5, foreground='white')])
                
                current_bar_bottom += score_value

                # Collect legend handles more reliably
                if not any(handle.get_label() == source_name for handle in legend_handles_from_bars):
                    # Create a proxy artist for the legend if we haven't already for this source
                    # This ensures each source appears once in the legend
                    proxy_handle = Line2D([0], [0], color=source_cfg["color"], lw=4, label=source_name)
                    legend_handles_from_bars.append(proxy_handle)
            
            max_height_in_option_group = max(max_height_in_option_group, current_bar_bottom)
            current_bar_idx += 1
        
        max_y_value_observed = max(max_y_value_observed, max_height_in_option_group)

        sum_final_radar_scores = 0.0
        radar_score_keys_for_sum_display = [ 
            "user_goal_responsiveness_score_final", "aesthetics_context_score_final",
            "functionality_flexibility_score_final", "durability_maintainability_score_final",
            "cost_efficiency_score_final", "green_building_score_final"
        ]
        for r_key in radar_score_keys_for_sum_display:
            score_val = option_info_original.get(r_key, 0.0) 
            try:
                sum_final_radar_scores += float(score_val)
            except (ValueError, TypeError): pass
        
        if sum_final_radar_scores > 0.01: 
            dot_x_pos = x_tick_positions_major[opt_idx]
            # Check if radar dot legend handle already exists
            radar_label_for_legend = '雷達圖評分加總'
            add_radar_legend = not any(h.get_label() == radar_label_for_legend for h in legend_handles_from_bars)

            ax.scatter(dot_x_pos, sum_final_radar_scores, marker='o', color='darkviolet', 
                       s=70, zorder=10, edgecolor='white', linewidth=0.75,
                       label=radar_label_for_legend if add_radar_legend else "") 
            
            if add_radar_legend:
                proxy_radar_dot_handle = Line2D([0], [0], marker='o', color='darkviolet', linestyle='None',
                                                markersize=7, markeredgecolor='white', markeredgewidth=0.75, 
                                                label=radar_label_for_legend)
                legend_handles_from_bars.append(proxy_radar_dot_handle)

            ax_min_curr, ax_max_curr = ax.get_ylim()
            text_offset_factor = 0.015
            current_y_range = ax_max_curr - ax_min_curr if ax_max_curr > ax_min_curr else 1.0
            text_offset = current_y_range * text_offset_factor
            if current_y_range <= 0: text_offset = 0.2

            ax.text(dot_x_pos, sum_final_radar_scores + text_offset, f"{sum_final_radar_scores:.1f}",
                    ha='center', va='bottom', color='darkviolet', fontsize=8,
                    fontproperties=cjk_font_prop,
                    path_effects=[PathEffects.withStroke(linewidth=0.75, foreground='white')])
            max_y_value_observed = max(max_y_value_observed, sum_final_radar_scores + text_offset + 1) 
            
        if opt_idx < num_options - 1:
            separator_x = (opt_idx + 1) * num_dimensions - 0.5
            ax.axvline(separator_x, color='lightgray', linestyle=':', linewidth=1)

    ax.set_ylabel("評分 (各來源0-10分，堆疊總分最高30分)", fontproperties=cjk_font_prop)
    ax.set_title(title, fontproperties=cjk_font_prop, fontsize=16, pad=40) 
    
    ax.set_xticks(x_indices)
    repeated_minor_labels = [textwrap.fill(dim_name, width=10) for _ in range(num_options) for dim_name in dimension_display_names]
    ax.set_xticklabels(repeated_minor_labels, fontproperties=cjk_font_prop, rotation=45, ha="right", fontsize=7)

    ax2 = ax.twiny() 
    ax2.set_xticks(x_tick_positions_major)
    ax2.set_xticklabels(x_tick_labels_major, fontproperties=cjk_font_prop, fontsize=9, color="blue")
    ax2.xaxis.set_ticks_position('bottom') 
    ax2.xaxis.set_label_position('bottom') 
    ax2.spines['bottom'].set_position(('outward', 60)) 
    ax2.set_xlim(ax.get_xlim()) 
    ax2.tick_params(axis='x', length=0) 
    ax2.set_xlabel("方案 (Options)", fontproperties=cjk_font_prop, color="blue", labelpad=20)

    ax.set_ylim(bottom=0, top=max_y_value_observed * 1.05 if max_y_value_observed > 0 else 30)

    if legend_handles_from_bars:
        # Ensure unique handles by label before creating the legend
        unique_handles_dict = {handle.get_label(): handle for handle in legend_handles_from_bars if handle.get_label()}
        final_handles = list(unique_handles_dict.values())
        
        ax.legend(handles=final_handles, title="評分來源/項目",
                  prop=cjk_font_prop if cjk_font_prop else FontProperties(size=8), 
                  loc="upper center", bbox_to_anchor=(0.5, 1.1),
                  ncol=min(len(final_handles), 4),
                  title_fontproperties=cjk_font_prop if cjk_font_prop else FontProperties(size=9, weight='bold'))
    
    plt.subplots_adjust(bottom=0.25, top=0.78) 

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) 

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Stacked Bar Chart generated: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"  - Error generating or saving stacked bar chart: {e}")
        traceback.print_exc() 
        plt.close(fig)
        return None

# --- <<< NEW NODE: Generate Evaluation Visualization >>> ---
def generate_evaluation_visualization_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    (For SpecialEvaAgent/FinalEvaAgent)
    Merges detailed assessments from LLM, Image, and Video evaluation branches.
    Calculates final scores for each option.
    Generates visualizations (e.g., radar chart, stacked bar chart).
    Sets final task status.
    """
    node_name = "Generate Visualization & Final Scores"
    print(f"--- Running Node: {node_name} ---")
    # Access tasks list from state directly to modify in place for current_task
    tasks = state["tasks"]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)):
        print(f"  - Error: Invalid task index {current_idx}. Skipping.")
        return {"tasks": tasks} # Still return tasks to propagate state, even if invalid index

    current_task = tasks[current_idx]
    selected_agent = current_task.get("selected_agent")

    if selected_agent not in ["SpecialEvaAgent", "FinalEvaAgent"]:
        print(f"  - Skipping node {node_name}, not a Special/Final agent.")
        _append_feedback(current_task, "Visualization/Final Scoring skipped: Not a Special/Final agent.", node_name)
        # If an EvaAgent somehow reaches here, it's a routing error. Mark task as failed.
        if current_task.get("status") != "failed":
             _set_task_failed(current_task, "Routing error: EvaAgent reached final scoring node.", node_name)
        return {"tasks": tasks}

    # If the task is already marked as failed (e.g., from prep node), skip processing but still go to end
    if current_task.get("status") == "failed":
        print(f"  - Task {current_task.get('task_id')} already failed. Proceeding to merge branch outputs before final status update.")
        pass # Do not return early if failed, proceed to merge branch outputs


    # --- Retrieve temporary outputs from parallel branches (from WorkflowState) ---
    # Use the new _temp_output keys from WorkflowState
    llm_branch_payload = state.get("llm_temp_output", {})
    image_branch_payload = state.get("image_temp_output", {})
    video_branch_payload = state.get("video_temp_output", {})

    # --- Store these raw payloads into TaskState new fields ---
    # Use the new _branch_payload keys in TaskState
    current_task["llm_branch_payload"] = llm_branch_payload
    current_task["image_branch_payload"] = image_branch_payload
    current_task["video_branch_payload"] = video_branch_payload
    
    print(f"  - Merged payloads stored in TaskState for task {current_task.get('task_id')}.")


    # --- Merge evaluation data into current_task["evaluation"] from stored branch_outputs ---
    if "evaluation" not in current_task:
        current_task["evaluation"] = {}
    
    # Merge detailed assessments lists
    current_task["evaluation"]["llm_detailed_assessment"] = llm_branch_payload.get("llm_detailed_assessment", [])
    current_task["evaluation"]["image_detailed_assessment"] = image_branch_payload.get("image_detailed_assessment", [])
    current_task["evaluation"]["video_detailed_assessment"] = video_branch_payload.get("video_detailed_assessment", [])

    # Merge other specific outputs from LLM branch (e.g., overall feedback, selected option)
    # Only update if the payload has these keys, don't overwrite if already exists from a previous run/retry
    if "feedback_llm_overall" in llm_branch_payload:
        current_task["evaluation"]["feedback_llm_overall"] = llm_branch_payload["feedback_llm_overall"]
    # --- MODIFICATION: Do NOT directly set "assessment" here from branch payload ---
    # The overall task assessment should be determined at the end based on the final merged results
    # if "assessment" in llm_branch_payload:
    #     current_task["evaluation"]["assessment"] = llm_branch_payload["assessment"] # LLM's overall assessment (e.g., Score X/10)
    # --- END MODIFICATION ---
    if "selected_option_identifier" in llm_branch_payload:
        current_task["evaluation"]["selected_option_identifier"] = llm_branch_payload["selected_option_identifier"]
    
    # --- Merge feedback_log ---
    # Start with existing feedback, then append branch feedback in order: Text > Image > Video
    aggregated_feedback_log_parts = [current_task.get("feedback_log", "") or ""] # Start with existing, ensure it's a string
    if llm_branch_payload.get("feedback_log_from_branch"):
        aggregated_feedback_log_parts.append(llm_branch_payload["feedback_log_from_branch"])
    if image_branch_payload.get("feedback_log_from_branch"):
        aggregated_feedback_log_parts.append(image_branch_payload["feedback_log_from_branch"])
    if video_branch_payload.get("feedback_log_from_branch"):
        aggregated_feedback_log_parts.append(video_branch_payload["feedback_log_from_branch"])

    # Filter out empty strings before joining
    current_task["feedback_log"] = "\n".join(filter(None, aggregated_feedback_log_parts)).strip()
    print(f"  - Merged feedback log.")


    # --- Merge subgraph_error ---
    # Start with existing error, then append branch errors in order: Text > Image > Video
    initial_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or "" # Ensure string
    error_parts = [initial_subgraph_error]

    if llm_branch_payload.get("subgraph_error_from_branch"):
        error_parts.append(f"LLM_Branch_Error: {llm_branch_payload['subgraph_error_from_branch']}")
    if image_branch_payload.get("subgraph_error_from_branch"):
        error_parts.append(f"Image_Branch_Error: {image_branch_payload['subgraph_error_from_branch']}")
    if video_branch_payload.get("subgraph_error_from_branch"):
        error_parts.append(f"Video_Branch_Error: {video_branch_payload['subgraph_error_from_branch']}")

    merged_subgraph_error = "; ".join(filter(None, error_parts)).strip("; ")
    if merged_subgraph_error:
        # Only update if there are errors to merge, don't overwrite a clean state with empty string
        current_task["evaluation"]["subgraph_error"] = merged_subgraph_error
        print(f"  - Merged Subgraph Errors: {merged_subgraph_error}")
    # else: # If merged_subgraph_error is empty, ensure the key is removed or None if desired
    #     current_task["evaluation"].pop("subgraph_error", None) # Option to clear if no errors


    # --- Check if any branch explicitly failed the task ---
    # If any branch payload has task_failed_in_branch = True, mark the main task as failed
    if llm_branch_payload.get("task_failed_in_branch") or \
       image_branch_payload.get("task_failed_in_branch") or \
       video_branch_payload.get("task_failed_in_branch"):
        if current_task.get("status") != "failed":
            # Use the merged subgraph error for the failure message if available
            fail_msg_details = merged_subgraph_error if merged_subgraph_error else 'Failure reported by one or more parallel evaluation branches. Check branch logs.'
            print(f"  - At least one parallel evaluation branch reported critical failure. Setting task to failed. Details: {fail_msg_details}")
            _set_task_failed(current_task, f"Failure in parallel evaluation branch. Details: {fail_msg_details}", node_name)
        else:
             print(f"  - One or more branches reported failure, but task was already marked failed.")


    # --- Proceed with visualization logic using the merged data ---
    # Use the lists stored in current_task["evaluation"]
    llm_assessments = llm_branch_payload.get("llm_detailed_assessment", [])
    image_assessments = image_branch_payload.get("image_detailed_assessment", [])
    video_assessments = video_branch_payload.get("video_detailed_assessment", [])

    prepared_inputs = current_task.get("task_inputs", {})
    options_data_master_list = prepared_inputs.get("options_data", [])

    if current_task.get("status") == "failed" and not options_data_master_list:
        print(f"  - Task is marked failed, and no options_data to visualize. Skipping visualization specific steps.")
        # Proceed to status update at the end.
    elif not isinstance(options_data_master_list, list):
        msg = f"Critical Error: options_data in task_inputs is not a list, but type {type(options_data_master_list)}. Cannot merge scores or visualize."
        print(f"  - {msg}")
        _append_feedback(current_task, msg, node_name)
        _set_task_failed(current_task, msg, node_name) # This failure overrides prior ones if more critical
        return {"tasks": tasks}
    elif not options_data_master_list:
        msg = "No options_data found in task_inputs. Cannot merge scores or visualize."
        print(f"  - {msg}")
        _append_feedback(current_task, msg, node_name)
        # Consider this a failure only if it's a Special/Final EvaAgent task that *must* evaluate options
        if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
             _set_task_failed(current_task, msg, node_name)
        # Proceed to status update at the end even if no options.

    print(f"  - Master options list for merge: {len(options_data_master_list)} items.")
    print(f"  - Data for merge: LLM ({len(llm_assessments)} recs), Image ({len(image_assessments)} recs), Video ({len(video_assessments)} recs)")

    # List to hold consolidated data for each option, BEFORE final score calculation
    consolidated_options_data = []

    # Score keys that we expect from the parallel evaluation branches
    # Ensure these match the keys returned by the LLM/tool prompts
    # IMPORTANT: Stacked bar chart relies on _score_llm, _score_img, _score_vid suffixes.
    # The merging process below calculates averages and stores them with 'collected_' prefix.
    # The final radar chart uses '_score_final' from the calculated scores.
    # The stacked bar chart needs the raw _score_llm, _score_img, _score_vid from the branch payloads.
    # We already retrieved llm_assessments, image_assessments, video_assessments from the state.

    # Iterate through each option defined in the master list from prepare_evaluation_inputs_node
    for master_opt_data_item in options_data_master_list:
        # Ensure master_opt_data_item is a dictionary
        if not isinstance(master_opt_data_item, dict):
            print(f"  - Warning: Item in options_data_master_list is not a dict: {master_opt_data_item}. Skipping.")
            continue

        opt_id = master_opt_data_item.get("option_id")
        if not opt_id:
            print(f"  - Warning: Master option data item missing option_id. Item: {str(master_opt_data_item)[:100]}. Skipping.")
            continue

        print(f"\n  --- Merging results for Option ID: {opt_id} ---")

        # Find corresponding assessments from each tool's output list
        # Use next with a default empty dict to ensure we always get a dict even if no match
        llm_opt_eval = next((item for item in llm_assessments if isinstance(item, dict) and item.get("option_id") == opt_id), {})
        img_opt_eval = next((item for item in image_assessments if isinstance(item, dict) and item.get("option_id") == opt_id), {})
        vid_opt_eval = next((item for item in video_assessments if isinstance(item, dict) and item.get("option_id") == opt_id), {})

        print(f"    - Found results: LLM ({bool(llm_opt_eval)}), Image ({bool(img_opt_eval)}), Video ({bool(vid_opt_eval)})")

        # Dictionary to hold collected values for this option from all sources BEFORE final calculation/mapping
        # Initialize with metadata from the master list
        option_collected_data: Dict[str, Any] = {
            "option_id": opt_id, # opt_id is already confirmed to exist
            "description": master_opt_data_item.get("description", "N/A"),
            "architecture_type": master_opt_data_item.get("architecture_type", "General"),
            # Initialize collected score/value keys to lists to collect values
            # These keys will temporarily store values from different branches
            "temp_collected_scores": {key: [] for key in ["user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text"]},
            # Collected texts/errors
            "collected_feedback_texts": [], # List to collect feedback in order (Text > Image > Video)
            "collected_errors": [], # List to collect errors
            # {{ Ensure image_paths and video_paths from master_opt_data_item are carried here if needed for intermediate steps, though likely not }}
            # "image_paths": master_opt_data_item.get("image_paths", []), # Usually not needed in this intermediate dict
            # "video_paths": master_opt_data_item.get("video_paths", []), # Usually not needed in this intermediate dict
        }

        # --- Collect Scores and Values from Branches ---
        # Iterate through the branches IN ORDER (LLM > Image > Video)
        # This ensures that if the same key appears in multiple branches,
        # we collect them for averaging.
        branches_to_process = [
            ("Text Eval", llm_opt_eval),
            ("Image Eval", img_opt_eval),
            ("Video Eval", vid_opt_eval),
        ]

        for branch_name, branch_data in branches_to_process:
            if not branch_data:
                print(f"    - No {branch_name} data for option {opt_id}.")
                continue

            # Collect numeric scores for averaging later
            for score_key in ["user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text"]:
                value = branch_data.get(score_key)
                if isinstance(value, (int, float)):
                    option_collected_data["temp_collected_scores"][score_key].append(float(value))
                    # print(f"      - Collected {branch_name} score for '{score_key}': {value}")
                # else:
                    # print(f"      - {branch_name} did not provide valid numeric score for '{score_key}' (value: {value}).")


            # Collect feedback text (order matters for concatenation)
            if branch_data.get("llm_feedback_text"):
                 option_collected_data["collected_feedback_texts"].append(f"{branch_name} Feedback: {branch_data['llm_feedback_text']}")


            # Collect errors
            if branch_data.get("error"):
                 option_collected_data["collected_errors"].append(f"{branch_name} Error: {branch_data['error']}")


        # --- Calculate Averages and Consolidate Data for Final Calculation ---
        # Dictionary to pass to _calculate_final_scores_for_options
        consolidated_data_for_final_calc: Dict[str, Any] = {
             "option_id": opt_id, # opt_id is confirmed
             "description": master_opt_data_item.get("description", "N/A"), # Directly from master
             "architecture_type": master_opt_data_item.get("architecture_type", "General"), # Directly from master
             # --- MODIFICATION: Copy image_paths and video_paths from master_opt_data_item ---
             "image_paths": master_opt_data_item.get("image_paths", []),
             "video_paths": master_opt_data_item.get("video_paths", []),
             # --- END MODIFICATION ---
        }

        # Calculate averages from collected values
        # Use new keys prefixed with 'collected_' for clarity in the next step
        for score_key, values in option_collected_data["temp_collected_scores"].items():
            # --- MODIFICATION: Calculate average and store with 'collected_' prefix ---
            collected_key_name = f"collected_{score_key.replace('_llm', '')}" # Map e.g., user_goal_responsiveness_score_llm -> collected_user_goal_responsiveness_score
            if values:
                consolidated_data_for_final_calc[collected_key_name] = sum(values) / len(values)
                print(f"    - Averaged '{score_key}': {consolidated_data_for_final_calc[collected_key_name]:.2f} (from {len(values)} sources)")
            else:
                 # If no tool provided this score, try to get initial value from master inputs if applicable (like cost/green)
                 initial_key_map = {
                     "estimated_cost": "initial_estimated_cost",
                     "green_building_potential_percentage": "initial_green_building_percentage"
                 }
                 master_input_key = initial_key_map.get(score_key)
                 initial_val = master_opt_data_item.get(master_input_key) if master_input_key else None

                 if initial_val is not None and isinstance(initial_val, (int, float, str)) and str(initial_val).replace('.', '', 1).isdigit():
                     consolidated_data_for_final_calc[collected_key_name] = float(initial_val)
                     print(f"    - No tool scores for '{score_key}', using initial value from inputs: {initial_val}")
                 else:
                     consolidated_data_for_final_calc[collected_key_name] = 0.0 # Default to 0.0 if no values found or calculable
                     # print(f"    - No values found or calculable for '{score_key}'. Defaulting to 0.0.")

            # --- END MODIFICATION ---

        # Combine collected feedback texts
        consolidated_data_for_final_calc["scoring_rationale"] = "\n".join(filter(None, option_collected_data["collected_feedback_texts"])).strip()
        if not consolidated_data_for_final_calc["scoring_rationale"]:
             consolidated_data_for_final_calc["scoring_rationale"] = "No detailed feedback available from evaluation branches."

        # Combine collected errors
        consolidated_data_for_final_calc["errors_during_processing"] = "; ".join(filter(None, option_collected_data["collected_errors"])).strip("; ")
        if consolidated_data_for_final_calc["errors_during_processing"]:
             print(f"    - Option {opt_id} had errors during parallel processing: {consolidated_data_for_final_calc['errors_during_processing']}")

        # Add this consolidated data (with averaged/collected values) to the list for the final step
        consolidated_options_data.append(consolidated_data_for_final_calc)
    # --- End of loop through master_options_data_list ---


    if not consolidated_options_data: # This means options_data_master_list was empty or all items were invalid/skipped
        msg = "After attempting to consolidate evaluation results, no valid options remained for final scoring/visualization."
        print(f"  - {msg}")
        _append_feedback(current_task, msg, node_name)
        # Mark task as failed if it's a Special/Final agent and no options were processable
        if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
             _set_task_failed(current_task, msg, node_name)
        # Proceed to status update at the end even if no options.


    # --- Calculate Final Scores (Cost Efficiency, Green Building) ---
    # Now call _calculate_final_scores_for_options with the consolidated data
    budget_limit_raw = prepared_inputs.get("budget_limit_overall")
    # Safely parse budget_limit_raw to a float
    budget_limit = None
    if budget_limit_raw is not None:
        try:
            # Allow parsing from int, float, or string containing a number
            budget_limit = float(budget_limit_raw)
        except (ValueError, TypeError):
            print(f"  - Warning: Could not parse budget_limit_overall '{budget_limit_raw}' to a float. Cost efficiency score will not use budget context.")
            budget_limit = None


    options_with_calculated_final_scores = _calculate_final_scores_for_options(
        consolidated_options_data, # Use the consolidated data with collected raw values
        budget_limit # Pass the parsed budget limit
    )

    # Check if calculation function returned valid data
    if not options_with_calculated_final_scores and options_data_master_list:
         msg = "Failed to calculate final scores for any options, despite options being in inputs."
         print(f"  - {msg}")
         _append_feedback(current_task, msg, node_name)
         if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
             # Only fail the task if options were expected but couldn't be scored
             _set_task_failed(current_task, msg, node_name)
         # We might still want to generate an empty chart or log the failure, so continue


    # --- Store Final Detailed Assessment ---
    # This list contains options with their final _score_final values, merged feedback, errors etc.
    current_task["evaluation"]["detailed_assessment"] = options_with_calculated_final_scores
    current_task["outputs"] = current_task.get("outputs", {})
    current_task["outputs"]["detailed_option_scores"] = options_with_calculated_final_scores # Also store in outputs for consistency

    print(f"  - Final scores calculated and stored for {len(options_with_calculated_final_scores)} options.")

    # Retrieve the CJK font property found during initialization
    global cjk_font_prop
    if 'cjk_font_prop' not in globals() or cjk_font_prop is None:
         print("  - Warning: CJK font property not initialized or found. Charts may have rendering issues with Chinese characters.")
         # Attempt to get font property again if it's None
         cjk_font_prop = get_cjk_font()


    # --- Generate Radar Chart ---
    # Only attempt to generate chart if there are options with final scores
    if options_with_calculated_final_scores:
        print(f"  - Attempting to generate Radar Chart.")
        chart_filename_base = f"evaluation_radar_{current_task.get('task_id', uuid.uuid4().hex[:8])}.png"
        chart_title = f"方案綜合比較圖: {current_task.get('description', '評估任務')}"
        if not isinstance(chart_title, str): # Ensure title is a string
            chart_title = str(chart_title)
        if len(options_with_calculated_final_scores) > 7:
            print(f"  - Warning: More than 7 options ({len(options_with_calculated_final_scores)}). Radar chart might be cluttered.")

        # Call the radar chart generation function
        # <<< MODIFIED: Pass cjk_font_prop to radar chart function >>>
        saved_radar_chart_path = _generate_radar_chart_for_options(
            options_data=options_with_calculated_final_scores, # Use the cleaned, final scored options
            output_filename=os.path.join(DIAGRAM_CACHE_DIR, chart_filename_base), # Save to DIAGRAM_CACHE_DIR
            title=chart_title,
            cjk_font_prop=cjk_font_prop # Pass the CJK font property
        )
        # <<< END MODIFIED >>>

        if saved_radar_chart_path:
            # Ensure output_files list exists
            if "output_files" not in current_task: current_task["output_files"] = []
            radar_chart_base64 = ""
            try:
                # Read the saved chart file and encode it to base64
                with open(saved_radar_chart_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    radar_chart_base64 = f"data:image/png;base64,{encoded_string}"
                print(f"  - Successfully encoded radar chart to base64.")
            except Exception as e:
                print(f"  - Error encoding radar chart to base64: {e}")
                # Append error to subgraph_error
                current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
                current_task["evaluation"]["subgraph_error"] = (current_subgraph_error + f"; Radar chart base64 encoding failed: {e}").strip("; ")
                _append_feedback(current_task, f"Failed to encode radar chart to base64: {e}", node_name)


            # Add the generated chart file info to output_files
            current_task["output_files"].append({
                "filename": os.path.basename(saved_radar_chart_path),
                "path": saved_radar_chart_path, # Store the full path
                "type": "image/png",
                "description": f"Radar chart comparing design options for {current_task.get('description', 'Evaluation Task')}.",
                "source_agent": selected_agent,
                "base64_data": radar_chart_base64 # Include base64 for immediate display
            })
            _append_feedback(current_task, f"Radar chart generated: {os.path.basename(saved_radar_chart_path)}", node_name)
        else:
            error_msg_radar = "Failed to generate radar chart image. Check logs for details."
            _append_feedback(current_task, error_msg_radar, node_name)
            current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
            current_task["evaluation"]["subgraph_error"] = (current_subgraph_error + "; RadarChart generation failed").strip("; ")
            print(f"  - {error_msg_radar}")

    else:
         print(f"  - No options with final scores available. Skipping radar chart generation.")
         _append_feedback(current_task, "Skipped radar chart generation: No valid options to score.", node_name)
         current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
         current_task["evaluation"]["subgraph_error"] = (current_subgraph_error + "; Skipped RadarChart generation (No scores)").strip("; ")


    # --- Generate Stacked Bar Chart for Source Scores ---
    # This chart uses the raw per-branch assessments BEFORE final averaging for radar.
    # It requires llm_detailed_assessment, image_detailed_assessment, video_detailed_assessment
    # which should contain the 6 dimension scores (0-10) from each source.

    # Check if we have the necessary per-branch assessment data
    # and if there were options in the input to begin with.
    can_generate_stacked_bar = (
        options_data_master_list and # Original options were provided
        (llm_assessments or image_assessments or video_assessments) # At least one branch provided data
    )

    if can_generate_stacked_bar:
        print(f"  - Attempting to generate Stacked Bar Chart for score sources.")
        stacked_bar_filename_base = f"evaluation_stacked_bar_{current_task.get('task_id', uuid.uuid4().hex[:8])}.png"
        stacked_bar_title = f"評分來源比較: {current_task.get('description', '評估任務')}"
        if not isinstance(stacked_bar_title, str):
            stacked_bar_title = str(stacked_bar_title)

        saved_stacked_bar_path = _generate_stacked_bar_chart_for_options(
            options_input_data=options_data_master_list, # Use master list for option IDs/descriptions
            llm_assessments=llm_assessments,
            image_assessments=image_assessments,
            video_assessments=video_assessments,
            output_filename=os.path.join(DIAGRAM_CACHE_DIR, stacked_bar_filename_base),
            title=stacked_bar_title,
            cjk_font_prop=cjk_font_prop # Pass the CJK font property
        )

        if saved_stacked_bar_path:
            if "output_files" not in current_task: current_task["output_files"] = []
            stacked_bar_base64 = ""
            try:
                with open(saved_stacked_bar_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    stacked_bar_base64 = f"data:image/png;base64,{encoded_string}"
                print(f"  - Successfully encoded stacked bar chart to base64.")
            except Exception as e:
                print(f"  - Error encoding stacked bar chart to base64: {e}")
                current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error") or ""
                current_task["evaluation"]["subgraph_error"] = (current_subgraph_error + f"; StackedBar base64 encoding failed: {e}").strip("; ")
                _append_feedback(current_task, f"Failed to encode stacked bar chart to base64: {e}", node_name)

            current_task["output_files"].append({
                "filename": os.path.basename(saved_stacked_bar_path),
                "path": saved_stacked_bar_path,
                "type": "image/png",
                "description": f"Stacked bar chart comparing score details for {current_task.get('description', 'Evaluation Task')}.",
                "source_agent": selected_agent,
                "base64_data": stacked_bar_base64
            })
            _append_feedback(current_task, f"Stacked bar chart generated: {os.path.basename(saved_stacked_bar_path)}", node_name)
        else:
            error_msg_stacked = "Failed to generate stacked bar chart for score sources. Check logs."
            _append_feedback(current_task, error_msg_stacked, node_name)
            current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
            current_task["evaluation"]["subgraph_error"] = (current_subgraph_error + "; StackedBar generation failed").strip("; ")
            print(f"  - {error_msg_stacked}")
    else:
        print(f"  - Skipping stacked bar chart: Not enough data (no input options or no branch assessments).")
        _append_feedback(current_task, "Skipped stacked bar chart: Insufficient data.", node_name)
    # --- <<< END NEW STACKED BAR CHART GENERATION >>> ---


    # --- Final Assessment Summary and Status Update ---
    # If LLM provided an overall assessment, use it. Otherwise, generate a summary based on scored options.
    # --- MODIFICATION: Determine overall task assessment based on success/failure and scored options ---
    # If task was marked failed during merging or branches, it stays failed.
    # If not failed, set to "completed".
    if current_task.get("status") != "failed":
         current_task["status"] = "completed"
         print(f"  - Task not marked failed. Setting final task status to COMPLETED.")
    # Set a text summary for current_task["evaluation"]["assessment"]
    if options_with_calculated_final_scores:
         # Summarize the outcome based on the number of options processed
         current_task["evaluation"]["assessment"] = f"Multi-option Analysis Complete ({len(options_with_calculated_final_scores)} options evaluated)"
         # If the LLM provided a selected_option_identifier, mention it here
         selected_id = current_task["evaluation"].get("selected_option_identifier")
         if selected_id:
             current_task["evaluation"]["assessment"] += f" - Selected: {selected_id}"
         # Optional: Add average overall score to assessment text
         # Example: Calculate average of UserGoal, Aesthetics, Func, Dura, CostEff, Green scores
         total_avg_score = 0
         count_scores = 0
         score_dims_for_avg = ["user_goal_goal_responsiveness_score_final", "aesthetics_context_score_final", "functionality_flexibility_score_final", "durability_maintainability_score_final", "cost_efficiency_score_final", "green_building_score_final"]
         for opt in options_with_calculated_final_scores:
             for dim in score_dims_for_avg:
                 score = opt.get(dim)
                 if isinstance(score, (int, float)):
                     total_avg_score += score
                     count_scores += 1
         if count_scores > 0:
             overall_score_avg = total_avg_score / count_scores
             current_task["evaluation"]["assessment"] += f" (Avg Dim Score: {overall_score_avg:.2f})"

    elif options_data_master_list:
         # Options were provided in inputs, but none were successfully scored
         current_task["evaluation"]["assessment"] = "Multi-option Analysis Failed (Could not score options)"
    else:
         # No options were provided in inputs
         current_task["evaluation"]["assessment"] = "Multi-option Analysis Skipped (No options in inputs)"

    # If task was marked failed above, override the assessment text to reflect failure
    if current_task.get("status") == "failed":
         current_task["evaluation"]["assessment"] = f"TASK FAILED: {current_task['evaluation']['assessment']}"

    print(f"  - Final task assessment text set to: {current_task['evaluation']['assessment']}")
    # --- END MODIFICATION ---


    # Append overall feedback summary to feedback_log
    overall_feedback_parts = [f"Overall Evaluation Summary ({selected_agent}):"]

    # Add LLM's overall feedback if available
    if current_task["evaluation"].get("feedback_llm_overall"):
        overall_feedback_parts.append(f"  LLM Master Feedback: {current_task['evaluation']['feedback_llm_overall']}")

    # Add summary for each scored option
    if options_with_calculated_final_scores:
        overall_feedback_parts.append("\nDetailed Option Summaries:")
        for opt in options_with_calculated_final_scores: # Use cleaned data for feedback log
            opt_id_log = opt.get('option_id', 'Unknown Option')
            # Use description (scheme theme) for logging if available
            opt_theme_log = opt.get('description', opt_id_log)
            rationale_log = opt.get('scoring_rationale', 'N/A')
            errors_log = opt.get('errors_during_processing', '')
            error_log_str_log = f" (Processing Errors: {errors_log})" if errors_log else ""

            # Build score string dynamically from final scores
            score_string_parts = []
            for dim_key in ["user_goal_responsiveness_score_final", "aesthetics_context_score_final", "functionality_flexibility_score_final", "durability_maintainability_score_final", "cost_efficiency_score_final", "green_building_score_final"]:
                 score_value = opt.get(dim_key)
                 display_name = dim_key.replace('_score_final', '').replace('_', ' ').title() # Simple display name
                 # Use Chinese display names if available (match _generate_radar_chart_for_options categories_display)
                 # Example mapping (ensure consistency)
                 chinese_names = {
                     "user_goal_responsiveness_score_final": "使用者目標回應性",
                     "aesthetics_context_score_final": "美學與場域關聯性",
                     "functionality_flexibility_score_final": "機能性與適應彈性",
                     "durability_maintainability_score_final": "耐久性與維護性",
                     "cost_efficiency_score_final": "早期成本效益估算",
                     "green_building_score_final": "綠建築永續潛力"
                 }
                 display_name = chinese_names.get(dim_key, display_name)

                 if isinstance(score_value, (int, float)):
                      score_string_parts.append(f"{display_name}={score_value:.2f}")
                 else:
                      score_string_parts.append(f"{display_name}={score_value}") # Log N/A or None

            score_string = ", ".join(score_string_parts)


            overall_feedback_parts.append(
                f"  - Option: {opt_theme_log} (ID: {opt_id_log}){error_log_str_log}\n"
                f"    Final Scores (0-10): {score_string}\n"
                f"    Collected Est. Cost: {opt.get('collected_estimated_cost', 'N/A')}, Collected Green %: {opt.get('collected_green_building_potential_percentage', 'N/A')}\n"
                f"    Scoring Rationale: {rationale_log[:300]}{'...' if len(rationale_log)>300 else ''}" # Limit rationale length
            )
    else:
        overall_feedback_parts.append("\nNo options were successfully scored.")
        # If there were branch errors but no scored options, summarize those errors again
        if merged_subgraph_error:
             overall_feedback_parts.append(f"\nErrors encountered during parallel processing: {merged_subgraph_error}")


    # Append the overall feedback to the task's feedback log
    # Ensure the overall summary is added as a distinct block
    _append_feedback(current_task, "\n".join(filter(None, overall_feedback_parts)), node_name)


    # --- Finalize Task Status ---
    # The status is already set based on branch failures / completion likelihood above.
    # This helper just logs the final status message based on current_task['status'].
    _update_eval_status_at_end(current_task, node_name) # Pass node_name only
    print(f"  - Visualization and final scoring complete. Final task status: {current_task['status']}")

    # Clear temporary workflow state keys after merging
    # These keys are set to None in the return dictionary
    print(f"  - Cleared temporary WorkflowState keys: llm_temp_output, image_temp_output, video_temp_output.")


    # Return the updated tasks list (which includes the modified current_task) and reset temporary keys
    # Modifying tasks[current_idx] in place is sufficient as 'tasks' is accessed directly from state.
    # We return the temporary keys as None to explicitly clear them in the workflow state after this node runs.
    return {"tasks": tasks, "llm_temp_output": None, "image_temp_output": None, "video_temp_output": None}

# =============================================================================
# <<< Evaluation Subgraph Nodes (Refactored) >>>
# =============================================================================

async def prepare_evaluation_inputs_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Prepares inputs for the evaluation agent based on its type (EvaAgent, SpecialEvaAgent, FinalEvaAgent).
    - EvaAgent: Focuses on the immediately preceding task's outputs.
    - SpecialEvaAgent / FinalEvaAgent: Aggregates outputs from multiple preceding "option" tasks.
      It structures these into 'options_data' including text summaries, media paths, and any initial cost/green data.
    Updates `current_task['task_inputs']` including a 'needs_detailed_criteria' flag.
    """
    node_name = "Prepare Eval Inputs"
    tasks = [t.copy() for t in state['tasks']]
    current_idx = state['current_task_index']
    if not (0 <= current_idx < len(tasks)):
         print(f"Eval Subgraph Error ({node_name}): Invalid current_task_index {current_idx}")
         if 0 <= current_idx < len(tasks): 
            _set_task_failed(tasks[current_idx], f"Invalid current_task_index {current_idx}", node_name)
         return {"tasks": tasks}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')
    print(f"--- Running Node: {node_name} for Task {current_idx} (Agent: {selected_agent}, Objective: {current_task.get('description')}) ---")

    runtime_config = config.get("configurable", {})
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    # --- MODIFICATION: Pass agent name for default LLM lookup ---
    ea_llm_config_params = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config_params, agent_name_for_default_lookup="eva_agent") # eva_agent is the config key
    # --- END MODIFICATION ---

    current_task["task_inputs"] = {}
    prompt_inputs = {}
    prompt_template_name = ""
    is_standard_eval = selected_agent == "EvaAgent"
    is_special_or_final_eval = selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]

    # `options_for_evaluation_pre_llm` is used to pass a preliminary structuring to the LLM.
    # The LLM is then expected to refine this, especially for file associations and unique option_ids.
    options_for_evaluation_pre_llm = [] 

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
        
        full_task_summary_parts = ["Workflow History Summary:"]
        tasks_for_summary_prep = tasks[:current_idx]
        if not tasks_for_summary_prep:
            full_task_summary_parts.append("  No prior tasks in history.")
        else:
            for i, task_in_history in enumerate(tasks_for_summary_prep):
                task_id_hist = task_in_history.get("task_id", f"hist_task_{i}")
                desc_hist = task_in_history.get("description", "N/A")
                agent_hist = task_in_history.get("selected_agent", "N/A")
                status_hist = task_in_history.get("status", "N/A")
                full_task_summary_parts.append(
                    f"  - Task {i} (ID: {task_id_hist}): '{desc_hist}' | Agent: {agent_hist} | Status: {status_hist}"
                )
        prepared_workflow_summary_str = "\n".join(full_task_summary_parts)
        
        budget_limit_overall = state.get("user_budget_limit") 
        
        for i, task in enumerate(tasks):
             if i < current_idx and task.get("status") == "completed":
                 task_id = task.get("task_id", f"task_{i}")
                 task_outputs = task.get("outputs", {}) 
                 task_output_files = task.get("output_files", [])
                 
                 if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
                     # This section creates a PRELIMINARY options_for_evaluation_pre_llm
                     # The LLM will use this as a hint but is instructed to perform more detailed parsing
                     # and file association.
                     option_image_files_pre_llm = []
                     option_video_files_pre_llm = []
                     other_files_pre_llm = []
                     for f_info in task_output_files:
                         f_type = f_info.get("type", "").lower()
                         f_path = f_info.get("path")
                         if not f_path or not os.path.exists(f_path):
                             print(f"    - Warning (Eval Prep - PreLLM): File path '{f_path}' for task {task_id} does not exist. Skipping for pre-LLM options data.")
                             continue
                         # The file's structured description is now passed as-is. LLM will parse TaskDesc.
                         file_desc_for_llm = f_info.get("description", "") 

                         if "image" in f_type:
                             option_image_files_pre_llm.append({"path": f_path, "filename": f_info.get("filename", os.path.basename(f_path)), "structured_description": file_desc_for_llm})
                         elif "video" in f_type:
                             option_video_files_pre_llm.append({"path": f_path, "filename": f_info.get("filename", os.path.basename(f_path)), "structured_description": file_desc_for_llm})
                         else:
                             other_files_pre_llm.append({"path": f_path, "filename": f_info.get("filename", os.path.basename(f_path)), "structured_description": file_desc_for_llm})
                     
                     option_data_for_pre_llm_structure = {
                         "option_id": task_id, # LLM will re-assign if one task creates multiple options
                         "description": task.get("description", "N/A"), 
                         "task_objective": task.get("task_objective", "N/A"),
                         "textual_summary_from_outputs": "", 
                         "image_paths": option_image_files_pre_llm, # Preliminary paths
                         "video_paths": option_video_files_pre_llm, # Preliminary paths
                         "other_relevant_files": other_files_pre_llm, # Preliminary paths
                         "raw_outputs_for_llm_parsing": {}, 
                         "architecture_type": task_outputs.get("architecture_type", "General") 
                     }
                     
                     outputs_to_summarize = {k: v for k, v in task_outputs.items() if k not in [
                         "mcp_internal_messages", "grounding_sources", "search_suggestions",
                         "architecture_type" 
                     ]}
                     try:
                        option_data_for_pre_llm_structure["textual_summary_from_outputs"] = json.dumps(outputs_to_summarize, ensure_ascii=False, default=str, indent=None)[:1000] + "..."
                        option_data_for_pre_llm_structure["raw_outputs_for_llm_parsing"] = outputs_to_summarize 
                     except Exception as e_json:
                        option_data_for_pre_llm_structure["textual_summary_from_outputs"] = f"Error summarizing outputs for pre-LLM: {e_json}"
                        option_data_for_pre_llm_structure["raw_outputs_for_llm_parsing"] = {"error": f"Could not serialize for pre-LLM: {e_json}"}

                     options_for_evaluation_pre_llm.append(option_data_for_pre_llm_structure)
                     print(f"    - Prepared PRE-LLM option data hint for Task ID {task_id}: Images: {len(option_image_files_pre_llm)}, Videos: {len(option_video_files_pre_llm)}")

                 if task_outputs:
                     outputs_for_summary = task_outputs.copy()
                     outputs_for_summary.pop("mcp_internal_messages", None)
                     outputs_for_summary.pop("grounding_sources", None)
                     outputs_for_summary.pop("search_suggestions", None)
                     aggregated_outputs[task_id] = task_outputs 
                 task_files_from_history = task.get("output_files") 
                 if task_files_from_history:
                     # Pass structured description in aggregated_files_json as well
                     files_with_structured_desc_for_llm = []
                     for f_hist in task_files_from_history:
                         f_copy = f_hist.copy()
                         f_copy['source_task_id'] = task_id
                         # The description field should already be structured from when it was created.
                         # No need to re-construct it here if it's already like "SourceAgent: ...; TaskDesc: ..."
                         files_with_structured_desc_for_llm.append(f_copy)
                     aggregated_files_raw.extend(files_with_structured_desc_for_llm)

        filtered_aggregated_files = _filter_base64_from_files(aggregated_files_raw) # Base64 is removed
        prompt_inputs = {
            "selected_agent": selected_agent,
            "current_task_objective": current_task.get("task_objective", "N/A"),
            "user_input": state.get("user_input", "N/A"),
            "full_task_summary": prepared_workflow_summary_str,
            "aggregated_outputs_json": json.dumps(aggregated_outputs, ensure_ascii=False, default=str),
            "aggregated_files_json": json.dumps(filtered_aggregated_files, ensure_ascii=False, default=str), # Contains files with their structured descriptions
            "llm_output_language": llm_output_language,
            # Pass the PRELIMINARY options structure as a hint to the LLM.
            # The LLM is instructed to refine this, assign unique option_ids, and perform detailed file association.
            "options_data_json": json.dumps(options_for_evaluation_pre_llm, ensure_ascii=False, default=str), 
            "budget_limit_overall": budget_limit_overall if budget_limit_overall is not None else "N/A",
        }
    else:
        err_msg = f"Invalid or missing agent type '{selected_agent}' for evaluation task."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    prompt_template_config_key = f"ea_{prompt_template_name}_prompt"
    prompt_template_str = runtime_config.get(prompt_template_config_key) or \
                          config_manager.get_prompt_template("eva_agent", prompt_template_name)
    if not prompt_template_str:
        err_msg = f"Missing required prompt template '{prompt_template_name}' (config key: '{prompt_template_config_key}') for {selected_agent} input preparation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        _set_task_failed(current_task, err_msg, node_name)
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    try:
        print(f"Eval Subgraph ({node_name}): Formatting prompt '{prompt_template_name}' with inputs: {list(prompt_inputs.keys())}")
        prep_prompt = prompt_template_str.format(**prompt_inputs)
        print(f"Eval Subgraph ({node_name}): Invoking LLM for {selected_agent} input prep...")
        prep_response = await llm.ainvoke(prep_prompt)
        prep_content = prep_response.content.strip()

        if prep_content.startswith("```json"): prep_content = prep_content[7:-3].strip()
        elif prep_content.startswith("```"): prep_content = prep_content[3:-3].strip()

        try:
            prepared_eval_inputs = json.loads(prep_content) # This is the full JSON object from the LLM
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
                # `prepared_eval_inputs` is the entire dictionary returned by the LLM.
                # It should contain all keys defined in the prompt's "Required Output JSON Format",
                # including the refined `options_data`.
                current_task["task_inputs"] = prepared_eval_inputs 
                
                needs_detailed_criteria = prepared_eval_inputs.get("needs_detailed_criteria", False)
                print(f"Eval Subgraph ({node_name}): Evaluation inputs prepared successfully by LLM for {selected_agent}. Needs Detailed Criteria: {needs_detailed_criteria}")
                
                if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
                    # --- CRITICAL: Use options_data FROM THE LLM's response ---
                    if "options_data" in prepared_eval_inputs and isinstance(prepared_eval_inputs.get("options_data"), list):
                        llm_generated_options = prepared_eval_inputs["options_data"]
                        # The LLM should have already performed the file association and option ID assignment.
                        # current_task["task_inputs"]["options_data"] is already set because current_task["task_inputs"] = prepared_eval_inputs
                        print(f"    - Using LLM-generated 'options_data' with {len(llm_generated_options)} options for {selected_agent}.")
                        
                        if llm_generated_options:
                            first_opt_debug_llm = llm_generated_options[0]
                            print(f"      Debug (LLM options_data): First option ID: '{first_opt_debug_llm.get('option_id')}', Description: '{first_opt_debug_llm.get('description', '')[:50]}...', Images: {len(first_opt_debug_llm.get('image_paths',[]))}, Videos: {len(first_opt_debug_llm.get('video_paths',[]))}")
                            for img_file_info in first_opt_debug_llm.get('image_paths', []):
                                print(f"        - Image for opt {first_opt_debug_llm.get('option_id')}: {img_file_info.get('filename')}") # No structured_description here, LLM consumed it
                        else:
                            print(f"    - LLM returned an empty 'options_data' list for {selected_agent}.")
                    else:
                        # This case indicates a problem with the LLM's response or the prompt.
                        print(f"    - CRITICAL WARNING: 'options_data' key NOT FOUND or not a LIST in LLM response for {selected_agent} during input prep. This is unexpected. Check LLM output and prompt '{prompt_template_name}'. Raw content hint: {prep_content[:500]}")
                        _append_feedback(current_task, "LLM response missing 'options_data'; evaluation may be incomplete.", node_name)
                        # As a fallback, we might use the pre-LLM structured one, but it lacks the refined file association.
                        # current_task["task_inputs"]["options_data"] = options_for_evaluation_pre_llm # Fallback
                        if current_task.get("evaluation") is None: current_task["evaluation"] = {}
                        current_task["evaluation"]["subgraph_error"] = (current_task.get("evaluation", {}).get("subgraph_error", "") or "" + f"; LLM_options_data_missing_or_invalid_for_{selected_agent}").strip("; ")
                        # If options_data is critical and missing, we might consider failing the task here.
                        _set_task_failed(current_task, f"LLM failed to provide valid 'options_data' for {selected_agent} evaluation.", node_name)


                    # `prepared_workflow_summary_str` is generated in Python and should be stored if the LLM doesn't also output it.
                    # The prompt for prepare_final_evaluation_inputs asks for `evaluation_target_full_summary` which is derived from `full_task_summary`.
                    # So, `prepared_eval_inputs` should already have it. If not, we can add it.
                    if "evaluation_target_full_summary" not in current_task["task_inputs"]:
                         current_task["task_inputs"]["evaluation_target_full_summary"] = prepared_workflow_summary_str
                    print(f"    - 'options_data' (from LLM) and 'evaluation_target_full_summary' stored in task_inputs for {selected_agent}.")

                if "needs_detailed_criteria" not in current_task["task_inputs"]: # Should be set by LLM
                    current_task["task_inputs"]["needs_detailed_criteria"] = False # Default if LLM somehow missed it
                if "evaluation" not in current_task: current_task["evaluation"] = {}
                # Clear any previous subgraph error for this specific prep step if successful now
                current_task["evaluation"]["subgraph_error"] = None 

        except json.JSONDecodeError:
            err_msg = f"Could not parse LLM JSON response for {selected_agent}. Raw content: '{prep_content}'"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            _set_task_failed(current_task, err_msg, node_name)

    except KeyError as ke:
        err_msg = f"Formatting error (KeyError: {ke}). Check prompt template '{prompt_template_name}' and inputs for {selected_agent}."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        print(f"--- Problematic Prompt Template ({prompt_template_name}) ---")
        print(repr(prompt_template_str)) 
        print(f"--- Provided Inputs ---")
        print(json.dumps(prompt_inputs, indent=2, ensure_ascii=False, default=str)) 
        print(f"---------------------------------")
        traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)
    except Exception as e:
        err_msg = f"Unexpected error during LLM call or processing for {selected_agent}: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        traceback.print_exc()
        _set_task_failed(current_task, err_msg, node_name)

    tasks[current_idx] = current_task
    # Ensure 'current_task' is also updated in the state if other nodes expect it directly for some reason
    # (though usually direct modifications to state.tasks is the primary way)
    return {"tasks": tasks, "current_task": current_task.copy()}

async def gather_criteria_sources_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Gathers context from RAG/Search to potentially inform criteria generation.
    Updates `current_task["evaluation"]["criteria_sources"]`.
    If evaluating specific architecture types, queries will be tailored.
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

    # --- NEW: Check for specific architecture type for Special/Final Agents ---
    # 'options_data' comes from prepare_evaluation_inputs_node, which now includes 'architecture_type' per option.
    # For criteria generation, we might use the type of the first option, or a consensus if multiple.
    # For simplicity, let's assume the 'current_task_objective' for Special/Final eval implies the type,
    # or that the prep_eval_inputs node sets a 'primary_architecture_type' in task_inputs.
    # For now, we'll make a simpler check: if task_inputs has architecture_type.
    
    primary_architecture_type = None
    if current_task.get("selected_agent") in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # Try to get it from the first option if options_data exists and has it
        options_data = current_task.get("task_inputs", {}).get("options_data")
        if isinstance(options_data, list) and options_data:
            primary_architecture_type = options_data[0].get("architecture_type")
        if not primary_architecture_type: # Fallback if not in options_data
            primary_architecture_type = current_task.get("task_inputs",{}).get("primary_architecture_type") # LLM from prep might set this

    if primary_architecture_type and primary_architecture_type.lower() != "general":
        print(f"  - Specific architecture type identified for criteria search: {primary_architecture_type}")
        type_specific_query_part = f" for {primary_architecture_type} design"
    else:
        type_specific_query_part = ""
    # --- END NEW ---

    runtime_config = config["configurable"]
    retriever_k = runtime_config.get("retriever_k", 5) # Use runtime config for K

    # RAG Query
    rag_query = f"Find evaluation standards or relevant context for: {task_desc_for_query}{type_specific_query_part}" #MODIFIED
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
    - SpecialEvaAgent / FinalEvaAgent: Detailed rubric focusing on the 5 key dimensions,
                                       potentially tailored to architecture type.
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
    # --- MODIFICATION: Initialize subgraph_error as empty string if None ---
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
    # --- END MODIFICATION ---
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
        prompt_template_str_from_config = runtime_config.get("ea_generate_criteria_prompt") # Get from runtime config first
        if not prompt_template_str_from_config: # Fallback to static config
            prompt_config_obj = config_manager.get_agent_config("eva_agent").prompts.get(prompt_template_name)
            prompt_template = prompt_config_obj.template if prompt_config_obj else None
        else:
            prompt_template = prompt_template_str_from_config

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
        prompt_template_str_from_config = runtime_config.get("ea_generate_final_criteria_prompt") # Get from runtime config first
        if not prompt_template_str_from_config: # Fallback to static config
            prompt_config_obj = config_manager.get_agent_config("eva_agent").prompts.get(prompt_template_name)
            prompt_template = prompt_config_obj.template if prompt_config_obj else None
        else:
            prompt_template = prompt_template_str_from_config
        
        options_data_for_criteria = current_task.get("task_inputs", {}).get("options_data", [])
        architecture_type_for_criteria = "General" 
        if isinstance(options_data_for_criteria, list) and options_data_for_criteria:
            arch_type = options_data_for_criteria[0].get("architecture_type")
            if arch_type and isinstance(arch_type, str) and arch_type.lower() != "unknown":
                architecture_type_for_criteria = arch_type

        prompt_inputs = {
            "selected_agent": selected_agent,
            "current_task_objective": current_task.get("task_objective", "N/A"),
            "user_input": state.get("user_input", "N/A"),
            # Use the definitively prepared summary
            "full_task_summary": current_task.get("task_inputs", {}).get("prepared_workflow_summary", "Workflow summary not available."),
            "options_to_evaluate_json": json.dumps(options_data_for_criteria, ensure_ascii=False, indent=2, default=str),
            "architecture_type": architecture_type_for_criteria, 
            "rag_context": criteria_sources.get("rag", "Not gathered or available."),
            "search_context": criteria_sources.get("search", "Not gathered or available."),
            "llm_output_language": llm_output_language
        }
    else:
        err_msg = f"Invalid agent type '{selected_agent}' encountered in criteria generation."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip("; ") if subgraph_error else f"; {err_msg}".strip("; ")
        generated_output = err_msg
        current_task["evaluation"]["specific_criteria"] = generated_output
        if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error
        tasks[current_idx] = current_task
        return {"tasks": tasks, "current_task": current_task.copy()}

    if not prompt_template:
        err_msg = f"Missing required prompt template '{prompt_template_name}' for {selected_agent} criteria/rubric generation!"
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        subgraph_error = (subgraph_error + f"; {err_msg}").strip(";") if subgraph_error else f"; {err_msg}".strip(";")
        generated_output = f"Error: Prompt template '{prompt_template_name}' not found."
    else:
        try:
            # The KeyError for 'specific_criteria' would happen here if the template is wrong
            prompt = prompt_template.format(**prompt_inputs)
            print(f"  - Invoking LLM for {selected_agent} criteria/rubric...")
            response = llm.invoke(prompt) # Assuming synchronous invoke for simplicity here, adapt if async
            generated_output = response.content.strip()
            print(f"  - Generated Criteria/Rubric:\n{generated_output[:500]}...") 
        except KeyError as ke:
            # This is where the original KeyError was caught
            err_msg = f"Formatting error (KeyError: {ke}). Check prompt template '{prompt_template_name}' and its input_variables for {selected_agent}. Ensure '{ke}' is not mistakenly in the template expecting to be an input when it's an output."
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            print(f"--- Problematic Prompt Template ('{prompt_template_name}') Content ---")
            print(repr(prompt_template)) # Print the actual template string
            print(f"--- Provided Inputs for .format() ---")
            print(json.dumps(prompt_inputs, indent=2, ensure_ascii=False))
            print(f"---------------------------------")
            # --- MODIFICATION: Ensure subgraph_error is a string before concatenation ---
            current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
            subgraph_error = (current_subgraph_error + f"; Criteria Gen Formatting Error: {ke}").strip("; ")
            # --- END MODIFICATION ---
            generated_output = f"Error during generation (Formatting): {ke}"
        except Exception as e:
            err_msg = f"Criteria/Rubric generation LLM error for {selected_agent}: {e}"
            print(f"Eval Subgraph Error ({node_name}): {err_msg}")
            # --- MODIFICATION: Ensure subgraph_error is a string before concatenation ---
            current_subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
            subgraph_error = (current_subgraph_error + f"; {err_msg}").strip("; ")
            # --- END MODIFICATION ---
            generated_output = f"Error during generation: {e}" 

    current_task["evaluation"]["specific_criteria"] = generated_output 
    if subgraph_error: # Check if subgraph_error has content
        current_task["evaluation"]["subgraph_error"] = subgraph_error

    tasks[current_idx] = current_task
    return {"tasks": tasks, "current_task": current_task.copy()}

# --- NEW NODE: Distributor for Parallel Evaluation ---
def distribute_evaluations_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Prepares the state for potentially parallel evaluation tool runs.
    It doesn't return a routing decision itself but ensures the necessary
    information (like which tools are active) is available for the next routing function.
    This node mainly acts as a clear separation point before fanning out.
    """
    node_name = "Distribute Evaluations"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    current_task = state["tasks"][current_idx]
    selected_agent = current_task.get("selected_agent")

    if selected_agent not in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # This node is primarily for Special/Final agents. EvaAgent goes direct to a tool.
        print(f"  - Agent is {selected_agent}, not Special/Final. Distribution logic skipped.")
        return {} # No change to state needed from here for EvaAgent path.

    if current_task.get("status") == "failed":
        print(f"  - Task already failed. Skipping distribution.")
        return {}

    prepared_inputs = current_task.get("task_inputs", {})
    options_data = prepared_inputs.get("options_data", [])
    if not isinstance(options_data, list): options_data = []

    needs_text_eval = True # Always
    needs_image_eval = any(isinstance(opt, dict) and opt.get("image_paths") for opt in options_data)
    needs_video_eval = any(isinstance(opt, dict) and opt.get("video_paths") for opt in options_data)

    # Store flags in the task's evaluation dict for the router to use
    if "evaluation" not in current_task: current_task["evaluation"] = {}
    current_task["evaluation"]["active_eval_tools"] = {
        "text": needs_text_eval,
        "image": needs_image_eval,
        "video": needs_video_eval
    }
    print(f"  - Active evaluation tools determined: Text={needs_text_eval}, Image={needs_image_eval}, Video={needs_video_eval}")
    
    # Update the task in the state
    tasks = list(state["tasks"])
    tasks[current_idx] = current_task
    return {"tasks": tasks}


# --- MODIFIED: Routing function after distribute_evaluations_node ---
def route_to_parallel_eval_tools(state: WorkflowState) -> List[str]:
    """
    Routes to the necessary evaluation tool nodes for parallel execution
    based on flags set by `distribute_evaluations_node`.
    Returns a list of node names to be run in parallel.
    """
    node_name = "Route to Parallel Eval Tools"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    current_task = state["tasks"][current_idx]
    selected_agent = current_task.get("selected_agent")

    # If EvaAgent, this router should not be hit, it goes directly from criteria to a single tool.
    # This router is for SpecialEvaAgent / FinalEvaAgent.
    if selected_agent not in ["SpecialEvaAgent", "FinalEvaAgent"]:
        print(f"  - Warning: {node_name} reached by {selected_agent}. Should be Special/Final. Defaulting to text eval path.")
        return ["evaluate_with_text_agent"] # Fallback, though this path should ideally not be taken by EvaAgent

    if current_task.get("status") == "failed":
        print(f"  - Task failed. No parallel tools will be run. Routing to visualization/end.")
        # Even if failed, we might want to go to visualization to show partial results or errors.
        # Or, if we want to halt completely, this could return a key that leads to END.
        # For now, let's assume visualization node can handle failed upstream.
        return ["generate_evaluation_visualization"] # Go to final merge/viz to handle failed state

    active_tools_flags = current_task.get("evaluation", {}).get("active_eval_tools", {"text": True})
    
    parallel_nodes_to_run = []
    if active_tools_flags.get("text"):
        parallel_nodes_to_run.append("evaluate_with_text_agent")
    if active_tools_flags.get("image"):
        parallel_nodes_to_run.append("evaluate_with_image_agent")
    if active_tools_flags.get("video"):
        parallel_nodes_to_run.append("evaluate_with_video_agent")

    if not parallel_nodes_to_run: # Should not happen if text is always true
        print(f"  - Warning ({node_name}): No active tools identified for {selected_agent}. Defaulting to text eval.")
        return ["evaluate_with_text_agent"]

    print(f"  - Routing to parallel execution of: {parallel_nodes_to_run}")
    return parallel_nodes_to_run

# --- Individual Evaluation Tool Nodes (evaluate_with_llm_node, evaluate_with_image_node, evaluate_with_video_node) ---
# These nodes need to be robust. If an option doesn't have images/videos,
# the corresponding image/video tool node should gracefully skip that option
# and record an appropriate entry in its `*_detailed_assessment` output.

# Example modification for evaluate_with_image_node:
def evaluate_with_image_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    node_name = "Image Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # ... (initial checks for failure, template loading etc. remain similar) ...
    if current_task.get("status") == "failed" and selected_agent == "EvaAgent":
        print(f"  - Skipping node {node_name} for EvaAgent, task already failed.")
        return {"tasks": tasks}

    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""
    prepared_inputs = current_task.get("task_inputs", {})
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")
    # ... (template loading logic) ...
    img_tool_prompt_template_str = config_manager.get_prompt_template("eva_agent", "evaluate_option_with_image_tool") # Example

    if not img_tool_prompt_template_str: # Handle missing template
        err_msg = "Missing 'evaluate_option_with_image_tool' prompt template."
        # ... (error handling as before) ...
        if selected_agent == "EvaAgent": _set_task_failed(current_task, err_msg, node_name)
        current_task["evaluation"]["image_detailed_assessment"] = [{"option_id": "N/A", "error": err_msg, "llm_feedback_text": err_msg}]
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    all_options_img_tool_evaluations = []
    
    # Determine which options to process
    options_to_process_for_this_tool = []
    if selected_agent == "EvaAgent":
        target_image_paths = prepared_inputs.get("evaluation_target_image_paths", [])
        if target_image_paths:
            options_to_process_for_this_tool = [{"option_id": "evaluated_task_image", "description": "Evaluated Task", "image_paths": [{"path": p} for p in target_image_paths]}]
        else: # EvaAgent, but no images for it
            print(f"  - EvaAgent: No target images for image evaluation. Skipping.")
            current_task["evaluation"]["assessment"] = "Fail" 
            _update_eval_status_at_end(current_task, f"{node_name} (EvaAgent - No Images)")
            tasks[current_idx] = current_task
            return {"tasks": tasks}
    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        options_data_input = prepared_inputs.get("options_data", [])
        if not isinstance(options_data_input, list): options_data_input = []
        options_to_process_for_this_tool = options_data_input # Process all options, skip internally if no images

    print(f"  - Image Evaluation: Will attempt to process {len(options_to_process_for_this_tool)} options/targets.")

    for option_data in options_to_process_for_this_tool:
        if not isinstance(option_data, dict): 
            print(f"  - Warning (Image Eval): Option data is not a dict: {option_data}. Skipping.")
            all_options_img_tool_evaluations.append({"option_id": "unknown_malformed", "error": "Malformed option data.", "llm_feedback_text":"Malformed option data."})
            continue

        option_id = option_data.get("option_id", f"img_opt_anon_{uuid.uuid4().hex[:4]}")
        
        # --- MODIFICATION: Check if this specific option has image_paths ---
        option_image_files = option_data.get("image_paths", [])
        if not option_image_files or not isinstance(option_image_files, list):
            print(f"    - Option {option_id}: No image_paths found or invalid format. Skipping image evaluation for this option.")
            all_options_img_tool_evaluations.append({
                "option_id": option_id,
                "llm_feedback_text": "Skipped: No images provided for this option.",
                # No 'error' key here, as it's a valid skip, not an error in the tool itself.
                # We can add a 'status' key if needed: "status": "skipped_no_images"
            })
            continue 
        # --- END MODIFICATION ---

        image_paths_for_option = [img_info["path"] for img_info in option_image_files if isinstance(img_info, dict) and img_info.get("path")]
        if not image_paths_for_option:
            print(f"    - Option {option_id}: image_paths list exists but contains no valid paths. Skipping.")
            all_options_img_tool_evaluations.append({"option_id": option_id, "llm_feedback_text": "Skipped: No valid image paths."})
            continue
            
        print(f"\n  --- Evaluating Option (ID: {option_id}) with Image Tool ---")
        # ... (rest of the image evaluation logic for this single option: prompt formatting, tool call, parsing)
        # Ensure that `single_option_img_tool_result` always has "option_id"
        # Example of handling result:
        single_option_img_tool_result = {"option_id": option_id} 
        try:
            # ... (actual tool call and parsing) ...
            # parsed_llm_json = ...
            # single_option_img_tool_result.update(parsed_llm_json)
            # If error during tool call for this specific option:
            # single_option_img_tool_result["error"] = "Error details..."
            # single_option_img_tool_result["llm_feedback_text"] = "Failed to evaluate images for this option due to error."
            pass # Placeholder for actual image tool logic
        except Exception as e_opt:
            single_option_img_tool_result["error"] = f"Exception during image eval for option {option_id}: {e_opt}"
            single_option_img_tool_result["llm_feedback_text"] = f"Failed due to: {e_opt}"
        
        all_options_img_tool_evaluations.append(single_option_img_tool_result)
        
    current_task["evaluation"]["image_detailed_assessment"] = all_options_img_tool_evaluations
    if selected_agent == "EvaAgent":
        # ... (EvaAgent specific logic for assessment based on the single result in all_options_img_tool_evaluations)
        pass
    else: # SpecialEvaAgent / FinalEvaAgent
        print(f"  - {selected_agent}: Image evaluation part completed for {len(all_options_img_tool_evaluations)} processed option entries. Results stored.")

    if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error
    tasks[current_idx] = current_task
    return {"tasks": tasks}

# Similar robustness needs to be added to evaluate_with_video_node and evaluate_with_llm_node
# for Special/Final agents: they should iterate all options from options_data,
# but only *act* on those relevant to them (e.g. llm_node processes all text,
# video_node only processes options with video_paths).

def evaluate_with_llm_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs evaluation using LLM based on prepared inputs and criteria/rubric.
    For Special/Final agents, it processes options_data and stores results in 'llm_detailed_assessment'.
    For EvaAgent, it performs standard evaluation.
    """
    node_name = "LLM Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): 
         # In a parallel node, if something is fundamentally wrong with index,
         # we should return a payload indicating failure for this branch.
         return {"llm_temp_output": {"error": f"Invalid task index {current_idx}", "task_failed_in_branch": True, "subgraph_error_from_branch": f"Invalid task index {current_idx}"}}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    # If EvaAgent and already failed, pass through.
    # For Special/Final, let it run even if another branch failed, merge node handles overall status.
    if current_task.get("status") == "failed" and selected_agent == "EvaAgent":
        print(f"  - Skipping node {node_name} for EvaAgent, previous step failed.")
        return {"tasks": tasks} # EvaAgent path returns tasks directly (single path)

    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or ""

    print(f"  - Performing LLM Evaluation for Agent: {selected_agent}")

    prepared_inputs = current_task.get("task_inputs", {})
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")

    runtime_config = config["configurable"]
    # --- MODIFICATION: Pass agent name for default LLM lookup ---
    ea_llm_config_params = runtime_config.get("ea_llm", {})
    llm = initialize_llm(ea_llm_config_params, agent_name_for_default_lookup="eva_agent") 
    # --- END MODIFICATION ---
    llm_output_language = runtime_config.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    
    # --- MODIFICATION: Get prompt template using new key name from runtime config first ---
    evaluation_template_config_key = "ea_evaluation_prompt"
    evaluation_template_str = runtime_config.get(evaluation_template_config_key) or \
                           config_manager.get_prompt_template("eva_agent", "evaluation")
    # --- END MODIFICATION ---

    llm_eval_results_for_options = [] # For Special/Final Agents
    llm_overall_feedback = "No overall feedback from LLM."
    llm_assessment_summary = "N/A"
    llm_selected_option_identifier = None
    branch_error = ""
    branch_failed = False # Flag to indicate critical failure in this branch

    # Prepare prompt inputs common to all modes
    prompt_inputs = {
        "selected_agent": selected_agent,
        "evaluation_target_description": prepared_inputs.get("evaluation_target_description", "N/A"),
        "evaluation_target_objective": prepared_inputs.get("evaluation_target_objective", "N/A"),
        "specific_criteria": specific_criteria,
        "llm_output_language": llm_output_language,
        "user_input": state.get("user_input", "N/A"),
        # Standard Eval Inputs (will be empty/None if not standard)
        "evaluation_target_outputs_json": prepared_inputs.get("evaluation_target_outputs_json", "{}") if selected_agent == "EvaAgent" else "{}",
        "evaluation_target_image_paths_str": ", ".join(prepared_inputs.get("evaluation_target_image_paths", [])) or "None" if selected_agent == "EvaAgent" else "None",
        "evaluation_target_video_paths_str": ", ".join(prepared_inputs.get("evaluation_target_video_paths", [])) or "None" if selected_agent == "EvaAgent" else "None",
        "evaluation_target_other_files_str": str(prepared_inputs.get("evaluation_target_other_files", []) or "None") if selected_agent == "EvaAgent" else "None",
        # Final/Special Eval Inputs (will be empty/None if not Special/Final)
        "full_task_summary": prepared_inputs.get("evaluation_target_full_summary", "N/A") if selected_agent != "EvaAgent" else "N/A", # Use the prepared summary from inputs
        "evaluation_target_key_image_paths_str": ", ".join(prepared_inputs.get("evaluation_target_key_image_paths", [])) or "None" if selected_agent != "EvaAgent" else "None",
        "evaluation_target_key_video_paths_str": ", ".join(prepared_inputs.get("evaluation_target_key_video_paths", [])) or "None" if selected_agent != "EvaAgent" else "None",
        "evaluation_target_other_artifacts_summary_str": prepared_inputs.get("evaluation_target_other_artifacts_summary", "N/A") if selected_agent != "EvaAgent" else "N/A",
        "options_data_for_llm_eval_json": json.dumps(prepared_inputs.get("options_data", []), ensure_ascii=False, default=str) if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"] else "[]",
        # --- MODIFICATION: Read feedback summaries from task_inputs if already there, otherwise default. Merge node will consolidate. ---
        "image_tool_feedback": prepared_inputs.get("image_tool_feedback_summary", "N/A from prep inputs"), # This might need adjustment - maybe get from state directly before merge?
        "video_tool_feedback": prepared_inputs.get("video_tool_feedback_summary", "N/A from prep inputs")  # Same as above
        # It seems better to get these from state within the merge node, not pass them to LLM eval node.
        # Let's remove these two from prompt_inputs and template. The merge node will use them.
        # REMOVED: "image_tool_feedback" and "video_tool_feedback" from prompt_inputs and template
        # The LLM evaluation prompt should probably NOT include feedback from other branches, as it should operate independently.
        # The merge node is where cross-branch info should be consolidated for final scoring/summary.
        # Let's revert the prompt template modification for 'evaluation'.
    }
    
    # Re-checking the 'evaluation' prompt template - it DOES include 'image_tool_feedback' and 'video_tool_feedback'.
    # This implies the LLM *is* intended to see these summaries *before* final visualization.
    # This is a design choice - it means the LLM's final overall assessment can factor them in.
    # Let's keep them in prompt_inputs, but they are NOT expected to be in `prepared_inputs`.
    # They should be outputs from the Image/Video nodes.
    # Ah, the parallel nodes evaluate_with_image_node and evaluate_with_video_node are intended to
    # return these feedback summaries. The merge node needs to collect them.
    # Let's remove these two keys from `prompt_inputs` for *this* node (`evaluate_with_llm_node`).
    # The LLM node just needs the *task inputs* and *its own result*. The merging happens later.

    # Let's refine the payload structure and return values for parallel nodes.
    # Each parallel node should return *its own* result data plus error/feedback for its branch.

    llm_branch_payload = {
        "llm_detailed_assessment": [],
        "feedback_log_from_branch": "",
        "subgraph_error_from_branch": "",
        "task_failed_in_branch": False,
        "feedback_llm_overall": "", # Overall feedback from LLM eval
        "assessment": "N/A", # Overall assessment from LLM eval (e.g., Score 7/10)
        "selected_option_identifier": None # Selected option from LLM eval
    }

    if not evaluation_template_str:
        err_msg = f"Missing unified 'evaluation' prompt template for eva_agent (config key: '{evaluation_template_config_key}')."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        branch_error = err_msg
        branch_failed = True
        if selected_agent == "EvaAgent":
            _set_task_failed(current_task, err_msg, node_name) # EvaAgent fails task immediately
            tasks[current_idx] = current_task
            return {"tasks": tasks}
        else: # Special/Final returns branch payload indicating failure
            llm_branch_payload["subgraph_error_from_branch"] = branch_error
            llm_branch_payload["task_failed_in_branch"] = branch_failed
            llm_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: {err_msg}"
            return {"llm_temp_output": llm_branch_payload}


    try:
        print(f"  - Formatting unified evaluation prompt for {selected_agent}...")
        # --- MODIFICATION: Format prompt_inputs based on selected_agent ---
        # This was already somewhat handled, but let's make sure only relevant keys are passed.
        # The `evaluation` prompt template is designed to handle all three agent types by checking `selected_agent`.
        # So, we just need to ensure all *possible* input_variables are in prompt_inputs, and the template handles N/A.
        # The current prompt_inputs dictionary seems mostly correct based on the template variables.
        prompt = evaluation_template_str.format(**prompt_inputs)
        print(f"  - Invoking LLM for {selected_agent} evaluation...")
        response = llm.invoke(prompt)
        content = response.content.strip() if response and hasattr(response, 'content') else ""

        if not content:
            print(f"Eval Subgraph Warning ({node_name}): LLM returned empty content for {selected_agent}. Defaulting to Fail.")
            branch_error = "LLM returned empty response."
            # For Special/Final, populate error for options if possible
            if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
                 options_data = prepared_inputs.get("options_data", [])
                 llm_eval_results_for_options = [{"option_id": opt.get("option_id", "unknown_option"), "llm_feedback_text": branch_error, "error": branch_error} for opt in options_data]
            if selected_agent == "EvaAgent":
                 current_task["evaluation"]["assessment"] = "Fail"
                 current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + f"\n[{node_name}]: {branch_error}").strip()
                 _update_eval_status_at_end(current_task, node_name) # EvaAgent sets final status
                 tasks[current_idx] = current_task
                 return {"tasks": tasks} # EvaAgent path
            else: # Special/Final branch returns payload
                 llm_branch_payload["llm_detailed_assessment"] = llm_eval_results_for_options
                 llm_branch_payload["feedback_log_from_branch"] = f"[{node_name} Warning]: {branch_error}"
                 llm_branch_payload["subgraph_error_from_branch"] = branch_error
                 llm_branch_payload["task_failed_in_branch"] = False # Empty response isn't necessarily task failure, maybe just this branch
                 return {"llm_temp_output": llm_branch_payload}

        else:
            print(f"  - Raw LLM response content received for {selected_agent}.")
            if content.startswith("```json"): content = content[7:-3].strip()
            elif content.startswith("```"): content = content[3:-3].strip()

            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    assessment_type_from_llm = parsed_json.get("assessment_type")
                    print(f"  - LLM Parsed Assessment Type: {assessment_type_from_llm}")

                    if selected_agent == "EvaAgent":
                        assessment = parsed_json.get("assessment", "Fail")
                        feedback = parsed_json.get("feedback", "No feedback provided.")
                        improvement_suggestions = parsed_json.get("improvement_suggestions", "N/A")
                        current_task["evaluation"]["assessment"] = assessment
                        current_task["evaluation"]["assessment_type"] = assessment_type_from_llm or "Standard"
                        feedback_details = f"Assessment Type: {current_task['evaluation']['assessment_type']}\nAssessment: {assessment}\nFeedback: {feedback}\nSuggestions: {improvement_suggestions}"
                        _append_feedback(current_task, feedback_details, node_name)
                        _update_eval_status_at_end(current_task, node_name) # EvaAgent sets final status here
                        tasks[current_idx] = current_task
                        return {"tasks": tasks} # EvaAgent path

                    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
                        detailed_option_scores_from_llm = parsed_json.get("detailed_option_scores", [])
                        llm_selected_option_identifier = parsed_json.get("selected_option_identifier")
                        llm_assessment_summary = parsed_json.get("assessment", "N/A")
                        llm_overall_feedback = parsed_json.get("feedback", "No overall feedback from LLM.")

                        if isinstance(detailed_option_scores_from_llm, list):
                            llm_eval_results_for_options = detailed_option_scores_from_llm
                            print(f"  - {selected_agent} - Parsed {len(llm_eval_results_for_options)} option scores from LLM.")
                            # --- Validation: Check for required keys in each option score ---
                            required_score_keys = ["option_id", "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text"]
                            for i, option_score_dict in enumerate(llm_eval_results_for_options):
                                if not isinstance(option_score_dict, dict):
                                    print(f"    - Warning: Option score item {i} is not a dict: {option_score_dict}")
                                    llm_eval_results_for_options[i] = {"option_id": option_score_dict.get("option_id", f"malformed_opt_{i}"), "error": "Malformed score item", "llm_feedback_text": "Malformed score item"}
                                    branch_error = (branch_error + f"; Malformed LLM score item {i}").strip("; ")
                                    continue
                                missing_keys = [key for key in required_score_keys if key not in option_score_dict]
                                if missing_keys:
                                     print(f"    - Warning: Option score item {i} (ID: {option_score_dict.get('option_id', 'N/A')}) missing keys: {missing_keys}")
                                     # Add missing keys with default values or mark as error
                                     for mk in missing_keys:
                                         if mk in ["estimated_cost", "green_building_potential_percentage"]: option_score_dict[mk] = 0.0 # Default numeric
                                         elif mk == "option_id": option_score_dict[mk] = option_score_dict.get("option_id", f"missing_id_{i}")
                                         else: option_score_dict[mk] = "Missing" # Default string
                                     # Consider if missing required keys should fail the branch
                                     # branch_error = (branch_error + f"; Opt {option_score_dict.get('option_id', 'N/A')} missing keys: {', '.join(missing_keys)}").strip("; ")

                            if not llm_eval_results_for_options: # If list was valid but empty after parsing
                                print(f"  - Warning: {selected_agent} ran, but 'detailed_option_scores' list from LLM was empty.")
                                branch_error = (branch_error + f"; LLM returned empty 'detailed_option_scores' list.").strip("; ")
                                options_data_input = prepared_inputs.get("options_data", [])
                                # Populate with error entries based on original options data if possible
                                llm_eval_results_for_options = [{"option_id": opt.get("option_id", "unknown_option"), "llm_feedback_text": "LLM did not provide detailed_option_scores or list was empty.", "error": "Empty detailed_option_scores from LLM."} for opt in options_data_input]

                        else: # detailed_option_scores_from_llm is not a list
                            print(f"  - Warning: {selected_agent} ran, but 'detailed_option_scores' in LLM response was not a list (type: {type(detailed_option_scores_from_llm)}).")
                            branch_error = (branch_error + f"; LLM 'detailed_option_scores' is not a list (type: {type(detailed_option_scores_from_llm)}).").strip("; ")
                            options_data_input = prepared_inputs.get("options_data", [])
                            # Populate with error entries based on original options data if possible
                            llm_eval_results_for_options = [{"option_id": opt.get("option_id", "unknown_option"), "llm_feedback_text": branch_error, "error": branch_error} for opt in options_data_input]

                else: # parsed_json is not a dict
                    err_msg_json = f"LLM returned valid JSON, but not a dictionary: {type(parsed_json)}."
                    print(f"Eval Subgraph Warning ({node_name}): {err_msg_json}")
                    branch_error = (branch_error + f"; LLM JSON not dictionary: {type(parsed_json)}").strip("; ")
                    if selected_agent == "EvaAgent":
                        current_task["evaluation"]["assessment"] = "Fail"; current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + f"\n[{node_name}]: {err_msg_json}").strip()
                        _update_eval_status_at_end(current_task, node_name)
                        tasks[current_idx] = current_task
                        return {"tasks": tasks} # EvaAgent path
                    else: # Special/Final branch returns payload
                        options_data = prepared_inputs.get("options_data", [])
                        llm_eval_results_for_options = [{"option_id": opt.get("option_id","unknown"), "error": err_msg_json, "llm_feedback_text": err_msg_json} for opt in options_data]
                        branch_failed = True # Consider this a branch failure
            
            except json.JSONDecodeError as json_e:
                err_msg_json = f"Failed to parse LLM JSON response for {selected_agent}: {json_e}. Content hint: {content[:200]}..."
                print(f"Eval Subgraph Error ({node_name}): {err_msg_json}")
                branch_error = (branch_error + f"; LLM JSON parse error: {json_e}").strip("; ")
                if selected_agent == "EvaAgent":
                    current_task["evaluation"]["assessment"] = "Fail"; current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + f"\n[{node_name}]: {err_msg_json}").strip()
                    _update_eval_status_at_end(current_task, node_name)
                    tasks[current_idx] = current_task
                    return {"tasks": tasks} # EvaAgent path
                else: # Special/Final branch returns payload
                    options_data = prepared_inputs.get("options_data", [])
                    llm_eval_results_for_options = [{"option_id": opt.get("option_id","unknown"), "error": err_msg_json, "llm_feedback_text": err_msg_json} for opt in options_data]
                    branch_failed = True # Consider this a branch failure

    except KeyError as ke:
        err_msg_key = f"Formatting error (KeyError: {ke}). Check unified 'evaluation' prompt and inputs for {selected_agent}."
        print(f"Eval Subgraph Error ({node_name}): {err_msg_key}")
        # _set_task_failed(current_task, err_msg_key, node_name) # Do not fail task immediately for parallel branch
        branch_error = (branch_error + f"; LLM Prompt Formatting Error: {ke}").strip("; ")
        branch_failed = True # This is a branch failure
        # For Special/Final, still try to populate llm_eval_results_for_options with error
        if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
            options_data = prepared_inputs.get("options_data", [])
            llm_eval_results_for_options = [{"option_id": opt.get("option_id","unknown"), "error": err_msg_key, "llm_feedback_text": err_msg_key} for opt in options_data]
        
        if selected_agent == "EvaAgent": # EvaAgent path fails immediately
            current_task["evaluation"]["assessment"] = "Fail"; current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + f"\n[{node_name}]: {err_msg_key}").strip()
            _update_eval_status_at_end(current_task, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks} # EvaAgent path
        # Special/Final branch continues to return payload below

    except Exception as e:
        err_msg_llm_call = f"LLM evaluation call error for {selected_agent}: {e}"
        print(f"Eval Subgraph Error ({node_name}): {err_msg_llm_call}")
        traceback.print_exc()
        # _set_task_failed(current_task, err_msg_llm_call, node_name) # Do not fail task immediately for parallel branch
        branch_error = (branch_error + f"; {err_msg_llm_call}").strip("; ")
        branch_failed = True # This is a branch failure
        if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
            options_data = prepared_inputs.get("options_data", [])
            llm_eval_results_for_options = [{"option_id": opt.get("option_id","unknown"), "error": str(e), "llm_feedback_text": str(e)} for opt in options_data]

        if selected_agent == "EvaAgent": # EvaAgent path fails immediately
            current_task["evaluation"]["assessment"] = "Fail"; current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + f"\n[{node_name}]: {err_msg_llm_call}").strip()
            _update_eval_status_at_end(current_task, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks} # EvaAgent path
        # Special/Final branch continues to return payload below


    # --- Return Payload for Special/Final Agents ---
    # EvaAgent return is handled within its specific logic branches above.
    # For Special/Final, package results into the branch payload.
    if selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        llm_branch_payload["llm_detailed_assessment"] = llm_eval_results_for_options
        llm_branch_payload["feedback_log_from_branch"] = f"[{node_name} Status]: Completed LLM evaluation part.\nOverall Feedback: {llm_overall_feedback[:100]}..."
        if branch_error:
            llm_branch_payload["subgraph_error_from_branch"] = branch_error
        llm_branch_payload["task_failed_in_branch"] = branch_failed
        llm_branch_payload["feedback_llm_overall"] = llm_overall_feedback
        llm_branch_payload["assessment"] = llm_assessment_summary # Store summary like "Score (7/10)"
        llm_branch_payload["selected_option_identifier"] = llm_selected_option_identifier

        print(f"  - {selected_agent}: LLM evaluation part completed. Returning payload.")
        return {"llm_temp_output": llm_branch_payload}

    # Fallback return (should not be reached if logic is correct)
    print(f"  - Warning: evaluate_with_llm_node reached end without returning. Agent: {selected_agent}")
    # For safety, return current tasks list if EvaAgent, or empty payload if Special/Final (indicating error)
    if selected_agent == "EvaAgent":
        return {"tasks": tasks}
    else:
        llm_branch_payload["subgraph_error_from_branch"] = (branch_error + "; Logic fell through").strip("; ")
        llm_branch_payload["task_failed_in_branch"] = True
        llm_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: Node logic fell through for {selected_agent}."
        return {"llm_temp_output": llm_branch_payload}


def evaluate_with_image_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs image-based evaluation for each option (Special/Final) or a single target (EvaAgent).
    Stores results in 'image_detailed_assessment'.
    """
    node_name = "Image Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): 
         return {"image_temp_output": {"error": f"Invalid task index {current_idx}", "task_failed_in_branch": True, "subgraph_error_from_branch": f"Invalid task index {current_idx}"}}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    if current_task.get("status") == "failed" and selected_agent == "EvaAgent": 
        print(f"  - Skipping node {node_name} for EvaAgent, task already failed.")
        return {"tasks": tasks}

    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {} 
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or "" # Keep original error
    
    branch_error = "" # Error specific to this branch
    branch_failed = False # Flag to indicate critical failure in this branch

    prepared_inputs = current_task.get("task_inputs", {})
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")
    
    runtime_config_from_graph = config.get("configurable", {}) 
    llm_output_language = runtime_config_from_graph.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)
    
    # --- MODIFICATION: Get prompt template using new key name from runtime config first ---
    img_tool_prompt_template_config_key = "ea_evaluate_option_with_image_tool_prompt"
    img_tool_prompt_template_str = runtime_config_from_graph.get(img_tool_prompt_template_config_key) or \
                                 config_manager.get_prompt_template("eva_agent", "evaluate_option_with_image_tool")
    # --- END MODIFICATION ---

    image_branch_payload = {
        "image_detailed_assessment": [],
        "feedback_log_from_branch": "",
        "subgraph_error_from_branch": "",
        "task_failed_in_branch": False,
        "image_tool_feedback_summary": "" # Summary feedback from this branch
    }


    if not img_tool_prompt_template_str:
        err_msg = f"Missing 'evaluate_option_with_image_tool' prompt template (config key: '{img_tool_prompt_template_config_key}')."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        branch_error = err_msg
        branch_failed = True
        _append_feedback(current_task, err_msg, node_name) # Append to feedback
        if selected_agent == "EvaAgent":
            _set_task_failed(current_task, err_msg, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks} # EvaAgent path
        else: # Special/Final branch returns payload
            image_branch_payload["subgraph_error_from_branch"] = branch_error
            image_branch_payload["task_failed_in_branch"] = branch_failed
            image_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: {err_msg}"
            return {"image_temp_output": image_branch_payload}


    options_for_image_eval = []
    if selected_agent == "EvaAgent":
        print(f"  - Adapting inputs for standard EvaAgent (Image Evaluation)")
        target_image_paths = prepared_inputs.get("evaluation_target_image_paths", [])
        if target_image_paths: 
            options_for_image_eval = [{
                "option_id": "evaluated_task_image", 
                "description": prepared_inputs.get("evaluation_target_description", "N/A"),
                "textual_summary_from_outputs": prepared_inputs.get("evaluation_target_outputs_json", "{}"), 
                "image_paths": [{"path": p, "filename": os.path.basename(p)} for p in target_image_paths],
                "architecture_type": "General" 
            }]
        else:
            print(f"  - EvaAgent: No target images found in prepared_inputs. Skipping image evaluation logic for EvaAgent.")
            # EvaAgent path for image eval effectively does nothing if no images. Status will be based on this.
            # If it was supposed to be image eval, this means it "failed" to find images.
            current_task["evaluation"]["assessment"] = "Fail" 
            current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + "\nImage Eval: No target images provided for EvaAgent.").strip()
            _update_eval_status_at_end(current_task, f"{node_name} (EvaAgent - No Images)")
            tasks[current_idx] = current_task
            return {"tasks": tasks}

    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # --- MODIFICATION: Read options_data from task_inputs ---
        options_data_input = prepared_inputs.get("options_data", []) 
        if not isinstance(options_data_input, list): # Safeguard
            print(f"  - Warning ({node_name}): options_data is not a list for {selected_agent}. Skipping image evaluation branch.")
            branch_error = "Options data is not a list."
            branch_failed = True
            image_branch_payload["subgraph_error_from_branch"] = branch_error
            image_branch_payload["task_failed_in_branch"] = branch_failed
            image_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: {branch_error}"
            return {"image_temp_output": image_branch_payload}

        # Filter options to only those that have image paths for Special/Final
        options_for_image_eval = [
            opt for opt in options_data_input
            # --- MODIFICATION: Ensure image_paths is a non-empty list ---
            if isinstance(opt, dict) and isinstance(opt.get("image_paths"), list) and opt.get("image_paths")
            # --- END MODIFICATION ---
        ]
        if not options_for_image_eval:
            print(f"  - {selected_agent}: No options with image_paths found in options_data. Image evaluation part will be skipped for all options.")
            # Ensure empty list is returned in payload
            image_branch_payload["image_detailed_assessment"] = []
            image_branch_payload["feedback_log_from_branch"] = f"[{node_name} Status]: No options with images found."
            return {"image_temp_output": image_branch_payload}

    print(f"  - Performing Image Evaluation for {len(options_for_image_eval)} options/targets (Agent: {selected_agent})")
    
    all_options_img_tool_evaluations = [] 
    image_tool_feedback_summary_parts = [] 

    for option_idx, option_data in enumerate(options_for_image_eval):
        # --- MODIFICATION: Safely get option_id and image_paths ---
        option_id = option_data.get("option_id", f"img_option_{option_idx+1}")
        option_image_files = option_data.get("image_paths", []) # This should be the list of dicts with 'path'
        # --- END MODIFICATION ---

        print(f"\n  --- Evaluating Option {option_idx+1}/{len(options_for_image_eval)} (ID: {option_id}) with Image Tool ---")

        # --- MODIFICATION: Extract valid image paths ---
        image_paths_for_option = [img_info["path"] for img_info in option_image_files if isinstance(img_info, dict) and img_info.get("path")]
        # --- END MODIFICATION ---
        
        if not image_paths_for_option: 
            print(f"    - Skipping option {option_id}, no valid image paths.")
            feedback_msg = "Skipped by image tool (no valid paths)."
            image_tool_feedback_summary_parts.append(f"Option {option_id}: {feedback_msg}")
            all_options_img_tool_evaluations.append({
                "option_id": option_id, 
                "llm_feedback_text": feedback_msg,
                "error": feedback_msg # Mark as error for downstream merging
            })
            branch_error = (branch_error + f"; Opt {option_id} Img skipped: No valid paths").strip("; ")
            continue
            
        architecture_type_info = f"The architectural style/type is: {option_data.get('architecture_type', 'General')}. " if option_data.get('architecture_type') else ""

        prompt_inputs_for_img_tool = {
            "option_id": option_id,
            "option_description": option_data.get("description", "N/A"),
            "textual_summary": option_data.get("textual_summary_from_outputs", "N/A"),
            "architecture_type_info": architecture_type_info,
            "initial_estimated_cost": option_data.get("initial_estimated_cost", "N/A"), 
            "initial_green_building_percentage": option_data.get("initial_green_building_percentage", "N/A"), 
            "specific_criteria": specific_criteria,
            "image_paths_str": ', '.join(image_paths_for_option), # Pass as comma-separated string
            "llm_output_language": llm_output_language
        }
        current_img_tool_prompt = ""
        try:
            current_img_tool_prompt = img_tool_prompt_template_str.format(**prompt_inputs_for_img_tool)
        except KeyError as ke:
            err_msg_prompt = f"Formatting error for image tool prompt (KeyError: {ke})."
            print(f"    Eval Subgraph Error ({node_name}): {err_msg_prompt}")
            all_options_img_tool_evaluations.append({ "option_id": option_id, "llm_feedback_text": err_msg_prompt, "error": str(ke)})
            branch_error = (branch_error + f"; Opt {option_id} ImgPromptFormatErr: {ke}").strip("; ")
            image_tool_feedback_summary_parts.append(f"Option {option_id}: Image tool prompt formatting error - {ke}")
            continue

        single_option_img_tool_result = { 
            "option_id": option_id, "llm_feedback_text": "Image tool did not run or failed for this option."
        }

        try:
            print(f"    - Calling Image Recognition tool for option {option_id} (Images: {len(image_paths_for_option)})...")
            tool_input_dict = {"image_paths": image_paths_for_option, "prompt": current_img_tool_prompt} # Tool expects list of paths
            raw_llm_json_str = img_recognition.run(tool_input=tool_input_dict)

            if isinstance(raw_llm_json_str, str) and raw_llm_json_str.strip():
                content_to_parse = raw_llm_json_str.strip()
                if content_to_parse.startswith("```json"): content_to_parse = content_to_parse[7:-3].strip()
                elif content_to_parse.startswith("```"): content_to_parse = content_to_parse[3:-3].strip()
                
                try:
                    parsed_json_output = json.loads(content_to_parse)
                    # --- MODIFICATION: Add the new required keys ---
                    required_keys = ["option_id", "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text"]
                    # --- END MODIFICATION ---
                    
                    # Check if parsed_json_output is a dict and has all required keys
                    if isinstance(parsed_json_output, dict) and all(key in parsed_json_output for key in required_keys):
                        single_option_img_tool_result = parsed_json_output
                        # Ensure option_id is correct even if tool LLM gets it wrong
                        single_option_img_tool_result["option_id"] = option_id 
                        print(f"    - Successfully parsed Image tool evaluation for option {option_id}.")
                        image_tool_feedback_summary_parts.append(f"Option {option_id} (Image Tool): {single_option_img_tool_result.get('llm_feedback_text','N/A')[:100]}...")
                    else: # Missing keys or not a dict
                        missing_keys = [key for key in required_keys if key not in parsed_json_output] if isinstance(parsed_json_output, dict) else required_keys
                        err_msg_parse = f"Image tool (LLM) JSON output missing required keys: {missing_keys}. Output hint: {content_to_parse[:100]}"
                        print(f"    Eval Subgraph Error ({node_name}): {err_msg_parse}")
                        single_option_img_tool_result.update({"llm_feedback_text": err_msg_parse, "error": f"Missing keys: {missing_keys}"})
                        branch_error = (branch_error + f"; Opt {option_id} ImgTool JSON missing keys").strip("; ")
                        image_tool_feedback_summary_parts.append(f"Option {option_id}: Image tool JSON missing keys - {missing_keys}")

                except json.JSONDecodeError as json_e: # JSON parse error
                    err_msg_parse = f"Failed to parse JSON from Image tool (LLM) for option {option_id}: {json_e}. Raw hint: {content_to_parse[:100]}"
                    print(f"    Eval Subgraph Error ({node_name}): {err_msg_parse}")
                    single_option_img_tool_result.update({"llm_feedback_text": err_msg_parse, "error": str(json_e)})
                    branch_error = (branch_error + f"; Opt {option_id} ImgTool JSON parse error").strip("; ")
                    image_tool_feedback_summary_parts.append(f"Option {option_id}: Image tool JSON parse error.")
            else: # Empty/invalid response from tool
                err_msg_tool = f"Image tool returned empty or invalid response for option {option_id}."
                print(f"    Eval Subgraph Error ({node_name}): {err_msg_tool}")
                single_option_img_tool_result.update({"llm_feedback_text": err_msg_tool, "error": "Empty/invalid tool response"})
                branch_error = (branch_error + f"; Opt {option_id} ImgTool empty/invalid response").strip("; ")
                image_tool_feedback_summary_parts.append(f"Option {option_id}: Image tool returned empty response.")

        except Exception as e: # Error calling tool
            err_msg_call = f"Error calling Image tool for option {option_id}: {e}"
            print(f"    Eval Subgraph Error ({node_name}): {err_msg_call}")
            traceback.print_exc()
            single_option_img_tool_result.update({"llm_feedback_text": err_msg_call, "error": str(e)})
            branch_error = (branch_error + f"; Opt {option_id} ImgTool call error: {type(e).__name__}").strip("; ")
            image_tool_feedback_summary_parts.append(f"Option {option_id}: Image tool call error - {type(e).__name__}")
        
        all_options_img_tool_evaluations.append(single_option_img_tool_result)
    # --- End of loop for options ---

    image_branch_payload["image_detailed_assessment"] = all_options_img_tool_evaluations
    image_branch_payload["image_tool_feedback_summary"] = "\n".join(image_tool_feedback_summary_parts) # Store summary for merging
    image_branch_payload["feedback_log_from_branch"] = f"[{node_name} Status]: Processed {len(options_for_image_eval)} options/targets.\nSummary: {image_branch_payload['image_tool_feedback_summary'][:200]}..."

    if branch_error:
         image_branch_payload["subgraph_error_from_branch"] = branch_error
         # Decide if branch failure should be set based on total errors vs total options
         if len([res for res in all_options_img_tool_evaluations if res.get("error")]) > len(all_options_img_tool_evaluations) / 2: # Heuristic: more than half failed
             branch_failed = True
             print(f"  - {node_name}: More than half of options failed image evaluation. Marking branch as failed.")
         else:
              print(f"  - {node_name}: Some options failed, but not critically. Branch is not marked failed.")

    image_branch_payload["task_failed_in_branch"] = branch_failed

    if selected_agent == "EvaAgent": # EvaAgent path sets final status and returns tasks
        if all_options_img_tool_evaluations: 
            first_option_scores = all_options_img_tool_evaluations[0]
            if "error" in first_option_scores: 
                 current_task["evaluation"]["assessment"] = "Fail"
                 current_task["evaluation"]["feedback"] = (current_task.get("evaluation", {}).get("feedback","") + f"\nImage Eval Error: {first_option_scores['error']}").strip()
            else: # EvaAgent successful image evaluation (simplified pass/fail for single tool)
                # Assess based on required keys being present and possibly scores >= 5 (basic check)
                required_scores_check = ["user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm"]
                scores_ok = all(first_option_scores.get(k, 0) >= 5 for k in required_scores_check) and all(k in first_option_scores for k in required_scores_check)
                
                # If previous assessment (e.g., text eval) was Pass, and this one is also Pass criteria-wise
                prior_assessment_is_pass = current_task.get("evaluation", {}).get("assessment") == "Pass"
                
                if scores_ok and (selected_agent != "EvaAgent" or prior_assessment_is_pass or not current_task.get("evaluation", {}).get("assessment")):
                     current_task["evaluation"]["assessment"] = "Pass"
                else:
                     current_task["evaluation"]["assessment"] = "Fail" # Fail if scores too low or if prior was fail

                current_task["evaluation"]["feedback"] = (current_task.get("evaluation", {}).get("feedback","") + f"\nImage Eval Feedback: {first_option_scores.get('llm_feedback_text', 'N/A')}").strip()

        else: # No images were processed for EvaAgent (already handled, but defensive)
            current_task["evaluation"]["assessment"] = "Fail" 
            current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + "\nImage Eval: No image data processed for EvaAgent.").strip()
            
        # Update EvaAgent task status based on assessment
        _update_eval_status_at_end(current_task, f"{node_name} (EvaAgent Path)")
        if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error # Carry over any prior error
        tasks[current_idx] = current_task
        return {"tasks": tasks}

    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # Special/Final branch returns payload
        print(f"  - {selected_agent}: Image evaluation part completed. Returning payload.")
        return {"image_temp_output": image_branch_payload}

    # Fallback return
    print(f"  - Warning: evaluate_with_image_node reached end without returning. Agent: {selected_agent}")
    if selected_agent == "EvaAgent":
         return {"tasks": tasks}
    else:
         image_branch_payload["subgraph_error_from_branch"] = (branch_error + "; Logic fell through").strip("; ")
         image_branch_payload["task_failed_in_branch"] = True
         image_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: Node logic fell through for {selected_agent}."
         return {"image_temp_output": image_branch_payload}


def evaluate_with_video_node(state: WorkflowState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs video-based evaluation for each option (Special/Final) or a single target (EvaAgent).
    Stores results in 'video_detailed_assessment'.
    """
    node_name = "Video Evaluation"
    print(f"--- Running Node: {node_name} ---")
    tasks = [t.copy() for t in state["tasks"]]
    current_idx = state["current_task_index"]
    if not (0 <= current_idx < len(tasks)): 
        return {"video_temp_output": {"error": f"Invalid task index {current_idx}", "task_failed_in_branch": True, "subgraph_error_from_branch": f"Invalid task index {current_idx}"}}

    current_task = tasks[current_idx]
    selected_agent = current_task.get('selected_agent')

    if current_task.get("status") == "failed" and selected_agent == "EvaAgent":
        print(f"  - Skipping node {node_name} for EvaAgent, task already failed.")
        return {"tasks": tasks}

    if "evaluation" not in current_task: current_task["evaluation"] = {}
    if "task_inputs" not in current_task: current_task["task_inputs"] = {}
    subgraph_error = current_task.get("evaluation", {}).get("subgraph_error", "") or "" # Keep original error

    branch_error = "" # Error specific to this branch
    branch_failed = False # Flag to indicate critical failure in this branch

    prepared_inputs = current_task.get("task_inputs", {})
    specific_criteria = current_task.get("evaluation", {}).get("specific_criteria", "Default criteria apply / Rubric not generated.")
    
    runtime_config_from_graph = config.get("configurable", {})
    llm_output_language = runtime_config_from_graph.get("global_llm_output_language", LLM_OUTPUT_LANGUAGE_DEFAULT)

    # --- MODIFICATION: Get prompt template using new key name from runtime config first ---
    vid_tool_prompt_template_config_key = "ea_evaluate_option_with_video_tool_prompt"
    vid_tool_prompt_template_str = runtime_config_from_graph.get(vid_tool_prompt_template_config_key) or \
                                 config_manager.get_prompt_template("eva_agent", "evaluate_option_with_video_tool")
    # --- END MODIFICATION ---

    video_branch_payload = {
        "video_detailed_assessment": [],
        "feedback_log_from_branch": "",
        "subgraph_error_from_branch": "",
        "task_failed_in_branch": False,
        "video_tool_feedback_summary": "" # Summary feedback from this branch
    }


    if not vid_tool_prompt_template_str:
        err_msg = f"Missing 'evaluate_option_with_video_tool' prompt template (config key: '{vid_tool_prompt_template_config_key}')."
        print(f"Eval Subgraph Error ({node_name}): {err_msg}")
        branch_error = err_msg
        branch_failed = True
        _append_feedback(current_task, err_msg, node_name)
        if selected_agent == "EvaAgent":
            _set_task_failed(current_task, err_msg, node_name)
            tasks[current_idx] = current_task
            return {"tasks": tasks}
        else: # Special/Final branch returns payload
            video_branch_payload["subgraph_error_from_branch"] = branch_error
            video_branch_payload["task_failed_in_branch"] = branch_failed
            video_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: {err_msg}"
            return {"video_temp_output": video_branch_payload}


    options_for_video_eval = []
    if selected_agent == "EvaAgent":
        print(f"  - Adapting inputs for standard EvaAgent (Video Evaluation)")
        target_video_paths = prepared_inputs.get("evaluation_target_video_paths", [])
        if target_video_paths:
            options_for_video_eval = [{
                "option_id": "evaluated_task_video", 
                "description": prepared_inputs.get("evaluation_target_description", "N/A"),
                "textual_summary_from_outputs": prepared_inputs.get("evaluation_target_outputs_json", "{}"),
                "video_paths": [{"path": p, "filename": os.path.basename(p)} for p in target_video_paths],
                "architecture_type": "General"
            }]
        else:
            print(f"  - EvaAgent: No target videos found in prepared_inputs. Skipping video evaluation logic.")
            current_task["evaluation"]["assessment"] = "Fail"
            current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + "\nVideo Eval: No target videos provided for EvaAgent.").strip()
            _update_eval_status_at_end(current_task, f"{node_name} (EvaAgent - No Videos)")
            tasks[current_idx] = current_task
            return {"tasks": tasks}
            
    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # --- MODIFICATION: Read options_data from task_inputs ---
        options_data_input = prepared_inputs.get("options_data", [])
        if not isinstance(options_data_input, list): # Safeguard
            print(f"  - Warning ({node_name}): options_data is not a list for {selected_agent}. Skipping video evaluation branch.")
            branch_error = "Options data is not a list."
            branch_failed = True
            video_branch_payload["subgraph_error_from_branch"] = branch_error
            video_branch_payload["task_failed_in_branch"] = branch_failed
            video_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: {branch_error}"
            return {"video_temp_output": video_branch_payload}

        # Filter options to only those that have video paths for Special/Final
        options_for_video_eval = [
            opt for opt in options_data_input
            # --- MODIFICATION: Ensure video_paths is a non-empty list ---
            if isinstance(opt, dict) and isinstance(opt.get("video_paths"), list) and opt.get("video_paths")
            # --- END MODIFICATION ---
        ]
        if not options_for_video_eval:
            print(f"  - {selected_agent}: No options with video_paths found in options_data. Video evaluation part will be skipped.")
            # Ensure empty list is returned in payload
            video_branch_payload["video_detailed_assessment"] = []
            video_branch_payload["feedback_log_from_branch"] = f"[{node_name} Status]: No options with videos found."
            return {"video_temp_output": video_branch_payload}

    print(f"  - Performing Video Evaluation for {len(options_for_video_eval)} options/targets (Agent: {selected_agent})")
    all_options_vid_tool_evaluations = []
    video_tool_feedback_summary_parts = []

    for option_idx, option_data in enumerate(options_for_video_eval):
        # --- MODIFICATION: Safely get option_id and video_paths ---
        option_id = option_data.get("option_id", f"vid_option_{option_idx+1}")
        option_video_files = option_data.get("video_paths", []) # This should be the list of dicts with 'path'
        # --- END MODIFICATION ---

        print(f"\n  --- Evaluating Option {option_idx+1}/{len(options_for_video_eval)} (ID: {option_id}) with Video Tool ---")

        # --- MODIFICATION: Extract valid video paths ---
        video_paths_for_option = [vid_info["path"] for vid_info in option_video_files if isinstance(vid_info, dict) and vid_info.get("path")]
        # --- END MODIFICATION ---

        if not video_paths_for_option:
            print(f"    - Skipping option {option_id}, no valid video paths.")
            feedback_msg = "Skipped by video tool (no valid paths)."
            video_tool_feedback_summary_parts.append(f"Option {option_id}: {feedback_msg}")
            all_options_vid_tool_evaluations.append({
                "option_id": option_id, 
                "llm_feedback_text": feedback_msg,
                "error": feedback_msg # Mark as error for downstream merging
            })
            branch_error = (branch_error + f"; Opt {option_id} Vid skipped: No valid paths").strip("; ")
            continue
        
        video_path_to_eval = video_paths_for_option[0] # Process one video per option for now
        architecture_type_info = f"The architectural style/type is: {option_data.get('architecture_type', 'General')}. " if option_data.get('architecture_type') else ""

        prompt_inputs_for_vid_tool = {
            "option_id": option_id,
            "option_description": option_data.get("description", "N/A"),
            "textual_summary": option_data.get("textual_summary_from_outputs", "N/A"),
            "architecture_type_info": architecture_type_info,
            "initial_estimated_cost": option_data.get("initial_estimated_cost", "N/A"),
            "initial_green_building_percentage": option_data.get("initial_green_building_percentage", "N/A"),
            "specific_criteria": specific_criteria,
            "video_path_to_eval": video_path_to_eval,
            "llm_output_language": llm_output_language
        }
        current_vid_tool_prompt = ""
        try:
            current_vid_tool_prompt = vid_tool_prompt_template_str.format(**prompt_inputs_for_vid_tool)
        except KeyError as ke:
            err_msg_prompt = f"Formatting error for video tool prompt (KeyError: {ke})."
            print(f"    Eval Subgraph Error ({node_name}): {err_msg_prompt}")
            all_options_vid_tool_evaluations.append({ "option_id": option_id, "llm_feedback_text": err_msg_prompt, "error": str(ke)})
            branch_error = (branch_error + f"; Opt {option_id} VidPromptFormatErr: {ke}").strip("; ")
            video_tool_feedback_summary_parts.append(f"Option {option_id}: Video tool prompt formatting error - {ke}")
            continue
            
        single_option_vid_tool_result = {
            "option_id": option_id, "llm_feedback_text": "Video tool did not run or failed for this option."
        }

        try:
            print(f"    - Calling Video Recognition tool for option {option_id}, video: {video_path_to_eval}...")
            tool_input_dict = {"video_path": video_path_to_eval, "prompt": current_vid_tool_prompt}
            raw_llm_json_str = video_recognition.run(tool_input=tool_input_dict) 

            if isinstance(raw_llm_json_str, str) and raw_llm_json_str.strip():
                content_to_parse = raw_llm_json_str.strip()
                if content_to_parse.startswith("```json"): content_to_parse = content_to_parse[7:-3].strip()
                elif content_to_parse.startswith("```"): content_to_parse = content_to_parse[3:-3].strip()

                try:
                    parsed_json_output = json.loads(content_to_parse)
                    # --- MODIFICATION: Add the new required keys ---
                    required_keys = ["option_id", "user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm", "estimated_cost", "green_building_potential_percentage", "llm_feedback_text"]
                    # --- END MODIFICATION ---

                    if isinstance(parsed_json_output, dict) and all(key in parsed_json_output for key in required_keys):
                        single_option_vid_tool_result = parsed_json_output
                        single_option_vid_tool_result["option_id"] = option_id 
                        print(f"    - Successfully parsed Video tool evaluation for option {option_id}.")
                        video_tool_feedback_summary_parts.append(f"Option {option_id} (Video Tool): {single_option_vid_tool_result.get('llm_feedback_text','N/A')[:100]}...")
                    else: # Missing keys or not a dict
                        missing_keys = [key for key in required_keys if key not in parsed_json_output] if isinstance(parsed_json_output, dict) else required_keys
                        err_msg_parse = f"Video tool (LLM) JSON output missing required keys: {missing_keys}. Output hint: {content_to_parse[:100]}"
                        print(f"    Eval Subgraph Error ({node_name}): {err_msg_parse}")
                        single_option_vid_tool_result.update({"llm_feedback_text": err_msg_parse, "error": f"Missing keys: {missing_keys}"})
                        branch_error = (branch_error + f"; Opt {option_id} VidTool JSON missing keys").strip("; ")
                        video_tool_feedback_summary_parts.append(f"Option {option_id}: Video tool JSON missing keys - {missing_keys}")
                except json.JSONDecodeError as json_e: # JSON Parse error
                    err_msg_parse = f"Failed to parse JSON from Video tool (LLM) for option {option_id}: {json_e}. Raw hint: {content_to_parse[:100]}"
                    print(f"    Eval Subgraph Error ({node_name}): {err_msg_parse}")
                    single_option_vid_tool_result.update({"llm_feedback_text": err_msg_parse, "error": str(json_e)})
                    branch_error = (branch_error + f"; Opt {option_id} VidTool JSON parse error").strip("; ")
                    video_tool_feedback_summary_parts.append(f"Option {option_id}: Video tool JSON parse error.")
            else: # Empty/invalid response from tool
                err_msg_tool = f"Video tool returned empty or invalid response for option {option_id}."
                print(f"    Eval Subgraph Error ({node_name}): {err_msg_tool}")
                single_option_vid_tool_result.update({"llm_feedback_text": err_msg_tool, "error": "Empty/invalid tool response"})
                branch_error = (branch_error + f"; Opt {option_id} VidTool empty/invalid response").strip("; ")
                video_tool_feedback_summary_parts.append(f"Option {option_id}: Video tool returned empty response.")
        
        except Exception as e: # Error calling tool
            err_msg_call = f"Error calling Video tool for option {option_id}: {e}"
            print(f"    Eval Subgraph Error ({node_name}): {err_msg_call}")
            traceback.print_exc()
            single_option_vid_tool_result.update({"llm_feedback_text": err_msg_call, "error": str(e)})
            branch_error = (branch_error + f"; Opt {option_id} VidTool call error: {type(e).__name__}").strip("; ")
            video_tool_feedback_summary_parts.append(f"Option {option_id}: Video tool call error - {type(e).__name__}")
        
        all_options_vid_tool_evaluations.append(single_option_vid_tool_result)
    # --- End of loop for video options ---

    video_branch_payload["video_detailed_assessment"] = all_options_vid_tool_evaluations
    video_branch_payload["video_tool_feedback_summary"] = "\n".join(video_tool_feedback_summary_parts)
    video_branch_payload["feedback_log_from_branch"] = f"[{node_name} Status]: Processed {len(options_for_video_eval)} options/targets.\nSummary: {video_branch_payload['video_tool_feedback_summary'][:200]}..."


    if branch_error:
         video_branch_payload["subgraph_error_from_branch"] = branch_error
         # Decide if branch failure should be set
         if len([res for res in all_options_vid_tool_evaluations if res.get("error")]) > len(all_options_vid_tool_evaluations) / 2: # Heuristic: more than half failed
             branch_failed = True
             print(f"  - {node_name}: More than half of options failed video evaluation. Marking branch as failed.")
         else:
              print(f"  - {node_name}: Some options failed, but not critically. Branch is not marked failed.")

    video_branch_payload["task_failed_in_branch"] = branch_failed


    if selected_agent == "EvaAgent": # EvaAgent path sets final status and returns tasks
         if all_options_vid_tool_evaluations: 
             first_option_scores = all_options_vid_tool_evaluations[0]
             if "error" in first_option_scores: 
                  current_task["evaluation"]["assessment"] = "Fail" 
                  current_task["evaluation"]["feedback"] = (current_task.get("evaluation", {}).get("feedback","") + f"\nVideo Eval Error: {first_option_scores['error']}").strip()
             else: # EvaAgent successful video evaluation
                 # Assess based on required keys being present and possibly scores >= 5 (basic check)
                 required_scores_check = ["user_goal_responsiveness_score_llm", "aesthetics_context_score_llm", "functionality_flexibility_score_llm", "durability_maintainability_score_llm"]
                 scores_ok = all(first_option_scores.get(k, 0) >= 5 for k in required_scores_check) and all(k in first_option_scores for k in required_scores_check)

                 # If previous assessment (e.g., image or text eval) was Pass, and this one is also Pass criteria-wise
                 prior_assessment_is_pass = current_task.get("evaluation", {}).get("assessment") == "Pass"
                 
                 if scores_ok and (selected_agent != "EvaAgent" or prior_assessment_is_pass or not current_task.get("evaluation", {}).get("assessment")):
                      current_task["evaluation"]["assessment"] = "Pass"
                 else: 
                      current_task["evaluation"]["assessment"] = "Fail"

                 current_task["evaluation"]["feedback"] = (current_task.get("evaluation", {}).get("feedback","") + 
                                                         f"\nVideo Eval Feedback: {first_option_scores.get('llm_feedback_text', 'N/A')}").strip()
         else: # No videos processed for EvaAgent
             # If an image eval happened before and set to Pass, video absence doesn't make it fail.
             # If no prior eval, or prior was fail, then it's fail.
             if current_task.get("evaluation", {}).get("assessment") != "Pass":
                  current_task["evaluation"]["assessment"] = "Fail"
             current_task["evaluation"]["feedback"] = (current_task.get("evaluation",{}).get("feedback","") + "\nVideo Eval: No video data processed for EvaAgent.").strip()

         _update_eval_status_at_end(current_task, f"{node_name} (EvaAgent Path)")
         if subgraph_error: current_task["evaluation"]["subgraph_error"] = subgraph_error # Carry over any prior error
         tasks[current_idx] = current_task
         return {"tasks": tasks}

    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        # Special/Final branch returns payload
        print(f"  - {selected_agent}: Video evaluation part completed. Returning payload.")
        return {"video_temp_output": video_branch_payload}

    # Fallback return
    print(f"  - Warning: evaluate_with_video_node reached end without returning. Agent: {selected_agent}")
    if selected_agent == "EvaAgent":
         return {"tasks": tasks}
    else:
         video_branch_payload["subgraph_error_from_branch"] = (branch_error + "; Logic fell through").strip("; ")
         video_branch_payload["task_failed_in_branch"] = True
         video_branch_payload["feedback_log_from_branch"] = f"[{node_name} Error]: Node logic fell through for {selected_agent}."
         return {"video_temp_output": video_branch_payload}

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

# --- Edges from Evaluation Tools ---
def route_after_single_eval_tool(state: WorkflowState) -> str:
    # This router is for nodes when they are part of an EvaAgent (single tool) path.
    # They should just end the subgraph.
    # For Special/Final, they go to visualization.
    current_idx = state["current_task_index"]
    current_task = state["tasks"][current_idx]
    selected_agent = current_task.get("selected_agent")
    if selected_agent == "EvaAgent":
        return END
    # For Special/Final, all parallel branches go to visualization
    return "generate_evaluation_visualization" 

# --- Routing function after criteria generation ---
def route_evaluation_flow(state: WorkflowState) -> str:
    """
    Routes to appropriate evaluation path based on Agent Type.
    - EvaAgent: Routes to a single tool (LLM, Image, or Video) based on output type.
    - SpecialEvaAgent/FinalEvaAgent: Determines which evaluation tools (text, image, video)
      are needed based on the prepared options_data and returns a specific key
      to trigger a sequence of these tools.
    """
    node_name = "Route Evaluation Flow"
    print(f"--- Running Node: {node_name} ---")
    current_idx = state["current_task_index"]
    tasks = state["tasks"]
    if not (0 <= current_idx < len(tasks)):
        print(f"  - Error ({node_name}): Invalid task index {current_idx}. Routing to END.")
        return END 

    current_task = tasks[current_idx]
    selected_agent = current_task.get("selected_agent")

    if current_task.get("status") == "failed":
        print(f"  - Routing Decision ({node_name}): Task failed in criteria generation or before. Routing to END.")
        return END

    if selected_agent == "EvaAgent":
        print(f"  - {node_name}: Agent is EvaAgent. Routing to single evaluation tool.")
        prepared_inputs = current_task.get("task_inputs", {})
        has_image_output = bool(prepared_inputs.get("evaluation_target_image_paths"))
        has_video_output = bool(prepared_inputs.get("evaluation_target_video_paths"))
        has_text_output = (prepared_inputs.get("evaluation_target_outputs_json") != '{}')

        if has_image_output:
            print(f"    - EvaAgent Routing: Image found -> Route to evaluate_with_image_agent")
            return "evaluate_with_image_agent"
        elif has_video_output:
            print(f"    - EvaAgent Routing: Video found (no image) -> Route to evaluate_with_video_agent")
            return "evaluate_with_video_agent"
        elif has_text_output:
            print(f"    - EvaAgent Routing: Only Text/Summary found -> Route to evaluate_with_text_agent")
            return "evaluate_with_text_agent"
        else:
            print(f"    - EvaAgent Routing: No specific outputs found -> Defaulting to evaluate_with_text_agent")
            return "evaluate_with_text_agent"

    elif selected_agent in ["SpecialEvaAgent", "FinalEvaAgent"]:
        print(f"  - {node_name}: Agent is {selected_agent}. Determining evaluation components.")
        
        prepared_inputs = current_task.get("task_inputs", {}) # Get prepared_inputs again
        options_data = prepared_inputs.get("options_data", []) 
        if not isinstance(options_data, list): # Safeguard
            print(f"    - Warning ({node_name}): options_data is not a list for {selected_agent}. Defaulting to text only.")
            options_data = []

        needs_text_eval = True # Always assume text/LLM evaluation is needed for Special/Final
        needs_image_eval = False
        needs_video_eval = False

        if options_data: # Only check for media if there are options to check
            if any(isinstance(opt, dict) and opt.get("image_paths") for opt in options_data):
                needs_image_eval = True
            if any(isinstance(opt, dict) and opt.get("video_paths") for opt in options_data):
                needs_video_eval = True
        
        print(f"    - Evaluation components: Text={needs_text_eval}, Image={needs_image_eval}, Video={needs_video_eval}")

        if needs_text_eval and needs_image_eval and needs_video_eval:
            routing_key = "eval_text_image_video"
        elif needs_text_eval and needs_image_eval:
            routing_key = "eval_text_image"
        elif needs_text_eval and needs_video_eval:
            routing_key = "eval_text_video"
        elif needs_text_eval: # Only text
            routing_key = "eval_text_only"
        else: # Should not happen if text is always true, but as a fallback
            print(f"    - Warning ({node_name}): No evaluation components determined for {selected_agent}. Defaulting to text only.")
            routing_key = "eval_text_only"
        
        print(f"    - {selected_agent} Routing: Determined routing key: {routing_key}")
        return routing_key
    
    else: 
        print(f"  - Error ({node_name}): Unknown agent type '{selected_agent}'. Routing to END.")
        _set_task_failed(current_task, f"Unknown agent type '{selected_agent}' in routing.", node_name)
        state["tasks"][current_idx] = current_task # Ensure failed task state is propagated
        return END

# =============================================================================
# Build and Compile Evaluation Subgraph
# =============================================================================
evaluation_subgraph_builder = StateGraph(WorkflowState)

evaluation_subgraph_builder.add_node("prepare_evaluation_inputs", prepare_evaluation_inputs_node)
evaluation_subgraph_builder.add_node("gather_criteria_sources", gather_criteria_sources_node)
evaluation_subgraph_builder.add_node("generate_specific_criteria", generate_specific_criteria_node)

# --- NEW: Distributor node ---
evaluation_subgraph_builder.add_node("distribute_evaluations", distribute_evaluations_node)

evaluation_subgraph_builder.add_node("evaluate_with_text_agent", evaluate_with_llm_node)
evaluation_subgraph_builder.add_node("evaluate_with_image_agent", evaluate_with_image_node)
evaluation_subgraph_builder.add_node("evaluate_with_video_agent", evaluate_with_video_node)
evaluation_subgraph_builder.add_node("generate_evaluation_visualization", generate_evaluation_visualization_node)

evaluation_subgraph_builder.set_entry_point("prepare_evaluation_inputs")

evaluation_subgraph_builder.add_conditional_edges(
    "prepare_evaluation_inputs",
    route_after_eval_prep,
    {"gather_criteria_sources": "gather_criteria_sources", "generate_specific_criteria": "generate_specific_criteria", "finished": END}
)
evaluation_subgraph_builder.add_edge("gather_criteria_sources", "generate_specific_criteria")


# --- MODIFIED EDGES for routing after criteria ---
evaluation_subgraph_builder.add_conditional_edges(
    "generate_specific_criteria",
    lambda state: "distribute_evaluations" if state["tasks"][state["current_task_index"]].get("selected_agent") in ["SpecialEvaAgent", "FinalEvaAgent"] else route_evaluation_flow(state), # route_evaluation_flow for EvaAgent direct path
    {
        "distribute_evaluations": "distribute_evaluations", # For Special/Final
        # EvaAgent direct paths (from route_evaluation_flow's original logic)
        "evaluate_with_text_agent": "evaluate_with_text_agent",
        "evaluate_with_image_agent": "evaluate_with_image_agent",
        "evaluate_with_video_agent": "evaluate_with_video_agent",
        END: END
    }
)

# --- NEW: From distributor, route to parallel tools ---
# This conditional edge will use the list of nodes returned by route_to_parallel_eval_tools
# LangGraph handles a list of strings as targets for parallel execution.
evaluation_subgraph_builder.add_conditional_edges(
    "distribute_evaluations",
    route_to_parallel_eval_tools,
    {
        "evaluate_with_text_agent": "evaluate_with_text_agent",
        "evaluate_with_image_agent": "evaluate_with_image_agent",
        "evaluate_with_video_agent": "evaluate_with_video_agent",
    }
)


# --- Edges from individual evaluation tools to the MERGE/VISUALIZATION node ---
# All parallel branches (and EvaAgent paths if they don't END sooner) must eventually lead to visualization or end.

# For EvaAgent, these tools might go to END directly if that's their final step.
# For Special/Final, they MUST go to generate_evaluation_visualization.
# The route_after_single_eval_tool handles this.

evaluation_subgraph_builder.add_conditional_edges("evaluate_with_text_agent", route_after_single_eval_tool, {
    "generate_evaluation_visualization": "generate_evaluation_visualization", END: END
})
evaluation_subgraph_builder.add_conditional_edges("evaluate_with_image_agent", route_after_single_eval_tool, {
    "generate_evaluation_visualization": "generate_evaluation_visualization", END: END
})
evaluation_subgraph_builder.add_conditional_edges("evaluate_with_video_agent", route_after_single_eval_tool, {
    "generate_evaluation_visualization": "generate_evaluation_visualization", END: END
})

evaluation_subgraph_builder.add_edge("generate_evaluation_visualization", END)

evaluation_teams = evaluation_subgraph_builder.compile()
evaluation_teams.name = "EvaluationSubgraph"
print("Evaluation Subgraph refactored and compiled successfully as 'evaluation_teams'.")

# =============================================================================
# End of File
# =============================================================================
