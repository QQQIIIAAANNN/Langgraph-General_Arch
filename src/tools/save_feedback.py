import os
import json



def save_feedback(current_round: int, report_content: str) -> str:
    """
    Save the feedback report into a file named final_feedback_report{current_round}.json.

    Args:
        current_round (int): The current round number for naming the file.
        report_content (str): The content of the feedback report to save. Should be a valid JSON string or JSON-like Python object.

    Returns:
        str: Success message with the saved file path.
    """
    # Create output directory if not exists
    output_dir = "./output/case_study"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file path
    file_name = f"final_feedback_report{current_round}.json"
    file_path = os.path.join(output_dir, file_name)
    
    # Ensure report_content is a valid JSON structure
    try:
        # If report_content is a string, try to parse it as JSON
        if isinstance(report_content, str):
            report_content = json.loads(report_content)
    except json.JSONDecodeError:
        # If parsing fails, wrap the string in a dictionary
        report_content = {"feedback": report_content}
    
    # Save report content as JSON
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report_content, file, ensure_ascii=False, indent=4)
    
    return f"Feedback report saved successfully as {file_path}"
