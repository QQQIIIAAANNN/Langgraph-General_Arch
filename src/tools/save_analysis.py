import os
import json


def save_analysis(current_round: int, analysis_content: str) -> str:
    """
    Save the analysis result into a file named shell_result{current_round}.json.

    Args:
        current_round (int): The current round number for naming the file.
        analysis_content (str): The content of the analysis result to save. Should be a valid JSON string or JSON-like Python object.

    Returns:
        str: Success message with the saved file path.
    """
    # Create output directory if not exists
    output_dir = "./output/case_study"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file path
    file_name = f"shell_result{current_round}.json"
    file_path = os.path.join(output_dir, file_name)
    
    # Ensure analysis_content is a valid JSON structure
    try:
        # If analysis_content is a string, try to parse it as JSON
        if isinstance(analysis_content, str):
            analysis_content = json.loads(analysis_content)
    except json.JSONDecodeError:
        # If parsing fails, wrap the string in a dictionary
        analysis_content = {"analysis": analysis_content}
    
    # Save analysis content as JSON
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(analysis_content, file, ensure_ascii=False, indent=4)
    
    return f"Analysis result saved successfully as {file_path}"
