import pandas as pd
from langchain.tools import tool

# Data is hard-coded so that the agent can call them without passing the dataframe as an argument.
# We cannot use a class because LangChain does not support tools inside classes.
PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


def reset_state():
    """
    Resets the project tasks to the original state.
    """
    global PROJECT_TASKS
    PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


@tool("project_management.get_task_information_by_id", return_direct=False)
def get_task_information_by_id(task_id=None, field=None):
    """
    Retrieves a specific field value from a task using its unique identifier.

    Purpose:
    Fetches individual task attributes by task ID. Use this when you need to get one specific 
    piece of information about a task you've already identified, rather than searching for tasks 
    or retrieving full task details.

    Usage Examples:
    get_task_information_by_id("00000042", "task_name")
    > Returns: {{"task_name": "Refactor authentication module"}}
    
    get_task_information_by_id("00000103", "assigned_to_email")
    > Returns: {{"assigned_to_email": "sarah@example.com"}}
    
    get_task_information_by_id("00000087", "due_date")
    > Returns: {{"due_date": "2023-07-15"}}

    Limitations:
    - Returns only one field at a time; use search_tasks for multiple fields
    - Requires exact task_id match (case-sensitive, 8 digits)
    - Returns error message if task_id doesn't exist
    - Cannot retrieve task_id field itself (redundant)

    Args:
    task_id (str): 8-digit unique task identifier (e.g., "00000042")
    field (str): Field name to retrieve. Valid options: "task_name", "assigned_to_email", 
                 "list_name", "due_date", "board"

    Returns:
    dict: Single-key dictionary with requested field and its value, or error message string
          if task not found or field invalid
    """
    if not task_id:
        return "Task ID not provided."
    if not field:
        return "Field not provided."
    task = PROJECT_TASKS[PROJECT_TASKS["task_id"] == task_id].to_dict(orient="records")
    if task:
        if field in task[0]:
            return {field: task[0][field]}
        else:
            return "Field not found."
    else:
        return "Task not found."


@tool("project_management.search_tasks", return_direct=False)
def search_tasks(task_name=None, assigned_to_email=None, list_name=None, due_date=None, board=None):
    """
    Searches for tasks matching specified criteria across one or more fields.

    Purpose:
    Finds all tasks that match your search criteria. Use this to discover tasks when you don't 
    know the task_id, to filter tasks by assignee, status, deadline, or other attributes, or to 
    get a list of tasks meeting multiple conditions. All searches are case-insensitive and support 
    partial matches.

    Usage Examples:
    search_tasks(task_name="API")
    > Returns all tasks with "API" in the name (e.g., "API integration", "Update API docs")
    
    search_tasks(assigned_to_email="sarah@example.com", list_name="In Progress")
    > Returns all in-progress tasks assigned to Sarah
    
    search_tasks(due_date="2023-06", board="Front end")
    > Returns all front-end tasks due in June 2023
    
    search_tasks(list_name="Completed")
    > Returns all completed tasks across all boards

    Limitations:
    - Returns empty list if no tasks match criteria
    - Requires at least one search parameter
    - Partial text matches may return more results than expected
    - Does not sort results (returned in arbitrary order)
    - Cannot use logical operators (AND/OR) between different field values

    Args:
    task_name (str, optional): Full or partial task name (case-insensitive)
    assigned_to_email (str, optional): Full or partial email address (case-insensitive)
    list_name (str, optional): Full or partial list name (case-insensitive)
    due_date (str, optional): Full or partial date in "YYYY-MM-DD" format
    board (str, optional): Full or partial board name (case-insensitive)

    Returns:
    list[dict]: List of task dictionaries, each containing all fields: "task_id", "task_name",
                "assigned_to_email", "list_name", "due_date", "board". Returns empty list if 
                no matches found, or error message string if no search parameters provided.
    """
    if not any([task_name, assigned_to_email, list_name, due_date, board]):
        return "No search parameters provided."
    tasks = PROJECT_TASKS.copy()
    if task_name:
        tasks = tasks[tasks["task_name"].str.contains(task_name, case=False)]
    if assigned_to_email:
        tasks = tasks[tasks["assigned_to_email"].str.contains(assigned_to_email, case=False)]
    if list_name:
        tasks = tasks[tasks["list_name"].str.contains(list_name, case=False)]
    if due_date:
        tasks = tasks[tasks["due_date"].str.contains(due_date, case=False)]
    if board:
        tasks = tasks[tasks["board"].str.contains(board, case=False)]
    return tasks.to_dict(orient="records")


@tool("project_management.create_task", return_direct=False)
def create_task(task_name=None, assigned_to_email=None, list_name=None, due_date=None, board=None):
    """
    Creates a new task with specified details and automatically generates a unique task ID.

    Purpose:
    Adds a new task to the project management system. Use this when planning new work, 
    assigning responsibilities, or tracking new deliverables. The system automatically assigns 
    a unique 8-digit task_id that can be used to reference this task in other operations.

    Usage Examples:
    create_task(
        "Integrate payment gateway API",
        "alex@example.com",
        "Backlog",
        "2023-08-15",
        "Back end"
    )
    > Returns: "00000156" (newly created task ID)
    
    create_task(
        "Design user profile page mockup",
        "jordan@example.com",
        "In Progress",
        "2023-07-01",
        "Design"
    )
    > Returns: "00000157"

    Limitations:
    - All five parameters are required; missing any will result in error
    - assigned_to_email must match an existing team member email
    - list_name must be exactly one of: "Backlog", "In Progress", "In Review", "Completed"
    - board must be exactly one of: "Back end", "Front end", "Design"
    - due_date must be in "YYYY-MM-DD" format (not validated for actual date)
    - Cannot create recurring or linked tasks
    - Cannot set additional metadata or custom fields

    Args:
    task_name (str): Descriptive name for the task
    assigned_to_email (str): Email address of team member (must exist in system)
    list_name (str): Task status - must be "Backlog", "In Progress", "In Review", or "Completed"
    due_date (str): Deadline in "YYYY-MM-DD" format
    board (str): Project board - must be "Back end", "Front end", or "Design"

    Returns:
    str: 8-digit task_id of the newly created task (e.g., "00000156"), or error message 
         string if validation fails
    """
    global PROJECT_TASKS

    if not all([task_name, assigned_to_email, list_name, due_date, board]):
        return "Missing task details."

    assigned_to_email = assigned_to_email.lower()
    if assigned_to_email not in PROJECT_TASKS["assigned_to_email"].str.lower().values:
        return "Assignee email not valid. Please choose from the list of team members."
    if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
        return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
    if board not in ["Back end", "Front end", "Design"]:
        return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."

    task_id = str(int(PROJECT_TASKS["task_id"].max()) + 1).zfill(8)
    new_task = pd.DataFrame(
        {
            "task_id": [task_id],
            "task_name": [task_name],
            "assigned_to_email": [assigned_to_email],
            "list_name": [list_name],
            "due_date": [due_date],
            "board": [board],
        }
    )
    PROJECT_TASKS = pd.concat([PROJECT_TASKS, new_task], ignore_index=True)
    return task_id


@tool("project_management.delete_task", return_direct=False)
def delete_task(task_id=None):
    """
    Permanently removes a task from the system by its unique identifier.

    Purpose:
    Deletes tasks that are no longer needed, were created in error, or are obsolete. Use this 
    to clean up the task list and remove cancelled work. This operation is permanent and cannot 
    be undone within the system.

    Usage Examples:
    delete_task("00000042")
    > Returns: "Task deleted successfully."
    
    delete_task("00000999")
    > Returns: "Task not found." (if task doesn't exist)

    Limitations:
    - Deletion is permanent and cannot be undone
    - Requires exact task_id (no partial matches or task name lookup)
    - Does not archive or create deletion history
    - Does not warn about deleting tasks with dependencies
    - Cannot bulk delete multiple tasks at once
    - Cannot delete based on criteria (must use task_id)

    Args:
    task_id (str): 8-digit unique identifier of the task to delete (e.g., "00000042")

    Returns:
    str: Success message "Task deleted successfully." if task found and deleted, 
         or error message "Task not found." if task_id doesn't exist, 
         or "Task ID not provided." if parameter missing
    """
    global PROJECT_TASKS

    if not task_id:
        return "Task ID not provided."

    if task_id in PROJECT_TASKS["task_id"].values:
        PROJECT_TASKS = PROJECT_TASKS[PROJECT_TASKS["task_id"] != task_id]
        return "Task deleted successfully."
    else:
        return "Task not found."


@tool("project_management.update_task", return_direct=False)
def update_task(task_id=None, field=None, new_value=None):
    """
    Modifies a single field of an existing task identified by its task ID.

    Purpose:
    Updates task details as work progresses or requirements change. Use this to reassign tasks, 
    update status, change deadlines, modify task names, or move tasks between boards. Updates 
    one field at a time with validation for field-specific constraints.

    Usage Examples:
    update_task("00000042", "list_name", "In Review")
    > Returns: "Task updated successfully." (moves task to review)
    
    update_task("00000103", "assigned_to_email", "newperson@example.com")
    > Returns: "Task updated successfully." (reassigns task)
    
    update_task("00000087", "due_date", "2023-08-30")
    > Returns: "Task updated successfully." (extends deadline)
    
    update_task("00000042", "board", "Back end")
    > Returns: "Task updated successfully." (moves to different board)

    Limitations:
    - Updates only one field per call; multiple updates require multiple calls
    - Cannot update task_id field itself (immutable identifier)
    - assigned_to_email must match existing team member
    - list_name must be exactly: "Backlog", "In Progress", "In Review", or "Completed"
    - board must be exactly: "Back end", "Front end", or "Design"
    - No validation for due_date format or logical validity
    - No change history or audit trail maintained
    - Cannot perform conditional updates or bulk updates

    Args:
    task_id (str): 8-digit unique task identifier (e.g., "00000042")
    field (str): Field to modify - must be "task_name", "assigned_to_email", "list_name", 
                 "due_date", or "board"
    new_value (str): New value for the specified field (validated based on field type)

    Returns:
    str: Success message "Task updated successfully." if update completed, 
         or error message if task not found, field invalid, value doesn't meet constraints,
         or required parameters missing
    """
    global PROJECT_TASKS

    if not task_id or not field or not new_value:
        return "Task ID, field, or new value not provided."

    if field == "assigned_to_email":
        new_value = new_value.lower()

    if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
        return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
    if field == "list_name" and new_value not in ["Backlog", "In Progress", "In Review", "Completed"]:
        return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
    if field == "assigned_to_email" and new_value not in PROJECT_TASKS["assigned_to_email"].str.lower().values:
        return "Assignee email not valid. Please choose from the list of team members."

    if task_id in PROJECT_TASKS["task_id"].values:
        if field in PROJECT_TASKS.columns:
            PROJECT_TASKS.loc[PROJECT_TASKS["task_id"] == task_id, field] = new_value
            return "Task updated successfully."
        else:
            return "Field not valid."
    else:
        return "Task not found."
