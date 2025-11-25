import pandas as pd
from smolagents import tool

# Data is hard-coded so that the agent can call them without passing the dataframe as an argument.
PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


def reset_state():
    """
    Resets the project tasks to the original state.
    """
    global PROJECT_TASKS
    PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


@tool
def get_task_information_by_id(task_id: str = None, field: str = None) -> dict:
    """
    Retrieves a specific field value from a task using its unique identifier.

    Purpose:
    Fetches a single piece of information (like 'due_date' or 'list_name') from one
    specific task using its known task_id. Use this for checking a specific detail
    about a task you've already identified. This tool does not search for tasks. To
    find tasks based on criteria (like assignee or status), use search_tasks.

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
    - Cannot be used to find tasks. It only retrieves data if the task_id is already known.

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


@tool
def search_tasks(task_name: str = None, assigned_to_email: str = None, list_name: str = None, due_date: str = None, board: str = None) -> list:
    """
    Searches for tasks matching specified criteria across one or more fields.

    Purpose:
    Finds all tasks that match your search criteria. Use this to discover tasks when you don't 
    know the task_id, to filter tasks by assignee, status, deadline, or other attributes, or to 
    get a list of tasks meeting multiple conditions. All searches are case-insensitive and support 
    partial matches. CRITICAL: This is the primary tool to find task_ids based on user
    criteria. The list of tasks returned (with their task_ids) is required to use
    update_task or delete_task in subsequent steps.

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
    - Does not sort results (returned in arbitrary order). It cannot find the 'most urgent'
      or 'newest' task.
    - Cannot use logical operators (AND/OR) between different field values
    - Cannot perform date-based comparisons (e.g., 'before today'). The due_date search
      is a simple text 'contains' match.
    - To act on the results (e.g., update or delete), you must call update_task or
      delete_task for each task_id returned by this search.

    Args:
    task_name (str, optional): Full or partial task name (case-insensitive)
    assigned_to_email (str, optional): Full or partial email address (case-insensitive)
    list_name (str, optional): Full or partial list name. Common values are 'Backlog'
                 (tasks not started), 'In Progress', 'In Review', 'Completed'. A request
                 for 'unfinished' tasks typically means searching for 'Backlog',
                 'In Progress', AND 'In Review'.
    due_date (str, optional): Full or partial date string (e.g., '2023-11' or '2023-11-15').
                 This is a simple text match, NOT a date comparison. It cannot
                 automatically find 'overdue' tasks (e.g., 'date < today'). To find
                 'overdue' tasks, you must search using a specific date string
                 (e.g., a past month like '2023-10').
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


@tool
def create_task(task_name: str = None, assigned_to_email: str = None, list_name: str = None, due_date: str = None, board: str = None) -> str:
    """
    Creates a new task with specified details and automatically generates a unique task ID.

    Purpose:
    Adds a new task to the project management system. Use this when planning new work, 
    assigning responsibilities, or tracking new deliverables. The system automatically assigns 
    a unique 8-digit task_id. IMPORTANT: If given a person's name (e.g., 'Fatima'),
    you MUST use the company_directory.find_email_address tool first to get their full
    email address before calling this tool.

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
    - assigned_to_email must match an existing team member email. You MUST use the
      company_directory.find_email_address tool to get the correct email from a name.
    - list_name must be exactly one of: "Backlog", "In Progress", "In Review", "Completed"
    - board must be exactly one of: "Back end", "Front end", "Design"
    - due_date must be in "YYYY-MM-DD" format (not validated for actual date)
    - Cannot create recurring or linked tasks
    - Cannot set additional metadata or custom fields

    Args:
    task_name (str): Descriptive name for the task
    assigned_to_email (str): CRITICAL: Must be a full, valid email address. If you
                 only have a name (e.g., 'Carlos'), you MUST use the
                 company_directory.find_email_address tool to find the correct
                 email BEFORE calling create_task. Do not guess
                 (e.g., 'carlos@example.com').
    list_name (str): Task status. 'Backlog' is the default for new, unstarted tasks.
                 Must be exactly: "Backlog", "In Progress", "In Review", or "Completed".
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


@tool
def delete_task(task_id: str = None) -> str:
    """
    Permanently removes a task from the system by its unique identifier.

    Purpose:
    Deletes tasks that are no longer needed. This operation is permanent and cannot 
    be undone. IMPORTANT: This tool requires an exact task_id. To delete tasks
    based on criteria (e.g., 'delete all of Carlos's completed tasks'), you must
    first use search_tasks to find their task_ids, and then call this tool for each
    task_id.

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
    - Cannot delete based on criteria (must use task_id); use search_tasks first to find the IDs.

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


@tool
def update_task(task_id: str = None, field: str = None, new_value: str = None) -> str:
    """
    Modifies a single field of an existing task identified by its task ID.

    Purpose:
    Updates task details. Use this to reassign tasks, update status (e.g., move list),
    change deadlines, etc. CRITICAL: This tool operates on a SINGLE task using its
    exact task_id. To update multiple tasks (e.g., 'move all of Carlos's tasks'),
    you must first call search_tasks to get the list of task_ids, and then call
    update_task repeatedly, once for each task_id.

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
    - Updates only one field per call. Bulk updates (e.g., 'move all tasks')
      require calling search_tasks first, then calling this tool multiple times
      (once per task_id).
    - Cannot update task_id field itself (immutable identifier)
    - assigned_to_email must match existing team member. You MUST use the
      company_directory.find_email_address tool first if you only have a name.
    - list_name must be exactly: "Backlog", "In Progress", "In Review", "Completed"
    - board must be exactly: "Back end", "Front end", or "Design"
    - No validation for due_date format or logical validity
    - No change history or audit trail maintained
    - Cannot perform conditional updates or bulk updates

    Args:
    task_id (str): 8-digit unique task identifier (e.g., "00000042")
    field (str): Field to modify - must be "task_name", "assigned_to_email", "list_name", 
                 "due_date", or "board". If updating assigned_to_email and you only
                 have a name (e.g., 'Kofi'), you MUST use the
                 company_directory.find_email_address tool to find the correct
                 email BEFORE calling this tool.
    new_value (str): New value for the specified field. For assigned_to_email
                 updates, this MUST be the full, correct email address. Do not guess.

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

project_management_tools = [
    get_task_information_by_id,
    search_tasks,
    create_task,
    delete_task,
    update_task,
]