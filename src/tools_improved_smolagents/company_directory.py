import pandas as pd
from smolagents import tool

EMAILS = pd.read_csv("data/raw/email_addresses.csv", header=None, names=["email_address"])


@tool
def find_email_address(name: str = "") -> list:
    """
    Searches the company directory for an employee's email address by matching their name.

    Purpose:
        Looks up and returns full email addresses from the company directory. **Use this tool to
        get the full, correct email address (e.g., 'jane.doe@company.com') whenever you need
        to use an employee's email in another tool and you only have their name (e.g., "Jane").**
        Performs a case-insensitive partial match.

    Usage Examples:
        find_email_address("Jane")
        # Returns: ["jane.doe@company.com"]  (One match found)
        
        find_email_address("Alex")
        # Returns: ["alex.smith@company.com", "alex.johnson@company.com"]  (Ambiguous match)
        
        find_email_address("Nonexistent Name")
        # Returns: []  (No match found)
        
        find_email_address("")
        # Returns: "Name not provided."

    Limitations:
        - Only searches within pre-loaded email addresses.
        - Performs substring matching, so partial names may return multiple results.
        - Does not verify if the person is an active employee.
        - **Returns all matches; it does not and *cannot* disambiguate between multiple employees
          with similar names. You must ask the user for clarification if you get multiple results.**
        - Requires at least part of the name to be provided.

    Args:
        name (str): Full or partial name of the employee to search for. 
                    Case-insensitive. Defaults to empty string.

    Returns:
        list[str] or str: A list of matching email addresses. Your next action depends on this list:
            - **List with 1 email (e.g., ["jane.doe@company.com"]):** Success. Use this email for your next step.
            - **List with 2+ emails (e.g., ["alex.smith@company.com", ...]):** Ambiguous. You MUST ask the user to clarify.
            - **Empty list ([]):** No match found. You MUST report this to the user.
            - **String "Name not provided.":** The 'name' argument was empty.
    """
    global EMAILS
    if name == "":
        return "Name not provided."
    name = name.lower()
    email_address = EMAILS[EMAILS["email_address"].str.contains(name)]
    return email_address["email_address"].values

directory_tools = [
    find_email_address,
]