import pandas as pd
from langchain.tools import tool

EMAILS = pd.read_csv("data/raw/email_addresses.csv", header=None, names=["email_address"])


@tool("company_directory.find_email_address", return_direct=False)
def find_email_address(name=""):
    """
    Searches the company directory for an employee's email address by matching their name.

    Purpose:
    Looks up and returns email addresses from the company directory that match the 
    provided name. Performs a case-insensitive partial match against email addresses 
    in the directory database.

    Usage Examples:
    find_email_address("John")
    # Returns: ["john.smith@example.com", "john.doe@example.com"]
    
    find_email_address("Sarah Chen")
    # Returns: ["sarah.chen@example.com"]
    
    find_email_address("")
    # Returns: "Name not provided."

    Limitations:
    - Only searches within pre-loaded email addresses from the CSV file
    - Performs substring matching, so partial names may return multiple results
    - Does not verify if the person currently works at the company
    - Returns all matches; does not disambiguate between multiple employees with similar names
    - Requires at least part of the name to be provided

    Args:
        name (str): Full or partial name of the employee to search for. 
                   Case-insensitive. Defaults to empty string.

    Returns:
        numpy.ndarray or str: Array of matching email addresses if found, or 
                             error message "Name not provided." if name is empty.
    """
    global EMAILS
    if name == "":
        return "Name not provided."
    name = name.lower()
    email_address = EMAILS[EMAILS["email_address"].str.contains(name)]
    return email_address["email_address"].values