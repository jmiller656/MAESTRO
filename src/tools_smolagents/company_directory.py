import pandas as pd
from smolagents import tool

EMAILS = pd.read_csv("data/raw/email_addresses.csv", header=None, names=["email_address"])


@tool
def find_email_address(name: str = "") -> list:
    """
    Finds the email address of an employee by their name.
    
    Args:
        name: Name of the person.
    
    Examples:
    >>> directory.find_email_address("John")
    ["john.smith@example.com"]
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