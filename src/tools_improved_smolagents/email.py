import pandas as pd
from smolagents import tool
from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
from typing import Any

# Data is hard-coded so that the agent can call them without passing the dataframe as an argument.
# We cannot use a class because LangChain does not support tools inside classes.
EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


def reset_state():
    """
    Resets the emails to the original state.
    """
    global EMAILS
    EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


@tool
def get_email_information_by_id(email_id: str = None, field: str = None) -> dict:
    """
    Retrieves a specific field from an email using its unique identifier.

    Purpose:
    Fetches a single piece of information (such as subject, sender, or body) from an email
    when you know the email's ID and need to extract one specific detail.

    Usage Examples:
    get_email_information_by_id("12345678", "subject")
    > Returns: {{"subject": "Project Update"}}
    
    get_email_information_by_id("12345678", "sent_date")
    > Returns: {{"sent_date": "2024-01-10 09:30:00"}}
    
    get_email_information_by_id("87654321", "sender")
    > Returns: {{"sender": "john@example.com"}}

    Limitations:
    - Requires exact email_id match (no partial matching)
    - Returns only one field at a time
    - Returns error message if email_id doesn't exist
    - Returns error message if field name is invalid

    Args:
    email_id (str): The unique identifier of the email to retrieve
    field (str): The specific field to extract. Valid options are:
                 "email_id", "sender", "subject", "sent_date", "body", "inbox/outbox"

    Returns:
    dict: A dictionary with the requested field name as key and its value, or
    str: An error message if email_id or field is missing/invalid
    """
    if not email_id:
        return "Email ID not provided."
    if not field:
        return "Field not provided."
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")
    if email:
        if field in email[0]:
            return {field: email[0][field]}
        else:
            return "Field not found."
    else:
        return "Email not found."

@tool
def search_emails(query: str = "", date_min: str = None, date_max: str = None) -> Any:
    """
    Searches for emails containing all specified keywords across subject, body, and sender fields.

    Purpose:
    Finds emails that match your search criteria. **This tool *always* sorts results
    by the most recent date, so the first result (index 0) in the list is always
    the 'latest' or 'most recent' one matching the query.**
    
    This tool is the crucial first step for any action that requires an `email_id`
    (e.g., deleting, replying, forwarding). You MUST use this tool to get the
    `email_id` *before* calling `delete_email`, `reply_email`, or `forward_email`.

    Usage Examples:
    search_emails("Project Update")
    > Returns emails containing both "Project" AND "Update"
    
    **search_emails("nadia")**
    **> Returns all emails from/to/about 'nadia', sorted by most recent first.**
    
    **search_emails("nadia budget")**
    **> Returns emails containing 'nadia' AND 'budget', sorted by most recent first.**
    
    search_emails("meeting", date_min="2024-01-01", date_max="2024-01-31")
    > Returns January 2024 emails containing "meeting"

    Limitations:
    - Returns maximum of 5 most recent matching emails
    - Requires ALL words in query to be present (AND logic, not OR)
    - **CRITICAL: If there are no matches, this tool returns the literal string "No emails found.", NOT an empty list. You MUST check for this string before attempting to iterate or access email IDs.**
    - **CRITICAL: Do NOT use search operators (e.g., `from:john`, `to:susie`). These will fail.**

    Args:
    query (str): Space-separated keywords to search for.
                 **To find an email from a person *about* a topic, include both
                 (e.g., `query='yuki budget'`).**
    date_min (str): Earliest sent date (format: "YYYY-MM-DD"). **Only use this if the
                 user *specifically* requests a time range (e.g., 'last week',
                 'from the last 5 days'). Do NOT use this for general 'latest'
                 or 'most recent' queries.**
    date_max (str): Latest sent date (format: "YYYY-MM-DD"). **Only use this if the
                 user *specifically* requests a time range. Do NOT use this
                 for general 'latest' or 'most recent' queries.**

    Returns:
    list: If there are any matches, up to 5 email dictionaries sorted by sent_datetime (most recent first),
          each containing: email_id, inbox/outbox, subject, sender/recipient,
          sent_datetime, body.
    str: **IMPORTANT: If there are no matches, this returns the literal string "No emails found.", NOT an empty list.**
    """

    query_words = query.lower().split()

    # Filter function to check if all query words are in any of the specified fields
    def filter_emails(row):
        combined_fields = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
        return all(word in combined_fields for word in query_words)

    # Apply filter function across all rows
    filtered_emails = EMAILS.apply(filter_emails, axis=1)
    emails = EMAILS[filtered_emails].sort_values("sent_datetime", ascending=False).to_dict(orient="records")
    if date_min:
        emails = [
            email for email in emails if pd.Timestamp(email["sent_datetime"]).date() >= pd.Timestamp(date_min).date()
        ]
    if date_max:
        # inclusive, remove time from timestamp
        emails = [
            email for email in emails if pd.Timestamp(email["sent_datetime"]).date() <= pd.Timestamp(date_max).date()
        ]
    if len(emails):
        return emails[:5]
    else:
        return "No emails found."
    
def non_tool_send(recipient: str = None, subject: str = None, body: str = None) -> str:
    """
    Sends an email to the specified recipient.
    
    Args:
        recipient: Email address of the recipient.
        subject: Subject line of the email.
        body: Body content of the email.
    
    Examples:
    >>> email.send_email("jane@example.com", "Meeting Reminder", "Don't forget our meeting at 10am tomorrow.")
    "Email sent successfully."
    """
    global EMAILS
    if not recipient or not subject or not body:
        return "Recipient, subject, or body not provided."
    if "@" not in recipient or "." not in recipient:
        return "Invalid recipient email address."
    recipient = recipient.lower()

    email_id = str(int(EMAILS["email_id"].max()) + 1)
    sent_datetime = HARDCODED_CURRENT_TIME
    EMAILS.loc[len(EMAILS)] = [
        email_id,
        "outbox",
        recipient,
        subject,
        sent_datetime,
        body,
    ]

    return "Email sent successfully."

@tool
def send_email(recipient: str = None, subject: str = None, body: str = None) -> str:
    """
    Composes and sends a new email to a specified recipient.

    Purpose:
    Creates and sends an email from the user's outbox with the provided subject line
    and message body.

    Usage Examples:
    send_email("jane@example.com", "Meeting Reminder", "Don't forget our meeting at 10am tomorrow.")
    > Returns: "Email sent successfully."
    
    send_email("invalid-email", "Test", "This will fail")
    > Returns: "Invalid recipient email address."

    Limitations:
    - Requires all three parameters (recipient, subject, body)
    - No support for CC, BCC, or attachments
    - **CRITICAL: This tool requires a *full email address* (e.g., 'dmitri.ivanov@atlas.com'). It cannot accept just a name (e.g., 'dmitri').**

    Args:
    recipient (str): Email address of the person receiving the message.
                     Must contain "@" and "." to pass validation.
                     **If you are given only a name (e.g., "dmitri"), you MUST use the `company_directory.find_email_address` tool first to find their full email address.**
    subject (str): Subject line that appears in the email header
    body (str): Main content/message of the email. **CRITICAL: Only include the text
                 of the email message itself.** Do NOT include conversational parts
                 of the user's request (e.g., 'Can you send this for me?').

    Returns:
    str: "Email sent successfully." if all parameters are valid and email is created, or
         an error message specifying what's missing or invalid
    """
    global EMAILS
    if not recipient or not subject or not body:
        return "Recipient, subject, or body not provided."
    if "@" not in recipient or "." not in recipient:
        return "Invalid recipient email address."
    recipient = recipient.lower()

    email_id = str(int(EMAILS["email_id"].max()) + 1)
    sent_datetime = HARDCODED_CURRENT_TIME
    EMAILS.loc[len(EMAILS)] = [
        email_id,
        "outbox",
        recipient,
        subject,
        sent_datetime,
        body,
    ]

    return "Email sent successfully."

@tool
def delete_email(email_id: str = None) -> str:
    """
    Permanently removes an email from the inbox or outbox.

    Purpose:
    Deletes a specific email from the user's email storage using its unique identifier.
    This action is permanent and cannot be undone. **You MUST use `search_emails` first
    to find the `email_id` of the email you want to delete.**

    Usage Examples:
    delete_email("12345678")
    > Returns: "Email deleted successfully."
    
    delete_email("99999999")
    > Returns: "Email not found."

    Limitations:
    - Deletion is permanent with no recovery option
    - Requires exact email_id match
    - **CRITICAL: This tool deletes only *one* email per call. If `search_emails` returns
      a list of multiple emails to be deleted (e.g., 'delete all emails from...'),
      you **MUST call `delete_email` repeatedly**, once for *each* `email_id` in the
      search results list.**
    - **CRITICAL: The `email_id` must be a valid ID (a string of numbers) obtained
      from `search_emails`. Do NOT pass the string "No emails found." to this tool.**

    Args:
    email_id (str): The unique identifier of the email to permanently delete

    Returns:
    str: "Email deleted successfully." if the email was found and removed, or
         an error message if the email_id is missing or not found
    """
    global EMAILS

    if not email_id:
        return "Email ID not provided."

    if email_id in EMAILS["email_id"].values:
        EMAILS = EMAILS[EMAILS["email_id"] != email_id]
        return "Email deleted successfully."
    else:
        return "Email not found."

@tool
def forward_email(email_id: str = None, recipient: str = None) -> str:
    """
    Forwards an existing email to a new recipient with "FW:" prefix in subject.

    Purpose:
    Sends a copy of an existing email to a different recipient.
    **You MUST use `search_emails` first to find the `email_id` of the email
    you want to forward.** To find the correct email, search for *all* keywords
    (e.g., sender and subject: `query='yuki appreciation gala'`).

    Usage Examples:
    forward_email("12345678", "jane@example.com")
    > Returns: "Email forwarded successfully."
    
    forward_email("invalid_id", "jane@example.com")
    > Returns: "Email not found."

    Limitations:
    - Cannot add additional message to the forwarded content
    - No support for forwarding to multiple recipients at once in a single call
    - **CRITICAL: This tool forwards only *one* email per call. If `search_emails`
      returns a list of multiple emails to be forwarded (e.g., 'forward all emails
      about X'), you **MUST call `forward_email` repeatedly**, once for *each*
      `email_id` in the search results list.**
    - **CRITICAL: The `email_id` must be a valid ID obtained from `search_emails`.
      Do NOT pass the string "No emails found." to this tool.**
    - **CRITICAL: This tool requires a *full email address* (e.g., 'yuki.tanaka@atlas.com')
      for the `recipient`. It cannot accept just a name (e.g., 'yuki').**

    Args:
    email_id (str): The unique identifier of the email to forward
    recipient (str): Email address of the person to receive the forwarded email.
                     Must contain "@" and "." to pass validation.
                     **If you are given only a name (e.g., "yuki"), you MUST use the
                     `company_directory.find_email_address` tool first to find their
                     full email address.**

    Returns:
    str: "Email forwarded successfully." if the operation completes, or
         an error message if parameters are missing, email not found, or
         recipient address is invalid
    """
    global EMAILS
    if not email_id or not recipient:
        return "Email ID or recipient not provided."
    if email_id not in EMAILS["email_id"].values:
        return "Email not found."
    if "@" not in recipient or "." not in recipient:
        return "Invalid recipient email address."
    recipient = recipient.lower()
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")[0]
    result = non_tool_send(recipient, f"FW: {email['subject']}", email["body"])
    return "Email forwarded successfully." if result == "Email sent successfully." else result

@tool
def reply_email(email_id: str = None, body: str = None) -> str:
    """
    Sends a reply to an existing email, addressing the original sender.

    Purpose:
    Responds to an email by automatically sending your reply to the original sender.
    **You MUST use `search_emails` first to find the `email_id` of the email
    you want to reply to.** To find the correct email, search for *all* keywords
    (e.g., sender and subject: `query='yuki appreciation gala'`).

    Usage Examples:
    reply_email("12345678", "Thank you for the update.")
    > Returns: "Email replied successfully."
    
    reply_email("invalid_id", "Thanks!")
    > Returns: "Email not found."

    Limitations:
    - Cannot modify the subject line (uses original subject)
    - Only replies to the direct sender (no "Reply All" functionality)
    - **CRITICAL: The `email_id` must be a valid ID (a string of numbers) obtained
      from `search_emails`. Do NOT pass the string "No emails found." to this tool.**

    Args:
    email_id (str): The unique identifier of the email being replied to
    body (str): Your reply message content. **CRITICAL: Only include the text
                 of the reply itself.** Do NOT include conversational phrases from
                 the user's request (e.g., 'Can you send...').

    Returns:
    str: "Email replied successfully." if the reply is sent, or
         an error message if parameters are missing, email not found, or
         the send operation fails
    """
    global EMAILS
    if not email_id or not body:
        return "Email ID or body not provided."
    if email_id not in EMAILS["email_id"].values:
        return "Email not found."
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")[0]
    result = non_tool_send(email["sender/recipient"], f"{email['subject']}", body)
    return "Email replied successfully." if result == "Email sent successfully." else result

email_tools = [
    get_email_information_by_id,
    search_emails,
    send_email,
    delete_email,
    forward_email,
    reply_email,    
]