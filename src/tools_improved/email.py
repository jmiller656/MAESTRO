import pandas as pd
from langchain.tools import tool

from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME

# Data is hard-coded so that the agent can call them without passing the dataframe as an argument.
# We cannot use a class because LangChain does not support tools inside classes.
EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


def reset_state():
    """
    Resets the emails to the original state.
    """
    global EMAILS
    EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


@tool("email.get_email_information_by_id", return_direct=False)
def get_email_information_by_id(email_id=None, field=None):
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


@tool("email.search_emails", return_direct=False)
def search_emails(query="", date_min=None, date_max=None):
    """
    Searches for emails containing all specified keywords across subject, body, and sender fields.

    Purpose:
    Finds emails that match your search criteria by checking if all words in your query
    appear anywhere in the email's subject, body, or sender fields. Results are sorted
    by most recent first and limited to 5 emails.

    Usage Examples:
    search_emails("Project Update")
    > Returns emails containing both "Project" AND "Update"
    
    search_emails("meeting budget", date_min="2024-01-01", date_max="2024-01-31")
    > Returns January 2024 emails containing both "meeting" AND "budget"
    
    search_emails(query="", date_min="2024-01-15")
    > Returns all emails sent on or after January 15, 2024

    Limitations:
    - Returns maximum of 5 most recent matching emails
    - Requires ALL words in query to be present (AND logic, not OR)
    - Case-insensitive matching only
    - No support for phrase matching or advanced search operators
    - Date filters are inclusive (includes boundary dates)

    Args:
    query (str): Space-separated keywords to search for. All words must appear in
                 the email's subject, body, or sender field
    date_min (str): Earliest sent date to include (format: "YYYY-MM-DD"). Inclusive
    date_max (str): Latest sent date to include (format: "YYYY-MM-DD"). Inclusive

    Returns:
    list: Up to 5 email dictionaries sorted by sent_datetime (most recent first),
          each containing: email_id, inbox/outbox, subject, sender/recipient,
          sent_datetime, body, or
    str: "No emails found." if no matches exist
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


@tool("email.send_email", return_direct=False)
def send_email(recipient=None, subject=None, body=None):
    """
    Composes and sends a new email to a specified recipient.

    Purpose:
    Creates and sends an email from the user's outbox with the provided subject line
    and message body. The email is timestamped with the current time and assigned
    a unique email ID for future reference.

    Usage Examples:
    send_email("jane@example.com", "Meeting Reminder", "Don't forget our meeting at 10am tomorrow.")
    > Returns: "Email sent successfully."
    
    send_email("team@company.com", "Weekly Update", "Here are this week's highlights...")
    > Returns: "Email sent successfully."
    
    send_email("invalid-email", "Test", "This will fail")
    > Returns: "Invalid recipient email address."

    Limitations:
    - Requires all three parameters (recipient, subject, body)
    - Validates email format but doesn't verify if address exists
    - No support for CC, BCC, or attachments
    - Cannot schedule emails for future delivery
    - Automatically converts recipient email to lowercase

    Args:
    recipient (str): Email address of the person receiving the message.
                     Must contain "@" and "." to pass validation
    subject (str): Subject line that appears in the email header
    body (str): Main content/message of the email

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


@tool("email.delete_email", return_direct=False)
def delete_email(email_id=None):
    """
    Permanently removes an email from the inbox or outbox.

    Purpose:
    Deletes a specific email from the user's email storage using its unique identifier.
    This action is permanent and cannot be undone.

    Usage Examples:
    delete_email("12345678")
    > Returns: "Email deleted successfully."
    
    delete_email("99999999")
    > Returns: "Email not found."
    
    delete_email("")
    > Returns: "Email ID not provided."

    Limitations:
    - Deletion is permanent with no recovery option
    - Requires exact email_id match
    - No bulk delete capability (one email at a time)
    - No confirmation prompt before deletion

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


@tool("email.forward_email", return_direct=False)
def forward_email(email_id=None, recipient=None):
    """
    Forwards an existing email to a new recipient with "FW:" prefix in subject.

    Purpose:
    Sends a copy of an existing email to a different recipient, preserving the original
    body content and adding "FW:" to the beginning of the subject line to indicate
    it's a forwarded message.

    Usage Examples:
    forward_email("12345678", "jane@example.com")
    > Returns: "Email forwarded successfully."
    
    forward_email("12345678", "team@company.com")
    → Sends email with subject "FW: [original subject]"
    
    forward_email("invalid_id", "jane@example.com")
    > Returns: "Email not found."

    Limitations:
    - Cannot add additional message to the forwarded content
    - No support for forwarding to multiple recipients at once
    - Validates email format but doesn't verify if address exists
    - Automatically converts recipient email to lowercase
    - Requires both email_id and recipient

    Args:
    email_id (str): The unique identifier of the email to forward
    recipient (str): Email address of the person to receive the forwarded email.
                     Must contain "@" and "." to pass validation

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
    result = send_email.func(recipient, f"FW: {email['subject']}", email["body"])
    return "Email forwarded successfully." if result == "Email sent successfully." else result


@tool("email.reply_email", return_direct=False)
def reply_email(email_id=None, body=None):
    """
    Sends a reply to an existing email, addressing the original sender.

    Purpose:
    Responds to an email by automatically sending your reply to the original sender
    while keeping the same subject line. The recipient is determined from the
    original email's sender/recipient field.

    Usage Examples:
    reply_email("12345678", "Thank you for the update.")
    → Sends reply to original sender with your message
    
    reply_email("12345678", "I'll review this and get back to you by Friday.")
    > Returns: "Email replied successfully."
    
    reply_email("invalid_id", "Thanks!")
    > Returns: "Email not found."

    Limitations:
    - Cannot modify the subject line (uses original subject)
    - Only replies to the direct sender (no "Reply All" functionality)
    - Requires both email_id and body content
    - No support for adding attachments
    - Does not quote original message in reply

    Args:
    email_id (str): The unique identifier of the email being replied to
    body (str): Your reply message content

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
    result = send_email.func(email["sender/recipient"], f"{email['subject']}", body)
    return "Email replied successfully." if result == "Email sent successfully." else result
