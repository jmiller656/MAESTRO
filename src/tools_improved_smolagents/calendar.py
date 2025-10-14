import pandas as pd
from smolagents import tool

# Data is hard-coded so that the agent can call them without passing the dataframe as an argument.
CALENDAR_EVENTS = pd.read_csv("data/processed/calendar_events.csv", dtype=str)


def reset_state():
    """
    Resets the calendar events to the original state.
    """
    global CALENDAR_EVENTS
    CALENDAR_EVENTS = pd.read_csv("data/processed/calendar_events.csv", dtype=str)


@tool
def get_event_information_by_id(event_id: str = None, field: str = None) -> dict:
    """
    Retrieves a specific field value from a calendar event using its unique identifier.

    Purpose:
    Fetches specific information (such as name, start time, or duration) for a calendar event 
    when you know the event's ID. This is useful for retrieving detailed information about a 
    particular event without searching through all events.

    Usage Examples:
    get_event_information_by_id("00000000", "event_name")
    # Returns: {{"event_name": "Meeting with Sam"}}
    
    get_event_information_by_id("00000000", "event_start")
    # Returns: {{"event_start": "2021-06-01 13:00:00"}}
    
    get_event_information_by_id("00000000", "duration")
    # Returns: {{"duration": "60"}}

    Limitations:
    - Requires knowing the exact 8-digit event ID
    - Only returns one field at a time
    - Returns error message if event ID doesn't exist
    - Returns error message if field name is invalid

    Args:
    event_id (str): 8-digit unique identifier for the event (required)
    field (str): Name of the field to retrieve. Valid options are: "event_id", "event_name", 
                 "participant_email", "event_start", "duration" (required)

    Returns:
    dict: Dictionary containing the requested field and its value, or an error message string 
          if the event is not found or parameters are missing
    """
    if not event_id:
        return "Event ID not provided."
    if not field:
        return "Field not provided."
    event = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] == event_id].to_dict(orient="records")
    if event:
        if field in event[0]:
            return {field: event[0][field]}
        else:
            return "Field not found."
    else:
        return "Event not found."


@tool
def search_events(query: str = "", time_min: str = None, time_max: str = None) -> list:
    """
    Searches for calendar events matching text criteria and/or time constraints.

    Purpose:
    Finds calendar events by searching event names and participant emails, optionally filtered 
    by a time range. Use this when you need to find events but don't know the exact event ID, 
    or when you want to see all events with a particular person or keyword within a timeframe.

    Usage Examples:
    search_events("Sam")
    # Returns events with "Sam" in the name or participant email
    
    search_events("meeting", time_min="2021-06-01 09:00:00", time_max="2021-06-01 17:00:00")
    # Returns meetings between 9 AM and 5 PM on June 1st
    
    search_events(time_min="2021-06-01 00:00:00", time_max="2021-06-02 00:00:00")
    # Returns all events on June 1st, 2021

    Limitations:
    - Returns maximum of 5 events (even if more match)
    - Case-insensitive text search only (no regex or advanced patterns)
    - Time filtering is inclusive on both bounds
    - Empty query with no time filters returns first 5 events in the dataset

    Args:
    query (str): Text to search for in event_name and participant_email fields. 
                 Partial matches accepted (optional, defaults to empty string)
    time_min (str): Earliest event start time to include in results. 
                    Format: "YYYY-MM-DD HH:MM:SS" (optional)
    time_max (str): Latest event start time to include in results. 
                    Format: "YYYY-MM-DD HH:MM:SS" (optional)

    Returns:
    list: List of up to 5 event dictionaries matching the criteria, each containing "event_id", 
          "event_name", "participant_email", "event_start", and "duration". Returns string 
          "No events found." if no matches exist.
    """
    events = CALENDAR_EVENTS[
        (CALENDAR_EVENTS["event_name"].str.contains(query, case=False))
        | (CALENDAR_EVENTS["participant_email"].str.contains(query, case=False))
    ].to_dict(orient="records")
    if time_min:
        events = [event for event in events if pd.Timestamp(event["event_start"]) >= pd.Timestamp(time_min)]
    if time_max:
        events = [event for event in events if pd.Timestamp(event["event_start"]) <= pd.Timestamp(time_max)]
    if events:
        return events[:5]
    else:
        return "No events found."


@tool
def create_event(event_name: str = None, participant_email: str = None, event_start: str = None, duration: str = None) -> str:
    """
    Creates a new calendar event with specified details and returns its unique identifier.

    Purpose:
    Adds a new event to the user's calendar with a title, participant, start time, and duration. 
    The system automatically generates a unique 8-digit event ID that can be used to reference, 
    update, or delete the event later.

    Usage Examples:
    create_event("Meeting with Sam", "sam@example.com", "2021-06-01 13:00:00", "60")
    # Returns: "00000123" (new event ID)
    
    create_event("Quarterly Review", "team@company.com", "2021-07-15 14:30:00", "90")
    # Returns: "00000124" (new event ID)

    Limitations:
    - Cannot create recurring events (each event must be created individually)
    - No validation for scheduling conflicts with existing events
    - No calendar availability checking for participants
    - Duration is stored as string representing minutes, not validated for reasonableness
    - All parameters are required; missing any parameter returns an error

    Args:
    event_name (str): Title or name of the event (required)
    participant_email (str): Email address of the event participant. Will be automatically 
                             converted to lowercase (required)
    event_start (str): Start date and time for the event. Format: "YYYY-MM-DD HH:MM:SS" (required)
    duration (str): Length of the event in minutes, e.g., "60" for one hour (required)

    Returns:
    str: 8-digit unique identifier (event_id) for the newly created event that can be used 
         for future reference, or an error message string if any required parameters are missing
    """
    global CALENDAR_EVENTS

    if not event_name:
        return "Event name not provided."
    if not participant_email:
        return "Participant email not provided."
    if not event_start:
        return "Event start not provided."
    if not duration:
        return "Event duration not provided."

    participant_email = participant_email.lower()

    event_id = str(int(CALENDAR_EVENTS["event_id"].max()) + 1).zfill(8)
    new_event = pd.DataFrame(
        {
            "event_id": [event_id],
            "event_name": [event_name],
            "participant_email": [participant_email],
            "event_start": [event_start],
            "duration": [duration],
        }
    )
    CALENDAR_EVENTS = pd.concat([CALENDAR_EVENTS, new_event])
    return event_id


@tool
def delete_event(event_id: str = None) -> str:
    """
    Permanently removes a calendar event from the system using its unique identifier.

    Purpose:
    Deletes an event from the calendar when it's been cancelled, completed, or is no longer 
    needed. This is a permanent operation that removes all information about the event from 
    the system.

    Usage Examples:
    delete_event("00000000")
    # Returns: "Event deleted successfully."
    
    delete_event("00000999")
    # Returns: "Event not found." (if this ID doesn't exist)

    Limitations:
    - Deletion is permanent and cannot be undone
    - No confirmation prompt before deletion
    - No notification sent to participants about cancellation
    - Requires exact 8-digit event ID (cannot delete by name or other criteria)

    Args:
    event_id (str): 8-digit unique identifier of the event to delete (required)

    Returns:
    str: Success message "Event deleted successfully." if the event was found and removed, 
         "Event not found." if the ID doesn't exist, or "Event ID not provided." if the 
         parameter is missing
    """
    global CALENDAR_EVENTS

    if not event_id:
        return "Event ID not provided."

    if event_id in CALENDAR_EVENTS["event_id"].values:
        CALENDAR_EVENTS = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] != event_id]
        return "Event deleted successfully."
    else:
        return "Event not found."


@tool
def update_event(event_id: str = None, field: str = None, new_value: str = None) -> str:
    """
    Modifies a specific field of an existing calendar event.

    Purpose:
    Updates individual details of a calendar event (such as changing the event name, participant, 
    start time, or duration) without needing to delete and recreate the entire event. This is 
    useful for rescheduling meetings, updating titles, or changing participants.

    Usage Examples:
    update_event("00000000", "event_name", "New Event Name")
    # Returns: "Event updated successfully."
    
    update_event("00000000", "event_start", "2021-06-02 14:00:00")
    # Returns: "Event updated successfully." (reschedules the event)
    
    update_event("00000000", "participant_email", "NewParticipant@Example.com")
    # Returns: "Event updated successfully." (email normalized to lowercase)

    Limitations:
    - Can only update one field at a time (multiple changes require multiple calls)
    - No validation that the new value is appropriate for the field type
    - Cannot update the event_id itself
    - Does not check for scheduling conflicts when updating event_start
    - No notification sent to participants about changes

    Args:
    event_id (str): 8-digit unique identifier of the event to update (required)
    field (str): Name of the field to modify. Valid options: "event_name", "participant_email", 
                 "event_start", "duration" (required)
    new_value (str): New value to set for the specified field. Participant emails are automatically 
                     converted to lowercase (required)

    Returns:
    str: Success message "Event updated successfully." if the update was applied, 
         "Event not found." if the event_id doesn't exist, or 
         "Event ID, field, or new value not provided." if any required parameters are missing
    """
    global CALENDAR_EVENTS

    if not event_id or not field or not new_value:
        return "Event ID, field, or new value not provided."
    if event_id in CALENDAR_EVENTS["event_id"].values:
        if field == "participant_email":
            new_value = new_value.lower()
        CALENDAR_EVENTS.loc[CALENDAR_EVENTS["event_id"] == event_id, field] = new_value
        return "Event updated successfully."
    else:
        return "Event not found."

calendar_tools = [
    get_event_information_by_id,
    search_events,
    create_event,
    delete_event,
    update_event,
]