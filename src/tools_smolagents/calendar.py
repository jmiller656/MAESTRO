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
    Retrieves a specific field value (e.g., name, start time) from an event using its 8-digit ID.

    Usage Scenarios:
    - Use when you have an `event_id` and need to know a *single piece* of information about it.
    - Use to get the 'event_name', 'participant_email', 'event_start', or 'duration' of a known event.

    Examples:
    >>> calendar.get_event_information_by_id("00000000", "event_name")
    {{"event_name": "Meeting with Sam"}}
    >>> calendar.get_event_information_by_id("00000000", "event_start")
    {{"event_start": "2021-06-01 13:00:00"}}

    Args:
    event_id (str): The 8-digit unique ID for the event (e.g., "00000123"). (Required)
    field (str): The specific field to retrieve. Must be one of: "event_id", "event_name", 
                 "participant_email", "event_start", "duration". (Required)

    Returns:
    dict: A dictionary with the requested field and its value, or an error string.
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
    Searches for events by keyword and/or time range. Returns a list of up to 5 matching events.

    Usage Scenarios:
    - Use when you *do not* know the `event_id`.
    - Use to find all meetings with a specific person (e.g., "Sam" or "sam@example.com").
    - Use to find all events on a specific day (e.g., by setting `time_min` and `time_max` to the start and end of the day).
    - Use to find an event with a keyword (e.g., "Meeting").

    Examples:
    >>> calendar.search_events("Sam")
    [{{"event_id": "00000000", ...}}]
    >>> calendar.search_events(time_min="2021-06-01 00:00:00", time_max="2021-06-01 23:59:59")
    [...all events on June 1st...]

    Args:
    query (str): Text to search in event names and participant emails. (Optional)
    time_min (str): The *earliest* event start time. Format: "YYYY-MM-DD HH:MM:SS". (Optional)
    time_max (str): The *latest* event start time. Format: "YYYY-MM-DD HH:MM:SS". (Optional)

    Returns:
    list: A list of up to 5 event dictionaries, or the string "No events found."
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
    Creates a new event with the specified details and returns its 8-digit event_id.

    Usage Scenarios:
    - Use to add a new event to the calendar.
    - Use when a user asks to 'schedule', 'book', or 'add' a meeting or event.

    Examples:
    >>> calendar.create_event("Meeting with Sam", "sam@example.com", "2021-06-01 13:00:00", "60")
    "00000123"

    Args:
    event_name (str): Title of the event (e.g., "Meeting with Sam"). (Required)
    participant_email (str): Email of the participant (e.g., "sam@example.com"). (Required)
    event_start (str): Start time. Format: "YYYY-MM-DD HH:MM:SS". (Required)
    duration (str): Duration in *minutes* (e.g., "60" for 1 hour). (Required)

    Returns:
    str: The 8-digit unique identifier (event_id) for the new event, or an error string.
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
    Permanently deletes an event using its 8-digit event_id.

    Usage Scenarios:
    - Use when a user asks to 'delete', 'cancel', or 'remove' a specific event.
    - This tool requires an `event_id`. If you don't have it, use `search_events` first to find it.

    Examples:
    >>> calendar.delete_event("00000123")
    "Event deleted successfully."

    Args:
    event_id (str): The 8-digit unique ID of the event to delete (e.g., "00000123"). (Required)

    Returns:
    str: "Event deleted successfully." or an error string ("Event not found.").
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
    Updates a *single field* of an existing event using its 8-digit event_id.

    Usage Scenarios:
    - Use to 'reschedule', 'rename', 'change', or 'move' an event.
    - Use to change the event name, start time, duration, or participant.
    - Note: Can only update one field at a time. Multiple changes require multiple calls.
    - This tool requires an `event_id`. If you don't have it, use `search_events` first.

    Examples:
    >>> calendar.update_event("00000000", "event_name", "New Event Name")
    "Event updated successfully."
    >>> calendar.update_event("00000000", "event_start", "2021-06-02 14:00:00")
    "Event updated successfully."

    Args:
    event_id (str): The 8-digit unique ID of the event to update. (Required)
    field (str): The field to change. Must be one of: "event_name", "participant_email", 
                 "event_start", "duration". (Required)
    new_value (str): The new value for the field. (Required)

    Returns:
    str: "Event updated successfully." or an error string.
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