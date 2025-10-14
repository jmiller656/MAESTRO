import pandas as pd
from langchain.tools import tool

ANALYTICS_DATA = pd.read_csv("data/processed/analytics_data.csv", dtype=str)
ANALYTICS_DATA["user_engaged"] = ANALYTICS_DATA["user_engaged"] == "True"  # Convert to boolean
PLOTS_DATA = pd.DataFrame(columns=["file_path"])
METRICS = ["total_visits", "session_duration_seconds", "user_engaged"]
METRIC_NAMES = ["total visits", "average session duration", "engaged users"]


def reset_state():
    """
    Resets the analytics data to the original state.
    """
    global ANALYTICS_DATA
    ANALYTICS_DATA = pd.read_csv("data/processed/analytics_data.csv", dtype=str)
    ANALYTICS_DATA["user_engaged"] = ANALYTICS_DATA["user_engaged"] == "True"  # Convert to boolean
    global PLOTS_DATA
    PLOTS_DATA = pd.DataFrame(columns=["file_path"])


@tool("analytics.get_visitor_information_by_id", return_direct=False)
def get_visitor_information_by_id(visitor_id=None):
    """
    Retrieves complete analytics data for a specific visitor.

    Purpose:
    Looks up and returns all stored analytics information for a given visitor ID,
    including their visit date, page views, session duration, traffic source, and
    engagement status.

    Usage Examples:
    - get_visitor_information_by_id("000") -> Returns all data for visitor 000
    - get_visitor_information_by_id("12345") -> Returns all data for visitor 12345

    Limitations:
    - Returns "Visitor ID not provided." if no visitor_id is given
    - Returns "Visitor not found." if the visitor_id doesn't exist in the data
    - Only retrieves data; does not modify or aggregate information

    Args:
    visitor_id (str): Unique identifier for the visitor to look up

    Returns:
    dict or str: Dictionary containing visitor analytics data (date_of_visit, 
    visitor_id, page_views, session_duration_seconds, traffic_source, user_engaged),
    or an error message string if visitor not found or ID not provided
    """
    if not visitor_id:
        return "Visitor ID not provided."
    visitor_data = ANALYTICS_DATA[ANALYTICS_DATA["visitor_id"] == visitor_id].to_dict(orient="records")
    if visitor_data:
        return visitor_data
    else:
        return "Visitor not found."


@tool("analytics.create_plot", return_direct=False)
def create_plot(time_min=None, time_max=None, value_to_plot=None, plot_type=None):
    """
    Generates a visualization of analytics data for a specified time range and metric.

    Purpose:
    Creates and saves a plot file showing analytics trends over time. The plot can
    visualize various metrics like total visits, session duration, engagement, or
    traffic source breakdowns in different chart formats.

    Usage Examples:
    - create_plot("2023-10-01", "2023-12-31", "total_visits", "line") -> Creates line chart
    - create_plot("2023-11-01", "2023-11-30", "user_engaged", "bar") -> Creates bar chart
    - create_plot("2023-10-15", "2023-10-20", "session_duration_seconds", "scatter") -> Creates scatter plot

    Limitations:
    - Requires all four parameters (time_min, time_max, value_to_plot, plot_type)
    - Returns error message if any required parameter is missing
    - value_to_plot must be one of: "total_visits", "session_duration_seconds", 
      "user_engaged", "visits_direct", "visits_referral", "visits_search_engine", 
      "visits_social_media"
    - plot_type must be one of: "bar", "line", "scatter", "histogram"
    - Does not display the plot directly; only saves to file

    Args:
    time_min (str): Start date in "YYYY-MM-DD" format
    time_max (str): End date in "YYYY-MM-DD" format
    value_to_plot (str): Metric to visualize (see Limitations for valid options)
    plot_type (str): Type of chart to create ("bar", "line", "scatter", or "histogram")

    Returns:
    str: File path to the saved plot (format: "plots/{{time_min}}_{{time_max}}_{{value_to_plot}}_{{plot_type}}.png"),
    or an error message string if parameters are invalid or missing
    """
    global PLOTS_DATA
    if not time_min:
        return "Start date not provided."
    if not time_max:
        return "End date not provided."
    if value_to_plot not in [
        "total_visits",
        "session_duration_seconds",
        "user_engaged",
        "visits_direct",
        "visits_referral",
        "visits_search_engine",
        "visits_social_media",
    ]:
        return "Value to plot must be one of 'total_visits', 'session_duration_seconds', 'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
    if plot_type not in ["bar", "line", "scatter", "histogram"]:
        return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"

    # Plot the data here and save it to a file
    file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
    PLOTS_DATA.loc[len(PLOTS_DATA)] = [file_path]
    return file_path


@tool("analytics.total_visits_count", return_direct=False)
def total_visits_count(time_min=None, time_max=None):
    """
    Calculates the total number of visits for each day in a specified time range.

    Purpose:
    Counts and aggregates all visitor sessions on a per-day basis within the given
    date range. This provides a daily breakdown of site traffic volume.

    Usage Examples:
    - total_visits_count("2023-10-01", "2023-10-06") -> Returns daily visit counts for Oct 1-6
    - total_visits_count("2023-11-15", "2023-11-15") -> Returns visit count for single day
    - total_visits_count(time_min="2023-10-01") -> Returns counts from Oct 1 onwards
    - total_visits_count(time_max="2023-10-31") -> Returns counts up to Oct 31

    Limitations:
    - Both time_min and time_max are optional; omitting them includes all available data
    - Returns 0 for dates with no visits within the range
    - Does not filter by traffic source, engagement, or other attributes
    - Counts all visits regardless of session duration or quality

    Args:
    time_min (str, optional): Start date in "YYYY-MM-DD" format. If omitted, includes all dates from beginning
    time_max (str, optional): End date in "YYYY-MM-DD" format. If omitted, includes all dates through end

    Returns:
    dict: Dictionary mapping each date to its visit count 
    (format: {{"YYYY-MM-DD": count, ...}})
    """
    if time_min:
        data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min]
    else:
        data = ANALYTICS_DATA
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    return data.groupby("date_of_visit").size().to_dict()


@tool("analytics.engaged_users_count", return_direct=False)
def engaged_users_count(time_min=None, time_max=None):
    """
    Calculates the number of engaged users for each day in a specified time range.

    Purpose:
    Counts visitors who met engagement criteria (user_engaged = True) on a per-day
    basis. This metric helps identify days with higher quality traffic and user
    interaction levels.

    Usage Examples:
    - engaged_users_count("2023-10-01", "2023-10-06") -> Returns daily engaged user counts for Oct 1-6
    - engaged_users_count("2023-11-01", "2023-11-30") -> Returns engaged users for entire November
    - engaged_users_count(time_min="2023-10-15") -> Returns counts from Oct 15 onwards

    Limitations:
    - Both time_min and time_max are optional; omitting them includes all available data
    - Returns 0 for dates with no engaged users within the range
    - Only counts users marked as engaged (user_engaged = True)
    - Does not provide engagement rate or compare to total visits
    - Does not break down engagement by traffic source or other dimensions

    Args:
    time_min (str, optional): Start date in "YYYY-MM-DD" format. If omitted, includes all dates from beginning
    time_max (str, optional): End date in "YYYY-MM-DD" format. If omitted, includes all dates through end

    Returns:
    dict: Dictionary mapping each date to its engaged user count 
    (format: {{"YYYY-MM-DD": count, ...}})
    """
    if time_min:
        data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min]
    else:
        data = ANALYTICS_DATA[:]
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)

    return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()


@tool("analytics.traffic_source_count", return_direct=False)
def traffic_source_count(time_min=None, time_max=None, traffic_source=None):
    """
    Counts visits from a specific traffic source for each day in a specified time range.

    Purpose:
    Tracks daily visit volume from a particular traffic channel (direct, referral,
    search engine, or social media). This helps analyze which acquisition channels
    drive traffic over time.

    Usage Examples:
    - traffic_source_count("2023-10-01", "2023-10-06", "search engine") -> Daily search engine visits
    - traffic_source_count("2023-11-01", "2023-11-30", "social media") -> Daily social media visits
    - traffic_source_count("2023-10-01", "2023-10-31") -> Daily total visits (all sources)
    - traffic_source_count(time_max="2023-10-15", traffic_source="direct") -> Direct traffic up to Oct 15

    Limitations:
    - All parameters are optional; omitting time range includes all dates
    - If traffic_source is omitted, returns total visits for all sources combined
    - traffic_source must be one of: "direct", "referral", "search engine", "social media"
    - Returns 0 for dates with no visits from the specified source
    - Cannot query multiple traffic sources simultaneously

    Args:
    time_min (str, optional): Start date in "YYYY-MM-DD" format. If omitted, includes all dates from beginning
    time_max (str, optional): End date in "YYYY-MM-DD" format. If omitted, includes all dates through end
    traffic_source (str, optional): Specific traffic source to filter by. Valid options are 
    "direct", "referral", "search engine", "social media". If omitted, returns all visits.

    Returns:
    dict: Dictionary mapping each date to visit count from the specified source
    (format: {{"YYYY-MM-DD": count, ...}})
    """
    if time_min:
        data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min]
    else:
        data = ANALYTICS_DATA[:]
    if time_max:
        data = data[data["date_of_visit"] <= time_max]

    if traffic_source:
        data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
        return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
    else:
        return data.groupby("date_of_visit").size().to_dict()


@tool("analytics.get_average_session_duration", return_direct=False)
def get_average_session_duration(time_min=None, time_max=None):
    """
    Calculates the average session duration for each day in a specified time range.

    Purpose:
    Computes the mean session length in seconds across all visits for each day.
    This metric indicates user engagement depth and content consumption patterns
    over time.

    Usage Examples:
    - get_average_session_duration("2023-10-01", "2023-10-06") -> Daily averages for Oct 1-6
    - get_average_session_duration("2023-11-01", "2023-11-30") -> Daily averages for November
    - get_average_session_duration(time_min="2023-10-15") -> Averages from Oct 15 onwards
    - get_average_session_duration() -> Averages for all available dates

    Limitations:
    - Both time_min and time_max are optional; omitting them includes all available data
    - Returns average in seconds (not formatted as minutes or hours)
    - Does not filter out outliers or invalid session durations
    - Cannot break down by traffic source, page, or other dimensions
    - Does not provide median or other statistical measures

    Args:
    time_min (str, optional): Start date in "YYYY-MM-DD" format. If omitted, includes all dates from beginning
    time_max (str, optional): End date in "YYYY-MM-DD" format. If omitted, includes all dates through end

    Returns:
    dict: Dictionary mapping each date to average session duration in seconds
    (format: {{"YYYY-MM-DD": float, ...}})
    """
    if time_min:
        data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min]
    else:
        data = ANALYTICS_DATA
    if time_max:
        data = data[data["date_of_visit"] <= time_max]

    data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
    return (
        data[["date_of_visit", "session_duration_seconds"]]
        .groupby("date_of_visit")
        .mean()["session_duration_seconds"]
        .to_dict()
    )