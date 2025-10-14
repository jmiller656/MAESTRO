from src.tools_improved_smolagents import calendar, email, analytics, project_management, customer_relationship_manager, company_directory
from src.tools_smolagents.toolkits import WRAPPER
from src.tools_smolagents.tracking import wrap_tool

calendar_toolkit = calendar.calendar_tools
email_toolkit = email.email_tools
analytics_toolkit = analytics.analytics_tools
project_management_toolkit = project_management.project_management_tools
customer_relationship_manager_toolkit = customer_relationship_manager.crm_tools
company_directory_toolkit = company_directory.directory_tools

tool_name_prefixes = {}

for tool in calendar_toolkit:
    tool_name_prefixes[tool.name] = "calendar."
for tool in email_toolkit:
    tool_name_prefixes[tool.name] = "email."
for tool in analytics_toolkit:
    tool_name_prefixes[tool.name] = "analytics."
for tool in project_management_toolkit:
    tool_name_prefixes[tool.name] = "project_management."
for tool in customer_relationship_manager_toolkit:
    tool_name_prefixes[tool.name] = "customer_relationship_manager."
for tool in company_directory_toolkit:
    tool_name_prefixes[tool.name] = "company_directory."

toolkits = [
    calendar_toolkit,
    email_toolkit,
    analytics_toolkit,
    project_management_toolkit,
    customer_relationship_manager_toolkit,
    company_directory_toolkit
]

for toolkit in toolkits:
    for idx, tool in enumerate(toolkit):
        toolkit[idx] = wrap_tool(
            tool, wrapper=WRAPPER, prefix=tool_name_prefixes[tool.name]
        )

tools_with_side_effects = [
    calendar.create_event,
    calendar.delete_event,
    calendar.update_event,
    email.send_email,
    email.delete_email,
    email.forward_email,
    email.reply_email,
    analytics.create_plot,
    project_management.create_task,
    project_management.delete_task,
    project_management.update_task,
    customer_relationship_manager.update_customer,
    customer_relationship_manager.add_customer,
    customer_relationship_manager.delete_customer,
]

tools_without_side_effects = [
    calendar.get_event_information_by_id,
    calendar.search_events,
    email.get_email_information_by_id,
    email.search_emails,
    analytics.engaged_users_count,
    analytics.get_visitor_information_by_id,
    analytics.traffic_source_count,
    analytics.total_visits_count,
    analytics.get_average_session_duration,
    project_management.get_task_information_by_id,
    project_management.search_tasks,
    customer_relationship_manager.search_customers,
    company_directory.find_email_address,
]

all_tools = tools_with_side_effects + tools_without_side_effects