import pandas as pd
from langchain.tools import tool

CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


def reset_state():
    """
    Resets the CRM data to the original state.
    """
    global CRM_DATA
    CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


@tool("customer_relationship_manager.search_customers", return_direct=False)
def search_customers(
    customer_name=None,
    customer_email=None,
    product_interest=None,
    status=None,
    assigned_to_email=None,
    last_contact_date_min=None,
    last_contact_date_max=None,
    follow_up_by_min=None,
    follow_up_by_max=None,
):
    """
    Searches for customers in the CRM database based on flexible filter criteria.

    Purpose:
        Retrieves customer records matching any combination of search filters. Use this tool to find customers
        by name, contact information, status, assigned sales representative, or date ranges. Returns up to 5
        matching records to prevent overwhelming results.

    Usage Examples:
        - search_customers(customer_name="John") → Find all customers with "John" in their name
        - search_customers(status="Lead", assigned_to_email="sam@example.com") → Find all leads assigned to Sam
        - search_customers(follow_up_by_max="2023-12-31") → Find customers needing follow-up by year-end
        - search_customers(product_interest="Software", status="Qualified") → Find qualified software prospects

    Limitations:
        - Returns maximum of 5 records (oldest records first if more matches exist)
        - Requires at least one search parameter
        - Text searches are case-insensitive partial matches
        - Date comparisons use string comparison (ensure YYYY-MM-DD format)

    Args:
        customer_name (str, optional): Partial or full name of the customer (case-insensitive)
        customer_email (str, optional): Partial or full email address of the customer (case-insensitive)
        product_interest (str, optional): Product category: "Software", "Hardware", "Services", "Consulting", or "Training"
        status (str, optional): Customer status: "Lead", "Qualified", "Proposal", "Won", or "Lost"
        assigned_to_email (str, optional): Email of the sales representative assigned to the customer
        last_contact_date_min (str, optional): Earliest last contact date in YYYY-MM-DD format
        last_contact_date_max (str, optional): Latest last contact date in YYYY-MM-DD format
        follow_up_by_min (str, optional): Earliest follow-up date in YYYY-MM-DD format
        follow_up_by_max (str, optional): Latest follow-up date in YYYY-MM-DD format

    Returns:
        list[dict] or str: List of up to 5 customer records as dictionaries with keys: customer_id, customer_name,
            customer_email, customer_phone, last_contact_date, product_interest, status, assigned_to_email, notes,
            follow_up_by. Returns error message string if no search parameters provided.
    """
    customers = CRM_DATA.copy()
    if not any(
        [
            customer_name,
            customer_email,
            product_interest,
            status,
            assigned_to_email,
            last_contact_date_min,
            last_contact_date_max,
            follow_up_by_min,
            follow_up_by_max,
        ]
    ):
        return "No search parameters provided. Please provide at least one parameter."

    if customer_name:
        customers = customers[customers["customer_name"].str.contains(customer_name, case=False)]
    if customer_email:
        customers = customers[customers["customer_email"].str.contains(customer_email, case=False)]
    if product_interest:
        customers = customers[customers["product_interest"].str.contains(product_interest, case=False)]
    if status:
        customers = customers[customers["status"].str.contains(status, case=False)]
    if assigned_to_email:
        customers = customers[customers["assigned_to_email"].str.contains(assigned_to_email, case=False)]
    if last_contact_date_min:
        customers = customers[customers["last_contact_date"] >= last_contact_date_min]
    if last_contact_date_max:
        customers = customers[customers["last_contact_date"] <= last_contact_date_max]
    if follow_up_by_min:
        customers = customers[customers["follow_up_by"] >= follow_up_by_min]
    if follow_up_by_max:
        customers = customers[customers["follow_up_by"] <= follow_up_by_max]
    return customers.to_dict(orient="records")[:5]


@tool("customer_relationship_manager.update_customer", return_direct=False)
def update_customer(customer_id=None, field=None, new_value=None):
    """
    Updates a specific field in an existing customer record.

    Purpose:
        Modifies a single field of a customer record identified by customer ID. Use this to update contact
        information, change status, reassign customers, update follow-up dates, or add notes. Email addresses
        are automatically normalized to lowercase.

    Usage Examples:
        - update_customer("00000001", "status", "Won") → Mark customer as closed-won
        - update_customer("00000001", "notes", "Follow up needed urgently") → Add notes to customer record
        - update_customer("00000042", "assigned_to_email", "jane@example.com") → Reassign customer to Jane
        - update_customer("00000010", "follow_up_by", "2024-03-15") → Set new follow-up date

    Limitations:
        - Can only update one field at a time
        - Cannot modify customer_id field
        - Status must be one of: "Lead", "Qualified", "Proposal", "Won", "Lost"
        - Product interest must be one of: "Software", "Hardware", "Services", "Consulting", "Training"
        - Returns error if customer_id not found in database

    Args:
        customer_id (str): 8-digit zero-padded customer ID (e.g., "00000001")
        field (str): Name of field to update. Valid values: "customer_name", "assigned_to_email",
            "customer_email", "customer_phone", "last_contact_date", "product_interest", "status",
            "notes", "follow_up_by"
        new_value (str): New value to set for the specified field. For dates, use YYYY-MM-DD format.

    Returns:
        str: Success message "Customer updated successfully." or error message describing the issue
            (e.g., missing parameters, invalid field, invalid status/product, customer not found)
    """
    global CRM_DATA

    if not customer_id or not field or not new_value:
        return "Customer ID, field, or new value not provided."

    if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
        return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"

    if field == "product_interest" and new_value not in ["Software", "Hardware", "Services", "Consulting", "Training"]:
        return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"

    if field == "customer_email" or field == "assigned_to_email":
        new_value = new_value.lower()

    if customer_id in CRM_DATA["customer_id"].values:
        if field in CRM_DATA.columns:
            CRM_DATA.loc[CRM_DATA["customer_id"] == customer_id, field] = new_value
            return "Customer updated successfully."
        else:
            return "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', 'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'"
    else:
        return "Customer not found."


@tool("customer_relationship_manager.add_customer", return_direct=False)
def add_customer(
    customer_name=None,
    assigned_to_email=None,
    status=None,
    customer_email=None,
    customer_phone=None,
    last_contact_date=None,
    product_interest=None,
    notes="",
    follow_up_by=None,
):
    """
    Creates a new customer record in the CRM database.

    Purpose:
        Adds a new customer with specified details and automatically generates a unique customer ID.
        Use this when onboarding new leads or customers. The system ensures unique ID generation
        and normalizes email addresses to lowercase. Only three fields are required; all others
        are optional.

    Usage Examples:
        - add_customer("John Doe", "sam@example.com", "Lead") → Minimal new lead
        - add_customer("Acme Corp", "jane@example.com", "Qualified", customer_email="contact@acme.com",
                       product_interest="Software") → Qualified prospect with contact details
        - add_customer("Tech Startup", "sam@example.com", "Proposal", customer_phone="555-0123",
                       follow_up_by="2024-04-01", notes="Pitched enterprise package") → Detailed opportunity

    Limitations:
        - Requires exactly three mandatory fields: customer_name, assigned_to_email, status
        - Status must be one of: "Lead", "Qualified", "Proposal", "Won", "Lost"
        - Product interest (if provided) must be one of: "Software", "Hardware", "Services", "Consulting", "Training"
        - Customer ID is auto-generated and cannot be specified manually
        - Does not check for duplicate customer names or emails

    Args:
        customer_name (str): Full name or company name of the customer (required)
        assigned_to_email (str): Email address of the sales representative assigned to this customer (required)
        status (str): Initial customer status - must be one of: "Lead", "Qualified", "Proposal", "Won", "Lost" (required)
        customer_email (str, optional): Email address of the customer
        customer_phone (str, optional): Phone number of the customer (any format)
        last_contact_date (str, optional): Date of most recent contact in YYYY-MM-DD format
        product_interest (str, optional): Product category of interest: "Software", "Hardware", "Services", "Consulting", or "Training"
        notes (str, optional): Free-form text notes about the customer (default: empty string)
        follow_up_by (str, optional): Target date for next follow-up in YYYY-MM-DD format

    Returns:
        str: The auto-generated 8-digit zero-padded customer ID (e.g., "00000201") if successful,
            or an error message if required fields are missing
    """
    global CRM_DATA
    if not all([customer_name, assigned_to_email, status]):
        return "Please provide all required fields: customer_name, assigned_to_email, status."

    assigned_to_email = assigned_to_email.lower()
    if customer_email:
        customer_email = customer_email.lower()

    new_id = str(int(CRM_DATA["customer_id"].max()) + 1).zfill(8)
    new_customer = pd.DataFrame(
        {
            "customer_id": [new_id],
            "customer_name": [customer_name],
            "customer_email": [customer_email],
            "customer_phone": [customer_phone],
            "last_contact_date": [last_contact_date],
            "product_interest": [product_interest],
            "status": [status],
            "assigned_to_email": [assigned_to_email],
            "notes": [notes],
            "follow_up_by": [follow_up_by],
        }
    )
    CRM_DATA = pd.concat([CRM_DATA, new_customer], ignore_index=True)
    return new_id


@tool("customer_relationship_manager.delete_customer", return_direct=False)
def delete_customer(customer_id=None):
    """
    Permanently removes a customer record from the CRM database.

    Purpose:
        Deletes a customer record by ID. Use this to remove duplicate entries, test data, or customers
        that should no longer be tracked. This operation cannot be undone, so verify the customer ID
        before deletion.

    Usage Examples:
        - delete_customer("00000001") → Remove customer with ID 00000001
        - delete_customer("00000999") → Remove customer with ID 00000999

    Limitations:
        - Deletion is permanent and cannot be undone
        - Does not archive or create backup of deleted record
        - Returns error if customer_id not found in database
        - Requires exact customer ID match (case-sensitive)

    Args:
        customer_id (str): 8-digit zero-padded customer ID to delete (e.g., "00000001")

    Returns:
        str: Success message "Customer deleted successfully." if deletion succeeded, or error message
            "Customer ID not provided." if no ID given, or "Customer not found." if ID doesn't exist
    """
    global CRM_DATA
    if not customer_id:
        return "Customer ID not provided."
    if customer_id not in CRM_DATA["customer_id"].values:
        return "Customer not found."
    CRM_DATA = CRM_DATA[CRM_DATA["customer_id"] != customer_id]
    return "Customer deleted successfully."
