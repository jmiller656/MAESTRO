import pandas as pd
from smolagents import tool

CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


def reset_state():
    """
    Resets the CRM data to the original state.
    """
    global CRM_DATA
    CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


@tool
def search_customers(
    customer_name: str = None,
    customer_email: str = None,
    product_interest: str = None,
    status: str = None,
    assigned_to_email: str = None,
    last_contact_date_min: str = None,
    last_contact_date_max: str = None,
    follow_up_by_min: str = None,
    follow_up_by_max: str = None,
) -> list:
    """
    Searches for customers based on the given parameters.
    
    Args:
        customer_name: Name of the customer.
        customer_email: Email address of the customer.
        product_interest: Product interest of the customer.
        status: Current status of the customer.
        assigned_to_email: Email address of the person assigned to the customer.
        last_contact_date_min: Minimum last contact date. Format: "YYYY-MM-DD"
        last_contact_date_max: Maximum last contact date. Format: "YYYY-MM-DD"
        follow_up_by_min: Minimum follow up date. Format: "YYYY-MM-DD"
        follow_up_by_max: Maximum follow up date. Format: "YYYY-MM-DD"
    
    Examples:
    >>> crm.search_customers(customer_name="John")
    {"customer_id": "00000001", "assigned_to_email": "sam@example.com", "customer_name": "John Smith",
    "customer_email": "john.smith@example.com", "customer_phone": "123-456-7890", "last_contact_date": "2023-01-01",
    "product_interest": "Software", "status": "Qualified", "follow_up_by": "2023-01-15", "notes": "Had a call on 2023-01-01. "}
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


@tool
def update_customer(customer_id: str = None, field: str = None, new_value: str = None) -> str:
    """
    Updates a customer record by ID.
    
    Args:
        customer_id: ID of the customer.
        field: Field to update. Available fields are: "customer_name", "assigned_to_email", "customer_email", "customer_phone", "last_contact_date", "product_interest", "status", "notes", "follow_up_by"
        new_value: New value for the field.
    
    Examples:
    >>> crm.update_customer("00000001", "status", "Won")
    "Customer updated successfully."
    """
    global CRM_DATA

    if not customer_id or not field or not new_value:
        return "Customer ID, field, or new value not provided."

    if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
        return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"

    if field == "product_interest" and new_value not in ["Software", "Hardware", "Services", "Consulting", "Training"]:
        return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"

    if field in ["customer_email", "assigned_to_email"]:
        new_value = new_value.lower()

    if customer_id in CRM_DATA["customer_id"].values:
        if field in CRM_DATA.columns:
            CRM_DATA.loc[CRM_DATA["customer_id"] == customer_id, field] = new_value
            return "Customer updated successfully."
        else:
            return "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', 'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'"
    else:
        return "Customer not found."


@tool
def add_customer(
    customer_name: str = None,
    assigned_to_email: str = None,
    status: str = None,
    customer_email: str = None,
    customer_phone: str = None,
    last_contact_date: str = None,
    product_interest: str = None,
    notes: str = "",
    follow_up_by: str = None,
) -> str:
    """
    Adds a new customer record.
    
    Args:
        customer_name: Name of the customer.
        assigned_to_email: Email address of the person assigned to the customer.
        status: Current status of the customer. One of: "Qualified", "Won", "Lost", "Lead", "Proposal"
        customer_email: Email address of the customer.
        customer_phone: Phone number of the customer.
        last_contact_date: The last date the customer was contacted. Format: "YYYY-MM-DD"
        product_interest: Product interest of the customer. One of: "Software", "Hardware", "Services", "Consulting", "Training"
        notes: Notes about the customer.
        follow_up_by: Date for the next follow up. Format: "YYYY-MM-DD"
    
    Examples:
    >>> crm.add_customer("Sam Smith", "sam@example.com", "Lead", "sam.smith@example.com", "123-456-7890", "2023-01-01", "Software")
    "00000201"
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


@tool
def delete_customer(customer_id: str = None) -> str:
    """
    Deletes a customer record by ID.
    
    Args:
        customer_id: ID of the customer.
    
    Examples:
    >>> crm.delete_customer("00000001")
    "Customer deleted successfully."
    """
    global CRM_DATA
    if not customer_id:
        return "Customer ID not provided."
    if customer_id not in CRM_DATA["customer_id"].values:
        return "Customer not found."
    CRM_DATA = CRM_DATA[CRM_DATA["customer_id"] != customer_id]
    return "Customer deleted successfully."

crm_tools = [
    search_customers,
    update_customer,
    add_customer,
    delete_customer,
]