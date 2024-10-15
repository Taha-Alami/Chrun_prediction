import os
from simple_salesforce import Salesforce
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import datetime
from dateutil.relativedelta import relativedelta
import calendar
import pandas as pd
import numpy as np

# XOR encoding for string encoding/decoding operations
def encode_xor(string_to_encode):
    """
    Encodes a string using XOR operation with a fixed key.

    Args:
        string_to_encode (str): The string to be encoded.

    Returns:
        str: Encoded string.
    """
    key = 67
    return ''.join(chr(ord(char) ^ key) for char in string_to_encode)

# Logic to get activation date based on conditions
def get_activation_date(activation_date, deployment_date_1, deployment_date_2):
    """
    Determines the activation date by comparing with default values.

    Args:
        activation_date (str): The original activation date.
        deployment_date_1 (str): The first deployment date.
        deployment_date_2 (str): The second deployment date.

    Returns:
        str: The appropriate activation date.
    """
    if activation_date != '14/09/2016':
        return activation_date
    return min(deployment_date_1, deployment_date_2)

# Loading secrets from Azure Key Vault
def load_user_secrets(secret_key, vault_url, environment):
    """
    Loads user secrets from environment variables or Azure Key Vault.

    Args:
        secret_key (str): The key or username for the secret.
        vault_url (str): The Azure Key Vault URL.
        environment (str): The environment (e.g., 'local', 'production').

    Returns:
        str: The loaded secret.
    """
    if os.getenv("ENV", "local") == "local" or os.getenv("ENV", "local") != environment:
        return os.getenv("SNOWFLAKE_PASSWORD")
    
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    return secret_client.get_secret(secret_key.replace('_', '')).value

# Date preparation logic for different use cases
def get_last_day_of_month(target_date):
    """
    Retrieves the last day of the month for the given date.

    Args:
        target_date (datetime): The target date.

    Returns:
        datetime: The last day of the month.
    """
    _, last_day = calendar.monthrange(target_date.year, target_date.month)
    return target_date.replace(day=last_day)

def get_arret_preparation_date(latest_date):
    """
    Returns the preparation date for 'arret de consommation' by subtracting 7 months from the given date.

    Args:
        latest_date (datetime): The latest available date.

    Returns:
        datetime: The adjusted preparation date.
    """
    return get_last_day_of_month(latest_date) - relativedelta(months=7)

def get_resiliation_preparation_date(latest_date):
    """
    Returns the preparation date for 'resiliation' by subtracting 1 month from the given date.

    Args:
        latest_date (datetime): The latest available date.

    Returns:
        datetime: The adjusted preparation date.
    """
    return get_last_day_of_month(latest_date) - relativedelta(months=1)

def get_client_preparation_date(latest_date):
    """
    Returns the preparation date for 'client' data by subtracting 9 months from the given date.

    Args:
        latest_date (datetime): The latest available date.

    Returns:
        datetime: The adjusted preparation date.
    """
    return get_last_day_of_month(latest_date) - relativedelta(months=9)

def get_prediction_start_date(latest_date):
    """
    Returns the start date for predictions (first day of the month).

    Args:
        latest_date (datetime): The latest available date.

    Returns:
        datetime: The first day of the month.
    """
    return latest_date.replace(day=1)

def parse_last_date(date_string):
    """
    Converts a date string into a datetime object, handling multiple formats.

    Args:
        date_string (str): The date in string format.

    Returns:
        datetime: The parsed date.
    """
    try:
        date_obj = pd.to_datetime(date_string, format='%d/%m/%Y').date()
    except ValueError:
        date_obj = pd.to_datetime(date_string, format='%Y/%m/%d').date()
    
    last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
    return pd.to_datetime(date_obj.replace(day=last_day))

# Salesforce Data Retrieval
def fetch_salesforce_interlocutor_data():
    """
    Retrieves interlocutor data from Salesforce using the appropriate credentials.
    
    Returns:
        pd.DataFrame: DataFrame containing interlocutor information merged with client data.
    """
    # Note: Placeholders for credentials to be replaced with environment variables or vault retrievals.
    sf = Salesforce(
        username=os.getenv("SF_USERNAME", ""),
        password=os.getenv("SF_PASSWORD", ""),
        organizationId=os.getenv("SF_ORG_ID", "")
    )
    
    users_df = pd.DataFrame(sf.query_all("SELECT Id, Name FROM User")['records'])[['Id', 'Name']]
    accounts_df = pd.DataFrame(sf.query_all("SELECT Code_client, Interlocuteur_referent FROM Account")['records'])[['Code_client', 'Interlocuteur_referent']]
    
    merged_df = pd.merge(accounts_df, users_df, left_on='Interlocuteur_referent', right_on='Id', how='inner')
    return merged_df[['Code_client', 'Name']].rename(columns={'Name': 'Interlocuteur_referent'})

# Data Preparation for Churn Analysis
def prepare_churn_data(test_data, predictions):
    """
    Prepares and processes churn data by merging results with additional client information.

    Args:
        test_data (pd.DataFrame): The test data for churn analysis.
        predictions (np.array): Model predictions for churn risk.

    Returns:
        pd.DataFrame: Prepared and formatted churn data.
    """
    churn_df = pd.concat([test_data, pd.DataFrame(predictions[:, 1], columns=['Churn'])], axis=1)
    churn_df = churn_df[churn_df['Churn'] >= 0.5].sort_values(by='Churn', ascending=False)
    
    churn_df['Consumption Change'] = churn_df['Evolution_CA'].apply(
        lambda x: 'Increased consumption' if x > 0 else ('Stable consumption' if x == 0 else 'Decreased consumption'))
    
    churn_df['Months Invoiced (Last 12)'] = 12 - churn_df['months_inactive']
    churn_df['Customer Age (Years)'] = round(churn_df['tenure'] / 12, 0)
    
    interlocutor_data = fetch_salesforce_interlocutor_data()
    churn_df = pd.merge(churn_df, interlocutor_data, left_on='CLIENT_CODE', right_on='Code_client', how='left')
    
    renamed_columns = {
        'CA_6_months': 'Last 6 Months Revenue',
        'CA_total': 'Total Revenue (Since 2017)',
        'evolution_nbr_transaction': 'Transaction Volume Change',
        'Churn': 'Churn Risk'
    }
    
    churn_df.rename(columns=renamed_columns, inplace=True)
    selected_columns = [
        'CLIENT_ORIGIN_CODE', 'Last 6 Months Revenue', 'Total Revenue (Since 2017)', 'Consumption Change',
        'Transaction Volume Change', 'Months Invoiced (Last 12)', 'Customer Age (Years)', 'Churn Risk',
        'Interlocutor_Reference'
    ]
    
    return churn_df[selected_columns]

# XGBoost Prediction Data Preparation
def prepare_xgb_prediction_data(data):
    """
    Prepares the input data for XGBoost model prediction by performing cleaning and transformation.

    Args:
        data (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed data ready for XGBoost prediction.
    """
    # Handle negative or missing values
    data['age_business'] = np.where(data['age_business'] < 0, 0, data['age_business'])
    data['tenure'] = np.where(data['tenure'] < 0, np.nan, data['tenure'])
    
    # Fill missing values for categorical columns
    data['employees'].replace({
        'Unknown': '1 or 2 employees',
        '0 employees as of 31/12': '1 or 2 employees',
        '1 to 9': '1 or 2 employees',
        'Unit without employees': '1 or 2 employees'
    }, inplace=True)
    
    # Fill missing and invalid values for numerical columns
    data['tenure'].fillna(0, inplace=True)
    data['age_business'].fillna(0, inplace=True)
    data['employees'].fillna('1 or 2 employees', inplace=True)

    # Filtering data for XGBoost
    data = data[(data['tenure'] >= 6) & (data['months_inactive'] <= 9) & (data['CA_12_months'] >= 300)]
    
    return data
