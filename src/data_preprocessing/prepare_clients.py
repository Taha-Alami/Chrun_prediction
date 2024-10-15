import pandas as pd
from src.settings import LOGGER

class ClientsPreparer:
    def __init__(self, processor):
        self.processor = processor

    def prepare_clients(self, start_date, end_date, last_date):
        """
        Prepares client data for the given period.
        
        Parameters:
        - start_date: Starting date for client data
        - end_date: Ending date for client data
        - last_date: The most recent date in the dataset
        
        Returns: DataFrame with client data
        """
        LOGGER.info(f"Preparing client data from {start_date} to {end_date}")
        
        # Fetch client data using the processor
        data = self.processor.get_client_data(start_date, end_date, last_date)
        
        if data.empty:
            LOGGER.warning(f"No client data found from {start_date} to {end_date}")
        else:
            LOGGER.info("Client data fetched successfully.")
        
        return data
