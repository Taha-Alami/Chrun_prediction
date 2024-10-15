import pandas as pd
from src.settings import LOGGER

class ArretConsoPreparer:
    def __init__(self, processor):
        self.processor = processor

    def prepare_arret_conso(self, start_date, end_date, last_date):
        """
        Prepares arret de consommation data for the given period.
        
        Parameters:
        - start_date: Starting date for arret conso data
        - end_date: Ending date for arret conso data
        - last_date: The most recent date in the dataset for comparison
        
        Returns: DataFrame with arret conso data
        """
        LOGGER.info(f"Preparing arret conso data from {start_date} to {end_date}")
        
        # Fetch arret conso data using the processor
        data = self.processor.get_arret_conso_data(start_date, end_date, last_date)
        
        if data.empty:
            LOGGER.warning(f"No arret conso data found from {start_date} to {end_date}")
        else:
            LOGGER.info("Arret conso data fetched successfully.")
        
        return data
