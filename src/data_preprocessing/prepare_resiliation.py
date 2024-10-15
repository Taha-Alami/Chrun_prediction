import pandas as pd
from dateutil.relativedelta import relativedelta
from src.settings import LOGGER

class ResiliationPreparer:
    def __init__(self, processor):
        self.processor = processor

    def prepare_resiliation_data(self, start_date, end_date, months_offset, scenario):
        """
        Prepares resiliation data for a given scenario by offsetting the dates and fetching the relevant data.
        
        Parameters:
        - start_date: Starting date for resiliation period
        - end_date: Ending date for resiliation period
        - months_offset: Number of months to offset
        - scenario: Scenario number (used to handle multiple cases)
        
        Returns: DataFrame with resiliation data
        """
        LOGGER.info(f"Preparing resiliation data for scenario {scenario}, from {start_date} to {end_date}")
        
        # Offset dates by the given months
        period_start = start_date + relativedelta(months=months_offset)
        period_end = end_date + relativedelta(months=months_offset)
        
        # Fetch resiliation data using the processor
        data = self.processor.get_resiliation_data(period_start, period_end, scenario)
        
        if data.empty:
            LOGGER.warning(f"No resiliation data found for scenario {scenario}, from {period_start} to {period_end}")
        else:
            LOGGER.info(f"Resiliation data successfully fetched for scenario {scenario}")
        
        return data

    def prepare_all_resiliations(self, start_date, end_date):
        """
        Prepares all resiliation data by combining multiple scenarios.
        
        Returns: DataFrame with concatenated resiliation data across different scenarios
        """
        resiliation_6 = self.prepare_resiliation_data(start_date, end_date, 6, 0)
        resiliation_7 = self.prepare_resiliation_data(start_date, end_date, 6, 1)
        resiliation_8 = self.prepare_resiliation_data(start_date, end_date, 6, 2)
        
        combined_data = pd.concat([resiliation_6, resiliation_7, resiliation_8], ignore_index=True)
        LOGGER.info("Combined all resiliation data across scenarios.")
        
        return combined_data

    def prepare_test_data(self, date_res, last_date, arret_conso_preparer, clients_preparer):
        """
        Prepares test data by combining resiliation, arret conso, and client data.
        
        Parameters:
        - date_res: Date for resiliation test data
        - last_date: Last available date in the dataset
        - arret_conso_preparer: ArretConsoPreparer instance for fetching arret conso data
        - clients_preparer: ClientsPreparer instance for fetching client data
        
        Returns: DataFrame with concatenated test data
        """
        LOGGER.info("Preparing test resiliation data")
        test_resiliation = self.prepare_all_resiliations(date_res + relativedelta(days=1), last_date)
        
        LOGGER.info("Preparing test arret conso data")
        test_arret_conso = arret_conso_preparer.prepare_arret_conso(date_res + relativedelta(days=1), last_date)
        
        LOGGER.info("Preparing test client data")
        test_clients = clients_preparer.prepare_clients(date_res + relativedelta(days=1), last_date)
        
        # Concatenate all test data
        test_data = pd.concat([test_resiliation, test_arret_conso, test_clients], ignore_index=True)
        LOGGER.info("Test data prepared successfully.")
        
        return test_data
