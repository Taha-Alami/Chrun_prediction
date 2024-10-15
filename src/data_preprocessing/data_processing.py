import os
import pickle
import pandas as pd
import argparse
from src.data_preprocessing.snowflake_request import SnowflakeRequest
from src.data_preprocessing.preprocessing import Preprocessing
from src.data_preprocessing.prepare_resiliation import ResiliationPreparer
from src.data_preprocessing.prepare_arret_conso import ArretConsoPreparer
from src.data_preprocessing.prepare_clients import ClientsPreparer
from src.data_preprocessing.data_to_predict import DataPredictPreparer
from src.data_preprocessing import CLIENT_COLS, CONTRACT_COLS, ESTABLISHMENT_COLS, APE_CODE_COLS, APE_DIM_COLS, \
    EXCLUDED_PRODUCTS, PRICE_BOOK_COLS, EXCLUDED_CONTRACTS, CONTRACT_DATES, EXCLUDED_CONTRACT_NUMBERS, start_date, EXCLUDED_GROUPS
from src.resources.utils import get_resiliation_preparation_date, get_arret_preparation_date, get_client_preparation_date, get_prediction_date, get_last_date
from src.settings import settings, LOGGER
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore")

def prepare_data():
    # Initialize Snowflake connection
    snowflake_request = SnowflakeRequest(
        settings.SNOWFLAKE_USER,
        settings.SNOWFLAKE_PASSWORD,
        settings.SNOWFLAKE_ACCOUNT,
        settings.DATABASE,
        settings.SCHEMA,
        settings.DATAWAREHOUSE,
        settings.ROLE,
    )

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default="train_df.pkl", help="Path for the training dataset")
    parser.add_argument("--test_file_path", type=str, default="test_df.pkl", help="Path for the testing dataset")
    parser.add_argument("--prediction_file_path", type=str, help="Path for the prediction data")
    args = parser.parse_args()

    # Data Preprocessing
    processor = Preprocessing(
        snowflake_request,
        CLIENT_COLS,
        CONTRACT_COLS,
        ESTABLISHMENT_COLS,
        APE_CODE_COLS,
        APE_DIM_COLS,
        PRICE_BOOK_COLS,
        EXCLUDED_PRODUCTS,
        EXCLUDED_CONTRACTS,
        CONTRACT_DATES,
        EXCLUDED_GROUPS
    )

    processor.prepare_data()
    last_date = get_last_date(processor.last_date)

    # Resiliation preparation
    resiliation_preparer = ResiliationPreparer(processor)
    LOGGER.info("Preparing resiliations")
    date_res = get_resiliation_preparation_date(last_date.date())
    resiliation_data = resiliation_preparer.prepare_all_resiliations(start_date, date_res)

    # Arret conso preparation
    arret_conso_preparer = ArretConsoPreparer(processor)
    LOGGER.info("Preparing arret conso")
    date_arret = get_arret_preparation_date(last_date.date())
    arret_conso_data = arret_conso_preparer.prepare_arret_conso(start_date, date_arret, last_date)

    # Clients preparation
    clients_preparer = ClientsPreparer(processor)
    LOGGER.info("Preparing clients")
    date_client = get_client_preparation_date(last_date.date())
    clients_data = clients_preparer.prepare_clients(start_date, date_client, last_date)

    # Combine all data for training
    train_df = pd.concat([resiliation_data, arret_conso_data, clients_data])
    train_df.to_pickle(args.train_file_path)

    # Test data preparation
    LOGGER.info("Preparing test data")
    test_df = resiliation_preparer.prepare_test_data(date_res, last_date, arret_conso_preparer, clients_preparer)
    test_df.to_pickle(args.test_file_path)

    # Prediction data preparation
    LOGGER.info("Preparing prediction data")
    prediction_preparer = DataPredictPreparer(processor)
    prediction_data = prediction_preparer.prepare_predictions(get_prediction_date(last_date.date()))
    prediction_data.to_pickle(args.prediction_file_path)

if __name__ == "__main__":
    prepare_data()
