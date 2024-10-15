import pandas as pd
import argparse
from src.settings import LOGGER, settings
from src.modeling import min_inactivity_months, min_age, min_revenue, numerical_features
import numpy as np
import mlflow
import os

def load_data_for_prediction(file_path):
    """
    Load and preprocess data for prediction.

    Args:
        file_path (str): Path to the input data file (pickle format).

    Returns:
        x_data (DataFrame): Preprocessed feature set for prediction.
    """
    # Load the data
    data_df = pd.read_pickle(file_path)

    # Drop irrelevant columns
    data_df.drop(['Account', 'CLIENT_ORIGIN_CODE'], axis=1, inplace=True)

    # Set data types and handle missing values
    convert_dict = {
        "main_product": "category",
        "business_sector": "category",
        'workforce': 'category',
        "frequency": "int"
    }
    data_df = data_df.astype(convert_dict)

    # Filter the data based on predefined conditions
    data_df['tenure'].fillna(0, inplace=True)
    data_df = data_df[data_df['tenure'] >= min_age]
    data_df = data_df[data_df['inactive_months'] <= min_inactivity_months]
    data_df = data_df[data_df['revenue_12_months'] >= min_revenue]

    # Feature engineering: compute revenue growth
    data_df['revenue_growth'] = ((data_df['revenue_6_months'] / (data_df['revenue_6_months'] - data_df['revenue_growth'])) - 1)

    # Select numerical features
    x_data = data_df[numerical_features]

    # Reset index and round the data
    x_data = x_data.reset_index(drop=True)
    x_data = x_data.round(2)

    return x_data

def make_prediction():
    """
    Load data, make predictions with the trained model, and log results.
    """
    # Input arguments for prediction
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="predict_df.pkl", help="Input data file path")
    parser.add_argument("--model_name", type=str, default="xgb_churn", help="Registered model name")
    args = parser.parse_args()

    # Start MLflow logging
    LOGGER.info(f"Loading input data from {args.input_file_path}")
    x_data = load_data_for_prediction(args.input_file_path)

    # Load the registered model from MLflow
    LOGGER.info(f"Loading the model {args.model_name} from MLflow")
    model_uri = f"models:/{args.model_name}/latest"
    clf = settings.MLFLOW.pyfunc.load_model(model_uri)

    # Make predictions
    LOGGER.info("Making predictions...")
    predictions = clf.predict(x_data)

    # Output the results
    output_df = pd.DataFrame({
        "predictions": predictions
    })
    output_file_path = os.path.join("results", "predictions.csv")
    output_df.to_csv(output_file_path, index=False)
    LOGGER.info(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    make_prediction()
