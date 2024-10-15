import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from src.settings import LOGGER, settings
from src.modeling import params
from xgboost import XGBClassifier
from src.modeling import min_inactivity_months, min_age, min_revenue, numerical_features
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def load_data(file_path, label_column):
    """
    Load and preprocess the dataset for training and testing.

    Args:
        file_path (str): Path to the input dataset (pickle format).
        label_column (str): The name of the target column (e.g., 'churn').

    Returns:
        x_data (DataFrame): Feature set after preprocessing.
        y_data (Series): Target values.
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

    # Extract the target and feature columns
    y_data = data_df.pop(label_column)
    x_data = data_df[numerical_features]
    
    # Reset index and round the data
    x_data = x_data.reset_index(drop=True)
    x_data = x_data.round(2)
    
    return x_data, y_data

def train_model():
    """
    Train the XGBoost churn prediction model.
    """
    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default="train_df.pkl", help="Train data file path")
    parser.add_argument("--test_file_path", type=str, default="test_df.pkl", help="Test data file path")
    parser.add_argument("--model_path", type=str, help="Model path for saving")
    args = parser.parse_args()

    # Start MLflow logging
    settings.MLFLOW.start_run()
    settings.MLFLOW.xgboost.autolog()  # Enable XGBoost autologging

    LOGGER.info(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # Load and preprocess the training data
    x_train, y_train = load_data(args.train_file_path, "churn")
    
    # Handle class imbalance using oversampling
    ros = RandomOverSampler(random_state=0)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    # Load and preprocess the test data
    x_test, y_test = load_data(args.test_file_path, "churn")

    LOGGER.info(f"Training with data of shape {x_train.shape}")

    # Initialize and train the XGBoost model
    clf = XGBClassifier(
        objective="binary:logistic",
        **params  # Model parameters from src.modeling
    )
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='logloss', early_stopping_rounds=10)

    # Make predictions
    y_pred = clf.predict(x_test)

    # Log classification report and confusion matrix
    LOGGER.info(classification_report(y_test, y_pred))
    LOGGER.info(confusion_matrix(y_test, y_pred).ravel())

    # Register the model via MLflow
    client = settings.MLFLOW.MlflowClient()
    LOGGER.info("Registering the model via MLFlow")
    settings.MLFLOW.xgboost.log_model(
        xgb_model=clf,
        registered_model_name="xgb_churn",
        artifact_path="xgb_churn"
    )

    try:
        latest_version = max(
            int(m.version) for m in client.search_model_versions(f"name='xgb_churn'")
        )
    except Exception as e:
        LOGGER.info(e)
    LOGGER.info(f"Latest model version: {latest_version}")

    # End MLflow run
    settings.MLFLOW.end_run()

if __name__ == "__main__":
    train_model()
