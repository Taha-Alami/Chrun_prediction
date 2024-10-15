# Churn Prediction Pipeline

This project provides a complete end-to-end pipeline for customer churn prediction using an XGBoost model. It includes data loading, preprocessing, model training, evaluation, and prediction with MLOps best practices for tracking experiments and managing models using MLflow.

## Project Structure

├── src                         # Main source directory for all project files
│   ├── data_preprocessing      # Directory for data preprocessing scripts
│   │   ├── data_processing.py  # Script for general data processing functions
│   │   ├── prepare_arret_conso.py  # Script for preparing data related to consumption stoppages
│   │   ├── prepare_clients.py  # Script for preparing client-related data
│   │   └── prepare_resiliation.py  # Script for preparing data related to cancellations
│   ├── modeling                # Directory for modeling scripts
│   │   ├── train.py            # Script for training the churn prediction model
│   │   └── predict.py          # Script for making predictions using the trained model
│   ├── settings.py             # Script for managing application settings and configurations
│   └── utils.py                # Utility functions and helper methods used across the project
├── README.md                   # Documentation file providing an overview of the project and its usage
└── requirements.txt            # File listing project dependencies for easy installation
          

# Prerequisites

- Python 3.8+
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

### Data Preprocessing 

Data preprocessing includes cleaning and feature engineering, such as handling tenure, customer inactivity, and revenue. To preprocess data, refer to the utility functions in `utils.py`.

**Key preprocessing steps:**
- Handling missing values for features like tenure.
- Filtering customers based on inactivity (`min_mois_innactivite`) and revenue (`min_CA`).
- Feature engineering for churn-related predictors, such as `Evolution_CA`.

### Model Training

To train the churn prediction model, use the `train.py` script. This script loads data, applies oversampling to handle class imbalance, trains the XGBoost model, and logs the model to MLflow.

**Example Usage**:

```bash
python train.py --train_file_path data/train_df.pkl --test_file_path data/test_df.pkl --model_path models/xgb_churn.pkl
```

**Arguments**:
- `--train_file_path`: Path to the training dataset in pickle format.
- `--test_file_path`: Path to the test dataset in pickle format.
- `--model_path`: Path to save the trained model.

The model is tracked and logged using MLflow, which includes automatic logging of parameters, metrics, and artifacts (enabled through `mlflow.xgboost.autolog()`).

### Model Prediction

Once the model is trained, use the `predict.py` script to load the model and generate predictions on new data.

**Example Usage**:

```bash
python predict.py --model_path models/xgb_churn.pkl --data_file_path data/test_df.pkl
```

This script loads the trained model from the specified `model_path`, preprocesses the input data, and outputs the predicted churn probabilities.

## MLflow Integration

MLflow is integrated for experiment tracking and model lifecycle management. The training process logs all metrics, parameters, and models to the MLflow tracking server. You can visualize these experiments through the MLflow UI to compare different model runs and versions.

## Logging

Logging is configured using the Python logging module. Logs are captured for key actions such as data loading, model training, and evaluation. Use the `LOGGER` object defined in `settings.py` for capturing log messages throughout the pipeline.

