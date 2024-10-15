import logging
import os
from logging.config import dictConfig
from typing import Optional
from src.resources.utils import load_user_secrets
import mlflow
import warnings
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

class BaseSettings:
    """
    Base settings class that provides configurations for all environments.
    """
    # Snowflake Connection Parameters (Environment-specific values will override these)
    SNOWFLAKE_ACCOUNT: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    SNOWFLAKE_DATABASE: str = os.getenv("SNOWFLAKE_DATABASE", "")
    SNOWFLAKE_SCHEMA: str = os.getenv("SNOWFLAKE_SCHEMA", "")
    SNOWFLAKE_ROLE: Optional[str] = os.getenv("SNOWFLAKE_ROLE", "")
    SNOWFLAKE_WAREHOUSE: str = os.getenv("SNOWFLAKE_WAREHOUSE", "")
 
    
    # MLFlow Config
    MLFLOW = mlflow

    # Logging Configuration
    LOGGING = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "": {"handlers": ["console"], "level": "DEBUG"}
        },
        "disable_existing_loggers": True,
    }

class DevSettings(BaseSettings):
    """
    Development environment settings.
    """
    SNOWFLAKE_USER: Optional[str] = os.getenv("SNOWFLAKE_USER")
    VAULT_KEY_URL: str = os.getenv("DEV_VAULT_KEY_URL", "")
    SNOWFLAKE_PASSWORD: Optional[str] = load_user_secrets(SNOWFLAKE_USER, VAULT_KEY_URL, "dev")

    def __init__(self):
        warnings.filterwarnings("ignore")
        self.MLFLOW.set_tracking_uri('http://localhost:5000')

class ProdSettings(BaseSettings):
    """
    Production environment settings.
    """
    SNOWFLAKE_USER: Optional[str] = os.getenv("SNOWFLAKE_USER")
    VAULT_KEY_URL: str = os.getenv("PROD_VAULT_KEY_URL", "")
    SNOWFLAKE_PASSWORD: Optional[str] = load_user_secrets(SNOWFLAKE_USER, VAULT_KEY_URL, "prod")

# Environment-based settings initialization
settings: DevSettings| ProdSettings
match os.getenv("ENV", "local"):
    case "local":
        settings = DevSettings()
        settings.MLFLOW.set_tracking_uri('http://localhost:5000')
    case "dev":
        settings = DevSettings()
    case "prod":
        settings = ProdSettings()


