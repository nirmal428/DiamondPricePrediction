import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

# Create a DataIngestion class
class DataIngestion:
    def __init__(self):
        # ✅ Correct class reference
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        try:
            # ✅ Correct file path ('notebooks' not 'notebook')
            df = pd.read_csv(os.path.join('notebooks', 'data', 'gemstone.csv'))
            logging.info('✅ Data read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("🔄 Performing train-test split")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("✅ Ingestion of data is complete")

            # ✅ Return both paths properly
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"❌ Error occurred in Data Ingestion: {e}")
            raise CustomException(e, sys)  # ✅ Reraise properly so error is visible
