import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        logger.info(f"Data Ingestion started logging")
        print(f"Data Ingestion started print")

        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR , exist_ok=True)
        print(f"Data Ingestion started with {self.bucket_name} and file is  {self.file_name}")

        logger.info(f"Data Ingestion started with {self.bucket_name} and file is  {self.file_name}")

    def download_csv_from_gcp(self):
        try:
            print('download_csv_from_gcp', RAW_FILE_PATH)
            #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/naaz/data1/official/myfunprojects/MLops/myMLflowPrj/mlflowEnv/env/poetic-sentinel-448201-n2-31456dcc94c7.json"
            client = storage.Client()
    
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            if not blob.exists:
                logger.error("data ingestion blob doesnt exist")
            

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"{RAW_FILE_PATH=}")
            print(f"CSV file is sucesfully downloaded to {RAW_FILE_PATH}")

            logger.info(f"CSV file is sucesfully downloaded to {RAW_FILE_PATH}")

        except Exception as ce:
            logger.error("Error while downloading the csv file")
            raise CustomException("Failed to downlaod csv file ", ce)
        
    def split_data(self):
        try:
            logger.info("Starting the splitting process")
            DATA_SHEET_NAME = 'Raw Data'
            data = pd.read_excel(RAW_FILE_PATH, sheet_name=DATA_SHEET_NAME)
            logger.info(data.shape)

            train_data , test_data = train_test_split(data , test_size=1-self.train_test_ratio , random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException("Failed to split data into training and test sets ", e)
        
    def run(self):

        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion completed sucesfully")
        
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        
        finally:
            logger.info("Data ingestion completed")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()




        

