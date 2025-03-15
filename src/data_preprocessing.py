import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def handle_missing_values(self, df):
        # Fill missing values for categorical columns with 'Unknown'
        df['race'].fillna('UNKNOWN', inplace=True)
        df['ethnicity'].fillna('UNKNOWN', inplace=True)
        df['patient.lang'].fillna('UNKNOWN', inplace=True)
        df['patient.marital.status'].fillna('UNKNOWN', inplace=True)
        df['homelessstatus'].fillna('UNKNOWN', inplace=True)
        df['pblchouspat'].fillna('UNKNOWN', inplace=True)
        df['vetstatus'].fillna('UNKNOWN', inplace=True)
        df['ptnt.cnsnt.t.cll.ysn'].fillna('N', inplace=True)
        df['ptnt.cnsnt.t.txt.ysn'].fillna('N', inplace=True)

    def handle_age_column(self, df):
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']

        # Add the age group column
        df['age_group'] = pd.cut(df['patient.age'], bins=bins, labels=labels, right=False)

        # Drop the original 'patient.age' column
        df = df.drop(columns=['patient.age'])

    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data Processing step")

            logger.info("Dropping the columns")
            df.drop_duplicates(inplace=True)

            df.drop(columns=['Unnamed: 0', 'patientid', 'apptdate'] , inplace=True)

            self.handle_missing_values(df)

            df['patient.noshow'] = df['patient.noshow'].replace(2, 1)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoding")
            #TODO: check if age grp helps handle_age_column()
            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}

            logger.info("Label Mappings are : ")
            for col,mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Doing Skewness HAndling")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data", e)
        
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced Data")
            X = df.drop(columns='patient.noshow')
            y = df["patient.noshow"]

            smote = SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_resampled , columns=X.columns)
            balanced_df["patient.noshow"] = y_resampled

            logger.info("Data balanced sucesffuly")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e)
    
    def select_features(self,df):
        try:
            logger.info("Starting our Feature selection step")

            X = df.drop(columns='patient.noshow')
            y = df["patient.noshow"]

            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                        'feature':X.columns,
                        'importance':feature_importance
                            })
            top_features_importance_df = feature_importance_df.sort_values(by="importance" , ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]

            top_10_features = top_features_importance_df["feature"].head(num_features_to_select).values

            logger.info(f"Features selected : {top_10_features}")

            top_10_df = df[top_10_features.tolist() + ["patient.noshow"]]

            logger.info("Feature slection completed sucesfully")

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while feature selection", e)
    
    def save_data(self,df , file_path):
        try:
            logger.info("Saving our data in processed folder")

            df.to_csv(file_path, index=False)

            logger.info(f"Data saved sucesfuly to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]  

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df , PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed sucesfully")    
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)
              
    
    
if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()       
