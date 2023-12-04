import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import create_dataset
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path:str = os.path.join('artifacts', 'models', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        preprocessor = MinMaxScaler()
        logging.info('Defined the preprocessor scaler')
        return preprocessor
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path)
            logging.info('Read the training and testing data')

            useful_column = ['close']
            training_df = train_df[useful_column]
            testing_df = test_df[useful_column]
            logging.info('Selecting useful column')

            training_df = training_df.dropna()
            testing_df = testing_df.dropna()

            preprocessor_obj = self.get_data_transformer()
            logging.info('Obtaining preprocessing object')

            training_arr = preprocessor_obj.fit_transform(training_df)
            testing_arr = preprocessor_obj.transform(testing_df)
            logging.info('Preprocessing done')
            
            train_arr = training_arr.reshape(training_arr.shape[0], 1)
            test_arr = testing_arr.reshape(testing_arr.shape[0], 1)
            logging.info('Reshaped the data')

            save_object(
            file_path = self.data_transformation_config.preprocessor_obj_path,
            obj = preprocessor_obj
            )
            logging.info('Saved preprocessing object')

            timestep = 30
            X_train, y_train = create_dataset(dataset=train_arr, time_step=timestep)
            logging.info('Splitting the train and test data into input and target sets')

            X_test, y_test = create_dataset(dataset=test_arr, time_step=timestep)
            logging.info('Splitting the train and test data into input and target sets')

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            return (
                X_train, 
                y_train, 
                X_test, 
                y_test
                )
   
        except Exception as e:
            raise CustomException(e, sys)