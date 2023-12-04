import os 
import sys
from src.exception import CustomException
from src.logger import logging
from datetime import date

from yahoo_fin.stock_info import get_data

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import create_dataset
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngessionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')

class DataIngession:
    def __init__(self):
        self.ingession_config = DataIngessionConfig()

    def initiate_data_ingession(self):
        logging.info('Entered the data ingession method or component.')
        try:
            today = date.today()
            data = get_data("^NSEI", start_date="01/07/2018", end_date=today, index_as_date = True)
            logging.info('Read the data from Yahoo Finance API as dataframe👍')
            os.makedirs(os.path.dirname(self.ingession_config.train_data_path), exist_ok=True)
            
            data.to_csv(self.ingession_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(data, train_size=0.85)

            logging.info('Ingession Complete👍')

            train_set.to_csv(self.ingession_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingession_config.test_data_path, index=False, header=True)

            logging.info('Ingession Complete👍')

            return (
                self.ingession_config.train_data_path, 
                self.ingession_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    obj=DataIngession()
    train_data_path, test_data_path = obj.initiate_data_ingession()
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()    
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print(X_train.shape, y_train.shape, X_test.shape, y_train.shape)

    train_model = ModelTrainer()
    mean_squared_error = train_model.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print(f'Mean Squeared Error : {mean_squared_error}')