import os
import sys
import numpy 
import math
from dataclasses import dataclass
import pickle

from sklearn.metrics import mean_squared_error as mse, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop,Adam,Adagrad,Adamax,SGD,Adadelta
from yahoo_fin.stock_info import get_data
import kerastuner as kt

from src.exception import CustomException
from src.logger import logging
from src.utils import build_model
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path =  os.path.join('artifacts', 'models', 'model.pkl')
    logging.info('Path assigned for LSTM model')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_model_trainer(self):
        try:        
            model = Sequential([
                layers.Input((30, 1)),
                layers.LSTM(32, activation='tanh'),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='linear'),
                layers.Dense(1)
            ])

            model.compile(optimizer=Adadelta(learning_rate=0.001), metrics=['mean_squared_error', 'accuracy'], loss='mse')

            return model
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            model = self.get_model_trainer()
            
            model.compile(optimizer=Adam(learning_rate=0.001), metrics=['mean_squared_error', 'accuracy'], loss='mse')

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=400, batch_size=10, verbose=1)

            error = model.evaluate(X_test, y_test)

            return(
                error
            )
        
        except Exception as e:
            raise CustomException(e, sys)