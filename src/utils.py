import os
import sys
import dill
import numpy as np

from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop,Adam,Adagrad,Adamax,Nadam,SGD,Adadelta
from yahoo_fin.stock_info import get_data
import kerastuner as kt

from src.exception import CustomException



def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]  
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def build_model(hp):
    model = Sequential()
    model.add(layers.Input((30, 1)),)
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    learning_rate = hp.Choice('learning_rate',  values=[1e-2, 1e-3, 1e-4])

    # optimizers = hp.Choice('optimizer', values=['Adagrad', 'Adadelta', 'Adadelta', 'Rmsprop'])

    model.compile(optimizer=Adadelta(learning_rate=learning_rate), metrics=['mean_squared_error', 'accuracy'], loss='mse')
    
    return model