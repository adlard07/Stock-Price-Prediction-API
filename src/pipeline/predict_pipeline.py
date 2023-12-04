import pickle
import numpy as np
from datetime import date
from tensorflow.keras.models import load_model

from yahoo_fin.stock_info import get_data

today = date.today()
data = get_data("^NSEI", start_date="01/07/2023", end_date=today, index_as_date = True)
latest_price = np.array(data['close'][-30:]).reshape(-1,1)

model = load_model('C:/Users/Home/Desktop/i_hate_this/StockPricePrediction/artifacts/models/model.h5')
preprocessor = pickle.load(open('artifacts/models/preprocessor.pkl', 'rb'))

n_steps=30
processed_price = preprocessor.transform(latest_price)
prev_price = processed_price.reshape((1, n_steps,1))
new_price = model.predict(prev_price)
latest = data['close'][-1]
new_price = preprocessor.inverse_transform(new_price)[0][0]
print(new_price)