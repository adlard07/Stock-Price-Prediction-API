import numpy as np
import pickle
from datetime import date
from flask import Flask, render_template
from tensorflow.keras.models import load_model
from yahoo_fin.stock_info import get_data
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


this_day = date.today()
today = this_day.strftime("%d/%m/%Y")
data = get_data("^NSEI", start_date="01/01/2023", end_date=today, index_as_date = True)
latest_price = np.array(data['close'][-30:]).reshape(-1,1)

model = load_model('artifacts/models/model.h5')
tuned_model = load_model('artifacts/models/tuned_model.h5')
preprocessor = pickle.load(open('artifacts/models/preprocessor.pkl', 'rb'))

n_steps=30
processed_price = preprocessor.transform(latest_price)
prev_price = processed_price.reshape((1, n_steps,1))

new_price = model.predict(prev_price)
tuned_model_new_price = tuned_model.predict(prev_price)

new_price = str(preprocessor.inverse_transform(new_price)[0][0])
tuned_model_new_price = str(preprocessor.inverse_transform(tuned_model_new_price)[0][0])


@app.route('/price')
def index_page():
    prices = {}
    prices['Date'] = today
    prices['model_pred'] = float(new_price)
    prices['tuned_model_pred'] = float(tuned_model_new_price)
    return prices

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
