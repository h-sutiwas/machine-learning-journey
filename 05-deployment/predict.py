import pickle

from flask import Flask
from flask import request

model_file = 'modelC=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)