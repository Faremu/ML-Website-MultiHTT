# A very simple Flask Hello World app for you to get started with...
import time
import pandas as pd
import os
from flask import Flask, jsonify, request, render_template,make_response
import pickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

app = Flask(__name__)
def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)

MODEL_DIR = os.path.join(
    app.root_path,
    "static",
    "model",
    "3outputs-model.pkl"
)
# model_files = [
#     "1-model.pkl",
#     "2-model.pkl",
#     "3-model.pkl"
# ]
# models = []

# for fname in model_files:
#     model_path = os.path.join(MODEL_DIR, fname)
#     with open(model_path, "rb") as f:
#         models.append(pickle.load(f))

with open(MODEL_DIR, "rb") as f:
    model = pickle.load(f)

@app.route('/',methods=['GET', 'POST'])
def index():
    template = render_template('index.html')
    response = make_response(template)
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route("/predict", methods=["POST"])
def predict():

    numeric_columns = [
        "RT", "T", "TP", "BL", "RS", "W", "WtBL",
        "C", "H", "N", "O", "S", "HC", "OC", "HHV"
    ]
    data = request.json
    app.logger.info(f'output :{data}')
    # Operation Condition
    RT = data['RT']

    T = data['T']
    TP = data['TP']
    BL = data['BL']
    RS = data['RS']
    W = data['W']
    WtBL = data['WtBL']

    # Elemental Property
    C = data['C']
    H = data['H']
    N = data['N']
    O = data['O']
    S = data['S']
    HC = data['HC']
    OC = data['OC']
    HHV = data['HHV']

    if(RT == '0' and T == '0' and TP == '0' and BL == '0' and RS == '0' and W == '0' and WtBL == '0' and
       C == '0' and H == '0' and N == '0' and O == '0' and S == '0' and HC == '0' and OC == '0' and HHV == '0'):
        return jsonify({"error": "Invalid input"}), 400
    clean_data = {}
    for col in numeric_columns:
        try:
            clean_data[col] = float(data[col])
        except (KeyError, ValueError, TypeError):
            return jsonify({"error": f"Invalid or missing value for {col}"}), 400

    entry = pd.DataFrame([clean_data]).to_numpy()
    app.logger.info("Stucked at predict")
    # pred = [float(m.predict(entry)[0]) for m in models]
    pred = model.predict(entry)[0].tolist()
    app.logger.info("Passed at predict")
    res = {'prediction':pred}

    return jsonify(res)

@app.route('/Procedure',methods=['GET'])
def algorithm():
    # 1
    template = render_template('procedure.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route('/Datasets',methods=['GET'])
def dataset():
    # 1
    template = render_template('dataset.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route('/Researcher',methods=['GET'])
def researcher():
    # 1
    template = render_template('researcher.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response
@app.route('/About',methods=['GET'])
def about():

    # 1
    template = render_template('about.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


