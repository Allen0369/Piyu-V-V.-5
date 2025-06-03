from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import uuid, os, json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50*1024*1024 # Can handle up to 50 mb
TEMP_RESULTS_DIR = "temp_results"
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

lstm_model = keras.models.load_model('models/model_LSTM.keras')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    file_path = os.path.join(TEMP_RESULTS_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "Session ID not found"}), 404

    with open(file_path, "r") as f:
        result = json.load(f)

    return jsonify(result)
    
@app.route('/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist("files")
    
    if not uploaded_files:
        return jsonify({"error": "No files received"}), 400

    try:
        dfs = []

        for file in uploaded_files:
            filename = file.filename
            
            df = pd.read_csv(file)
            
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        
        model_instance = modelLSTM(combined_df, lstm_model)
        restructured_df = model_instance.restructureData()
        X, y = model_instance.assignFeatureTarget(restructured_df)
        X_scaled, y_scaled = model_instance.scaleValues(X, y)
        y_pred_scaled = model_instance.predictLSTM(X_scaled)
        y_pred, y_true = model_instance.scaleInverseTransform(y_pred_scaled, y_scaled)

        restructured_df['prediction'] = y_pred.flatten()
        
        prediction_lstm = restructured_df[['date', 'PULocationID', 'day_of_week', 'log_passenger_count_lag7', 'pct_change', 'prediction']].to_dict(orient='records')

        initial = combined_df.groupby('day_of_week').agg({
            'trip_distance': ['count', 'mean'],
            'passenger_count': 'sum',
            'trip_duration': 'mean'
        }).round(2)

        initial.columns = ['_'.join(col).strip() for col in initial.columns.values]
        initial = initial.reset_index()

        initial_data = initial.to_dict(orient='records')
        
        result = {
            "status": "success",
            "files_processed": [f.filename for f in uploaded_files],
            "lstm_results": prediction_lstm,
            "initial": initial_data
        }
        
        session_id = str(uuid.uuid4())
        with open(f"{TEMP_RESULTS_DIR}/{session_id}.json", "w") as f:
            json.dump(result, f)

        return jsonify({"session_id": session_id})

    except Exception as e:
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500

class modelLSTM:
    def __init__(self, df, model_1):
        self.df = df
        self.scaler_y = None
        self.model_1 = model_1

    def restructureData(self):
        res_df = self.df.groupby(['date', 'PULocationID', 'day_of_week'])['passenger_count'].sum().reset_index()
        res_df = res_df.sort_values(['PULocationID','day_of_week','date'])
        res_df['passenger_count_lag7'] = res_df.groupby(['PULocationID', 'day_of_week'])['passenger_count'].shift(1)
        res_df['pct_change'] = (res_df['passenger_count'] - res_df['passenger_count_lag7']) / (res_df['passenger_count_lag7'] + 1e-6)
        res_df['log_passenger_count'] = np.log1p(res_df['passenger_count'])
        res_df['log_passenger_count_lag7'] = np.log1p(res_df['passenger_count_lag7'])

        return res_df.dropna()

    def assignFeatureTarget(self, res_df):
        X = res_df[['log_passenger_count_lag7', 'pct_change']].values
        y = res_df['log_passenger_count'].values.reshape(-1, 1)
        return X, y

    def scaleValues(self, X, y):
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler_y = scaler_y
        self.scaler_X = scaler_X

        return X_scaled, y_scaled

    def predictLSTM(self, X_scaled):
        y_pred_scaled = self.model_1.predict(X_scaled)
        return y_pred_scaled

    def scaleInverseTransform(self, y_pred_scaled, y_scaled):
        y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.expm1(y_pred_log)

        y_true = np.expm1(self.scaler_y.inverse_transform(y_scaled))
        return y_pred, y_true

if __name__ == "__main__":
    app.run(debug=True)