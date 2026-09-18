from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# helper to load pickle if present
def try_load(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# Attempt to load artifacts (place them next to this file)
MODEL_PATH = 'rf_model.pkl'
FEATURES_PATH = 'feature_names.pkl'
ONEHOT_META = 'onehot_metadata.pkl'
IMPUTERS = 'imputers.pkl'
SCALERS = 'scalers.pkl'
PREPROCESSOR = 'preprocessor.pkl'

model = try_load(MODEL_PATH)
feature_names = try_load(FEATURES_PATH) or []
onehot_meta = try_load(ONEHOT_META) or {}
imputers = try_load(IMPUTERS) or {}
scalers = try_load(SCALERS) or {}
preprocessor = try_load(PREPROCESSOR)


def build_features(input_json):
    # If a full preprocessor exists, use it
    if preprocessor is not None:
        # Expect input_json to contain all original feature columns
        df = pd.DataFrame([input_json])
        arr = preprocessor.transform(df)
        return arr

    # Otherwise, reconstruct using saved metadata
    df = pd.DataFrame([input_json])

    # One-hot via metadata (get_dummies with train columns)
    for col, dummy_cols in onehot_meta.items():
        # ensure column present
        df[col] = df.get(col, '').astype(str)
        dummies = pd.get_dummies(df[col], prefix=col)
        # align to dummy_cols (train columns)
        dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # For numeric columns, apply imputer then scaler per column
    for col, imputer in (imputers or {}).items():
        if col not in df.columns:
            df[col] = np.nan
        arr = imputer.transform(df[[col]])
        scaler = scalers.get(col)
        if scaler is not None:
            arr = scaler.transform(arr)
        df[col] = arr.flatten()

    # Ensure final feature order matches feature_names if available
    if feature_names:
        for fn in feature_names:
            if fn not in df.columns:
                df[fn] = 0
        df = df[feature_names]

    return df.values


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file rf_model.pkl not found on server.'}), 500
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({'error': 'Invalid JSON payload'}), 400
        X = build_features(payload)
        preds = model.predict(X)
        out = float(preds[0])
        return jsonify({'prediction': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/', methods=['GET'])
def index():
    # simple form page
    return render_template('index.html')


@app.route('/predict_form', methods=['POST'])
def predict_form():
    # Read form fields and build payload
    try:
        fields = ['company_type','employees','revenue_year','net_income','net_income_year','market_cap','direct_subsidiaries_count']
        payload = {}
        for f in fields:
            v = request.form.get(f)
            # convert numeric-like fields to float if possible
            if v is None or v == '':
                payload[f] = None
                continue
            if f in ['employees','revenue_year','net_income','net_income_year','market_cap','direct_subsidiaries_count']:
                try:
                    payload[f] = float(v)
                except:
                    payload[f] = None
            else:
                payload[f] = v

        # build features and predict
        if model is None:
            return render_template('index.html', error='Model not found on server.')
        X = build_features(payload)
        pred = model.predict(X)[0]
        return render_template('index.html', prediction=float(pred))
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    # Allow configuring host/port/debug via environment variables
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('1', 'true', 'yes')
    print(f'Starting app on http://{host}:{port}  (debug={debug})')
    app.run(host=host, port=port, debug=debug)
