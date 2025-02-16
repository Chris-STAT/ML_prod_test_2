import os
import json
import xgboost as xgb
import numpy as np

def model_fn(model_dir):
    """Load the model for inference"""
    model_path = os.path.join(model_dir, 'model')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'text/csv':
        # Parse the string into a numpy array
        data = np.array([float(x) for x in request_body.decode().split(',')])
        # Reshape for single sample
        data = data.reshape(1, -1)
        return xgb.DMatrix(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    return model.predict(input_data)

def output_fn(prediction, accept):
    """Format output"""
    return json.dumps(prediction.tolist())
