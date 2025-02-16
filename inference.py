import os
import json
import xgboost as xgb

def model_fn(model_dir):
    """Load the model for inference"""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'model'))
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'text/csv':
        # Parse CSV
        return xgb.DMatrix(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    return model.predict(input_data)

def output_fn(prediction, accept):
    """Format output"""
    return json.dumps(prediction.tolist())
