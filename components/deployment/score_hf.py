import json
import mlflow
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")  # Assuming the model is in the "model" folder
    model = mlflow.pyfunc.load_model(model_path)

def run(data):
    try:
        input_data = json.loads(data)
        result = model.predict(input_data)
        return result.tolist()  # Convert to a serializable format
    except Exception as e:
        return json.dumps({"error": str(e)})