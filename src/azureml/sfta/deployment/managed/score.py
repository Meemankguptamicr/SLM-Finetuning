import os
!!!import onnx
from transformers import AutoTokenizer, AutoModel

def init():
    global model
    global tokenizer

    # Get the path where the model is stored
    model_path = os.getenv("AZUREML_MODEL_DIR")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

def run(data):
    try:
        # Assuming data is received as JSON with key 'input'
        if isinstance(data, str):
            import json
            data = json.loads(data)
        
        input_text = data.get('input', '')
        inputs = tokenizer.encode_plus(input_text, return_tensors="pt")
        outputs = model(**inputs)
        # Process outputs as needed
        result = outputs.last_hidden_state.tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}