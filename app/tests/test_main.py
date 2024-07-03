from fastapi.testclient import TestClient
from api import main
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Get the model version from the environment
model_version = os.getenv('MODEL_VERSION', 'v1')

# Import the correct model version dynamically
model = __import__(f'model.{model_version}.model', fromlist=[''])

client = TestClient(main.app)

API_KEY = os.getenv('API_KEY')

def test_health_check():
    headers = {"Authorization": API_KEY}
    response = client.get("home_endpoint/", headers=headers)
    
    print(response.json())
    print('\n'*9)
    
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK", "model_version": model.__version__}

def test_classifier_endpoint():
    headers = {"Authorization": API_KEY}
    params = {
        'Age': [44],
        'Gender': ['Male'],
        'Impluse': [44],
        'Pressure_Hight': [44],
        'Pressure_Low': [44],
        'Glucose': [44],
        'KCM': [44],
        'Troponin': [4],
        'class': ['unknown']
    }
    response = client.post("/classifier/", params=params, headers=headers)
    
    assert response.status_code == 200
    assert "class" in response.json()
    assert "details" in response.json()