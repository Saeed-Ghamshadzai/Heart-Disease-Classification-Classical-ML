from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, ValidationError, conint, confloat
from enum import Enum
import logging
from dotenv import load_dotenv
import pandas as pd
import os
from data.data import preprocessor

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app/api/app.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
# Get the model version from the environment
model_version = os.getenv('MODEL_VERSION')

# Import the correct model version dynamically
try:
    model = __import__(f'model.{model_version}.model', fromlist=[''])

except ImportError as e:
    logging.error(f'Failed to import model version {model_version}: {e}', exc_info=True)
    raise RuntimeError(f'Model version {model_version} not found')

# Initialize the application
app = FastAPI()

# GEt the API key from the environment
API_KEY = os.getenv('API_KEY')
# Add API key header
api_key_header = APIKeyHeader(name='Authorization')

# API key validation
async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail='Could not validate credentials')
    return api_key

# Gender feature datatype for post endpoint
class Gender_type(str, Enum):
    male = 'Male'
    female = 'Female'

@app.get('/home_endpoint/')
async def home(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Endpoint to perform a health check and get the model version.
    """
    logging.info("Home endpoint called from %s", request.client.host)
    return {"health_check": "OK", "model_version": model.__version__}

@app.post("/classifier/")
async def get_classifier(
    Age: conint(ge=preprocessor.min_max['age'][0], le=preprocessor.min_max['age'][1]),
    Gender: Gender_type,
    Impluse: confloat(ge=preprocessor.min_max['impluse'][0], le=preprocessor.min_max['impluse'][1]),
    Pressure_Hight: confloat(ge=preprocessor.min_max['pressurehight'][0], le=preprocessor.min_max['pressurehight'][1]),
    Pressure_Low: confloat(ge=preprocessor.min_max['pressurelow'][0], le=preprocessor.min_max['pressurelow'][1]),
    Glucose: confloat(ge=preprocessor.min_max['glucose'][0], le=preprocessor.min_max['glucose'][1]),
    KCM: confloat(ge=preprocessor.min_max['kcm'][0], le=preprocessor.min_max['kcm'][1]),
    Troponin: confloat(ge=preprocessor.min_max['troponin'][0], le=preprocessor.min_max['troponin'][1]),

    request: Request,
    api_key: APIKey = Depends(get_api_key)
    ):
    """
    Endpoint to get a classification from the model based on input data
    """
    try:
        input_data = {
            'age': [Age],
            'gender': [1 if Gender.value == 'Male' else 0],
            'impluse': [Impluse],
            'pressurehight': [Pressure_Hight],
            'pressurelow': [Pressure_Low],
            'glucose': [Glucose],
            'kcm': [KCM],
            'troponin': [Troponin],
            'class': ['unknown']
        }

        # Make dataframe
        input_data = pd.DataFrame.from_dict(input_data)

        # Get the model
        classifier_model = model.load_model()
        # Classify the data
        classification = model.predictor(input_data, classifier_model)

        logging.info("Prediction made successfully")

        # JSON serializable the transformed data
        del input_data['class']
        input_data_serializable = input_data.applymap(lambda x: x.item() if hasattr(x, 'item') else x).to_dict(orient='list')

        return {
            "class": classification,
            "details": {
                "input_data": input_data_serializable,
                "model_version": model.__version__
            }
        }

    except ValidationError as e:
        logging.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid data")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unecpected error occured")

if __name__ == "__main__":
    import uvicorn
    import subprocess

    uvicorn.run(app, host="localhost", port=8080)