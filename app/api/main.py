'''
Test main file, chaking if the scripts work fine.
'''

import logging
from dotenv import load_dotenv
import pandas as pd
import os
from data.data import preprocessor, dataframe

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

data = pd.DataFrame({
    'age': [14],
    'gender': [1],
    'impluse': [77],
    'pressurehight': [160],
    'pressurelow': [83],
    'glucose': [100.0],
    'kcm': [2.0],
    'troponin': [0.01],
    'class': ['unknown']
})

classifier = model.load_model()

prediction = model.predictor(data, classifier)

print(prediction)