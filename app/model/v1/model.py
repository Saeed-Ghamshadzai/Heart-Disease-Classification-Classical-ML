import joblib
import os
from pathlib import Path
from data.data import preprocessor
from dotenv import load_dotenv
import sklearn
import logging

__version__ = "0.1.0"

# Load environment variables
load_dotenv()
# # Get the model version from the environment
path_to_model = os.getenv('PATH_TO_MODEL')

def load_model():
    """
    Load the trained machine learning model.

    Returns:
        object: Loaded machine learning model.

    Raises:
        RuntimeError: If loading the model fails.
    """
    try:
        with open(f"{path_to_model}/trained_model-{__version__}.pkl", "rb") as f:
            model = joblib.load(f)
            logging.info("Model loaded successfully")
            return model

    except Exception as e:
        logging.error(f'Faild to load model: {e}')
        raise RuntimeError('Model loading failed')


def predictor(input_data, classifier):
    """
    Make predictions using the provided input data and classifier.

    Args:
        input_data (DataFrame): Input data for prediction.
        classifier (object): Machine learning classifier object.

    Returns:
        ndarray: Predicted classes.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        processed_data = preprocessor.transform(input_data, drop_target=True, scale=True)
        
        prediction = classifier.predict(processed_data.values)
        output = 'Positive' if prediction == 1 else 'Negative'

        return output

    except Exception as e:
        logging.error(f'Prediction failed: {e}')
        raise RuntimeError(f'Prediction failed: {e}')