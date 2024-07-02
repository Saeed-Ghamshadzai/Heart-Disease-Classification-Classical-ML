# FastAPI ML Model Deployment

This project deploys a machine learning model using FastAPI. The application provides a REST API to make predictions based on input data.

Link to the notebook on Kaggle: https://www.kaggle.com/code/saeedghamshadzai/heart-disease-classification-classical-ml

# Application setup commands

Run these commands to set up your virtual environment:

    Create virtual environment and activate it:
        python -m venv app/venv
        source app/venv/bin/activate     # On Windows use `app\venv\Scripts\activate` or `app\venv\Scripts\Activate.ps1`

    Install dependencies:
        pip install --upgrade pip
        pip install -r requirements.txt 

    Configure Environment Variables:
        cp app/.env.template app/.env
    
    Run the application:
        fastapi dev app/api/main.py
    Using docker:
        docker-compose -f docker-compose.yaml up --build

## Project Structure

```plaintext
Heart-Disease-Classification-Classical-ML/
│
├── app/
├── requirements.txt
├── dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .gitignore
└── README.md