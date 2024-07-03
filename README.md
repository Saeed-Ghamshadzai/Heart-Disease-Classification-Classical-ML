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
        pip install -e app\.

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
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.log
│   │   └── main.py
|   |
│   ├── venv/
|   |
│   ├── model/
│   │   ├── __init__.py
│   │   └── v1/
│   │      ├── __init__.py
│   │      ├── model.py
│   │      └── trained_model-0.1.0.pkl
|   |
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_main.py
|   |
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── csv_dataset/
│   │       └── Heart Attack.csv
|   |
│   ├── .env.template -> .env
│   ├── pytest.ini
│   ├── Notebook.ipynb
|   └── setup.py
|
├── requirements.txt
├── dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .gitignore
└── README.md