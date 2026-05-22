# FEMA Disaster Recovery Cost Forecasting System

## Project Overview

This project predicts FEMA disaster recovery costs using Machine Learning and an XGBoost regression model.

The system includes:

- A FastAPI backend
- A Streamlit dashboard
- Scenario simulation
- Budget gap analysis
- Disaster severity classification
- Feature importance visualization

## Features

- Predict disaster recovery costs
- Compare recovery costs across different durations
- Budget risk analysis
- Interactive dashboard
- Dockerized deployment
- Machine Learning powered forecasting

## Tech Stack

- Python
- FastAPI
- Streamlit
- XGBoost
- Pandas
- Plotly
- Docker

## Project Structure

```text
DISASTER_COST_PREDICTION
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── xgboost_model.pkl
│   └── model_features.pkl
│
├── reports/
│   ├── xgboost_feature_importance.png
│   └── feature_importance.csv
│
├── src/
│   ├── 01_data_ingestion.py
│   ├── 02_check_loaded_data.py
│   ├── 03_data_cleaning.py
│   ├── 04_eda.py
│   ├── 05_feature_engineering.py
│   ├── 06_linear_regression.py
│   ├── 07_random_forest.py
│   ├── 08_xgboost.py
│   ├── 09_feature_importance.py
│   └── 10_save_model.py
│
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
```

Move into the project folder:

```bash
cd DISASTER_COST_PREDICTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Start Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

## Docker Deployment

Build and run containers:

```bash
docker compose up --build
```

Backend API:

```text
http://localhost:8000
```

Dashboard:

```text
http://localhost:8501
```

## Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.52 |
| MAE | 1.04 |
| RMSE | 1.44 |

Final selected model:
- XGBoost Regressor

## Dashboard Features

- Disaster recovery cost forecasting
- Budget gap analysis
- Disaster severity classification
- Scenario simulation
- Feature importance visualization
- Interactive Plotly analytics

## Future Improvements

- Real-time FEMA API integration
- Cloud deployment with AWS or Azure
- Advanced geospatial analytics
- Climate risk integration
- Economic and infrastructure indicators
- Historical disaster comparison analytics

## Live Application

### Streamlit Dashboard
https://fema-disaster-dashboard.onrender.com

### FastAPI Documentation
https://fema-disaster-cost-forecasting-system.onrender.com/docs

---

## Dashboard Preview

### Main Interface
<img width="1243" height="597" alt="image" src="https://github.com/user-attachments/assets/eed2412c-d298-4a7c-babd-8994397da9e5" />

<img width="767" height="635" alt="image" src="https://github.com/user-attachments/assets/71d92267-cd58-4d54-84bf-daa3db8fb208" />


![Dashboard](reports/dashboard_main.png)

### Prediction Results
<img width="1228" height="641" alt="image" src="https://github.com/user-attachments/assets/17dcd13a-7b25-4e3e-982e-d03b12bdc462" />

![Prediction](reports/prediction_results.png)

### API Documentation
<img width="817" height="640" alt="image" src="https://github.com/user-attachments/assets/05717368-1dac-4b80-9f9d-f5372a548120" />
<img width="745" height="149" alt="image" src="https://github.com/user-attachments/assets/ab0a776f-0c03-4616-b4ea-dae569525b5d" />
<img width="776" height="143" alt="image" src="https://github.com/user-attachments/assets/6b6bb203-de18-48d4-ada1-da5df8726059" />
<img width="827" height="149" alt="image" src="https://github.com/user-attachments/assets/d80024bb-3b13-45c2-b5c6-8d6049edbde2" />



![API Docs](reports/api_docs.png)

## Author

Iheoma Nwuzor

## License

This project is for educational and portfolio purposes.
