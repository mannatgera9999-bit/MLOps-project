# MP150: Power Loom Output Prediction

## 1. Project Overview

This project develops an end-to-end Machine Learning Operations (MLOps) pipeline to predict the metres of fabric produced by a power loom during a shift.

The project is based on the MP150 Power Loom Output problem for the Patna Weavers' Cluster.

The objective is to build a reproducible machine learning workflow that includes data validation, preprocessing, model training, experiment tracking, model registration, prediction serving, containerization, continuous integration, Kubernetes deployment, and model monitoring.

**Problem ID:** MP150  
**Group ID:** G206  
**Target variable:** `metres_per_shift`  
**ML task:** Regression  
**Evaluation metric:** Mean Absolute Error (MAE)

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Experiment tracking | MLflow |
| Data version control | DVC |
| API development | FastAPI |
| Input validation | Pydantic |
| Model serialization | Joblib |
| Testing | Pytest, FastAPI TestClient |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Orchestration | Kubernetes |
| Configuration | YAML |

## 3. Repository Structure

```text
MLOps-project/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc
│   │   └── later.csv.dvc
│   └── processed/
│
├── src/
│   └── mp150/
│       ├── data_validation.py
│       ├── data_cleaning.py
│       ├── model_training.py
│       ├── monitoring.py
│       ├── retraining.py
│       └── predict_holdout.py
│
├── tests/
│   └── test_api.py
│
├── models/
│   └── model.joblib
│
├── predictions/
│   └── holdout_predictions.csv
│
├── reports/
│   └── findings.yaml
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── .dvc/
├── params.yaml
├── requirements.txt
├── mlflow.db
└── README.md
```

## 4. Environment Setup

Clone the repository:

```bash
git clone https://github.com/mannatgera9999-bit/MLOps-project.git
cd MLOps-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Dataset Management Using DVC

The project uses DVC to track the training and later datasets.

Check DVC status:

```bash
dvc status
```

Retrieve tracked data when the DVC cache or a configured remote is available:

```bash
dvc pull
```

The training and later datasets are represented by DVC pointer files in the repository. The holdout features file is maintained separately.

**Note:** The current repository does not have a configured DVC remote. The actual training and later datasets must therefore be available through the local DVC cache or another supplied copy before running the pipeline on a different machine.

## 6. Data Validation and Preprocessing

The data validation module checks the datasets for potential quality problems, including:

- Data leakage
- Unit inconsistencies
- Sentinel values
- Duplicate records
- Stuck sensor readings
- Target-value anomalies
- Category spelling variations

Run validation:

```bash
python -m src.mp150.data_validation
```

Run preprocessing:

```bash
python -m src.mp150.data_cleaning
```

The cleaning module generates processed datasets while preserving the original raw files.

## 7. Model Training and MLflow Tracking

The project evaluates machine learning models for regression using a training and validation split.

Run model training:

```bash
python -m src.mp150.model_training
```

MLflow is configured to use a local SQLite tracking database.

The experiments record model parameters, validation MAE, training-data SHA-256 information, and model artifacts.

The selected model is registered under:

`MP150_PowerLoom_Model`

The selected model is identified through the `champion` alias.

To launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

`http://127.0.0.1:5000`

## 8. Model Monitoring and Retraining

Run monitoring:

```bash
python -m src.mp150.monitoring
```

The monitoring module evaluates Population Stability Index (PSI) for numeric input features and calculates prediction errors across shed groups.

Run the retraining workflow:

```bash
python -m src.mp150.retraining
```

The retraining process uses the first 70% of later-period data for training and the remaining 30% for evaluation.

The new model is promoted only when its MAE is at least 12% lower than the previous model's MAE on the evaluation portion.

## 9. Holdout Prediction

Generate predictions using the registered champion model:

```bash
python -m src.mp150.predict_holdout
```

The output is saved to:

`predictions/holdout_predictions.csv`

The output contains exactly two columns:

- `record_id`
- `prediction`

## 10. FastAPI Model Serving

Start the API:

```bash
uvicorn app.main:app --reload
```

The application runs at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks application health |
| POST | `/predict` | Predicts power loom output |

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Prediction request

Example request body:

```json
{
  "loom_age_yr": 10,
  "thread_count": 40,
  "power_cuts_min": 20,
  "weaver_exp_yr": 8,
  "humidity_pct": 60,
  "shift_hours": 8,
  "fabric": "cotton",
  "shed": "Shed 1"
}
```

The response contains the predicted production in metres per shift.

### Request validation

The API applies the following mandatory client-side rules:

- `loom_age_yr` must not exceed 55.
- `thread_count` must be at least 20.

Invalid requests are rejected with HTTP 422 responses.

Prediction events are recorded using structured JSON logging.

## 11. Automated Testing

Run the test suite:

```bash
python -m pytest -v
```

The API tests cover:

- Health endpoint response
- Valid prediction request
- Invalid loom age
- Invalid thread count

## 12. Docker Deployment

Build the Docker image:

```bash
docker build -t mp150-api:1.0 .
```

Run the container:

```bash
docker run -d --name mp150-container -p 8000:8000 mp150-api:1.0
```

Check running containers:

```bash
docker ps
```

Check container logs:

```bash
docker logs mp150-container
```

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

## 13. Kubernetes Deployment

The Kubernetes configuration is available in the `k8s/` directory.

Apply the deployment:

```bash
kubectl apply -f k8s/deployment.yaml
```

Apply the service:

```bash
kubectl apply -f k8s/service.yaml
```

Check deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

The deployment uses two application replicas.

The Service exposes port 80 and forwards requests to application port 8000.

Readiness and liveness probes use the `/health` endpoint.

### Access the application

Forward the Kubernetes Service:

```bash
kubectl port-forward service/mp150-svc 8080:80
```

Test the application:

```bash
curl http://127.0.0.1:8080/health
```

## 14. Continuous Integration

GitHub Actions is configured in:

`.github/workflows/ci.yml`

The workflow executes when code is pushed to or a pull request is opened against the `main` branch.

The automated pipeline:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs project dependencies.
4. Runs the pytest test suite.

Successful workflow execution verifies that the automated tests pass in the GitHub Actions environment.

## 15. Team Contributions

The project work is divided between two participating members.

| Member | Contribution | Share |
|---|---|---:|
| Mannat Gera | Data validation, DVC, preprocessing, data-quality analysis, model development, MLflow tracking, model registration, monitoring, retraining and holdout predictions | 60% |
| Payal Kaushik | FastAPI model serving, request validation, API testing, Docker configuration, GitHub Actions CI, Kubernetes deployment, structured logging and documentation | 40% |

Both members are responsible for understanding the complete MLOps pipeline and maintaining their individual GitHub commit histories.

## 16. Project Repository

GitHub repository:

https://github.com/mannatgera9999-bit/MLOps-project

## 17. Important Notes

- Keep original raw datasets unchanged.
- Use the supplied dataset associated with problem MP150.
- Maintain the DVC tracking files and MLflow database.
- Ensure the registered champion model is available before starting model serving.
- Keep the Docker image and Kubernetes configuration synchronized.
- Run the tests after making code changes.
- Do not commit virtual environments, temporary files or unnecessary generated artifacts.
