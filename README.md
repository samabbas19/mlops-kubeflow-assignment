# MLOps Pipeline with DVC, MLflow/Kubeflow, and CI/CD

## Project Overview

This project demonstrates a complete Machine Learning Operations (MLOps) pipeline for housing price prediction. The pipeline implements industry-standard practices including:

- **Data Versioning** with DVC (Data Version Control)
- **Pipeline Orchestration** with MLflow (alternative to Kubeflow for easier setup)
- **Continuous Integration** with GitHub Actions/Jenkins
- **Model Training** using Random Forest Regression
- **Automated Testing** and validation

### ML Problem
Predicting median housing prices using the California Housing dataset (alternative to the deprecated Boston Housing dataset). The model achieves an R² score of approximately 0.80, indicating good predictive performance.

## Repository Structure

```
mlops-kubeflow-assignment/
├── data/                          # Data directory (tracked by DVC)
│   ├── raw_data.csv              # Original dataset (tracked by DVC)
│   ├── extracted_data.csv        # Extracted data
│   ├── train_data.csv            # Training set
│   └── test_data.csv             # Test set
├── src/                           # Source code
│   ├── pipeline_components.py    # Pipeline component functions
│   ├── model_training.py         # Standalone training script
│   └── generate_data.py          # Dataset generation script
├── components/                    # Compiled Kubeflow components
│   ├── data_extraction.yaml      # Data extraction component
│   ├── data_preprocessing.yaml   # Data preprocessing component
│   ├── model_training.yaml       # Model training component
│   └── model_evaluation.yaml     # Model evaluation component
├── models/                        # Trained models and artifacts
│   ├── model.pkl                 # Trained Random Forest model
│   ├── metrics.json              # Evaluation metrics
│   └── scaler.pkl                # Feature scaler
├── mlruns/                        # MLflow tracking directory
├── .github/workflows/             # GitHub Actions workflows
│   └── ml-pipeline.yml           # CI/CD pipeline
├── pipeline.py                    # Main MLflow pipeline definition
├── compile_components.py          # Script to compile Kubeflow components
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container definition
├── Jenkinsfile                    # Jenkins CI/CD pipeline
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized deployment)
- Minikube and kubectl (optional, for Kubeflow deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize DVC and Pull Data

```bash
# Initialize DVC (if not already initialized)
dvc init

# Configure DVC remote storage
# Option 1: Local directory
dvc remote add -d myremote ../dvc-storage

# Option 2: Google Drive
# dvc remote add -d myremote gdrive://<folder-id>

# Option 3: AWS S3
# dvc remote add -d myremote s3://mybucket/path

# Pull the data
dvc pull
```

### 4. Generate Dataset (if needed)

```bash
python src/generate_data.py
```

## Running the Pipeline

### Option 1: Using MLflow (Recommended)

MLflow is easier to set up and doesn't require Kubernetes.

```bash
# Run the complete pipeline
python pipeline.py --mode mlflow

# View MLflow UI
mlflow ui
# Then open http://localhost:5000 in your browser
```

### Option 2: Using Kubeflow Pipelines

#### Setup Minikube and Kubeflow

```bash
# Start Minikube
minikube start --cpus 4 --memory 8192 --disk-size=40g

# Install Kubeflow Pipelines standalone
export PIPELINE_VERSION=2.0.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

# Port-forward to access the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

#### Compile and Run Pipeline

```bash
# Compile Kubeflow components
python compile_components.py

# The components are now in the components/ directory
# Upload them to the Kubeflow UI at http://localhost:8080
```

### Option 3: Using Standalone Script

```bash
# Run the training pipeline without MLflow/Kubeflow
python src/model_training.py
```

## Pipeline Walkthrough

### 1. Data Extraction

**File**: `src/pipeline_components.py::data_extraction()`

- Loads the raw dataset from DVC-tracked storage
- Validates data integrity
- Outputs extracted data for preprocessing

**Input**: `data/raw_data.csv` (DVC tracked)
**Output**: `data/extracted_data.csv`

### 2. Data Preprocessing

**File**: `src/pipeline_components.py::data_preprocessing()`

- Handles missing values using mean imputation
- Splits data into training (80%) and test (20%) sets
- Applies StandardScaler for feature normalization
- Saves the scaler for future inference

**Input**: `data/extracted_data.csv`
**Output**:
- `data/train_data.csv`
- `data/test_data.csv`
- `models/scaler.pkl`

### 3. Model Training

**File**: `src/pipeline_components.py::model_training()`

- Trains a Random Forest Regressor with 100 estimators
- Uses scikit-learn's RandomForestRegressor
- Saves the trained model as a pickle file

**Input**: `data/train_data.csv`
**Output**: `models/model.pkl`

**Hyperparameters**:
- `n_estimators`: 100
- `random_state`: 42
- `n_jobs`: -1 (use all CPU cores)

### 4. Model Evaluation

**File**: `src/pipeline_components.py::model_evaluation()`

- Evaluates model on test set
- Calculates metrics:
  - **MSE** (Mean Squared Error)
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R² Score** (Coefficient of Determination)
- Saves metrics to JSON file

**Input**:
- `models/model.pkl`
- `data/test_data.csv`

**Output**: `models/metrics.json`

**Expected Performance**:
- R² Score: ~0.80
- RMSE: ~0.50
- MAE: ~0.33

## Continuous Integration

### GitHub Actions

The project includes a GitHub Actions workflow that automatically:

1. **Environment Setup**: Installs Python and dependencies
2. **Pipeline Compilation**: Compiles and validates the pipeline
3. **Syntax Validation**: Checks Python syntax
4. **Artifact Upload**: Saves compiled components

**File**: `.github/workflows/ml-pipeline.yml`

**Trigger**: Push or Pull Request to main/master branch

### Jenkins

Alternative CI/CD pipeline using Jenkins:

**File**: `Jenkinsfile`

**Stages**:
1. **Environment Setup**: Install dependencies
2. **Pipeline Compilation**: Compile and validate pipeline
3. **Pipeline Validation**: Check file structure and syntax

**Setup**:
```bash
# Create a new Pipeline job in Jenkins
# Point it to your GitHub repository
# Set the script path to: Jenkinsfile
```

## DVC Workflow

### Track New Data

```bash
# Add new data file
dvc add data/new_data.csv

# Commit the .dvc file
git add data/new_data.csv.dvc
git commit -m "Track new data"

# Push data to remote storage
dvc push
```

### Pull Data from Remote

```bash
# Pull all tracked data
dvc pull

# Pull specific file
dvc pull data/raw_data.csv.dvc
```

### Check DVC Status

```bash
dvc status
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t mlops-pipeline:latest .
```

### Run Pipeline in Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models mlops-pipeline:latest
```

## Model Training Component Details

### Inputs and Outputs

**Data Training Component**:

**Inputs**:
- `train_data_path` (str): Path to the training dataset CSV file
- `model_output_path` (str): Path where the trained model should be saved
- `n_estimators` (int, default=100): Number of trees in the Random Forest
- `random_state` (int, default=42): Random seed for reproducibility

**Outputs**:
- `model_output_path` (str): Path to the saved model file (`.pkl` format)

**Process**:
1. Loads training data from CSV
2. Separates features (X) from target variable (y)
3. Initializes RandomForestRegressor with specified hyperparameters
4. Trains the model using `fit(X_train, y_train)`
5. Serializes and saves the model using `joblib.dump()`

## Monitoring and Logging

### MLflow Tracking

All pipeline runs are tracked in MLflow:

- **Experiment Name**: `housing_price_prediction`
- **Tracked Metrics**: MSE, RMSE, MAE, R² Score
- **Tracked Parameters**: test_size, n_estimators, random_state
- **Artifacts**: model, scaler, datasets

View experiments:
```bash
mlflow ui
# Open http://localhost:5000
```

## Troubleshooting

### Issue: DVC push fails

**Solution**: Check your remote storage configuration
```bash
dvc remote list
dvc remote modify myremote --local url /path/to/storage
```

### Issue: MLflow won't start

**Solution**: Clear the MLflow directory and restart
```bash
rm -rf mlruns/
python pipeline.py --mode mlflow
```

### Issue: Kubeflow connection refused

**Solution**: Check if port-forwarding is active
```bash
kubectl get pods -n kubeflow
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

## Performance Metrics

Current model performance on California Housing dataset:

| Metric | Value |
|--------|-------|
| R² Score | 0.8053 |
| RMSE | 0.5051 |
| MAE | 0.3274 |
| MSE | 0.2552 |

## Future Improvements

1. **Hyperparameter Tuning**: Implement GridSearch or RandomSearch
2. **Model Registry**: Add MLflow Model Registry for version control
3. **A/B Testing**: Deploy multiple model versions
4. **Real-time Inference**: Add Flask/FastAPI API endpoint
5. **Monitoring**: Add model drift detection
6. **Advanced Preprocessing**: Feature engineering and selection

## License

This project is for educational purposes as part of the Cloud MLOps course.

## Author

[Your Name] - [Your Student ID]

## Acknowledgments

- Scikit-learn for ML algorithms
- MLflow for experiment tracking
- DVC for data versioning
- Kubeflow for pipeline orchestration
