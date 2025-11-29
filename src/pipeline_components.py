"""
Pipeline components for the MLOps pipeline
Can be used with both Kubeflow and MLflow
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json


def data_extraction(data_path: str, output_path: str) -> str:
    """
    Extract data from DVC tracked source

    Args:
        data_path: Path to the raw data file
        output_path: Path to save extracted data

    Returns:
        output_path: Path where data was saved
    """
    print(f"Extracting data from {data_path}")

    # Read the data
    df = pd.read_csv(data_path)

    # Save to output path
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Data extracted successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return output_path


def data_preprocessing(input_path: str, train_output: str, test_output: str,
                       test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Preprocess data: clean, scale, and split into train/test sets

    Args:
        input_path: Path to raw data
        train_output: Path to save training data
        test_output: Path to save test data
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        dict with paths to train and test data
    """
    print(f"Preprocessing data from {input_path}")

    # Load data
    df = pd.read_csv(input_path)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Handle missing values
    X = X.fillna(X.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Create output directories
    os.makedirs(os.path.dirname(train_output) if os.path.dirname(train_output) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(test_output) if os.path.dirname(test_output) else '.', exist_ok=True)

    # Save train data
    train_df = X_train_scaled.copy()
    train_df['target'] = y_train.values
    train_df.to_csv(train_output, index=False)

    # Save test data
    test_df = X_test_scaled.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(test_output, index=False)

    # Save scaler
    scaler_path = os.path.join(os.path.dirname(train_output), 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    return {
        'train_path': train_output,
        'test_path': test_output,
        'scaler_path': scaler_path
    }


def model_training(train_data_path: str, model_output_path: str,
                   n_estimators: int = 100, random_state: int = 42) -> str:
    """
    Train a Random Forest model

    Args:
        train_data_path: Path to training data
        model_output_path: Path to save trained model
        n_estimators: Number of trees in the forest
        random_state: Random seed

    Returns:
        model_output_path: Path where model was saved
    """
    print(f"Training model with data from {train_data_path}")

    # Load training data
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    print(f"Training Random Forest with {n_estimators} estimators...")
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else '.', exist_ok=True)
    joblib.dump(model, model_output_path)

    print(f"Model trained and saved to {model_output_path}")
    print(f"Feature importances: {dict(zip(X_train.columns, model.feature_importances_))}")

    return model_output_path


def model_evaluation(model_path: str, test_data_path: str, metrics_output_path: str) -> dict:
    """
    Evaluate the trained model on test data

    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        metrics_output_path: Path to save evaluation metrics

    Returns:
        dict with evaluation metrics
    """
    print(f"Evaluating model from {model_path}")

    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2_score': float(r2_score(y_test, y_pred))
    }

    # Save metrics
    os.makedirs(os.path.dirname(metrics_output_path) if os.path.dirname(metrics_output_path) else '.', exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R² Score: {metrics['r2_score']:.4f}")

    return metrics


# Kubeflow component wrappers (if using Kubeflow)
try:
    from kfp.dsl import component

    @component(
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn', 'joblib']
    )
    def kfp_data_extraction(data_path: str, output_path: str) -> str:
        return data_extraction(data_path, output_path)

    @component(
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn', 'joblib']
    )
    def kfp_data_preprocessing(input_path: str, train_output: str, test_output: str) -> dict:
        return data_preprocessing(input_path, train_output, test_output)

    @component(
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn', 'joblib']
    )
    def kfp_model_training(train_data_path: str, model_output_path: str) -> str:
        return model_training(train_data_path, model_output_path)

    @component(
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn', 'joblib', 'numpy']
    )
    def kfp_model_evaluation(model_path: str, test_data_path: str, metrics_output_path: str) -> dict:
        return model_evaluation(model_path, test_data_path, metrics_output_path)

except ImportError:
    print("Kubeflow Pipelines not installed. Skipping KFP component definitions.")
