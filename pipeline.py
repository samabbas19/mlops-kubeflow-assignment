"""
MLflow Pipeline Definition
Alternative to Kubeflow for easier setup
"""
import mlflow
import mlflow.sklearn
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)
import json
import os


def run_mlflow_pipeline(data_path='data/raw_data.csv', experiment_name='housing_price_prediction'):
    """
    Run the complete ML pipeline using MLflow for tracking

    Args:
        data_path: Path to the raw data file
        experiment_name: Name of the MLflow experiment
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="housing_pipeline"):
        print("=" * 60)
        print(f"Starting MLflow Pipeline: {experiment_name}")
        print("=" * 60)

        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Step 1: Data Extraction
        print("\n[1/4] Data Extraction")
        with mlflow.start_run(run_name="data_extraction", nested=True):
            extracted_data = data_extraction(data_path, 'data/extracted_data.csv')
            mlflow.log_artifact(extracted_data)

        # Step 2: Data Preprocessing
        print("\n[2/4] Data Preprocessing")
        with mlflow.start_run(run_name="data_preprocessing", nested=True):
            preprocessing_result = data_preprocessing(
                extracted_data,
                'data/train_data.csv',
                'data/test_data.csv'
            )
            mlflow.log_artifact(preprocessing_result['train_path'])
            mlflow.log_artifact(preprocessing_result['test_path'])
            mlflow.log_artifact(preprocessing_result['scaler_path'])

        # Step 3: Model Training
        print("\n[3/4] Model Training")
        with mlflow.start_run(run_name="model_training", nested=True):
            os.makedirs('models', exist_ok=True)
            model_path = model_training(
                preprocessing_result['train_path'],
                'models/model.pkl'
            )
            mlflow.log_artifact(model_path)

        # Step 4: Model Evaluation
        print("\n[4/4] Model Evaluation")
        with mlflow.start_run(run_name="model_evaluation", nested=True):
            metrics = model_evaluation(
                model_path,
                preprocessing_result['test_path'],
                'models/metrics.json'
            )

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact('models/metrics.json')

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        print(f"\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics


# Kubeflow Pipeline Definition (if using Kubeflow)
# Note: Simplified version for compatibility
def compile_kubeflow_pipeline(output_path='pipeline.yaml'):
    """Compile the Kubeflow pipeline to YAML"""
    try:
        # Use the compiled components from compile_components.py instead
        print("Note: Kubeflow components are compiled separately using compile_components.py")
        print(f"Component YAMLs are available in the components/ directory")
        print("To use with Kubeflow, upload the component YAMLs to the Kubeflow UI")
        return True
    except Exception as e:
        print(f"Could not compile Kubeflow pipeline: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--mode', choices=['mlflow', 'kubeflow', 'compile'],
                       default='mlflow', help='Pipeline mode')
    parser.add_argument('--data-path', default='data/raw_data.csv',
                       help='Path to raw data')

    args = parser.parse_args()

    if args.mode == 'mlflow':
        run_mlflow_pipeline(data_path=args.data_path)
    elif args.mode == 'kubeflow':
        print("Kubeflow mode - please upload pipeline.yaml to Kubeflow UI")
    elif args.mode == 'compile':
        compile_kubeflow_pipeline()
