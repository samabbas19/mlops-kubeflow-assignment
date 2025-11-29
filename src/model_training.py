"""
Standalone model training script
"""
import argparse
from pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data-path', default='data/raw_data.csv', help='Path to raw data')
    parser.add_argument('--model-output', default='models/model.pkl', help='Path to save model')
    parser.add_argument('--metrics-output', default='models/metrics.json', help='Path to save metrics')

    args = parser.parse_args()

    print("=" * 50)
    print("Starting ML Training Pipeline")
    print("=" * 50)

    # Step 1: Data Extraction
    print("\n[1/4] Data Extraction")
    extracted_data = data_extraction(args.data_path, 'data/extracted_data.csv')

    # Step 2: Data Preprocessing
    print("\n[2/4] Data Preprocessing")
    preprocessing_result = data_preprocessing(
        extracted_data,
        'data/train_data.csv',
        'data/test_data.csv'
    )

    # Step 3: Model Training
    print("\n[3/4] Model Training")
    model_path = model_training(
        preprocessing_result['train_path'],
        args.model_output
    )

    # Step 4: Model Evaluation
    print("\n[4/4] Model Evaluation")
    metrics = model_evaluation(
        model_path,
        preprocessing_result['test_path'],
        args.metrics_output
    )

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)
    print(f"\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
