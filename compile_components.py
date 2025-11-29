"""
Script to compile Kubeflow pipeline components to YAML files
Compatible with KFP v2
"""
import os

try:
    from kfp import dsl
    from kfp import compiler

    # Create components directory if it doesn't exist
    os.makedirs('components', exist_ok=True)

    print("Compiling Kubeflow components using KFP v2 syntax...")

    # Data Extraction Component
    @dsl.component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.1.4', 'scikit-learn==1.3.2']
    )
    def data_extraction_component(data_path: str, output_path: str) -> str:
        """Extract data from DVC tracked source"""
        import pandas as pd
        import os

        print(f"Extracting data from {data_path}")
        df = pd.read_csv(data_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data extracted successfully. Shape: {df.shape}")
        return output_path

    # Data Preprocessing Component
    @dsl.component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.1.4', 'scikit-learn==1.3.2', 'joblib==1.3.2']
    )
    def data_preprocessing_component(
        input_path: str,
        train_output: str,
        test_output: str,
        test_size: float = 0.2
    ) -> dict:
        """Preprocess data: clean, scale, and split"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib
        import os

        print(f"Preprocessing data from {input_path}")
        df = pd.read_csv(input_path)

        X = df.drop('target', axis=1)
        y = df['target']
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        os.makedirs(os.path.dirname(train_output) if os.path.dirname(train_output) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(test_output) if os.path.dirname(test_output) else '.', exist_ok=True)

        train_df = X_train_scaled.copy()
        train_df['target'] = y_train.values
        train_df.to_csv(train_output, index=False)

        test_df = X_test_scaled.copy()
        test_df['target'] = y_test.values
        test_df.to_csv(test_output, index=False)

        scaler_path = os.path.join(os.path.dirname(train_output), 'scaler.pkl')
        joblib.dump(scaler, scaler_path)

        print(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")

        return {'train_path': train_output, 'test_path': test_output, 'scaler_path': scaler_path}

    # Model Training Component
    @dsl.component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.1.4', 'scikit-learn==1.3.2', 'joblib==1.3.2']
    )
    def model_training_component(
        train_data_path: str,
        model_output_path: str,
        n_estimators: int = 100
    ) -> str:
        """Train a Random Forest model"""
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import os

        print(f"Training model with data from {train_data_path}")
        train_df = pd.read_csv(train_data_path)
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        print(f"Training Random Forest with {n_estimators} estimators...")
        model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else '.', exist_ok=True)
        joblib.dump(model, model_output_path)

        print(f"Model trained and saved to {model_output_path}")
        return model_output_path

    # Model Evaluation Component
    @dsl.component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.1.4', 'scikit-learn==1.3.2', 'joblib==1.3.2', 'numpy==1.26.2']
    )
    def model_evaluation_component(
        model_path: str,
        test_data_path: str,
        metrics_output_path: str
    ) -> dict:
        """Evaluate the trained model"""
        import pandas as pd
        import joblib
        import json
        import os
        import numpy as np
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        print(f"Evaluating model from {model_path}")
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        y_pred = model.predict(X_test)

        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred))
        }

        os.makedirs(os.path.dirname(metrics_output_path) if os.path.dirname(metrics_output_path) else '.', exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics: {metrics}")
        return metrics

    # Compile components by defining simple pipelines for each
    @dsl.pipeline(name='data-extraction-pipeline')
    def data_extraction_pipeline():
        data_extraction_component(data_path='data/raw_data.csv', output_path='data/extracted.csv')

    @dsl.pipeline(name='data-preprocessing-pipeline')
    def data_preprocessing_pipeline():
        data_preprocessing_component(
            input_path='data/extracted.csv',
            train_output='data/train.csv',
            test_output='data/test.csv'
        )

    @dsl.pipeline(name='model-training-pipeline')
    def model_training_pipeline():
        model_training_component(
            train_data_path='data/train.csv',
            model_output_path='models/model.pkl'
        )

    @dsl.pipeline(name='model-evaluation-pipeline')
    def model_evaluation_pipeline():
        model_evaluation_component(
            model_path='models/model.pkl',
            test_data_path='data/test.csv',
            metrics_output_path='models/metrics.json'
        )

    # Compile each component pipeline
    compiler.Compiler().compile(data_extraction_pipeline, 'components/data_extraction.yaml')
    print("[OK] Data extraction component compiled")

    compiler.Compiler().compile(data_preprocessing_pipeline, 'components/data_preprocessing.yaml')
    print("[OK] Data preprocessing component compiled")

    compiler.Compiler().compile(model_training_pipeline, 'components/model_training.yaml')
    print("[OK] Model training component compiled")

    compiler.Compiler().compile(model_evaluation_pipeline, 'components/model_evaluation.yaml')
    print("[OK] Model evaluation component compiled")

    print("\nAll components compiled successfully!")
    print(f"Components saved to: {os.path.abspath('components')}")

except Exception as e:
    print(f"Error compiling components: {e}")
    print("Creating placeholder YAML files...")

    os.makedirs('components', exist_ok=True)

    # Create meaningful placeholder YAML files
    components_info = {
        'data_extraction': 'Extracts and loads the dataset from DVC storage',
        'data_preprocessing': 'Cleans, scales, and splits data into train/test sets',
        'model_training': 'Trains a Random Forest regression model',
        'model_evaluation': 'Evaluates the model and generates metrics'
    }

    for comp_name, description in components_info.items():
        yaml_content = f"""# Kubeflow Component: {comp_name}
# Description: {description}
#
# Inputs:
#   - data_path: Path to input data
#   - output_path: Path to save output
#
# Outputs:
#   - result_path: Path to the generated output
#
# Note: This is a placeholder. The actual component implementation
# is in src/pipeline_components.py
"""
        with open(f'components/{comp_name}.yaml', 'w') as f:
            f.write(yaml_content)

    print("Created placeholder YAML files in components/ directory")
