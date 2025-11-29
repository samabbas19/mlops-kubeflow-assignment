pipeline {
    agent any

    stages {
        // Stage 1: Environment Setup
        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                sh '''
                    python --version
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        // Stage 2: Pipeline Compilation
        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling pipeline...'
                sh '''
                    # Validate Python syntax
                    python -m py_compile src/pipeline_components.py
                    python -m py_compile src/model_training.py
                    python -m py_compile pipeline.py

                    # Try to compile Kubeflow pipeline if available
                    python pipeline.py --mode compile || echo "Kubeflow not available, using MLflow"
                '''
            }
        }

        // Stage 3: Pipeline Validation
        stage('Pipeline Validation') {
            steps {
                echo 'Validating pipeline components...'
                sh '''
                    # Check if required files exist
                    test -f src/pipeline_components.py
                    test -f src/model_training.py
                    test -f pipeline.py
                    test -f requirements.txt

                    echo "All required files are present"
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline compilation and validation completed successfully!'
        }
        failure {
            echo 'Pipeline compilation or validation failed.'
        }
        always {
            echo 'Cleaning up workspace...'
            cleanWs()
        }
    }
}
