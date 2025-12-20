pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        MLFLOW_TRACKING_URI = "https://dagshub.com/salsabil812/llm_mlops1.mlflow"
        PYTHONUTF8 = "1"
        PIP_DISABLE_PIP_VERSION_CHECK = "1"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Python & venv') {
            steps {
                sh '''
                    set -e
                    python3 --version

                    if [ ! -d ".venv" ]; then
                        python3 -m venv .venv
                    fi

                    # Activer le virtualenv
                    . .venv/bin/activate

                    # Upgrade pip
                    python -m pip install --upgrade pip

                    # Installer toutes les dépendances
                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    else
                        pip install "dvc[s3]" mlflow pandas numpy scikit-learn torch transformers sentence-transformers
                    fi
                '''
            }
        }

        stage('Load credentials') {
          steps {
            withCredentials([
              string(credentialsId: 'DAGSHUB_USER', variable: 'DAGSHUB_USER'),
              string(credentialsId: 'DAGSHUB_TOKEN', variable: 'DAGSHUB_TOKEN')
            ]) {
              sh '''
                export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USER
                export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
        
                export DVC_HTTP_USER=$DAGSHUB_USER
                export DVC_HTTP_PASSWORD=$DAGSHUB_TOKEN
        
                echo "Credentials loaded"
              '''
            }
          }
        }


        stage('Train + Evaluate + Gate + Auto Promote') {
            steps {
                sh '''
                    set -e
                    . .venv/bin/activate

                    # Exécuter la pipeline DVC complète
                    dvc repro -f prepare train evaluate deepchecks_gate promote_auto
                '''
            }
        }
    }

    post {
        always {
            sh '''
                if [ -f metrics/scores.json ]; then
                    echo "---- metrics/scores.json ----"
                    cat metrics/scores.json
                fi

                if [ -f metrics/deepchecks.json ]; then
                    echo "---- metrics/deepchecks.json ----"
                    cat metrics/deepchecks.json
                fi
            '''
            archiveArtifacts artifacts: 'metrics/*.json', fingerprint: true
        }

        failure {
            echo "❌ Pipeline failed. Check the DVC stage shown above."
        }
    }
}
