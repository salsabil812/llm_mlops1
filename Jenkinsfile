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
                    set -euo pipefail
                    python3 --version

                    if [ ! -d ".venv" ]; then
                        python3 -m venv .venv
                    fi

                    . .venv/bin/activate
                    python -m pip install --upgrade pip

                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    else
                        pip install dvc mlflow pandas numpy scikit-learn torch transformers sentence-transformers
                    fi
                '''
            }
        }

        stage('DVC pull (DagsHub auth)') {
            steps {
                withCredentials([
                    string(credentialsId: 'DAGSHUB_USER', variable: 'DAGSHUB_USER'),
                    string(credentialsId: 'DAGSHUB_TOKEN', variable: 'DAGSHUB_TOKEN')
                ]) {
                    sh '''
                        set -euo pipefail
                        . .venv/bin/activate

                        dvc --version

                        # (Optionnel mais recommandé) cache DVC persistant
                        # Si ton agent n'est pas persistant, tu peux commenter ces lignes
                        dvc config cache.dir /var/jenkins_home/dvc-cache
                        mkdir -p /var/jenkins_home/dvc-cache

                        # Auth DVC vers DagsHub (remote = "dagshub" car dvc remote default = dagshub)
                        dvc remote modify dagshub user "$DAGSHUB_USER"
                        dvc remote modify dagshub password "$DAGSHUB_TOKEN"

                        # Pull des données/modèles/metrics versionnés par DVC
                        dvc pull -v
                    '''
                }
            }
        }

        stage('Train + Evaluate + Gate + Auto Promote') {
            steps {
                withCredentials([
                    string(credentialsId: 'DAGSHUB_USER', variable: 'DAGSHUB_USER'),
                    string(credentialsId: 'DAGSHUB_TOKEN', variable: 'DAGSHUB_TOKEN')
                ]) {
                    sh '''
                        set -euo pipefail
                        . .venv/bin/activate

                        # Auth MLflow vers DagsHub (si tes scripts loggent sur MLflow)
                        export MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER"
                        export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"

                        dvc repro -f prepare train evaluate deepchecks_gate promote_auto
                    '''
                }
            }
        }
    }

    post {
        always {
            sh '''
                set +e
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
