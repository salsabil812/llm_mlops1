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
                sh '''
                  set -euo pipefail
                  git status --porcelain || true
                '''
            }
        }
        stage('Clean workspace') {
              steps {
                sh '''
                  set -euo pipefail
                  # Nettoyer outputs générés par les anciens builds
                  rm -rf models/transformer_classifier || true
                  rm -f metrics/train_metrics.json || true
            
                  # Optionnel: si tu veux repartir clean à 100%
                  # dvc checkout -f || true
                '''
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

                        # ✅ Auth HTTP fiable pour DagsHub (évite 401)
                        cat > ~/.netrc <<EOF
machine dagshub.com
login $DAGSHUB_USER
password $DAGSHUB_TOKEN
EOF
                        chmod 600 ~/.netrc

                        # (Optionnel) cache DVC persistant
                        dvc config cache.dir /var/jenkins_home/dvc-cache
                        mkdir -p /var/jenkins_home/dvc-cache

                        # Garder aussi l’auth dans la config DVC (OK)
                        dvc remote modify dagshub user "$DAGSHUB_USER"
                        dvc remote modify dagshub password "$DAGSHUB_TOKEN"

                        # Petit test rapide (facultatif)
                        dvc list . models -R | head -n 20 || true

                        dvc pull -v

                        # Vérif
                        ls -la data/raw/train.csv metrics/scores.json models/transformer_classifier >/dev/null
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

                        export MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER"
                        export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"

                        # 🔎 Debug utile : voir si git pense que src/*.py sont modifiés
                        git status --porcelain || true

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
