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

          # Installer les packages dans le même shell
          . .venv/bin/activate
          pip install --upgrade pip

          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            pip install "dvc[s3]" mlflow pandas numpy scikit-learn torch transformers sentence-transformers
          fi
        '''
      }
    }

    stage('Load credentials & DVC pull') {
      steps {
        withCredentials([
            usernamePassword(
                credentialsId: 'dagshub-creds',
                usernameVariable: 'DAGSHUB_USERNAME',
                passwordVariable: 'DAGSHUB_PASSWORD'
            )
        ]) {
          sh '''
            set -e
            cd $WORKSPACE
            echo "✅ Credentials loaded"

            # Activer le venv et exporter les credentials
            . .venv/bin/activate
            export DAGSHUB_USERNAME=$DAGSHUB_USERNAME
            export DAGSHUB_PASSWORD=$DAGSHUB_PASSWORD
            export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
            export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_PASSWORD

            # Pull des données DVC
            dvc pull -v
          '''
        }
      }
    }

    stage('Train + Evaluate + Gate + Auto Promote') {
      steps {
        sh '''
          set -e
          cd $WORKSPACE
          . .venv/bin/activate

          # Exécution de la pipeline DVC
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
