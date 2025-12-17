pipeline {
  agent any

  options {
    timestamps()
  }

  environment {
    // DAGsHub MLflow tracking
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
        powershell '''
          $ErrorActionPreference = "Stop"
          python -V

          if (!(Test-Path ".venv")) { python -m venv .venv }

          # Upgrade pip inside venv
          .\\.venv\\Scripts\\python -m pip install --upgrade pip

          # Install deps
          if (Test-Path "requirements.txt") {
            .\\.venv\\Scripts\\pip install -r requirements.txt
          } else {
            # Fallback minimal deps
            .\\.venv\\Scripts\\pip install dvc mlflow pandas numpy scikit-learn torch transformers sentence-transformers
          }
        '''
      }
    }

    stage('Load credentials') {
      steps {
        withCredentials([
          string(credentialsId: 'DAGSHUB_USER', variable: 'DAGSHUB_USER'),
          string(credentialsId: 'DAGSHUB_TOKEN', variable: 'DAGSHUB_TOKEN')
        ]) {
          powershell '''
            $ErrorActionPreference = "Stop"

            # MLflow auth (DAGsHub uses basic auth style)
            $env:MLFLOW_TRACKING_USERNAME = $env:DAGSHUB_USER
            $env:MLFLOW_TRACKING_PASSWORD = $env:DAGSHUB_TOKEN

            # Optionally used by some DVC remotes / scripts
            $env:DAGSHUB_USERNAME = $env:DAGSHUB_USER
            $env:DAGSHUB_PASSWORD = $env:DAGSHUB_TOKEN

            echo "✅ Credentials loaded (values are masked by Jenkins)."
          '''
        }
      }
    }

    stage('DVC pull') {
      steps {
        powershell '''
          $ErrorActionPreference = "Stop"
          .\\.venv\\Scripts\\dvc --version

          # Pull data/artifacts tracked by DVC
          .\\.venv\\Scripts\\dvc pull -v
        '''
      }
    }

    stage('Pipeline to Staging') {
      steps {
        powershell '''
          $ErrorActionPreference = "Stop"

          # Run until Staging promotion
          .\\.venv\\Scripts\\dvc repro -f prepare train evaluate deepchecks_gate promote_staging
        '''
      }
    }

    stage('Promote to Production (manual approval)') {
      steps {
        script {
          input message: 'Deepchecks PASSED. Promote model to Production (and archive old Production)?', ok: 'Promote'
        }
        powershell '''
          $ErrorActionPreference = "Stop"
          .\\.venv\\Scripts\\dvc repro -f promote_production
        '''
      }
    }
  }

  post {
    always {
      powershell '''
        if (Test-Path "metrics\\scores.json") { echo "---- metrics/scores.json ----"; type metrics\\scores.json }
        if (Test-Path "metrics\\deepchecks.json") { echo "---- metrics/deepchecks.json ----"; type metrics\\deepchecks.json }
      '''
      archiveArtifacts artifacts: 'metrics/*.json', fingerprint: true
    }

    failure {
      echo "❌ Pipeline failed. Check console output above (DVC stage that failed will be shown)."
    }
  }
}
