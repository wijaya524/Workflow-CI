import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

# 1. ATUR PATH ABSOLUT SECARA OTOMATIS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder MLProject
PARENT_DIR = os.path.dirname(BASE_DIR)                 # Root repo (Workflow-CI)
mlruns_path = os.path.join(PARENT_DIR, "mlruns")

# Pastikan tracking URI dan autolog dikunci ke file lokal
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.sklearn.autolog(disable=True)

# 2. LOCK ARTIFACT LOCATION EKSPERIMEN KE LOKAL DISK
experiment_name = "Credit Scoring"
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(experiment_name)

if exp is None:
    exp_id = client.create_experiment(
        name=experiment_name,
        artifact_location=f"file://{mlruns_path}"
    )
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(experiment_name=experiment_name)

# 3. PROSES DATA & MODEL TRAINING
data_path = os.path.join(BASE_DIR, 'data_train.csv')
data = pd.read_csv(data_path)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("average_rating", axis=1),
    data['average_rating'],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]
model = RandomForestRegressor(n_estimators=505, max_depth=37, random_state=42)

# 4. EKSEKUSI RUN LOKAL MURNI
with mlflow.start_run(experiment_id=exp_id):
  
    mlflow.log_param("n_estimator", 505)
    mlflow.log_param("max_depth", 37)

    model.fit(X_train, y_train)

    # Log model ke folder mlruns lokal dengan aman tanpa tuntutan server HTTP
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    r2_score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", r2_score)

print("Berjalan dengan sukses melalui MLflow Proyek!")
