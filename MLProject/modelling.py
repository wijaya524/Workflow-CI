import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

# 1. SETUP PATH ABSOLUT LOKAL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder MLProject
PARENT_DIR = os.path.dirname(BASE_DIR)                 # Root repo (Workflow-CI)
mlruns_path = os.path.join(PARENT_DIR, "mlruns")

# Kunci tracking URI ke berkas lokal murni
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.sklearn.autolog(disable=True)

# 2. LOAD DATA MENGGUNAKAN PATH ABSOLUT
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

# 3. GUNAKAN NESTED=TRUE AGAR MENGIKUTI RUN DARI CLI MLFLOW
# Trik ini menghindari konflik Active Run ID dengan terminal
with mlflow.start_run(nested=True):
  
    mlflow.log_param("n_estimator", 505)
    mlflow.log_param("max_depth", 37)

    # Latih model
    model.fit(X_train, y_train)

    # Log model artifact (Akan masuk ke mlruns lokal secara aman)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    r2_score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", r2_score)

print("Berjalan dengan sukses melalui MLflow Proyek!")
