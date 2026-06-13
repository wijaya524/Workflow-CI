import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

# 1. SETUP PATH ABSOLUT DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data_train.csv')
data = pd.read_csv(data_path)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("average_rating", axis=1),
    data['average_rating'],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]
model = RandomForestRegressor(n_estimators=505, max_depth=37, random_state=42)

# 2. CATAT PARAMETER LANGSUNG (Tanpa start_run, karena sudah otomatis dibuka oleh CLI mlflow run)
mlflow.log_param("n_estimator", 505)
mlflow.log_param("max_depth", 37)

# Latih model
model.fit(X_train, y_train)

# 3. LOG MODEL ARTIFACT
# Karena menggunakan run aktif dari CLI, ia akan langsung menulis ke folder mlruns lokal
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

# 4. LOG METRIK EVALUASI
r2_score = model.score(X_test, y_test)
mlflow.log_metric("r2_score", r2_score)

print("Berjalan dengan sukses melalui MLflow Proyek lokal!")
