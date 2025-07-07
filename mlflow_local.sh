mlflow server \
  --backend-store-uri sqlite:///mydb.sqlite \
  --default-artifact-root file:./mlflow_experiments_store \
  --host 0.0.0.0 \
  --port 5000