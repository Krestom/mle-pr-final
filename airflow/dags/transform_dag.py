from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime
from pathlib import Path
import sys, logging, os

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "project"

# чтобы импортировать project.data_preprocessing
sys.path.append(str(ROOT_DIR))

os.environ.setdefault("DEBUG_FRACTION", "0.05")

def run_transform():
    logging.info("▶️  running project.data_preprocessing.run_full_pipeline()")
    from project.data_preprocessing import run_full_pipeline
    run_full_pipeline()
    logging.info("✅  preprocessing finished")

with DAG(
    dag_id="transform_events",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    wait_extract = ExternalTaskSensor(
        task_id="wait_for_extract",
        external_dag_id="extract_raw_events",
        external_task_id="copy_raw_files",
        poke_interval=30,
        mode="reschedule",
    )

    transform = PythonOperator(
        task_id="run_notebook_code",
        python_callable=run_transform,
    )

    wait_extract >> transform