from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_retraining():
    import sys
    sys.path.append('/opt/airflow/scripts')
    from retrain_script import run_full_pipeline
    run_full_pipeline()

with DAG(
    'retrain_model',
    default_args=default_args,
    schedule_interval=None,  # Ручной запуск
    catchup=False,
    tags=['retraining'],
) as dag:
    
    retrain_task = PythonOperator(
        task_id='run_retraining',
        python_callable=run_retraining,
    )

    retrain_task