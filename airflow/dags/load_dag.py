from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

with DAG(
    dag_id="load_and_deploy",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    wait_transform = ExternalTaskSensor(
        task_id="wait_for_transform",
        external_dag_id="transform_events",
        external_task_id="run_notebook_code",
        poke_interval=30,
        mode="reschedule",
    )

    rebuild_docker = BashOperator(
        task_id="rebuild_compose",
        bash_command="docker compose down --volumes --remove-orphans && docker compose build && docker compose up -d --force-recreate",
        cwd=str(ROOT_DIR),
        env={"DOCKER_BUILDKIT": "1"},
    )

    wait_transform >> rebuild_docker