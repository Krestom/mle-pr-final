from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path
import shutil, logging

# ── вычисляем <repo root>/ с учётом того, что файл лежит airflow/dags/this_file.py
ROOT_DIR = Path(__file__).resolve().parents[2]

# откуда берём «сырае» события; поправьте под себя
RAW_SRC = ROOT_DIR
# куда кладём, чтобы видел data_preprocessing.py
RAW_DST = ROOT_DIR / "data"
print(ROOT_DIR, RAW_SRC, RAW_DST)

def extract():
    RAW_DST.mkdir(parents=True, exist_ok=True)

    # копируем ВСЁ *.csv *.parquet
    for src in RAW_SRC.glob("*.[cp][as][rvq]*"):
        dst = RAW_DST / src.name
        shutil.copy2(src, dst)
        logging.info(f"Copied {src.relative_to(ROOT_DIR)} → {dst.relative_to(ROOT_DIR)}")

with DAG(
    dag_id="extract_raw_events",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    PythonOperator(task_id="copy_raw_files", python_callable=extract)