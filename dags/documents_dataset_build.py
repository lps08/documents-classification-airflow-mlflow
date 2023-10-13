import sys
sys.path.append('/opt/airflow/')
sys.path.append('/opt/airflow/src')
sys.path.append('/opt/airflow/src/preprocessing/dataset')
sys.path.append('/opt/airflow/src/preprocessing/ocr')
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from src.dataset.data_mapping import mapping_files
from src.preprocessing.ocr.documents_ocr import documents_ocr
from src import constants

dag = DAG(
    'documents_dataset_build',
    description='Build the csv dataset from the documents',
    schedule_interval=None,
    start_date=datetime(2023,9,20),
    default_view='graph',
    catchup=False,
)

dataset_file_sensor = FileSensor(
    task_id="dataset_file_sensor",
    filepath=constants.DATASET_PATH,
    poke_interval = 10,
    timeout = 60,
    dag=dag
)

mapping_files_task = PythonOperator(
    task_id='mapping_files',
    python_callable=mapping_files,
    dag=dag
)

mapped_files_sensor = FileSensor(
    task_id = 'mapped_files_sensor',
    filepath=f'{constants.DATA_DIR}{constants.MAPPED_FILES_CSV}',
    poke_interval = 10,
    timeout = 60,
    dag=dag
)

ocr_task = PythonOperator(
    task_id = 'ocr',
    python_callable=documents_ocr,
    outlets=constants.OCR_DOCUMENTS_DATASET,
    dag=dag,
)

dataset_file_sensor >> mapping_files_task >> mapped_files_sensor >> ocr_task