import sys
sys.path.append('/opt/airflow/')
sys.path.append('/opt/airflow/src')
sys.path.append('/opt/airflow/src/preprocessing/dataset')
sys.path.append('/opt/airflow/src/preprocessing/ocr')
from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from src.dataset.data_mapping import mapping_files
from src.preprocessing.ocr.documents_ocr import documents_ocr
from src import constants

@dag(
    dag_id='documents_dataset_build',
    description='Build the csv dataset from the documents',
    schedule_interval=None,
    start_date=datetime(2023,9,20),
    catchup=False,
)
def dataset_build():
    dataset_file_sensor = FileSensor(
        task_id="dataset_file_sensor",
        filepath=constants.DATASET_PATH,
        poke_interval = 10,
        timeout = 60,
    )

    mapped_files_sensor = FileSensor(
        task_id = 'mapped_files_sensor',
        filepath=f'{constants.DATA_DIR}{constants.MAPPED_FILES_CSV}',
        poke_interval = 10,
        timeout = 60,
    )

    @task
    def mapping_files_task():
        mapping_files()

    @task(outlets=constants.OCR_DOCUMENTS_DATASET)
    def ocr():
        documents_ocr()

    dataset_file_sensor >> mapping_files_task() >> mapped_files_sensor >> ocr()

dataset_build()