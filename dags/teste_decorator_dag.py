import sys
sys.path.append('/opt/airflow/')
sys.path.append('/opt/airflow/src')
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from datetime import datetime
import pandas as pd
from src import constants

@dag(
    dag_id="consumer",
    start_date=datetime(2023,9,20),
    schedule=[constants.OCR_DOCUMENTS_DATASET],  # Scheduled on both Datasets
    catchup=False,
)
def datasets_consumer_dag():
    @task
    def read_dataset():
        df = pd.read_csv(f'{constants.DATA_DIR}{constants.OCR_DOCUMENTS_CSV}')

        return df
    
    bash_task = BashOperator(
        task_id="date_task",
        bash_command='echo "Hello, Airflow!"'
    )

    read_dataset() >> bash_task

datasets_consumer_dag()