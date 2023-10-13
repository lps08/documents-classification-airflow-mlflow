import sys
sys.path.append('/opt/airflow/')
sys.path.append('/opt/airflow/src')
sys.path.append('/opt/airflow/src/preprocessing/nlp')
from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from src import constants
from src.preprocessing.nlp.docs_nlp import dataset_nlp_preprocessing
import os
import math
from src.model.train import mlflow_train_track, mlflow_registry_best_model
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

@dag(
    dag_id="documents_training",
    start_date=datetime(2023,9,20),
    schedule=[constants.OCR_DOCUMENTS_DATASET],  # Scheduled on both Datasets
    catchup=False,
)
def datasets_consumer_dag():

    documents_ocr_file_sensor_task = FileSensor(
        task_id="documents_ocr_file_sensor",
        filepath=constants.DOCUMENTS_OCR,
        poke_interval = 10,
        timeout = 60
    )

    @task
    def nlp_preprocessing_task():
        df_processed = dataset_nlp_preprocessing(os.path.join(constants.DATA_DIR, constants.OCR_DOCUMENTS_CSV))
        return df_processed
    
    @task
    def knn_train_task(df):
        k = int(math.sqrt(df.shape[0]))
        knn_hyperparameters = {
            'n_neighbors':[int(k/4), int(k/3), int(k/2), k, k*2, k*3],
            'weights':['uniform', 'distance'],
            'p':[1, 2],
        }

        mlflow_train_track(
            model=KNeighborsClassifier(),
            df=df,
            hyperparameters_space=knn_hyperparameters,
            # out_model_path=constants.KNN_MODEL_PATH,
            model_name='knn',
        )
    
    @task
    def rf_train_task(df):
        rf_hyperparameters = {
            'n_estimators': [180, 200, 220, 240],
            'criterion': ['gini', 'entropy'],
            'max_depth': [80, 90, 100, 110],
            'min_samples_split': [4, 5, 6],
            'min_samples_leaf': [1, 2],
            'oob_score': [True, False],
        }

        mlflow_train_track(
            model=RandomForestClassifier(),
            df=df,
            hyperparameters_space=rf_hyperparameters,
            # out_model_path=constants.RF_MODEL_PATH,
            model_name='random_forest',
        )

    @task
    def mlp_train_task(df):
        mlp_hyperparameters = {
            'hidden_layer_sizes': [(2,), (3,), (4,), (5,)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }

        mlflow_train_track(
            model=MLPClassifier(),
            df=df,
            hyperparameters_space=mlp_hyperparameters,
            # out_model_path=constants.MLP_MODEL_PATH,
            model_name='MLP',
        )

    @task
    def xgb_train_task(df):
        xgb_hyperparameters = {
            'max_depth': [6,10, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 300, 500],
            'colsample_bytree': [0.3, 0.7],
        }

        mlflow_train_track(
            model=XGBClassifier(),
            df=df,
            hyperparameters_space=xgb_hyperparameters,
            # out_model_path=constants.XGB_MODEL_PATH,
            model_name='XGBoost',
        )

    @task
    def select_best_model_task():
        mlflow_registry_best_model(
            server_uri=constants.MLFLOW_URI,
            experiment_name=constants.EXPERIMENT_NAME,
            model_registry_name=constants.MODEL_REGISTRY_NAME,
        )

    documents_preprocessed_xcom = nlp_preprocessing_task()
    knn_consumer = knn_train_task(documents_preprocessed_xcom)
    # rf_consumer = rf_train_task(documents_preprocessed_xcom)
    mlp_consumer = mlp_train_task(documents_preprocessed_xcom)
    # xgb_consumer = xgb_train_task(documents_preprocessed_xcom)

    documents_ocr_file_sensor_task >> documents_preprocessed_xcom
    # documents_preprocessed_xcom >> [knn_consumer, rf_consumer, mlp_consumer, xgb_consumer] >> select_best_model_task()
    documents_preprocessed_xcom >> [knn_consumer, mlp_consumer] >> select_best_model_task()

datasets_consumer_dag()