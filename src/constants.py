from airflow.models import Variable
from airflow.datasets import Dataset

MIN_DOC_CLASS = int(Variable.get('min_doc_class'))
DATASET_PATH = Variable.get('dataset_path')
DATA_DIR = Variable.get('data_dir')
SAMPLE_SIZE = int(Variable.get('sample_size'))
DOCUMENTS_OCR = Variable.get('documents_ocr_path')
MAPPED_FILES_CSV= f'mapped_files.csv'
OCR_DOCUMENTS_CSV = f'ocr_docs.csv'
OCR_DOCUMENTS_DATASET = Dataset(f'file://{DOCUMENTS_OCR}')
KNN_MODEL_PATH = Variable.get('knn_path')
RF_MODEL_PATH = Variable.get('rf_path')
MLP_MODEL_PATH = Variable.get('mlp_path')
XGB_MODEL_PATH = Variable.get('xgb_path')
BEST_MODEL_PATH = Variable.get('best_model_path')
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = Variable.get('experiment_name')
MODEL_REGISTRY_NAME = Variable.get('model_registry_name')
TEST_SIZE = float(Variable.get('test_size'))

# MLFLOW_URI = "http://localhost:5000"