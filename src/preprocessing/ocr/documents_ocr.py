#%%
import pandas as pd
from datetime import datetime
from src.preprocessing.ocr.pdf_to_text import PDF2Text
from src.dataset.utils import get_documents_mapped
from datetime import datetime
import constants
import os

#%%
def documents_ocr(documents_mapped_file:str = constants.MAPPED_FILES_CSV, sample_size:int = None if constants.SAMPLE_SIZE < 0 else constants.SAMPLE_SIZE, csv_out_name = constants.OCR_DOCUMENTS_CSV, dir_to_csv=constants.DATA_DIR):
    """
    Perform Optical Character Recognition (OCR) on documents and save the results to a CSV file.

    Parameters:
    documents_mapped_file (str, optional): The path to the CSV file containing information about the documents.
    sample_size (int, optional): The number of documents to process. If None, all documents are processed.
    csv_out_name (str): The name of the output CSV file.
    dir_to_csv (str): The directory where the output CSV file will be saved.

    Example:
    ```
    # Perform OCR on documents and save the results to a CSV file
    documents_ocr(documents_mapped_file='mapped_documents.csv', sample_size=100, csv_out_name='ocr_results.csv')
    ```
    This function performs Optical Character Recognition (OCR) on a set of documents specified in a CSV file. It extracts
    text from each document, filters and preprocesses the text, and then saves the results to a CSV file with the specified name.
    The function allows processing a subset of documents specified by 'sample_size' or all documents if 'sample_size' is None.
    """
    df = get_documents_mapped(path=os.path.join(constants.DATA_DIR, documents_mapped_file), sample_size=sample_size)

    documents_texts = []
    documents_class = []

    for i in range(df.shape[0]):
        try:
            pdf = PDF2Text(pdf_path=df.iloc[i, 1], num_pages=1)
            pdf_text = pdf.get_texts()
            pdf_class = df.iloc[i, 0]
        except:
            print(f'Error in this doc: {df.iloc[i, 1]}')
            continue
        
        documents_texts.append(pdf_text)
        documents_class.append(pdf_class)

        if i % 50 == 0:
            print(f'{i}/{df.shape[0]} ocr of documents completed')

    print('OCR preprocess completed!')
    print(f'Saving file {csv_out_name}...!')

    df_documents_dataset = pd.DataFrame(
        {
            'text' : documents_texts,
            'class' : documents_class,
            'date' : datetime.now().date()
        }
    )

    df_documents_dataset.to_csv(os.path.join(dir_to_csv, csv_out_name), index=False)