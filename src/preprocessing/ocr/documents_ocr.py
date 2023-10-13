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