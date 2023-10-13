#%%
import os
from collections import defaultdict
import pandas as pd
import constants
import unidecode
# %%
# find out all the documents subfolder name and track them
def mapping_files(dataset_path:str=constants.DATASET_PATH, dir_to_csv:str=constants.DATA_DIR):
    """Find all files from the dataset directory

    Parameters
    ----------
    dataset_path: path to the dataset

    Return
    ------
    list: all the files mapped to csv
    """
    files_dir = defaultdict(list)
    for i in os.listdir(dataset_path):
        class_files_path = os.path.join(dataset_path, i)

        for dir in os.listdir(class_files_path):
            file_path = os.path.join(class_files_path, dir)

            for doc_class in os.listdir(file_path):
                doc_class_dir = os.path.join(file_path, doc_class)

                if os.path.isdir(doc_class_dir):
                    files_dir[unidecode.unidecode(dir)].extend([os.path.join(doc_class_dir, doc) for doc in os.listdir(doc_class_dir)])
                else:
                    files_dir[unidecode.unidecode(dir)].append(os.path.join(doc_class_dir))

    df_docs_path = pd.DataFrame(
        {
            'classes' : [key.lower() for key in files_dir.keys() for _ in files_dir[key]],
            'paths' : [file for key in files_dir.keys() for file in files_dir[key]]
        }
    )

    if os.path.isdir(dir_to_csv):
        print(f'Saving {constants.MAPPED_FILES_CSV}')
        df_docs_path.to_csv(os.path.join(dir_to_csv, constants.MAPPED_FILES_CSV), index=False)
    else:
        raise ValueError('Invalid directory!')
