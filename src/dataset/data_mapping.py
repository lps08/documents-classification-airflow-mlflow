#%%
import os
from collections import defaultdict
import pandas as pd
import constants
import unidecode
# %%
# find out all the documents subfolder name and track them
def mapping_files(dataset_path:str=constants.DATASET_PATH, dir_to_csv:str=constants.DATA_DIR):
    """
    Create a mapping of file paths within a dataset directory and save it to a CSV file.

    Parameters:
    dataset_path (str, optional): The path to the dataset directory. Default is constants.DATASET_PATH.
    dir_to_csv (str, optional): The directory where the generated CSV file will be saved. Default is constants.DATA_DIR.

    Returns:
    None

    Example:
    ```
    mapping_files(dataset_path="/path/to/dataset", dir_to_csv="/output/directory")
    ```
    This function creates a mapping of file paths within a dataset directory. It lists files in subdirectories,
    associates them with their respective class labels, and saves this mapping as a CSV file. You can specify
    the dataset path and the directory for saving the CSV file. If the specified output directory does not exist,
    a ValueError is raised.
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
