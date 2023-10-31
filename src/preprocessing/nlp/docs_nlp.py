# %%
from src.preprocessing.nlp.nlp_preprocessing import NLPreprocessing
import pandas as pd

# %%
def dataset_nlp_preprocessing(docs_csv_path:str):
    """
    Perform NLP preprocessing on a dataset stored in a CSV file.

    Parameters:
    docs_csv_path (str): The path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: A DataFrame with NLP-preprocessed text data.

    Example:
    ```
    # Assuming 'csv_path' is the path to the CSV file containing the dataset
    preprocessed_df = dataset_nlp_preprocessing(csv_path)
    print(preprocessed_df.head())
    ```
    This function reads a dataset from a CSV file, applies NLP preprocessing to the text data, and returns a DataFrame
    with the preprocessed text. The preprocessing steps include removing punctuation, digits, accents, misspelled words,
    URLs, HTML tags, stopwords, unknown words, and performing stemming.
    """
    df = pd.read_csv(docs_csv_path)
    df.dropna(axis=0, inplace=True)

    # df_test = df.iloc[:200, :].copy()

    nlp = NLPreprocessing()
    print(f'Preprocessing {len(df)} documents...')

    df['text'] = [nlp.filter_text(text) for text in df['text'].values]

    print('Preprocessing finished!')
    return df
