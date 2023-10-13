# %%
from src.preprocessing.nlp.nlp_preprocessing import NLPreprocessing
import pandas as pd

# %%
def dataset_nlp_preprocessing(docs_csv_path:str):
    df = pd.read_csv(docs_csv_path)
    df.dropna(axis=0, inplace=True)

    # df_test = df.iloc[:200, :].copy()

    nlp = NLPreprocessing()
    print(f'Preprocessing {len(df)} documents...')

    df['text'] = [nlp.filter_text(text) for text in df['text'].values]

    print('Preprocessing finished!')
    return df
