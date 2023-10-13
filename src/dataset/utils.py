import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import re
from src import constants

class Utils:

    @staticmethod
    def read_docs(path)->pd.DataFrame:
        """Read a csv mapped with dataset files and will be filtered 
        by document extensions (pdf or PDF).

        Parameters
        ----------
        path: path of the dataset

        Returns
        -------
        Dataframe containing only documents files
        """
        df = pd.read_csv(path)
        return df[df['paths'].apply(
            lambda x: True if re.match(r'.+\.(pdf|PDF)$', x) else False
        )]
    
    @staticmethod
    def remove_few_docs(df:pd.DataFrame, min_data:int)->pd.DataFrame:
        """Remove classes from the documents dataset that contain a minimum 
        amount of data according to the parameter 'min_data'.

        Parameters
        ----------
        df: Dataframe to filter
        min_data: minimum amount of the data to remove a class

        Returns
        -------
        Dataframe with filtered data
        """
        df_counted_classes = df.groupby('classes').count().sort_values('paths', ascending=False)
        
        classes_to_remove = df_counted_classes[df_counted_classes['paths'] < min_data].index.tolist()
        mask = ~df['classes'].isin(classes_to_remove) # ~ = inverte the bool value of the isin result
        return df[mask]
    
    @staticmethod
    def stratified_sampling(df:pd.DataFrame, n_sample:int, column:str)->pd.DataFrame:
        """Choose a sample data according to the stratifield method.

        Parameters
        ----------
        df: Dataframe get the sample
        n_sample: sample size
        column: column to stratify the sample

        Returns
        Dataframe with stratified sample
        """
        population = len(df)
        sample_size = n_sample / population
        stratified_sampling = StratifiedShuffleSplit(test_size=sample_size)
        stratified_sample = stratified_sampling.split(X=df, y=df[column])
        for _, y in stratified_sample:
            df_stratified_sample = df.iloc[y]
        return df_stratified_sample
    
def get_documents_mapped(path:str, min_doc_class:int=constants.MIN_DOC_CLASS, sample_size:int=None)->pd.DataFrame:
    """Read a csv file with a mapped files and apply some preprocessing step on it

    Parameters
    ----------
    path: path of the csv file
    min_doc_class: Identify the minimum value within each class of documents to be
                   removed from the dataframe.
    sample_size: if passed, return the dataframe sampled
    
    Returns
    -------
    Dataframe preprocessed and ready to work with
    """
    df_files = Utils.read_docs(path)
    df_filtered = Utils.remove_few_docs(df_files, min_doc_class)
    if sample_size:
        return Utils.stratified_sampling(df_filtered, n_sample=sample_size, column='classes')
    return df_filtered