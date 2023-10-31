from sklearn.preprocessing import LabelEncoder
import pickle

class Utils:
    @staticmethod
    def data_prep(df):
        """
        Prepare and preprocess the input DataFrame for machine learning by encoding the target variable.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the dataset to be preprocessed.

        Returns:
        X (pandas.DataFrame): The feature matrix after removing 'class' and 'date' columns.
        y (numpy.ndarray): The encoded target variable as a NumPy array.
        target_encoder_model (sklearn.preprocessing.LabelEncoder): The label encoder for the target variable.

        Example:
        ```
        import pandas as pd

        # Assuming 'df' is a pandas DataFrame
        X, y, target_encoder = Utils.data_prep(df)
        ```
        This static method preprocesses the input DataFrame by dropping 'class' and 'date' columns and encoding
        the target variable 'class' using LabelEncoder. It returns the feature matrix, the encoded target variable,
        and the label encoder model for further use in machine learning tasks.
        """
        df.dropna(axis=0, inplace=True)
        X = df.drop(['class', 'date'], axis=1)
        y = df['class'].values

        target_encoder_model = LabelEncoder()
        y = target_encoder_model.fit_transform(y)

        return X, y, target_encoder_model

    @staticmethod
    def save_pickel_model(model, out_path):
        """
        Save a Python object (model) as a pickle file at the specified path.

        Parameters:
        model (object): The Python object (e.g., a machine learning model) to be saved.
        out_path (str): The file path where the model will be saved as a pickle file.

        Returns:
        None

        Example:
        ```
        # Assuming 'model' and 'out_path' are defined
        Utils.save_pickel_model(model, out_path)
        ```
        This static method saves the provided Python object (e.g., a machine learning model) as a pickle file
        at the specified file path. It allows you to serialize and store the object for future use or sharing.
        """
        with open(out_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_pickle_model(model_path):
        """
        Load a Python object (model) from a pickle file at the specified path.

        Parameters:
        model_path (str): The file path from which to load the pickle model.

        Returns:
        model (object): The Python object (e.g., a machine learning model) loaded from the pickle file.

        Example:
        ```
        # Assuming 'model_path' is defined
        loaded_model = Utils.load_pickle_model(model_path)
        ```
        This static method loads a Python object (e.g., a machine learning model) from a pickle file
        at the specified file path. It allows you to deserialize and retrieve the object for use in your code.
        """
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    
    # @staticmethod
    # def get_estimator_from_pipeline(pipeline):
    #     return pipeline.steps[1][1]
    
    @staticmethod
    def has_mlflow_registered_model(model_name:str, client):
        """
        Check if an MLflow registered model with the specified name exists.

        Parameters:
        model_name (str): The name of the MLflow registered model to check for.
        client (mlflow.tracking.MlflowClient): The MLflow client used for tracking and managing models.

        Returns:
        bool: True if the model with the given name exists in the model registry, False otherwise.

        Example:
        ```
        from mlflow.tracking import MlflowClient

        # Assuming 'model_name' and 'client' are defined
        if Utils.has_mlflow_registered_model(model_name, client):
            print(f"Model '{model_name}' exists in the model registry.")
        else:
            print(f"Model '{model_name}' does not exist in the model registry.")
        ```
        This static method checks if an MLflow registered model with the specified name exists in the model registry
        using the provided MLflow client. It returns True if the model exists and False if it doesn't.
        """
        try:
            client.get_registered_model(name=model_name)
            return True
        except Exception:
            return False