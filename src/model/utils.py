from sklearn.preprocessing import LabelEncoder
import pickle

class Utils:
    @staticmethod
    def data_prep(df):
        df.dropna(axis=0, inplace=True)
        X = df.drop(['class', 'date'], axis=1)
        y = df['class'].values

        target_encoder_model = LabelEncoder()
        y = target_encoder_model.fit_transform(y)

        return X, y, target_encoder_model

    @staticmethod
    def save_pickel_model(model, out_path):
        with open(out_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_pickle_model(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    
    @staticmethod
    def get_estimator_from_pipeline(pipeline):
        return pipeline.steps[1][1]
    
    @staticmethod
    def has_mlflow_registered_model(model_name:str, client):
        try:
            client.get_registered_model(name=model_name)
            return True
        except Exception:
            return False