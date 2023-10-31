# %%
from sklearn.compose import ColumnTransformer
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from src import constants
from src.model.utils import Utils
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from src.model.evaluate import Evaluate
from datetime import datetime
import os

# %%

def build_model(model, hyperparameters_space:dict):
    """
    Build a machine learning model pipeline for hyperparameter tuning.

    Parameters:
    model (estimator): The machine learning model to be used for the pipeline.
    hyperparameters_space (dict): A dictionary specifying the hyperparameter search space
        to be explored during hyperparameter tuning. It should be in the form of {'parameter_name': [values]}.

    Returns:
    Pipeline: A Scikit-learn pipeline that includes data preprocessing and hyperparameter tuning
        steps. The pipeline consists of the following components:
        - 'preprocessor': Data preprocessing step, including text feature selection and TF-IDF vectorization.
        - 'grid_search': Hyperparameter tuning step using HalvingGridSearchCV.
        
    The pipeline can be used for fitting and evaluating the model with various hyperparameter settings.

    Example usage:
    model = RandomForestClassifier()  # Example estimator
    hyperparameters = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
    }
    model_pipeline = build_model(model, hyperparameters)
    model_pipeline.fit(X_train, y_train)  # Fit the model with hyperparameter tuning
    y_pred = model_pipeline.predict(X_test)  # Make predictions using the tuned model.
    """
    features = ['text']

    text_transform = Pipeline([
        ('column_selector', ColumnSelector(cols=features, drop_axis=True)),
        ('tfidf_vectorizer', TfidfVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', text_transform, features),
        ]
    )

    grid_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=hyperparameters_space,
        scoring="accuracy",
        cv=2,
        return_train_score=True,
        n_jobs=-1,
        # error_score='raise'
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('grid_search', grid_search)
    ])

    return model_pipeline

def mlflow_train(model, df, active_run:mlflow.ActiveRun, hyperparameters_space:dict, model_name:str=None):
    """
    Train a machine learning model, log metrics, and artifacts to MLflow.

    Parameters:
    model (sklearn.base.BaseEstimator): The scikit-learn machine learning model to be trained.
    df (pandas.DataFrame): The input DataFrame containing the dataset for training.
    active_run (mlflow.ActiveRun): The active MLflow run to log information to.
    hyperparameters_space (dict): A dictionary specifying hyperparameter configurations for model tuning.
    model_name (str, optional): A name for the model to use when logging artifacts and metrics. Default is None.

    Returns:
    None

    Example:
    ```
    import mlflow
    from sklearn.model_selection import train_test_split

    # Assuming 'df' and 'hyperparameters_space' are defined
    with mlflow.start_run() as active_run:
        mlflow_train(model, df, active_run, hyperparameters_space, model_name="MyModel")
    ```
    This function trains the specified model on the provided dataset, logs various metrics, hyperparameters,
    and artifacts to the active MLflow run. The logged metrics and artifacts can be viewed in the MLflow UI.
    """
    X, y, target_encoder_model = Utils.data_prep(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69, stratify=y)
    
    with active_run:
        mlflow.sklearn.autolog()

        model_pipeline = build_model(model, hyperparameters_space)

        print('Trainig model...')
        model_pipeline.fit(X_train, y_train)

        model_estimator = Utils.get_estimator_from_pipeline(model_pipeline)
        print(f'Model best params: {model_estimator.best_params_}')
        print(f'Model best score: {model_estimator.best_score_}')

        evaluate = Evaluate(
            model=model_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        print('Logging metrics to MLFlow...')
        score = evaluate.score_model()
        for key in score.keys():
            mlflow.log_param(key=key, value=str(score[key]))

        print('Logging label encoder to MLFlow...')
        label_encoder_name = f'label-encoder-{model_name}.pickel'
        label_encoder_path = os.path.join(constants.DATA_DIR, label_encoder_name)
        Utils.save_pickel_model(model=target_encoder_model, out_path=label_encoder_path)
        mlflow.log_artifact(label_encoder_path)

        # log charts
        fig = evaluate.plot_confusion_matrix()
        mlflow.log_figure(figure=fig, artifact_file=f'confusion-{model_name}-{datetime.now().date()}.png')
        print('The models configurations and params can be seen on MLFlow UI http://localhost:5000/')

def mlflow_train_track(model, df, hyperparameters_space:dict, experiment_name:str="creating_pipeline", model_name:str=None):
    """
    Train a machine learning model, log metrics, and artifacts to MLflow within a specified experiment.

    Parameters:
    model (sklearn.base.BaseEstimator): The scikit-learn machine learning model to be trained.
    df (pandas.DataFrame): The input DataFrame containing the dataset for training.
    hyperparameters_space (dict): A dictionary specifying hyperparameter configurations for model tuning.
    experiment_name (str, optional): The name of the MLflow experiment to use. Default is "creating_pipeline."
    model_name (str, optional): A name for the model to use when logging artifacts and metrics. Default is None.

    Returns:
    None

    Example:
    ```
    from mlflow.tracking import MlflowClient

    # Assuming 'model', 'df', and 'hyperparameters_space' are defined
    mlflow_train_track(model, df, hyperparameters_space, experiment_name="MyExperiment", model_name="MyModel")
    ```
    This function sets up and tracks the machine learning training process within the specified MLflow experiment.
    It logs various metrics, hyperparameters, and artifacts to the active MLflow run within the chosen experiment.
    If the experiment doesn't exist, it creates a new one with the provided name.
    """
    mlflow.set_tracking_uri(constants.MLFLOW_URI)
    mlflow.set_experiment(experiment_name=experiment_name)

    active_run = mlflow.active_run() if mlflow.active_run() else mlflow.start_run()
    mlflow_train(
        model=model,
        df=df,
        hyperparameters_space=hyperparameters_space,
        active_run=active_run,
        model_name=model_name,
    )

def mlflow_registry_best_model(server_uri:str, experiment_name:str, model_registry_name:str, registry_by:str='cross validation mean'):
    """
    Register the best-performing model from an MLflow experiment into the model registry.

    Parameters:
    server_uri (str): The URI of the MLflow tracking server.
    experiment_name (str): The name of the MLflow experiment to search for the best model.
    model_registry_name (str): The name for the registered model in the model registry.
    registry_by (str, optional): The parameter to determine the best model by. Default is 'cross validation mean'.

    Returns:
    None

    Example:
    ```
    server_uri = "http://localhost:5000"  # Replace with your MLflow tracking server URI
    experiment_name = "MyExperiment"
    model_registry_name = "MyModel"
    mlflow_registry_best_model(server_uri, experiment_name, model_registry_name)
    ```
    This function searches for the best-performing model within the specified MLflow experiment, determined by the
    specified parameter. It then registers the best model into the model registry with the given name.
    If the model registry or model version already exists, it will update the existing version.
    """
    client = MlflowClient(tracking_uri=server_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    best_run = runs.pop()
    cv_best = best_run.data.params[registry_by]

    for run in runs:
        cv_run = run.data.params[registry_by]

        if cv_run > cv_best:
            best_run = run
            cv_best = cv_run

    print(f'Best model run: {best_run.info.run_name} | run id: {best_run.info.artifact_uri}')
    print('Registering best model...')

    if not Utils.has_mlflow_registered_model(model_name=model_registry_name, client=client):
        client.create_registered_model(name=model_registry_name)
    
    model_registered = client.create_model_version(
        name=model_registry_name,
        source=f"{best_run.info.artifact_uri}/model",
        run_id=best_run.info.run_id,
    )
    print(f'Model version {model_registered.version} registered!')