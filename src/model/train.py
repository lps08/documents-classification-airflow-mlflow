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

        # log charts
        fig = evaluate.plot_confusion_matrix()
        mlflow.log_figure(figure=fig, artifact_file=f'confusion-{model_name}-{datetime.now().date()}.png')
        print('The models configurations and params can be seen on MLFlow UI http://localhost:5000/')

def mlflow_train_track(model, df, hyperparameters_space:dict, experiment_name:str="creating_pipeline", model_name:str=None):
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

def select_best_model():
    knn_model = Utils.get_estimator_from_pipeline(Utils.load_pickle_model(constants.KNN_MODEL_PATH))
    rf_model = Utils.get_estimator_from_pipeline(Utils.load_pickle_model(constants.RF_MODEL_PATH))
    mlp_model = Utils.get_estimator_from_pipeline(Utils.load_pickle_model(constants.MLP_MODEL_PATH))
    xgb_model = Utils.get_estimator_from_pipeline(Utils.load_pickle_model(constants.XGB_MODEL_PATH))

    best_model = knn_model

    for model in [rf_model, mlp_model, xgb_model]:
        if knn_model.best_score_ < model.best_score_:
            best_model = model

    print(f'Best model is {best_model}')
    Utils.save_pickel_model(model=best_model, out_path=constants.BEST_MODEL_PATH)

# preds = model.predict(X_test)
# [target_encoder_model.inverse_transform([pred]) for pred in preds]