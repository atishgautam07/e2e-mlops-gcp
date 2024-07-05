import os
from mlProject import logger
import pickle
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from mlProject.entity.config_entity import (ModelTrainerConfig)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def load_pickle(self, filename: str):
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)
    
    
    def train(self):

        # def run_optimization(data_path: str, num_trials: int):

        X_train, y_train = self.load_pickle(os.path.join(self.config.train_data_path, "train.pkl"))
        X_val, y_val = self.load_pickle(os.path.join(self.config.test_data_path, "val.pkl"))


        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("random-forest-hyperopt")

        def objective(params):
            
            with mlflow.start_run():
                mlflow.set_tag("model", "randomforest")
                mlflow.log_params(params)

                pipeline = make_pipeline(
                    DictVectorizer(),
                    RandomForestRegressor(**params)
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)

                rmse = mean_squared_error(y_val, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
            return {'loss': rmse, 'status': STATUS_OK}


        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 2)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 2)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
            'random_state': 42
        }

        rstate = np.random.default_rng(42)  # for reproducible results
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.config.num_trials,
            trials=Trials(),
            rstate=rstate
        )
        
        # joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

