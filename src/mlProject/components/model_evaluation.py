import os
import mlflow
import pickle
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlProject import logger
from mlProject.entity.config_entity import (ModelEvaluationConfig)



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def load_pickle(self, filename):
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)


    def run_register_model(self):

        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        mlflow.sklearn.autolog()
        client = MlflowClient(tracking_uri=self.config.ml_uri)

        # X_train, y_train = self.load_pickle(os.path.join(self.config.data_path, "train.pkl"))
        X_val, y_val = self.load_pickle(os.path.join(self.config.data_path, "val.pkl"))
        X_test, y_test = self.load_pickle(os.path.join(self.config.data_path, "test.pkl"))

        # Retrieve the top_n model runs and log the models
        logger.info("Retrieve the top_n model runs and log the models.")
        experiment = client.get_experiment_by_name(self.config.hpo_exp)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.rmse ASC"]
        )
        logger.info(len(runs))
        logger.info("logging top_n models wiht test metrics.")
        
        for run in runs:
            logger.info((str(run.info.run_id), str(run.data.metrics), str(run.data.params)))
            modelPath = client.download_artifacts(run_id=run.info.run_id, path="model")
            pipeLine = self.load_pickle(os.path.join(modelPath, "model.pkl"))

            with mlflow.start_run():

                mlflow.set_tag("model", "randomforest_topN")
                mlflow.log_params(run.data.params)
                
                logger.info("Evaluate model on the validation and test sets")
                val_rmse = mean_squared_error(y_val, pipeLine.predict(X_val), squared=False)
                mlflow.log_metric("val_rmse", val_rmse)
                test_rmse = mean_squared_error(y_test, pipeLine.predict(X_test), squared=False)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.sklearn.log_model(pipeLine, artifact_path="model")

        logger.info("Selecting the model with the lowest test RMSE")
        experiment = client.get_experiment_by_name(self.config.exp_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.test_rmse ASC"]
        )[0]

        # Register the best model
        logger.info("Registering the best model")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")