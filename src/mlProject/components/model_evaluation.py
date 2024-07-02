import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle
from mlProject import logger
from mlProject.entity.config_entity import (ModelEvaluationConfig)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def predict_data(self):
        logger.info (f"predicting using model from {self.config.model_path}")
        with open(self.config.model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)

        dftmp = pd.read_csv(self.config.pred_data_path)
        categorical = ['PULocationID', 'DOLocationID']
        dicts = dftmp[categorical].to_dict(orient='records')
        X_val = dv.transform(dicts)
        y_pred = model.predict(X_val)

        prdStd = y_pred.std()
        prdMean = y_pred.mean()
        logger.info (f"Standard deviation of preds - {prdStd}")
        logger.info (f"Mean of preds - {prdMean}")

        logger.info ("creating results dataframe...")
        df_res = pd.DataFrame({
            'ride_id': dftmp['ride_id'],
            'predicted_duration': y_pred
        })

        logger.info (f'writing results to - {self.config.root_dir}')
        df_res.to_parquet(
            self.config.root_dir + f'/yellow_{self.config.year:04d}-{self.config.month:02d}.parquet',
            engine='pyarrow',
            compression=None,
            index=False
        )
        sz = os.path.getsize(self.config.root_dir) / (1024*1024)
        logger.info(f'df_results file-size - {sz}')
        logger.info("Results file written.")