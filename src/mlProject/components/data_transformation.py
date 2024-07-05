import os
from mlProject import logger
import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_extraction import DictVectorizer
from mlProject.entity.config_entity import (DataTransformationConfig)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    
    def dump_pickle(self, obj, filename: str):
        with open(filename, "wb") as f_out:
            return pickle.dump(obj, f_out)


    def read_dataframe(self, filename: str):
        df = pd.read_parquet(filename)

        df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        df['ride_id'] = f'{self.config.year_train:04d}/{self.config.month_train:02d}_' + df.index.astype('str')

        return df


    def preprocess(self, df: pd.DataFrame,):  
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')
        return dicts

    def process_data(self):
        logger.info(f'reading file - {self.config.data_path}')
        df_train = self.read_dataframe(
        os.path.join(self.config.data_path, f'{self.config.color}_tripdata_{self.config.year_train:04d}-{self.config.month_train:02d}.parquet')
        )
        df_val = self.read_dataframe(
            os.path.join(self.config.data_path, f'{self.config.color}_tripdata_{self.config.year_val:04d}-{self.config.month_val:02d}.parquet')
        )
        df_test = self.read_dataframe(
            os.path.join(self.config.data_path, f'{self.config.color}_tripdata_{self.config.year_test:04d}-{self.config.month_test:02d}.parquet')
        )

        # Extract the target
        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values
        y_test = df_test[target].values

        # Preprocess data
        logger.info("preprocess data.")
        X_train = self.preprocess(df_train) 
        X_val = self.preprocess(df_val) 
        X_test = self.preprocess(df_test) 

        # Save DictVectorizer and datasets
        self.dump_pickle((X_train, y_train), os.path.join(self.config.root_dir, "train.pkl"))
        self.dump_pickle((X_val, y_val), os.path.join(self.config.root_dir, "val.pkl"))
        self.dump_pickle((X_test, y_test), os.path.join(self.config.root_dir, "test.pkl"))

        logger.info("Data transformation completed")
