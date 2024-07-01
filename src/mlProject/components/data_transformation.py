import os
from mlProject import logger
import pandas as pd
from pathlib import Path
from mlProject.entity.config_entity import (DataTransformationConfig)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def process_data(self):
        logger.info(f'reading file - {self.config.data_path}')
        df = pd.read_parquet(self.config.data_path)
        
        categorical = ['PULocationID', 'DOLocationID']
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        df['ride_id'] = f'{self.config.year:04d}/{self.config.month:02d}_' + df.index.astype('str')

        df.to_csv(os.path.join(self.config.root_dir, "predSet.csv"),index = False)
        logger.info("Data transformation completed")
        logger.info(df.shape)
        
        return df