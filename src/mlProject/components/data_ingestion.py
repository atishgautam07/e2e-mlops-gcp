import os
from mlProject import logger
from mlProject.utils.common import get_size
import pandas as pd
from pathlib import Path
from mlProject.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):

        filename_train = f'{self.config.color}_tripdata_{self.config.year_train:04d}-{self.config.month_train:02d}.parquet'
        pd.read_parquet(self.config.source_URL + filename_train).to_parquet(self.config.root_dir + "/" + filename_train)
        logger.info(f"{filename_train} download! \n")

        filename_val = f'{self.config.color}_tripdata_{self.config.year_val:04d}-{self.config.month_val:02d}.parquet'
        pd.read_parquet(self.config.source_URL + filename_val).to_parquet(self.config.root_dir + "/" + filename_val)
        logger.info(f"{filename_val} download! \n")

        filename_test = f'{self.config.color}_tripdata_{self.config.year_test:04d}-{self.config.month_test:02d}.parquet'
        pd.read_parquet(self.config.source_URL + filename_test).to_parquet(self.config.root_dir + "/" + filename_test)
        logger.info(f"{filename_test} download! \n")
