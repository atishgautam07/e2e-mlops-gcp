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
        if not os.path.exists(self.config.local_data_file):
            filename = self.config.local_data_file
            pd.read_parquet(self.config.source_URL).to_parquet(filename)
            logger.info(f"{filename} download! \n")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
