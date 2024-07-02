from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.dataDetails

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            color=params.color,
            year=params.year,
            month=params.month,
            root_dir=config.root_dir,
            source_URL=config.source_URL + f'{params.color}_tripdata_{params.year:04d}-{params.month:02d}.parquet',
            local_data=config.local_data,
            local_data_file=config.local_data_file + f'{params.color}_tripdata_{params.year:04d}-{params.month:02d}.parquet'
        )

        return data_ingestion_config
    
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.dataDetails

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            year=params.year,
            month=params.month,
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.dataDetails

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            pred_data_path=config.pred_data_path,
            model_path = config.model_path,
            year=params.year,
            month=params.month            
        )

        return model_evaluation_config



    