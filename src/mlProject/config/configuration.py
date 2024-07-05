from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)

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
            year_train=params.year_train,
            month_train=params.month_train,
            year_val=params.year_val,
            month_val=params.month_val,
            year_test=params.year_test,
            month_test=params.month_test,
            
            root_dir=config.root_dir,
            source_URL=config.source_URL,
        )

        return data_ingestion_config    
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.dataDetails

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            color=params.color,
            year_train=params.year_train,
            month_train=params.month_train,
            year_val=params.year_val,
            month_val=params.month_val,
            year_test=params.year_test,
            month_test=params.month_test,
            
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            num_trials=config.num_trials,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            mlflow_uri = config.mlflow_uri
            
        )

        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            top_n=config.top_n,
            ml_uri=config.ml_uri,
            hpo_exp=config.hpo_exp,
            exp_name=config.exp_name,
            rf_params=config.rf_params,
            
        )

        return model_evaluation_config



    