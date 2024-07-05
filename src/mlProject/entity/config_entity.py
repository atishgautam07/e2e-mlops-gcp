from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    color: str
    year_train: int
    month_train: int
    year_val: int
    month_val: int
    year_test: int
    month_test: int
    root_dir: Path
    source_URL: str



@dataclass(frozen=True)
class DataTransformationConfig:
    color: str
    year_train: int
    month_train: int
    year_val: int
    month_val: int
    year_test: int
    month_test: int
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    num_trials: int
    train_data_path: Path
    test_data_path: Path
    mlflow_uri: str
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    top_n: int
    ml_uri: str
    hpo_exp: str
    exp_name: str
    rf_params: list