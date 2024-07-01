from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    color: str
    year: int
    month: int
    root_dir: Path
    source_URL: str
    local_data: Path
    local_data_file: Path



@dataclass(frozen=True)
class DataTransformationConfig:
    year: int
    month: int
    root_dir: Path
    data_path: Path