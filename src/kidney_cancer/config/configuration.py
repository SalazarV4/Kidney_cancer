from pathlib import Path
import os
from kidney_cancer.constants import  CONFIG_FILE_PATH, PARAMS_FILE_PATH
from kidney_cancer.utils.common import read_yaml, create_directories
from kidney_cancer.entity.config_entity import (DataIngestionConfig,
                                                BaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS
        )

        return base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.model_training
        base_model = self.config.base_model
        params = self.params
        data_path = os.path.join(self.config.data_ingestion.unzip_dir,"Kidney_dataset")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(base_model.updated_base_model_path),
            training_data=Path(data_path),
            params_epoch=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation_config= EvaluationConfig(
            path_of_model=Path("artifacts/model_training/model.pth"),
            training_data=Path("artifacts/data_ingestion/Kidney_dataset"),
            mlflow_uri="https://dagshub.com/SalazarV4/Kidney_cancer.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return evaluation_config
