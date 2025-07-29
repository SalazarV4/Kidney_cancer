from kidney_cancer.config.configuration import ConfigurationManager
from kidney_cancer.components.model_training import ModelTraining
from kidney_cancer import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = ModelTraining(config=training_config)
        training.get_base_model()
        training.train_val_loader()
        training.train()


if __name__ == "__main__":
    try:
        logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME)
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
