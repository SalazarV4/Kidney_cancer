from kidney_cancer.config.configuration import ConfigurationManager
from kidney_cancer.components.base_model import PrepareBaseModel
from kidney_cancer import logger

STAGE_NAME = "Base Model"

class BaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()



if __name__ == '__main__':
    try:
        logger.info("*"*10)
        logger.info(">>>>>>>> Preparing %s <<<<<<<<", STAGE_NAME)
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>>>> %s Created <<<<<<<<\n\n", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
