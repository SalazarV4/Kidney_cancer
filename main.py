from kidney_cancer import logger
from kidney_cancer.pipeline.data_ingestion import DataIngestionTrainingPipeline
from kidney_cancer.pipeline.base_model import BaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(">>>>>>> Initializing %s <<<<<<<", STAGE_NAME)
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>>>>>> %s Completed <<<<<<<", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Base Model"

try:
    logger.info(">>>>>>>> Preparing %s <<<<<<<<", STAGE_NAME)
    base_model = BaseModelTrainingPipeline()
    base_model.main()
    logger.info(">>>>>>>> %s Succesfully Prepared <<<<<<<<", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e
