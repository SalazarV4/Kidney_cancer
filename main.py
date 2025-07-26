from kidney_cancer import logger
from kidney_cancer.pipeline.data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>>>>>> stage %s completed <<<<<<<\n\n ================", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e
