from kidney_cancer.config.configuration import ConfigurationManager
from kidney_cancer.components.data_ingestion import DataIngestion
from kidney_cancer import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass      
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(">>>>>>> stage %s started <<<<<<<", STAGE_NAME)
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(">>>>>>> stage %s completed <<<<<<<\n\n ================", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
