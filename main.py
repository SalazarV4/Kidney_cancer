from kidney_cancer import logger
from kidney_cancer.pipeline.data_ingestion import DataIngestionTrainingPipeline
from kidney_cancer.pipeline.base_model import BaseModelTrainingPipeline
from kidney_cancer.pipeline.model_training import ModelTrainingPipeline
from kidney_cancer.pipeline.model_evaluation import ModelEvaluationPipeline
from argparse import ArgumentParser
import dagshub

parser = ArgumentParser(description="Main file for Kidney Cancer detector")
parser.add_argument("-t",
                    "--training",
                    help="Skip model training from pipeline",
                    default=False)

args = parser.parse_args()
dagshub.init(repo_owner='SalazarV4', repo_name='Kidney_cancer', mlflow=True)

STAGE_NAME_1 = "Data Ingestion"
try:
    logger.info(">>>>>>> Initializing %s <<<<<<<", STAGE_NAME_1)
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>>>>>> %s Completed <<<<<<<", STAGE_NAME_1)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_2 = "Base Model"

try:
    logger.info(">>>>>>>> Preparing %s <<<<<<<<", STAGE_NAME_2)
    base_model = BaseModelTrainingPipeline()
    base_model.main()
    logger.info(">>>>>>>> %s Succesfully Prepared <<<<<<<<", STAGE_NAME_2)
except Exception as e:
    logger.exception(e)
    raise e

if args.training:
    STAGE_NAME_3 = "Training"
    try:
        logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME_3)
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME_3)
    except Exception as e:
        logger.exception(e)
        raise e
else:
    print("[INFO]: Omitting model training")


STAGE_NAME_4 = "Evaluation"

try:
    logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME_4)
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME_4)
except Exception as e:
    logger.exception(e)
    raise e
