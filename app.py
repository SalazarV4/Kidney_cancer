import gradio as gr
from kidney_cancer.pipeline.prediction import PredictionPipeline

import dagshub
from kidney_cancer import logger
from kidney_cancer.pipeline.data_ingestion import DataIngestionTrainingPipeline
from kidney_cancer.pipeline.base_model import BaseModelTrainingPipeline
from kidney_cancer.pipeline.model_training import ModelTrainingPipeline
from kidney_cancer.pipeline.model_evaluation import ModelEvaluationPipeline

#dagshub.init(repo_owner='SalazarV4', repo_name='Kidney_cancer', mlflow=True)

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

STAGE_NAME_3 = "Training"
try:
    logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME_3)
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME_3)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME_4 = "Evaluation"

try:
    logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME_4)
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME_4)
except Exception as e:
    logger.exception(e)
    raise e


demo = gr.Interface(fn=PredictionPipeline.predict,
             inputs=gr.Image(type='pil'),
             outputs=gr.Label(num_top_classes=2),
             examples=["artifacts/data_ingestion/Kidney_dataset/Test/Normal/Normal- (449).jpg",
                       "artifacts/data_ingestion/Kidney_dataset/Test/Tumor/Tumor- (450).jpg"])

demo.launch(server_name="0.0.0.0", server_port=8080)
