from kidney_cancer.config.configuration import ConfigurationManager
from kidney_cancer.components.model_evaluation import Evaluation
from kidney_cancer import logger

STAGE_NAME = "Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(">>>>>>>> %s Started <<<<<<<<", STAGE_NAME)
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(">>>>>>>> %s Finished <<<<<<<<\n\n", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
