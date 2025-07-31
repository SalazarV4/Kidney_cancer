from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from kidney_cancer import logger
from kidney_cancer.utils.common import save_json
from kidney_cancer.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.CrossEntropyLoss()

    def test_loader(self):
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=[224,224]),
            v2.ToDtype(dtype=torch.float32,scale=True)])

        self.test_data = ImageFolder(
            root=self.config.training_data / "Test",
            transform=self.transforms)

        self.test_dataloader = DataLoader(self.test_data,
                                           batch_size=self.config.params_batch_size,
                                           shuffle=True)

    def test(self):
        self.model.eval()

        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for batch, (X_test, y_test) in tqdm(enumerate(self.test_dataloader)):

                X_test, y_test = X_test.to(self.device), y_test.to(self.device)

                test_pred_logits = self.model(X_test)

                loss = self.loss_fn(test_pred_logits, y_test)
                test_loss += loss.item()

                test_pred_labels = test_pred_logits.argmax(dim=1)

                test_acc += ((test_pred_labels == y_test).sum().item()/len(test_pred_labels))

            test_loss = test_loss / len(self.test_dataloader)
            test_acc = test_acc / len(self.test_dataloader)
            logger.info("Testing Loss: %s | Testing Accuracy: %s",test_loss,test_acc)

            return test_loss, test_acc


    def load_model(self):
        return torch.load(self.config.path_of_model, weights_only=False)

    def evaluation(self):
        self.model = self.load_model()
        self.test_loader()
        self.score = self.test()
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0],
                 "accuracy": self.score[1]}
            )

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(pytorch_model=self.model,
                                         artifact_path="model",
                                         registered_model_name="VGG16")
            else:
                mlflow.pytorch.log_model(self.model, "model")
