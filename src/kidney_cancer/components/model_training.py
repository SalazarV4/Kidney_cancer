import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
from kidney_cancer import logger
from kidney_cancer.config.configuration import ConfigurationManager
from kidney_cancer.entity.config_entity import TrainingConfig
import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import mlflow

class ModelTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_base_model(self):
        self.model = torch.load(
            self.config.updated_base_model_path,
            weights_only=False
        )
    
    def train_val_loader(self):
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=[224,224]),
            v2.ToDtype(dtype=torch.float32,scale=True)])

        self.train_data = ImageFolder(
            root=self.config.training_data / "Train",
            transform=self.transforms)

        self.val_data = ImageFolder(
            root=self.config.training_data / "Val",
            transform=self.transforms)
        
        if self.config.params_is_augmentation:
            pass
            
        self.train_dataloader = DataLoader(self.train_data,
                                           batch_size=self.config.params_batch_size,
                                           shuffle=True)
        
        self.val_dataloader = DataLoader(self.val_data,
                                          batch_size=self.config.params_batch_size,
                                          shuffle=True)
    
    @staticmethod
    def save_model(path: Path, model: torchvision.models):
        torch.save(obj=model,f=path)

    def train(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.model.parameters(),lr=self.config.params_learning_rate)
        self.model = self.model.to(self.device)
        for epoch in range(self.config.params_epoch):

            self.model.train()

            train_loss, train_acc = 0, 0
            logger.info("Training Started")
            for batch, (X, y) in tqdm(enumerate(self.train_dataloader)):

                X, y = X.to(self.device), y.to(self.device)

                y_pred = self.model(X)

                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

                self.optim.zero_grad()

                loss.backward()

                self.optim.step()

                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_class == y).sum().item()/len(y_pred)

                train_loss = train_loss / len(self.train_dataloader)
                train_acc = train_acc / len(self.train_dataloader)

                if batch + 1 %10 == 0:
                    logger.info("Training loss: %s | Train Accuracy: %s",train_loss,train_acc)
            logger.info("Evaluating model..")
            self.model.eval()

            val_loss, val_acc = 0, 0

            with torch.inference_mode():
                for batch, (X_val, y_val) in tqdm(enumerate(self.val_dataloader)):
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                    val_pred_logits = self.model(X_val)

                    loss = self.loss_fn(val_pred_logits, y_val)
                    val_loss += loss.item()

                    val_pred_labels = val_pred_logits.argmax(dim=1)

                    val_acc += ((val_pred_labels == y_val).sum().item()/len(val_pred_labels))


            val_loss = val_loss / len(self.val_dataloader)
            val_acc = val_acc / len(self.val_dataloader)
            logger.info("Training Loss: %s | Train Accuracy: %s | Val Loss: %s | Val Accuracy: %s",
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc)

            self.save_model(self.config.trained_model_path, self.model)
            logger.info("Model Saved at %s",self.config.trained_model_path)

            self.score = (train_loss, train_acc, val_loss, val_acc)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"train_loss": self.score[0],
                 "train_accuracy": self.score[1],
                 "val_loss": self.score[2],
                 "val_accuracy":self.score[3]}
            )

            # if tracking_url_type_store != "file":
            #     mlflow.pytorch.log_model(pytorch_model=self.model,
            #                              artifact_path="model",
            #                              registered_model_name="VGG16")
            # else:
            #     mlflow.pytorch.log_model(self.model, "model")

    def copy_model(self):
        os.makedirs(self.config.model_dir,exist_ok=True)
        shutil.copyfile(self.config.trained_model_path, self.config.model_dir / "model.pth")
