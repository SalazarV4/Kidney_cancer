from pathlib import Path
from torchvision import models
import torchvision
import torch
from torch import nn
from kidney_cancer.entity.config_entity import BaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    @staticmethod
    def save_model(path: Path, model: torchvision.models):
        torch.save(model, path)

    def get_base_model(self):
        self.model = models.vgg16(weights=self.config.params_weights)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_features, freeze_till=None):
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in list(model.parameters())[:-freeze_till]:
                param.requires_grad = False

        model.classifier[6] = nn.Linear(
            in_features=model.classifier[6].in_features,
            out_features=classes)

        print(model)
        return model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_features=True
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
