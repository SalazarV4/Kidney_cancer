import os
import numpy as np
import torch
from torchvision.transforms import v2
import torchvision.io as io

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def predict(image):
        model = torch.load(f="model/model.pth", weights_only=False)

        # image = io.read_image(self.filename)
        labels = ["Normal", "Tumor"]


        transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=[224,224]),
            v2.ToDtype(dtype=torch.float32,scale=True)])
        image = image.convert('RGB')
        x = transforms(image).unsqueeze(dim=0)
        model.eval()
        with torch.inference_mode():
            logits = model(x)
            y_pred = torch.sigmoid(logits)
            return {labels[i]:float(y_pred[0][i]) for i in range(2)}
