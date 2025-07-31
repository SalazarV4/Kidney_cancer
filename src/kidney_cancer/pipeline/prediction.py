import os
import numpy as np
import torch
from torchvision.transforms import v2
import torchvision.io as io

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = torch.load(f="artifacts/model_training/model.pth", weights_only=False)

        image = io.read_image(self.filename)

        transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=[224,224]),
            v2.ToDtype(dtype=torch.float32,scale=True)])

        model.eval()
        x = transforms(image).unsqueeze(dim=0)
        y_pred = model(x)
        result = np.argmax(y_pred.detach().numpy(),axis=1)

        if result[0] == 1:
            return [{'image':'Tumor'}]
        else:
            return [{'image':'Normal'}]
