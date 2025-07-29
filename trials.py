from torchvision.transforms import v2
from PIL import Image

img = Image.open(r"C:\Users\User\Desktop\DS\Kidney_cancer\artifacts\data_ingestion\Kidney_dataset\Train\Normal\Normal- (2282).jpg")
transform = v2.Compose([
    v2.Resize(size=[224,224]),
    v2.ConvertImageDtype(),
    v2.PILToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]       )
])

transform(img)