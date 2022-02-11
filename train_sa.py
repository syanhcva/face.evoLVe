import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision import datasets
from PIL import Image
import os
import shutil

from backbone.model_irse import IR_50
from head.metrics import ArcFace

input_size = [112, 112]
embedding_size = 512
output_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")
device_id = 0
weight_path = "./weights/backbone_ir50_asia.pth"
test_img_path = "./test_images/1.jpg"
img = Image.open(test_img_path)

backbone = IR_50(input_size)
if device.type == "cpu":
    backbone.load_state_dict(torch.load(weight_path, map_location='cpu'))
else:
    backbone.load_state_dict(torch.load(weight_path))

head = ArcFace(in_features=embedding_size, out_features=output_size, device_id=device_id)
train_transform = transforms.Compose(
    [  # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
path = "/Users/anhnguyen/Desktop/dataset/VGGFace2/test"
dataset_train = datasets.ImageFolder(path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=16,
)

images, labels = next(iter(train_loader))

features = backbone(images)
outputs = head(features, labels.long())
print(outputs.shape)
