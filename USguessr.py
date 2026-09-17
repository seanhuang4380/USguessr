#TODO
# split train/test
# loading model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCHS = 10
DATA_DIR = "US/data"
data_dir = Path(DATA_DIR)

def get_dataset_info(dataset):
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class Names: {dataset.classes}")
    print(f"Number of Images: {len(dataset)}")

    class_counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1
    print("Class Distribution: ")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

#-------------------------------------training----------------------------------------

train_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#------------------------------------------validation---------------------------------------------

val_transform = v2.Compose([
v2.Resize((224, 224)),
v2.ToTensor(),
v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])

val_dataset = datasets.ImageFolder(data_dir / 'test', transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


class DuplicateFileException(Exception):
    def __init__(self, file_name: str) -> None:
        self.message = f"File: {file_name} already exists"
        super().__init__(self.message)


def main():
    #-----------------------------------userinput---------------------------------

    while (version_name := input("What name do you want to save this model as?")) == "":
        print("Name cannot be empty")

    # disallow overwriting models 
    check_folders = [Path("versions"), Path("in_progress")]
    prefix = Path(version_name).stem
    for folder in check_folders:
        for f in folder.iterdir():
            if (f.name.startswith(prefix) and f.suffix == ".pth"): 
                raise DuplicateFileException(version_name)

    while (epochs := input("Number of epochs")) == "":
        print("Invalid number")
    epochs = int(epochs)

    # -----------------------------------------load model------------------------

    print("Training Dataset Info: ")
    get_dataset_info(train_dataset)
    print("Validation Dataset Info: ")
    get_dataset_info(val_dataset)

    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    #------------------------------------training/validation-----------------------------

    best_model = None
    best_accuracy = 0.0

    start_time = time.time()
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}: "
            f"Validation Accuracy = {val_acc:.2f}%"
        )
        
        checkpoint_path = Path("in_progress") / Path(f"{version_name}_epoch_{epoch}.ckpt") 
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':val_loss, 
            'accuracy': val_acc
        }
        torch.save(checkpoint, checkpoint_path) # save checkpoint each epoch
        if (val_acc > best_accuracy):
            best_model = model.state_dict()

        print(f"Checkpoint saved to {checkpoint_path}.")


    # save the model with best accuracy
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    opath = Path("saves") / Path(f"{version_name}.pt") 
    save = {
        'model_state_dict': best_model,
        'accuracy': best_accuracy 
    }
    torch.save(save, opath) # save model to saves 
    print(f"Model saved to {opath}") 

if __name__ == "__main__":
    main()