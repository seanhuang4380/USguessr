#TODO
# epochs from checkpoint doesn't work

import torch
torch.backends.cudnn.enabled = False # disables miopen, which batchnorm2d, and subsequently, resnet, uses during forward pass
                                        # crashes otherwise
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision import models, datasets
import numpy as np
from pathlib import Path
from PIL import Image
import time
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCHS = 10 # ending epoch
DATA_DIR = "US/data"
BATCH_SIZE = 32
data_dir = Path(DATA_DIR)
start_epoch = 0 # may change if a ckpt is loaded

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
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / total
#------------------------------------------validation---------------------------------------------

val_transform = v2.Compose([
v2.Resize((224, 224)),
v2.ToImage(),
v2.ToDtype(torch.float32, scale=True),
v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])

val_dataset = datasets.ImageFolder(data_dir / 'validation', transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
assert train_dataset.class_to_idx == val_dataset.class_to_idx

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

            val_loss += loss.item() * labels.size(0) 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    val_loss = val_loss / total 
    return val_loss, accuracy

class InvalidFileTypeException(Exception):
    """Exception raised when a file type is not supported."""
    def __init__(self, message="The provided file type is invalid."):
        self.message = message
        super().__init__(self.message)


class DuplicateFileException(Exception):
    """Exception raised when a model state already exists"""
    def __init__(self, file_name: str) -> None:
        self.message = f"File: {file_name} already exists"
        super().__init__(self.message)


def main():
    print(DEVICE)
    #-----------------------------------userinput---------------------------------
    for folder in ["saves", "checkpoints"]:
        Path(folder).mkdir(exist_ok=True)

    # model can either be empty, pt, or ckpt file
    model_to_load = input("Enter model to load") 
    if model_to_load and not model_to_load.endswith((".pt", ".ckpt")):
        raise InvalidFileTypeException() 

    while (version_name := input("What name do you want to save this model as? ")) == "":
        print("Name cannot be empty")

    # disallow overwriting models 
    check_folders = [Path("saves"), Path("checkpoints")]
    version_name = Path(version_name).stem # normalize version_name to the prefix

    for folder in check_folders:
        for f in folder.iterdir():
            if (f.name.startswith(version_name) and (f.suffix == ".pt" or f.suffix == ".ckpt")): 
                raise DuplicateFileException(version_name)

    #-------------------------------------dataset info-----------------------------------

    # print("Training Dataset Info: ")
    # get_dataset_info(train_dataset)
    # print("Validation Dataset Info: ")
    # get_dataset_info(val_dataset)

    # -----------------------------------------load model------------------------

    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    for param in model.parameters(): # freeze before adding FC layer
        param.requires_grad = False

    model.fc = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)


    best_model = None
    best_accuracy = 0.0
    if model_to_load: # resume training with last accuracy and last epoch loaded
        checkpoint = torch.load(model_to_load, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        if ('optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['val_accuracy']
            best_model = copy.deepcopy(model.state_dict())

    #------------------------------------training/validation-----------------------------

    start_time = time.time()
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}: "
            f"Train loss: {train_loss:.4f}\n"
            f"Validation Loss: {val_loss:.4f}\n"
            f"Validation Accuracy: {val_acc:.4f}%\n"
        )
        
        checkpoint_path = Path("checkpoints") / Path(f"{version_name}_epoch_{epoch + 1}.ckpt") 
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':val_loss, 
            'val_accuracy': val_acc
        }
        torch.save(checkpoint, checkpoint_path) # save checkpoint each epoch
        if (val_acc > best_accuracy) or best_model is None:
            best_model = copy.deepcopy(model.state_dict())
            best_accuracy = val_acc

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
