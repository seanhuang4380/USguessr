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
from matplotlib import pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
END_EPOCH = 15 
DATA_DIR = "US/data"
BATCH_SIZE = 32
data_dir = Path(DATA_DIR)
start_epoch = 0 # may change if .ckpt is loaded

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
    v2.RandomResizedCrop(
        (224, 224),
        scale=(0.8, 1.0)
    ),
    v2.RandomHorizontalFlip(p=0.5),
    #  v2.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.05
    # ),
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
    top5_correct_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0) 

            # top 1
            _, predicted = outputs.max(1)

            # top 5
            top5 = outputs.topk(5, dim=1).indices
            top5_correct = top5.eq(labels.unsqueeze(1)).any(dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            top5_correct_count += top5_correct.sum().item()

    accuracy = 100 * correct / total
    top5_accuracy = 100 * top5_correct_count/total
    val_loss = val_loss / total 

    return val_loss, accuracy, top5_accuracy 

def confusion_matrix(model, val_loader, num_classes):
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype = int)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)
            for actual, pred in zip(labels, predicted):
                cm[actual.item(), pred.item()] += 1 

    return cm


def plot_cm(cm, class_names):
    fig, ax = plt.subplots(figsize = (16,16))

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names, rotation = 90)
    ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.show()


def print_prediction_distribution(cm, class_names):

    predicted_counts = cm.sum(axis=0)

    for i, class_name in enumerate(class_names):
        print(
            f"{class_name}: "
            f"{predicted_counts[i]} predictions"
        )


def print_state_acc(cm, class_names):
    for i, class_name in enumerate(class_names):
        correct = cm[i,i]
        total = cm[i].sum()

        accuracy = 100 * correct / total
        print(f"{class_name}: {accuracy:.2f}%")


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
    model_to_load = input("Enter model to load: ") 
    if model_to_load and not model_to_load.endswith((".pt", ".ckpt")):
        raise InvalidFileTypeException() 

    while (version_name := input("What name do you want to save this model as? ")) == "":
        print("Name cannot be empty.")

    # disallow overwriting models 
    check_folders = [Path("saves"), Path("checkpoints")]
    version_name = Path(version_name).stem # normalize version_name to the prefix

    exists  = False
    for folder in check_folders:
        for f in folder.iterdir():
            if (f.name.startswith(version_name) and (f.suffix == ".pt" or f.suffix == ".ckpt")): 
                exists = True
                break

    if (exists):
        check = input("This save already exists. Override it? (y/n): ")
        if (check.lower() != "y"):
            raise DuplicateFileException(version_name)

    #dataset info

    # print("Training Dataset Info: ")
    # get_dataset_info(train_dataset)
    # print("Validation Dataset Info: ")
    # get_dataset_info(val_dataset)

    #load model

    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    for param in model.parameters(): # freeze before adding FC layer
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_features, len(train_dataset.classes))
        )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr':LR},
        {'params': model.layer4.parameters(), 'lr':0.0001}
    ]) 


    # default values
    best_model = None
    best_accuracy = 0.0
    best_top5_acc = 0.0
    start_epoch = 0
    best_epoch = None
    best_train_loss = None
    best_val_loss = None

    # loaded values
    if model_to_load: 
        checkpoint = torch.load(model_to_load, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        if ('optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['val_accuracy']
            best_model = copy.deepcopy(model.state_dict())

    #training/validation

    start_time = time.time()
    for epoch in range(start_epoch, END_EPOCH):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_top5_acc = validate(model, val_loader, criterion)

        # confusion matrix and state accuracy
        # cm = confusion_matrix(model ,val_loader, len(val_dataset.classes))
        # cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims = True)
        # print_state_acc(cm_normalized, val_dataset.classes)
        # print_prediction_distribution(cm, val_dataset.classes)
        # plot_cm(cm_normalized, val_dataset.classes)
        
        print(
            f"Epoch {epoch+1}: "
            f"Train loss: {train_loss:.4f}\n"
            f"Validation Loss: {val_loss:.4f}\n"
            f"Top 1 Accuracy: {val_acc:.4f}%\n"
            f"Top-5 Accuracy: {val_top5_acc:.2f}%"
        )
        
        checkpoint_path = Path("checkpoints") / Path(f"{version_name}_epoch_{epoch + 1}.ckpt") 
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':val_loss, 
            'val_accuracy': val_acc,
            'val_top5_acc': val_top5_acc
        }

        torch.save(checkpoint, checkpoint_path) # save checkpoint each epoch
        if (val_acc > best_accuracy) or best_model is None:
            best_model = copy.deepcopy(model.state_dict())
            best_accuracy = val_acc
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_top5_acc = val_top5_acc

        print(f"Checkpoint saved as {checkpoint_path}.\n")


    # save the model with best accuracy
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    opath = Path("saves") / Path(f"{version_name}.pt") 
    save = {
        'model_state_dict': best_model,
        'accuracy': best_accuracy 
    }
    torch.save(save, opath)
    print(f"Model saved as {opath}\n") 


    print(f"Best Model stats: Name, Epochs, Best Epoch, Train Loss, Val Loss, Top 1 Acc, Top 5 Acc, Minutes\n")
    print(f"{version_name} {END_EPOCH} {best_epoch} {best_train_loss} {best_val_loss} {best_accuracy} {best_top5_acc} {total_time/60:.2f}")

    with open("stats.txt", mode="a") as f:
        f.write(f"{version_name} {END_EPOCH} {best_epoch} {best_train_loss:.4f} {best_val_loss:.4f} {best_accuracy:.4f} {total_time/60:.2f}\n")


if __name__ == "__main__":
    main()
