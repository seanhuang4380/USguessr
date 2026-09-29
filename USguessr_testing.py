from torchvision.transforms import v2
from pathlib import Path
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision import models
import USguessr as usg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DATA_DIR = "US/data"
data_dir = Path(DATA_DIR)
test_dataset = datasets.ImageFolder(data_dir / 'test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()


def main():
    model_to_load = input("Enter model to load: ") 
    if not model_to_load.endswith((".pt", ".ckpt")):
        raise usg.InvalidFileTypeException() 

    
    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_features, len(test_dataset.classes))
        )
    model = model.to(DEVICE)

    checkpoint = torch.load(model_to_load, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    

    val_loss, top1_acc, top5_acc = usg.validate(model, test_loader, criterion)

    print(
            f"Validation Loss: {val_loss:.4f}\n"
            f"Top 1 Accuracy: {top1_acc:.4f}%\n"
            f"Top-5 Accuracy: {top5_acc:.2f}%"
        )
        
if __name__ == "__main__":
    main()