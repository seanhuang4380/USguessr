import pathlib 
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random

def visualize(image) -> None:
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(image)

def create_augmentations() -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.RandomGamma(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.VerticalFlip(p=0.3),
        A.ToGray(p=0.2),
        A.ColorJitter(p=0.5),
        ToTensorV2()
    ])

def augment_images(input_folder: str, output_folder: str, num_aug_images: int):
    ifolder = Path(input_folder)
    ofolder = Path(output_folder)

    if not Path.exists(ofolder):
        ofolder.mkdir()


    image_files = [f for f in ifolder.iterdir()]
    augmenter = create_augmentations()

    for image_path in tqdm(image_files, desc="Processing Images"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 

        base_name = image_path.stem

        for i in range(num_aug_images):
            augmented = augmenter(image=image)['image']
            augmented = augmented.permute(1,2,0).numpy() #CHW to HWC
            # augmented = (augmented*255).astype(np.uint8) # only needed if normalization occurs inside augmenter
            output_path = ofolder / Path(f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print(f"Images saved to {ofolder}")


def main():
    input_folder = input("Input folder: ")
    output_folder = input("Output folder: ")
    num_images = input("number of aug images: ")

    if not input_folder:
        input_folder = "aug_testing/src_images"
    if not output_folder:
        output_folder = "aug_testing/aug_images"
    if num_images:
        num_images = int(num_images)
    else:
        num_images = 5


    augment_images(input_folder, output_folder, num_images)
    


if __name__ == "__main__":
    main()
