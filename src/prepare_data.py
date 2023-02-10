import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import config

# Define data augmentation transforms
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load all normal images
normal_images = []
normal_labels = []

for file_name in os.listdir(f"images/{config.NORMAL_DATA_STR}"):
    image = Image.open(os.path.join(f"images/{config.NORMAL_DATA_STR}", file_name))
    image = data_transform(image)
    normal_images.append(image)
    normal_labels.append(0)

# Load all diseased images
diseased_images = []
diseased_labels = []

for file_name in os.listdir(f"images/{config.DISEASE_DATA_STR}"):
    image = Image.open(os.path.join(f"images/{config.DISEASE_DATA_STR}", file_name))
    image = data_transform(image)
    diseased_images.append(image)
    diseased_labels.append(1)

# Concatenate normal and diseased images and labels
images = normal_images + diseased_images
labels = normal_labels + diseased_labels

# Convert the lists to PyTorch tensors
images = torch.stack(images)
labels = torch.tensor(labels)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=config.VAL_SPLIT_SIZE, random_state=42)


torch.save(
    {
        "train" : {

            "train_images":train_images,
            "train_labels" : train_labels,

        },
        "validation" : {

            "val_images":val_images,
            "val_labeld" : val_labels
        },
        "test" :{

            "test_images" : val_images[:config.TEST_SPLIT_SIZE],
            "test_labels": val_labels[:config.TEST_SPLIT_SIZE]
        },
        "metadata" : {
            "train_size" : len(train_labels),
            "test_size" : len(val_labels),
        }
    },"dataset/train_val_test.pt"

)
print("Data Prepared.")
