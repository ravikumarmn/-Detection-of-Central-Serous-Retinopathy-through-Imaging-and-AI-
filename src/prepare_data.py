# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, Dataset
# import config
# import argparse
# import multiprocessing as mp
# import pickle

# def data_transforms():
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return transform

# def preprocess_image(file_name):
#     image = Image.open(file_name)
#     label = 0 if "normal" in file_name else 1
#     image = data_transforms()(image)
#     return image, label

# class CustomDataset(Dataset):
#     def __init__(self, image_files, transform=None, cache=False):
#         self.image_files = image_files
#         self.transform = transform
#         self.cache = cache
#         if self.cache:
#             self.cached_images = self.cache_images()

#     def cache_images(self):
#         if os.path.exists('cached_images.pkl'):
#             with open('cached_images.pkl', 'rb') as f:
#                 cached_images = pickle.load(f)
#             return cached_images
#         else:
#             print('Caching images...')
#             with mp.Pool() as p:
#                 cached_images = p.map(preprocess_image, self.image_files)
#             with open('cached_images.pkl', 'wb') as f:
#                 pickle.dump(cached_images, f)
#             print('Done caching images.')
#             return cached_images

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         if self.cache:
#             image, label = self.cached_images[idx]
#         else:
#             file_name = self.image_files[idx]
#             image, label = preprocess_image(file_name)
#         return image, label

# def create_data(args):
#     dataset = args.dataset
#     normal_str, disease_str = None, None

#     if dataset == 'fundus':
#         normal_str = "normal"
#         disease_str = 'csc'
#     elif dataset == 'macular':
#         normal_str = "OCTID_NORMAL"
#         disease_str = "OCTID_MH"

#     normal_image_files = [
#         os.path.join(f"images/{normal_str}", file_name)
#         for file_name in os.listdir(f"images/{normal_str}")
#     ]
#     normal_dataset = CustomDataset(
#         normal_image_files,
#         transform=None,
#         cache=True
#     )

#     diseased_image_files = [
#         os.path.join(f"images/{disease_str}", file_name)
#         for file_name in os.listdir(f"images/{disease_str}")
#     ]
#     diseased_dataset = CustomDataset(
#         diseased_image_files,
#         transform=data_transforms(),
#         cache=True
#     )

#     # concatenate datasets and labels
#     images = torch.cat(
#         [normal_dataset[i][0] for i in range(len(normal_dataset))] +
#         [diseased_dataset[i][0] for i in range(len(diseased_dataset))]
#     )
#     labels = torch.cat(
#         [normal_dataset[i][1] for i in range(len(normal_dataset))] +
#         [diseased_dataset[i][1] for i in range(len(diseased_dataset))]
#     )

#     # split data into train, validation, and test sets
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         images, labels, test_size=config.VAL_SPLIT_SIZE, random_state=42)
    
#     test_images, test_labels = val_images[:config.TEST_SPLIT_SIZE], val_labels[:config.TEST_SPLIT_SIZE]

#     # create data dictionary
#     data_dict = {
#         "train": {
#             "train_images": train_images,
#             "train_labels": train_labels
#         },
#         "validation": {
#             "val_images": val_images,
#             "val_labels": val_labels
#         },
#         "test": {
#             "test_images": test_images,
#             "test_labels": test_labels
#         },
#         "metadata": {
#             "train_size": len(train_labels),
#             "val_size": len(val_labels),
#             "test_size": len(test_labels),
#             "normal_str": normal_str,
#             "disease_str": disease_str
#         }
#     }

#     # save data dictionary to file
#     file_path = f"dataset/{args.dataset}_train_val_test.pt"
#     torch.save(data_dict, file_path)
#     print(f"Data prepared and saved to {file_path}")

# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import config
# import argparse
# import pickle
# import multiprocessing as mp


# def data_transforms():
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return transform


# def preprocess_image(file_name):
#     image = Image.open(file_name)
#     label = 0 if "normal" in file_name else 1
#     image = data_transforms()(image)
#     return image, label


# class CustomDataset(Dataset):
#     def __init__(self, image_files, transform=None, cache=False):
#         self.image_files = image_files
#         self.transform = transform
#         self.cache = cache
#         if self.cache:
#             self.cached_images = self.cache_images()

#     def cache_images(self):
#         if os.path.exists('cached_images.pkl'):
#             with open('cached_images.pkl', 'rb') as f:
#                 cached_images = pickle.load(f)
#             return cached_images
#         else:
#             print('Caching images...')
#             with mp.Pool() as p:
#                 cached_images = p.map(preprocess_image, self.image_files)
#             with open('cached_images.pkl', 'wb') as f:
#                 pickle.dump(cached_images, f)
#             print('Done caching images.')
#             return cached_images

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         if self.cache:
#             image, label = self.cached_images[idx]
#         else:
#             file_name = self.image_files[idx]
#             image, label = preprocess_image(file_name)
#         return image, label


# def load_dataset(dataset, cache=False):
#     print("Running load_dataset function.")

#     normal_str, disease_str = None, None

#     if dataset == 'fundus':
#         normal_str = "normal"
#         disease_str = 'csr'
#     elif dataset == 'macular':
#         normal_str = "OCTID_NORMAL"
#         disease_str = "OCTID_MH"

#     normal_image_files = [
#         os.path.join(f"images/{normal_str}", file_name)
#         for file_name in os.listdir(f"images/{normal_str}")
#     ]
#     diseased_image_files = [
#         os.path.join(f"images/{disease_str}", file_name)
#         for file_name in os.listdir(f"images/{disease_str}")
#     ]

#     train_normal_files, val_normal_files = train_test_split(normal_image_files, test_size=config.VAL_SPLIT_SIZE,
#                                                             random_state=42)
#     train_diseased_files, val_diseased_files = train_test_split(diseased_image_files, test_size=config.VAL_SPLIT_SIZE,
#                                                                 random_state=42)

#     train_files = train_normal_files + train_diseased_files
#     val_files = val_normal_files + val_diseased_files

#     train_dataset = CustomDataset(
#         train_files,
#         transform=data_transforms(),
#         cache=cache
#     )
#     val_dataset = CustomDataset(
#         val_files,
#         transform=data_transforms(),
#         cache=cache
#     )

#     return train_dataset, val_dataset

# def create_data(args):
#     print("Running create_data function.")

#     dataset = args.dataset
#     normal_str, disease_str = None, None

#     if dataset == 'fundus':
#         normal_str = "normal"
#         disease_str = 'csr'
#     elif dataset == 'macular':
#         normal_str = "OCTID_NORMAL"
#         disease_str = "OCTID_MH"

#     normal_image_files = [
#         os.path.join(f"images/{normal_str}", file_name)
#         for file_name in os.listdir(f"images/{normal_str}")
#     ]
#     normal_dataset = CustomDataset(
#         normal_image_files,
#         transform=data_transforms()
#     )
#     normal_loader = DataLoader(
#         normal_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=0
#     )

#     diseased_image_files = [
#         os.path.join(f"images/{disease_str}", file_name)
#         for file_name in os.listdir(f"images/{disease_str}")
#     ]
#     diseased_dataset = CustomDataset(
#         diseased_image_files,
#         transform=data_transforms()
#     )
#     diseased_loader = DataLoader(
#         diseased_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=0
#     )

#     # Check if saved data file exists, if not create it
#     file_path = f"dataset/{args.dataset}_train_val_test.pt"
#     if os.path.exists(file_path):
#         data_dict = torch.load(file_path)
#     else:
#         images = torch.cat(
#             [image for image, _ in normal_loader] +
#             [image for image, _ in diseased_loader]
#         )
#         labels = torch.cat(
#             [label for _, label in normal_loader] +
#             [label for _, label in diseased_loader]
#         )

#         train_images, val_images, train_labels, val_labels = train_test_split(
#             images, labels, test_size=config.VAL_SPLIT_SIZE, random_state=42
#         )

#         data_dict = {
#             "train": {
#                 "train_images": train_images,
#                 "train_labels": train_labels
#             },
#             "validation": {
#                 "val_images": val_images,
#                 "val_labels": val_labels
#             },
#             "test": {
#                 "test_images": val_images[:config.TEST_SPLIT_SIZE],
#                 "test_labels": val_labels[:config.TEST_SPLIT_SIZE]
#             },
#             "metadata": {
#                 "train_size": len(train_labels),
#                 "test_size": len(val_labels)
#             }
#         }

#         torch.save(data_dict, file_path)
#         print(f"Data prepared and saved to {file_path}")


# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset",
#         choices=["fundus", "macular"],
#         required=True,
#         help="Choose dataset."
#     )
#     args = parser.parse_args()
#     create_data(args)
        
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import config
import argparse
from tqdm import tqdm

# Define data augmentation transforms
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_data(args_params):
    # Determine the dataset and associated strings
    if args_params.dataset == 'fundus':
        normal_data_str = "normal"
        disease_data_str = 'csr'
    elif args_params.dataset == 'macular':
        normal_data_str = "OCTID_NORMAL"
        disease_data_str = "OCTID_MH"

    # Load normal images
    normal_images = []
    normal_labels = []
    normal_images_files = os.listdir(f"images/{normal_data_str}")
    for file_name in tqdm(normal_images_files, total=len(normal_images_files), postfix="Normal"):
        image = Image.open(os.path.join(f"images/{normal_data_str}", file_name))
        normal_images.append(data_transform(image))
        normal_labels.append(0)

    # Load diseased images
    diseased_images = []
    diseased_labels = []
    diseased_images_files = os.listdir(f"images/{disease_data_str}")
    for file_name in tqdm(diseased_images_files, total=len(diseased_images_files), postfix="Disease"):
        image = Image.open(os.path.join(f"images/{disease_data_str}", file_name))
        diseased_images.append(data_transform(image))
        diseased_labels.append(1)
    print("stacking")
    # Concatenate images and labels
    images = torch.stack(normal_images + diseased_images)
    labels = torch.tensor(normal_labels + diseased_labels)
    print("splitting")

    # Split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=config.VAL_SPLIT_SIZE, random_state=42)
    print("saving")

    # Create dictionary with data
    data_dict = {
        "train": {
            "train_images": train_images,
            "train_labels": train_labels
        },
        "validation": {
            "val_images": val_images,
            "val_labels": val_labels
        },
        "test": {
            "test_images": val_images[:config.TEST_SPLIT_SIZE],
            "test_labels": val_labels[:config.TEST_SPLIT_SIZE]
        },
        "metadata": {
            "train_size": len(train_labels),
            "test_size": len(val_labels)
        }
    }

    # Save data dictionary to file
    torch.save(data_dict, f"dataset/{args_params.dataset}_train_val_test.pt")
    print("Data prepared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["fundus", "macular"],
        required=True,
        help="Choose dataset."
    )
    args = parser.parse_args()
    create_data(args)