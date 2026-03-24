import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_dataloader(data_dir='/dataset/tiny-imagenet-200', batch_size=32, num_workers=4):
    """
    This functiion creates e returns both the training and testing dataloaders
    """
    # Define transformations for the dataset
    train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    tiny_imagenet_dataset_test = ImageFolder(root=f'{data_dir}/val', transform=val_transform)

    # Create a DataLoader
    dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader_train, dataloader_test