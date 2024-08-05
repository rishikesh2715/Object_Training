import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class CIFAR10YChannel(Dataset):
    def __init__(self, cifar10_dataset, transform=None):
        self.dataset = cifar10_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Convert to grayscale (Y channel)
        y_image = transforms.functional.to_grayscale(image, num_output_channels=1)
        
        if self.transform:
            y_image = self.transform(y_image)
        
        return y_image, label

def get_cifar10_dataloaders(batch_size=32, use_y_channel=False):
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_y = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform_rgb)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_rgb)

    # Split the training set into train and validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    if use_y_channel:
        train_dataset = CIFAR10YChannel(train_dataset, transform=transform_y)
        val_dataset = CIFAR10YChannel(val_dataset, transform=transform_y)
        test_dataset = CIFAR10YChannel(test_dataset, transform=transform_y)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader