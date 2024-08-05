import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Assuming all previous code is in a file named 'image_classification.py'
from image_classification import get_model, train_model, evaluate_model

class CIFAR10YChannel(Dataset):
    def __init__(self, cifar10_dataset, transform=None):
        self.dataset = cifar10_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert to YCbCr and keep only Y channel
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

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set hyperparameters
    num_classes = 10  # CIFAR-10 has 10 classes
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # Train and evaluate RGB model
    print("Training RGB model...")
    train_loader_rgb, val_loader_rgb, test_loader_rgb = get_cifar10_dataloaders(batch_size, use_y_channel=False)
    model_rgb = get_model(num_classes, use_y_channel=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_rgb.parameters(), lr=learning_rate)

    model_rgb = train_model(model_rgb, train_loader_rgb, val_loader_rgb, criterion, optimizer, num_epochs, device)
    test_loss_rgb, test_acc_rgb = evaluate_model(model_rgb, test_loader_rgb, criterion, device)

    # Train and evaluate Y-channel model
    print("\nTraining Y-channel model...")
    train_loader_y, val_loader_y, test_loader_y = get_cifar10_dataloaders(batch_size, use_y_channel=True)
    model_y = get_model(num_classes, use_y_channel=True)
    optimizer = optim.Adam(model_y.parameters(), lr=learning_rate)

    model_y = train_model(model_y, train_loader_y, val_loader_y, criterion, optimizer, num_epochs, device)
    test_loss_y, test_acc_y = evaluate_model(model_y, test_loader_y, criterion, device)

    # Compare results
    print("\nFinal Results:")
    print(f"RGB Model - Test Loss: {test_loss_rgb:.4f}, Test Accuracy: {test_acc_rgb:.4f}")
    print(f"Y-channel Model - Test Loss: {test_loss_y:.4f}, Test Accuracy: {test_acc_y:.4f}")

if __name__ == "__main__":
    main()