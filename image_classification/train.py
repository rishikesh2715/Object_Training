import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
from model import get_model
from dataset import get_cifar10_dataloaders
from evaluate import evaluate_model
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history

def save_history(history, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 10
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    rgb_time, y_time = 0, 0

    for use_y_channel in [False, True]:
        channel_type = "Y" if use_y_channel else "RGB"
        print(f"\nTraining {channel_type} model...")
        
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size, use_y_channel)
        model = get_model(num_classes, use_y_channel)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start_time = time.time()
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        end_time = time.time()
        training_time = end_time - start_time

        if use_y_channel:
            y_time = training_time
        else:
            rgb_time = training_time

        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"{channel_type} Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"{channel_type} Model Training Time: {training_time:.2f} seconds")

        # Save model
        torch.save(model.state_dict(), f'cifar10_{channel_type.lower()}_model.pth')

        # Save history to CSV
        save_history(history, f'history_{channel_type.lower()}.csv')

    # Plot training history
    from visualize import plot_training_history
    plot_training_history('history_rgb.csv', 'history_y.csv', rgb_time, y_time)

if __name__ == "__main__":
    main()