import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_training_history(rgb_csv, y_csv, rgb_time, y_time):
    rgb_df = pd.read_csv(rgb_csv)
    y_df = pd.read_csv(y_csv)

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    
    # Loss plots
    axs[0, 0].plot(rgb_df['epoch'], rgb_df['train_loss'], label='RGB Train')
    axs[0, 0].plot(rgb_df['epoch'], rgb_df['val_loss'], label='RGB Val')
    axs[0, 0].plot(y_df['epoch'], y_df['train_loss'], label='Y Train')
    axs[0, 0].plot(y_df['epoch'], y_df['val_loss'], label='Y Val')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    
    # Accuracy plots
    axs[0, 1].plot(rgb_df['epoch'], rgb_df['train_acc'], label='RGB Train')
    axs[0, 1].plot(rgb_df['epoch'], rgb_df['val_acc'], label='RGB Val')
    axs[0, 1].plot(y_df['epoch'], y_df['train_acc'], label='Y Train')
    axs[0, 1].plot(y_df['epoch'], y_df['val_acc'], label='Y Val')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].legend()
    
    # Training time comparison
    axs[0, 2].bar(['RGB', 'Y'], [rgb_time, y_time])
    axs[0, 2].set_title('Training Time (seconds)')
    

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



if __name__ == "__main__":
    plot_training_history('history_rgb.csv', 'history_y.csv', 100, 90)  # Example usage