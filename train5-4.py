import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import matplotlib.pyplot as plt

# Set up Random-Erasing in model training
transform_with_erasing = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing()
])

transform_without_erasing = transforms.Compose([
   transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

# Load training and validation datasets
train_dataset_path = "C:/Users/user/OneDrive/Desktop/Dataset_OpenCvDl_Hw2_Q5/dataset/training_dataset"
val_dataset_path = "C:/Users/user/OneDrive/Desktop/Dataset_OpenCvDl_Hw2_Q5/dataset/validation_dataset"

train_dataset_with_erasing = ImageFolder(root=train_dataset_path, transform=transform_with_erasing)
train_dataset_without_erasing = ImageFolder(root=train_dataset_path, transform=transform_without_erasing)

val_dataset_with_erasing = ImageFolder(root=val_dataset_path, transform=transform_with_erasing)
val_dataset_without_erasing = ImageFolder(root=val_dataset_path, transform=transform_without_erasing)

batch_size = 32
train_loader_with_erasing = DataLoader(train_dataset_with_erasing, batch_size=batch_size, shuffle=True)
train_loader_without_erasing = DataLoader(train_dataset_without_erasing, batch_size=batch_size, shuffle=True)

val_loader_with_erasing = DataLoader(val_dataset_with_erasing, batch_size=batch_size, shuffle=False)
val_loader_without_erasing = DataLoader(val_dataset_without_erasing, batch_size=batch_size, shuffle=False)

# Define the model architecture
class MyResNetModel(nn.Module):
    def __init__(self):
        super(MyResNetModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # Adjust the output layer according to your specific task

    def forward(self, x):
        x = self.resnet(x)
        return x

# Train and validate the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples * 100  # Convert to percentage
    return accuracy

# Plot accuracy values with a bar chart and save the figure
def plot_and_save_accuracy(acc_with_erasing, acc_without_erasing):
    accuracy_values = [acc_with_erasing, acc_without_erasing]
    labels = ['With Random-Erasing', 'Without Random-Erasing']

    fig, ax = plt.subplots()
    bars = plt.bar(labels, accuracy_values, color=['blue', 'orange'])
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # Set Y-axis limit to 0-100%
    plt.title('Model Accuracy Comparison')

    # Annotate bars with actual accuracy values
    for bar, acc_value in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 1, f'{acc_value:.2f}%', fontsize=9)

    plt.savefig('accuracy_comparison.png')
    plt.show()

# Load the trained model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == "__main__":
    # Initialize the models
    model_with_erasing = MyResNetModel()
    model_without_erasing = MyResNetModel()

    # Train the models with Adam optimizer
    optimizer_with_erasing = torch.optim.Adam(model_with_erasing.parameters(), lr=0.001)
    optimizer_without_erasing = torch.optim.Adam(model_without_erasing.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_erasing.to(device)
    model_without_erasing.to(device)

    train(model_with_erasing, train_loader_with_erasing, optimizer_with_erasing, criterion, device)
    train(model_without_erasing, train_loader_without_erasing, optimizer_without_erasing, criterion, device)

    # Validate the models
    acc_with_erasing = validate(model_with_erasing, val_loader_with_erasing, criterion, device)
    acc_without_erasing = validate(model_without_erasing, val_loader_without_erasing, criterion, device)

    # Plot and save accuracy values
    plot_and_save_accuracy(acc_with_erasing, acc_without_erasing)

    # Save trained models
    torch.save(model_with_erasing.state_dict(), 'trained_model_with_erasing.pth')
    torch.save(model_without_erasing.state_dict(), 'trained_model_without_erasing.pth')

    # Load trained models (if needed)
    # load_model(model_with_erasing, 'path_to_trained_model_with_erasing.pth')
    # load_model(model_without_erasing, 'path_to_trained_model_without_erasing.pth')
