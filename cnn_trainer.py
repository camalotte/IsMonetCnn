# cnn_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),  # Adjust the linear layer input size to match your conv output
            nn.ReLU(),
            nn.Linear(512, 2)  # Assuming 2 classes for classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


def main():
    # Set the base directory where the split datasets are located
    base_dir = 'split_data'

    # Define transformations for the images
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to the correct size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=transformations)
    val_dataset = datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=transformations)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model
    model = SimpleCNN()

    # # Move the model to the GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Function to perform training
    def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=25):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {running_loss / len(train_loader):.4f}, '
                  f'Validation Loss: {val_loss / len(val_loader):.4f}, '
                  f'Validation Accuracy: {100 * correct / total:.2f}%')

        print('Finished Training')

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=25)

    # Save the trained model
    torch.save(model.state_dict(), 'monet_model.pth')


if __name__ == '__main__':
    main()