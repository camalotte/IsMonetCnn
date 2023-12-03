# eval_model.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_trainer import SimpleCNN  # Make sure the training logic is protected by if __name__ == '__main__'
import os

# Define transformations for the evaluation images
evaluation_transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the evaluation dataset
evaluation_dataset = datasets.ImageFolder(root='photos', transform=evaluation_transformations)
evaluation_loader = DataLoader(evaluation_dataset, batch_size=1, shuffle=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load('monet_model.pth', map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Evaluate the model on the evaluation dataset
with torch.no_grad():
    for inputs, _ in evaluation_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Get the image filename from the dataset
        image_path, _ = evaluation_loader.dataset.samples[evaluation_loader._dataset_kind]
        image_filename = os.path.basename(image_path)

        # Check the prediction
        prediction_class = 'Monet' if predicted.item() == 0 else 'Not Monet'
        print(f"{image_filename}: {prediction_class}")