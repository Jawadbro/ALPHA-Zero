# Import necessary libraries
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import DataLoader, TensorDataset, random_split  # Data handling utilities
import torchvision.transforms as transforms  # Image transformation utilities
import numpy as np  # Numerical computing library
import cv2  # OpenCV library for image processing
import pytesseract  # OCR engine
from PIL import Image  # Python Imaging Library for image opening and manipulation
import os  # Operating system interface
from tqdm import tqdm  # Progress bar library

# 1. Data Loading and Preprocessing

def load_dataset(dataset_path, is_ekush=True):
    """
    Load and preprocess the dataset from the given path.
    
    Args:
    dataset_path (str): Path to the dataset directory
    is_ekush (bool): Flag to indicate if the dataset is Ekush (not used in current implementation)
    
    Returns:
    tuple: Preprocessed images and corresponding labels as PyTorch tensors
    """
    images, labels = [], []  # Initialize empty lists for images and labels
    for label in os.listdir(dataset_path):  # Iterate through subdirectories (each representing a class)
        label_path = os.path.join(dataset_path, label)  # Full path to the class directory
        if os.path.isdir(label_path):  # Check if it's a directory
            for img_name in os.listdir(label_path):  # Iterate through images in the class directory
                img_path = os.path.join(label_path, img_name)  # Full path to the image file
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                img = cv2.resize(img, (28, 28))  # Resize image to 28x28 pixels
                img = img / 255.0  # Normalize pixel values to range [0, 1]
                images.append(img)  # Add preprocessed image to the list
                labels.append(int(label))  # Add corresponding label to the list
    # Convert lists to PyTorch tensors
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1), torch.tensor(labels, dtype=torch.long)

# Load datasets
ekush_images, ekush_labels = load_dataset('path/to/ekush_dataset', is_ekush=True)  # Load Ekush dataset
bangla_lekha_images, bangla_lekha_labels = load_dataset('path/to/bangla_lekha_dataset', is_ekush=False)  # Load Bangla Lekha dataset

# Combine datasets
combined_images = torch.cat((ekush_images, bangla_lekha_images), 0)  # Concatenate image tensors
combined_labels = torch.cat((ekush_labels, bangla_lekha_labels), 0)  # Concatenate label tensors

# Split dataset into train and validation sets
dataset = TensorDataset(combined_images, combined_labels)  # Create a TensorDataset
train_size = int(0.8 * len(dataset))  # Calculate size of training set (80% of total)
val_size = len(dataset) - train_size  # Calculate size of validation set (20% of total)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Randomly split the dataset

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)  # DataLoader for training set
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)  # DataLoader for validation set

# 2. Model Architecture

class BanglaOCRModel(nn.Module):
    """
    CNN model for Bangla OCR
    """
    def __init__(self, num_classes):
        """
        Initialize the model architecture
        
        Args:
        num_classes (int): Number of output classes
        """
        super(BanglaOCRModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the convolutional layers
            nn.Linear(128 * 3 * 3, 512),  # Fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
        x (Tensor): Input tensor
        
        Returns:
        Tensor: Output tensor
        """
        x = self.features(x)  # Pass input through feature extraction layers
        x = self.classifier(x)  # Pass through classifier layers
        return x

# 3. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
model = BanglaOCRModel(num_classes=len(set(combined_labels.numpy()))).to(device)  # Initialize model and move to device
criterion = nn.CrossEntropyLoss()  # Define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)  # Learning rate scheduler

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
    model (nn.Module): The neural network model
    loader (DataLoader): DataLoader for the training set
    criterion: Loss function
    optimizer: Optimization algorithm
    device: Device to run the training on (CPU or GPU)
    
    Returns:
    tuple: Average loss and accuracy for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_images, batch_labels in tqdm(loader, desc="Training"):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)  # Move batch to device
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(batch_images)  # Forward pass
        loss = criterion(outputs, batch_labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()  # Accumulate loss
        _, predicted = outputs.max(1)  # Get the index of the max log-probability
        total += batch_labels.size(0)  # Increase total count
        correct += predicted.eq(batch_labels).sum().item()  # Increase correct count
    
    return running_loss / len(loader), correct / total  # Return average loss and accuracy

def validate(model, loader, criterion, device):
    """
    Validate the model
    
    Args:
    model (nn.Module): The neural network model
    loader (DataLoader): DataLoader for the validation set
    criterion: Loss function
    device: Device to run the validation on (CPU or GPU)
    
    Returns:
    tuple: Average loss and accuracy for the validation set
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for batch_images, batch_labels in tqdm(loader, desc="Validating"):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)  # Move batch to device
            outputs = model(batch_images)  # Forward pass
            loss = criterion(outputs, batch_labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss
            _, predicted = outputs.max(1)  # Get the index of the max log-probability
            total += batch_labels.size(0)  # Increase total count
            correct += predicted.eq(batch_labels).sum().item()  # Increase correct count
    
    return running_loss / len(loader), correct / total  # Return average loss and accuracy

num_epochs = 30  # Number of training epochs
best_val_acc = 0.0  # Initialize best validation accuracy

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)  # Train for one epoch
    val_loss, val_acc = validate(model, val_loader, criterion, device)  # Validate the model
    
    scheduler.step(val_loss)  # Adjust learning rate based on validation loss
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:  # If current validation accuracy is the best so far
        best_val_acc = val_acc  # Update best validation accuracy
        torch.save(model.state_dict(), 'best_bangla_ocr_model.pth')  # Save the model
    
    print(f"Best Val Acc: {best_val_acc:.4f}")

# 4. Inference Function

def preprocess_image(image_path):
    """
    Preprocess an image for inference
    
    Args:
    image_path (str): Path to the image file
    
    Returns:
    Tensor: Preprocessed image tensor
    """
    image = Image.open(image_path).convert('L')  # Open image and convert to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28 pixels
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Convert to tensor and add batch dimension
    return image_tensor

def detect_and_recognize_bangla(image_path, model, device):
    """
    Perform OCR and classification on a Bangla image
    
    Args:
    image_path (str): Path to the image file
    model (nn.Module): Trained neural network model
    device: Device to run the inference on (CPU or GPU)
    
    Returns:
    dict: Dictionary containing OCR results, CNN classification, and confidence scores
    """
    # Load and preprocess the image
    image_tensor = preprocess_image(image_path).to(device)  # Preprocess image and move to device

    # OCR using pytesseract
    ocr_text = pytesseract.image_to_string(Image.open(image_path), lang='ben')  # Perform OCR
    
    # CNN classification
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(image_tensor)  # Forward pass
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Compute probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get the class with highest probability
        cnn_class = predicted.item()  # Get the predicted class
        cnn_confidence = confidence.item()  # Get the confidence score

    # Combine OCR and CNN outputs
    ocr_confidence = pytesseract.image_to_data(Image.open(image_path), lang='ben', output_type=pytesseract.Output.DICT)['conf'][0]  # Get OCR confidence
    
    # Determine document type based on OCR and CNN confidences
    if ocr_confidence > 80 and cnn_confidence > 0.9:
        document_type = "clean"
    else:
        document_type = "messy"

    # Return results
    return {
        "ocr_text": ocr_text,
        "cnn_classification": cnn_class,
        "cnn_confidence": cnn_confidence,
        "ocr_confidence": ocr_confidence,
        "document_type": document_type
    }

# Test the function
model.load_state_dict(torch.load('best_bangla_ocr_model.pth'))  # Load the best model
model.eval()  # Set model to evaluation mode

result = detect_and_recognize_bangla("example_image.jpg", model, device)  # Perform OCR and classification
print(result)  # Print the results