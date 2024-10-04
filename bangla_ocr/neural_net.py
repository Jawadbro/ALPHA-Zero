# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import cv2
import pytesseract
from PIL import Image
import os
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Data Loading and Preprocessing
def load_dataset(dataset_path, is_ekush=True):
    """
    Load and preprocess the dataset from the given path.
    
    Args:
    dataset_path (str): Path to the dataset directory
    is_ekush (bool): Flag to indicate if the dataset is Ekush (affects label extraction)
    
    Returns:
    tuple: Preprocessed images and corresponding labels as PyTorch tensors
    """
    images, labels = [], []
    try:
        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
                        if img is None:
                            raise ValueError(f"Failed to read image: {img_path}")
                        
                        # Ensure consistent resizing for both datasets
                        img = cv2.resize(img, (28, 28))  # Resize all images to 28x28
                        
                        # Normalize images to [0, 1] range
                        img = img / 255.0
                        
                        images.append(img)
                        labels.append(int(label))
                    except Exception as e:
                        logging.error(f"Error processing {img_path}: {str(e)}")
        
        # Convert to PyTorch tensors and ensure 4D shape: [batch_size, 1 (channel), 28, 28]
        images_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return images_tensor, labels_tensor
    except Exception as e:
        logging.error(f"Error loading dataset from {dataset_path}: {str(e)}")
        return None, None


# Load datasets
# Load datasets
try:
    ekush_images, ekush_labels = load_dataset(r'E:\Code\f2\ALPHA-Zero\bangla_ocr\ekush\dataset\dataset', is_ekush=True)
    bangla_lekha_images, bangla_lekha_labels = load_dataset(r'E:\Code\f2\ALPHA-Zero\bangla_ocr\bangla_lekha', is_ekush=False)

    if ekush_images is None or bangla_lekha_images is None:
        raise ValueError("Failed to load one or both datasets")

    # Combine datasets
    combined_images = torch.cat((ekush_images, bangla_lekha_images), 0)  # Safe concatenation now
    combined_labels = torch.cat((ekush_labels, bangla_lekha_labels), 0)

    # Split dataset into train and validation sets
    dataset = TensorDataset(combined_images, combined_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
except Exception as e:
    logging.error(f"Error in dataset preparation: {str(e)}")
    raise


# 2. Model Architecture

class BanglaOCRModel(nn.Module):
    """
    CNN model for Bangla OCR
    """
    def __init__(self, num_classes):
        super(BanglaOCRModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BanglaOCRModel(num_classes=len(set(combined_labels.numpy()))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_images, batch_labels in tqdm(loader, desc="Training"):
        try:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        except Exception as e:
            logging.error(f"Error in training batch: {str(e)}")
    
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(loader, desc="Validating"):
            try:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            except Exception as e:
                logging.error(f"Error in validation batch: {str(e)}")
    
    return running_loss / len(loader), correct / total

num_epochs = 30
best_val_acc = 0.0

for epoch in range(num_epochs):
    try:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bangla_ocr_model.pth')
        
        logging.info(f"Best Val Acc: {best_val_acc:.4f}")
    except Exception as e:
        logging.error(f"Error in epoch {epoch+1}: {str(e)}")

# 4. Inference Function

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def detect_and_recognize_bangla(image_path, model, device):
    try:
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            raise ValueError("Failed to preprocess image")
        image_tensor = image_tensor.to(device)

        # OCR using pytesseract
        ocr_text = pytesseract.image_to_string(Image.open(image_path), lang='ben')
        
        # CNN classification
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            cnn_class = predicted.item()
            cnn_confidence = confidence.item()

        # Get OCR confidence
        ocr_confidence = pytesseract.image_to_data(Image.open(image_path), lang='ben', output_type=pytesseract.Output.DICT)['conf'][0]
        
        # Determine document type
        document_type = "clean" if ocr_confidence > 80 and cnn_confidence > 0.9 else "messy"

        return {
            "ocr_text": ocr_text,
            "cnn_classification": cnn_class,
            "cnn_confidence": cnn_confidence,
            "ocr_confidence": ocr_confidence,
            "document_type": document_type
        }
    except Exception as e:
        logging.error(f"Error in detect_and_recognize_bangla: {str(e)}")
        return None

# Test the function
try:
    model.load_state_dict(torch.load('best_bangla_ocr_model.pth'))
    model.eval()

    result = detect_and_recognize_bangla(r"E:\Code\f2\ALPHA-Zero\bangla_ocr\pngtree-abar-dekha-hobe-bangla-text-png-image_225469.jpg", model, device)
    if result:
        logging.info("OCR and Classification Results:")
        for key, value in result.items():
            logging.info(f"{key}: {value}")
    else:
        logging.error("Failed to perform OCR and classification")
except Exception as e:
    logging.error(f"Error in testing: {str(e)}")