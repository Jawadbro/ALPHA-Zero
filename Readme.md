# Real-Life Object Detection Model with AI

## Table of Contents
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Model](#running-the-model)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

This project implements an **Object Detection Model** using a camera to detect objects in real-time. The camera captures an image at a regular interval, which is then sent to an AI model that identifies what the object is and provides the output text. 
This can be used in a variety of real-world applications, such as surveillance systems, smart devices and medical applications.

## How It Works

1. **Capture Image:**
   A camera connected to the system captures frames at specific intervals.

2. **Send to AI Model:**
   The captured image is processed and sent to a pre-trained object detection model for analysis.

3. **Model Prediction:**
   The AI model processes the image and returns predictions, identifying the object along with a confidence score.

4. **Output:**
   The system displays or logs the object detected, providing insights for further actions.

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

- Python 3.9.x or higher
- OpenCV
- TensorFlow or PyTorch (based on the model)
- Firebase (optional, for storing image metadata)
- And others included in the `requirements.txt` file


## Setup instructions

Follow these steps to set up the project in your local machine

### **1. Clone the repository**

```cmd
git clone https://github.com/Jawadbro/ALPHA-Zero
cd AlPHA-Zero
```

### **2. Create a virtual Environment(Optional but not required)**

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

### **3. Install dependencies**

```cmd
pip install -r requirements.txt
```