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

This project implements an **Object Detection Model** using a camera/webcam to detect objects in real-time. The camera captures images, which is then sent to an AI model that identifies what the object is and provides a response to it. The response is converted to speech in Bangla(Bangladesh) Language. This can be used in a variety of real-world applications, such as surveillance systems, smart devices and medical applications.

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
- PyTorch (based on the model you can use TensorFlow too, we used torch)
- Firebase (optional, for storing image metadata)
- And others included in the `requirements.txt` file


## Setup instructions

Follow these steps to set up the project in your local machine

### **1. Clone the repository**
```cmd
git clone https://github.com/Jawadbro/ALPHA-Zero.git
cd AlPHA-Zero
```
### **2. Create a virtual Environment(Optional but not required)**
```cmd
python -m venv venv_folder_name
venv_folder_name\Scripts\activate
```
### **3. Install dependencies**

```cmd
pip install -r requirements.txt
```

## Running the model
To run the model enter the following to your console
```cmd
python main.py
```

## Usage
The usage of our proposed solution not only spans in medical but also in surveillance. This AI solution can be used to detect objects in real time and get speech recognition of the seen object. This can be helpful for visually impaired people, surveillance systems, smart devcies, medical applications and also shows talent in defence systems if used for that purpose.

## Contributing
Following are the people who contributed to the project:
- [Jawadbro](https://github.com/Jawadbro)
- [rafibuilds](https://github.com/rafibuilds)
- [TahmidAqib](https://github.com/TahmidAqib)