# ALPHA-Zero: Multimodal Virtual Assistant

## Overview
ALPHA-Zero is a real-time, multimodal virtual assistant that "sees and talks." It captures visual inputs via camera, detects objects using deep learning, and generates Bangla speech descriptions—designed for accessibility (e.g., visually impaired support) or surveillance. Developed by Jawadul Karim, this project leverages computer vision and AI inference, inspired by my passion for accessible tech (e.g., Pose-to-Text research).

## Features
- **Real-Time Object Detection**: Identifies objects (e.g., "chair" with 0.95 confidence) from live camera feeds.
- **Bangla Speech Output**: Converts detections into spoken descriptions using gTTS for accessibility.
- **Basic Memory**: Logs metadata via Firebase for interaction recall (e.g., past detections).
- **Lightweight Pipeline**: Runs on standard hardware with minimal setup.

## Tech Stack
- **Python 3.9+**: Core language for the end-to-end workflow.
- **OpenCV**: Captures and processes real-time video frames.
- **PyTorch**: Powers pre-trained object detection models for inference.
- **gTTS**: Synthesizes Bangla speech from detection outputs.
- **Firebase**: Optional persistence for metadata logging.
- **Dependencies**: numpy, pillow, pyaudio (inferred from requirements.txt).

## Installation
1. Clone the repo: `git clone https://github.com/Jawadbro/ALPHA-Zero.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the assistant: `python main.py`

## Future Work
- Add speech input for two-way interaction (e.g., using Whisper).
- Enhance memory with a vector DB (e.g., Pinecone) for multi-turn context.



