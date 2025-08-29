# Liveness Detection with LwFLNeT

This project implements a liveness detection system using deep learning, specifically the LwFLNeT model. Liveness detection is a crucial task in biometric security, aiming to distinguish between real (live) faces and spoofed (fake) faces, such as photos or videos.

## Project Structure

- `train.py`: Main training script for the liveness detection model.
- `datasets.py`: Dataset loader for CelebA_Spoof, including face detection and preprocessing.
- `nuaa.py`: Loader for the NUAA dataset, used for cross-dataset evaluation.
- `models/face_detector.py`: Face detection utility used to crop faces from images.
- `models/liveness.py`: Contains the LwFLNeT model implementation.
- `webcam.py`: (Optional) Webcam utility for real-time inference.

## Datasets

- **CelebA_Spoof**: Used for training and testing. The dataset is preprocessed to detect and crop faces, then normalized for input to the model.
- **NUAA**: Used for cross-dataset evaluation to test generalization.

## Model

- **LwFLNeT**: A lightweight neural network designed for fast and efficient liveness detection. The model is trained using cross-entropy loss with class weighting to address class imbalance.

## Training

- Training is performed using PyTorch, with data loaded via `DataLoader`.
- Confusion matrices and metrics (FAR, FRR, HTER) are computed for each epoch and visualized.
- Model weights are saved after training.

## Usage

1. Prepare the datasets (CelebA_Spoof and NUAA) in the expected directory structure.
2. Run `train.py` to train the model and evaluate its performance.
3. Confusion matrices and metrics are saved in the `temp/` directory for each epoch.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- seaborn
- OpenCV
- matplotlib

Install dependencies with:

```bash
pip install torch torchvision scikit-learn seaborn opencv-python matplotlib
```

## Results

- Training and evaluation metrics are printed to the console and saved as images.
- The trained model is saved as `model.pth`.

## Notes

- Ensure the datasets are correctly placed and paths are updated as needed.
- The face detector is used to crop faces before feeding images to the model.
