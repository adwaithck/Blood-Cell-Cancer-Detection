# Blood-Cell-Cancer-Detection

# Blood Cell Cancer Detection

A deep learning project for classifying different types of blood cells to aid in cancer detection using transfer learning with MobileNetV2 and EfficientNetB0 architectures.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a computer vision solution for automated blood cell classification using deep learning. The system can classify blood cells into 5 different categories, which is crucial for medical diagnosis and cancer detection. The project employs transfer learning with two state-of-the-art convolutional neural network architectures: MobileNetV2 and EfficientNetB0.

### Key Features
- **Multi-class classification** of blood cells into 5 categories
- **Transfer learning** approach using pre-trained models
- **Data augmentation** techniques to improve model generalization
- **Comprehensive evaluation** with detailed performance metrics
- **High accuracy** achieved by both model architectures

## Dataset

### Dataset Structure
The dataset contains blood cell images organized into training, validation, and test sets:

- **Training Set**: 3,495 images
- **Validation Set**: 998 images  
- **Test Set**: 501 images
- **Image Dimensions**: 224×224 pixels
- **Number of Classes**: 5

### Blood Cell Types
The dataset includes the following blood cell categories:
1. **Basophil**
2. **Erythroblast**
3. **Monocyte**
4. **Myeloblast**
5. **Segmented Neutrophil**

### Data Preprocessing
- Images resized to 224×224 pixels for model compatibility
- Pixel values normalized to [0,1] range
- Data augmentation applied to training set:
  - Random rotations
  - Width and height shifts
  - Zoom transformations
  - Horizontal flips
  - Brightness adjustments

## Models

### 1. MobileNetV2 Architecture

**Base Model Configuration:**
- Pre-trained MobileNetV2 from ImageNet
- Top layers excluded for custom classification head
- Last 30 layers made trainable for fine-tuning
- Frozen layers: ~0.73M parameters
- Trainable layers: ~1.89M parameters

**Custom Classification Head:**
```
GlobalAveragePooling2D()
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(128, activation='relu') 
BatchNormalization()
Dropout(0.4)
Dense(5, activation='softmax')  # 5 classes
```

**Training Configuration:**
- Optimizer: Adam (learning rate: 1e-4)
- Loss function: Categorical Crossentropy
- Callbacks: EarlyStopping (patience=4), ModelCheckpoint
- Total Parameters: ~2.62M
- Trainable Parameters: ~1.89M

### 2. EfficientNetB0 Architecture

**Base Model Configuration:**
- Pre-trained EfficientNetB0 from ImageNet
- Top layers excluded for custom classification head
- Last 30 layers made trainable for fine-tuning
- Total Parameters: ~4.41M
- Trainable Parameters: ~1.86M

**Custom Classification Head:**
```
GlobalAveragePooling2D()
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(128, activation='relu')
BatchNormalization()
Dropout(0.4)
Dense(5, activation='softmax')  # 5 classes
```

**Training Configuration:**
- Optimizer: Adam (learning rate: 1e-4)
- Loss function: Categorical Crossentropy
- Callbacks: EarlyStopping (patience=4), ModelCheckpoint
- Maximum Epochs: 50

## Performance

### MobileNetV2 Results
- **Test Accuracy**: 99.80%
- **Test Precision**: 99.80%
- **Test Recall**: 99.80%
- **Test F1-Score**: 99.80%

**Per-Class Performance:**
- Basophil: 1.00 precision, 1.00 recall, 1.00 f1-score
- Erythroblast: 1.00 precision, 1.00 recall, 1.00 f1-score
- Monocyte: 1.00 precision, 1.00 recall, 1.00 f1-score
- Myeloblast: 1.00 precision, 1.00 recall, 1.00 f1-score
- Segmented Neutrophil: 1.00 precision, 1.00 recall, 1.00 f1-score

### EfficientNetB0 Results
- **Test Accuracy**: 99.80%
- **Test Precision**: 99.80%
- **Test Recall**: 99.80%
- **Test F1-Score**: 99.80%

**Per-Class Performance:**
- Basophil: 1.00 precision, 1.00 recall, 1.00 f1-score
- Erythroblast: 1.00 precision, 1.00 recall, 1.00 f1-score
- Monocyte: 1.00 precision, 1.00 recall, 1.00 f1-score
- Myeloblast: 0.99 precision, 1.00 recall, 1.00 f1-score
- Segmented Neutrophil: 1.00 precision, 0.99 recall, 0.99 f1-score

### Training Progress
Both models demonstrated excellent convergence:
- **Validation accuracy** reached 99.2%+ within ~9 epochs
- **Training accuracy** consistently improved across epochs
- **Loss values** decreased steadily with minimal overfitting
- **Early stopping** mechanism prevented overfitting

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Required packages listed in requirements.txt

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Blood-Cell-Cancer-Detection
```

2. **Create virtual environment:**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
tensorflow>=2.0.0
keras
numpy
matplotlib
seaborn
scikit-learn
efficientnet
pillow
```

## Usage

### Training the Models

1. **Prepare your dataset** with the following structure:
```
dataset/
├── train/
│   ├── basophil/
│   ├── erythroblast/
│   ├── monocyte/
│   ├── myeloblast/
│   └── seg_neutrophil/
├── validation/
│   ├── basophil/
│   ├── erythroblast/
│   ├── monocyte/
│   ├── myeloblast/
│   └── seg_neutrophil/
└── test/
    ├── basophil/
    ├── erythroblast/
    ├── monocyte/
    ├── myeloblast/
    └── seg_neutrophil/
```

2. **Run the training notebook:**
```bash
jupyter notebook models.ipynb
```

### Model Inference

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('mobilenet.keras')  # or 'efficientnet.keras'

# Prepare image for prediction
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Class names
class_names = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class]:.4f}")
```

## Results

### Model Comparison

| Model | Parameters | Trainable | Test Accuracy | Training Time |
|-------|------------|-----------|---------------|---------------|
| MobileNetV2 | 2.62M | 1.89M | 99.80% | Fast |
| EfficientNetB0 | 4.41M | 1.86M | 99.80% | Medium |

### Key Findings

1. **Exceptional Performance**: Both models achieved outstanding 99.80% accuracy on the test set
2. **Perfect Classification**: Near-perfect precision and recall across all blood cell types
3. **Robust Generalization**: Strong performance indicates good model generalization
4. **Efficient Architecture**: MobileNetV2 offers similar performance with fewer parameters
5. **Quick Convergence**: Both models converged rapidly with early stopping around epoch 26

### Confusion Matrix Analysis
Both models demonstrated perfect or near-perfect classification with minimal misclassification across all blood cell categories.

## Technical Implementation

### Data Pipeline
- **ImageDataGenerator** for real-time data augmentation
- **Flow from directory** for efficient batch loading
- **Categorical encoding** for multi-class classification
- **Normalization** applied to validation and test sets

### Transfer Learning Strategy
- Pre-trained ImageNet weights as starting point
- Frozen base layers to preserve learned features
- Fine-tuning of top layers for domain adaptation
- Custom classification head for 5-class prediction

### Training Optimization
- **Early Stopping**: Monitoring validation loss with patience=4
- **Model Checkpointing**: Saving best weights during training
- **Learning Rate**: Conservative 1e-4 for stable convergence
- **Batch Processing**: Efficient memory management

## Future Work

### Potential Improvements
1. **Model Ensemble**: Combine predictions from multiple architectures
2. **Advanced Augmentation**: Implement more sophisticated data augmentation
3. **Attention Mechanisms**: Integrate attention layers for better feature focus
4. **Larger Dataset**: Expand training data for improved robustness
5. **Cross-Validation**: Implement k-fold validation for better evaluation
6. **Model Compression**: Optimize models for deployment environments

### Clinical Integration
1. **Real-time Inference**: Develop API for clinical workflow integration
2. **Batch Processing**: Support for multiple image analysis
3. **Confidence Thresholding**: Implement uncertainty quantification
4. **Explainable AI**: Add visualization for model decision explanation

## Contributing

We welcome contributions to improve this blood cell classification system:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/improvement`)
3. **Commit changes** (`git commit -am 'Add new feature'`)
4. **Push to branch** (`git push origin feature/improvement`)
5. **Create Pull Request**

### Areas for Contribution
- Model architecture improvements
- Data preprocessing enhancements
- Evaluation metric additions
- Documentation improvements
- Bug fixes and optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **TensorFlow/Keras** for deep learning framework
- **ImageNet** for pre-trained model weights
- **Dataset Contributors** for providing blood cell images
- **Scientific Community** for transfer learning research

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{blood_cell_detection_2024,
  title={Blood Cell Cancer Detection using Deep Learning},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-username/Blood-Cell-Cancer-Detection}}
}
```

---

**Note**: This project is intended for educational and research purposes. For clinical applications, please ensure proper validation and regulatory compliance.
