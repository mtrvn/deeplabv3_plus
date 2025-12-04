# DeepLabv3+ for Medical Image Segmentation

## Overview

This repository implements **DeepLabv3+**, a state-of-the-art semantic segmentation architecture, specifically configured for medical image segmentation tasks. DeepLabv3+ combines atrous (dilated) convolutions with spatial pyramid pooling to capture multi-scale contextual information, making it highly effective for precise organ and lesion segmentation in medical imaging.

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Quick Start](#quick-start)
6. [Model Components](#model-components)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Inference & Visualization](#inference--visualization)
10. [Results](#results)

## Features

- **Efficient Architecture**: ResNet50 backbone with atrous spatial pyramid pooling (ASPP) for multi-scale feature extraction
- **Medical Imaging Focus**: Optimized for binary segmentation tasks (medical region vs. background)
- **Advanced Loss Functions**: Implements DiceBCE and IoU loss for better handling of imbalanced medical datasets
- **Data Augmentation**: Uses Albumentations for robust data preprocessing and augmentation
- **GPU Acceleration**: CUDA support with mixed precision training for faster convergence
- **Comprehensive Metrics**: Tracks Dice coefficient, IoU, and pixel accuracy during training

## Architecture Overview

### DeepLabv3+ Components

```
Input Image (3 channels)
    ↓
ResNet50 Backbone (Layer 1-3)
    ├─ Low-level features (Layer 1: 256 channels)
    └─ Deep features (Layer 3: 1024 channels)
         ↓
    Atrous Spatial Pyramid Pooling (ASPP)
         ↓
    Decoder
    ├─ Conv 1×1 on low-level features (48 channels)
    └─ Concatenate & refine with 3×3 convolutions
         ↓
    Final Classification (1 channel for binary segmentation)
         ↓
    Output: Segmentation Map
```

### Key Architectural Features

1. **Backbone Network**: ResNet50 with ImageNet pretrained weights for transfer learning
2. **Atrous Convolution**: Dilated convolutions (rates: 6, 12, 18) capture receptive fields without reducing resolution
3. **ASPP Module**: Multi-scale feature fusion using parallel atrous convolutions
4. **Decoder**: Upsamples and combines high-level semantic and low-level spatial features
5. **Skip Connections**: Preserves fine-grained details from early layers

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 10.2+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd deeplabv3_plus

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=1.9.0
torchvision>=0.10.0
numpy
Pillow
matplotlib
albumentations>=1.1.0
tqdm
```

## Dataset Preparation

### Dataset Structure

Organize your medical imaging dataset as follows:

```
datasets/
└── icis_2018/
    ├── train/
    │   ├── images/          # Training images (RGB)
    │   └── masks/           # Segmentation masks (grayscale)
    └── test/
        ├── images/          # Test images (RGB)
        └── masks/           # Ground truth masks (grayscale)
```

### File Naming Convention

- **Images**: `sample_001.jpg`, `sample_002.jpg`, etc.
- **Masks**: `sample_001_segmentation.png`, `sample_002_segmentation.png`, etc.
  - Mask values: 0 (background), 255 or 1 (foreground/region of interest)

### Data Format

- **Images**: RGB PNG/JPG files (any resolution, resized to 256×256 during training)
- **Masks**: Grayscale PNG files with binary values (0 and 255)
- **Pixel values**: Images are normalized; masks are converted to float and thresholded at 255→1.0

## Quick Start

### 1. Prepare Your Data

Place your training and validation datasets in the `datasets/icis_2018/` directory following the structure above.

### 2. Run Training

Open `main.ipynb` in Jupyter Notebook and execute cells in order:

```python
# Cell 1: Import libraries and set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cell 2: Configure dataset paths
TRAIN_IMG_DIR = "datasets/icis_2018/train/images"
TRAIN_MASK_DIR = "datasets/icis_2018/train/masks"
VAL_IMG_DIR = "datasets/icis_2018/test/images"
VAL_MASK_DIR = "datasets/icis_2018/test/masks"
BATCH_SIZE = 16

# Cell 3+: Load data, build model, and train
```

### 3. Monitor Training

The notebook displays:
- Training loss (Dice-BCE)
- IoU scores
- Dice coefficients
- Prediction visualizations

## Model Components

### 1. `dataset_processor.py`

Handles data loading and augmentation:

- **DatasetProcessor Class**: Custom PyTorch Dataset for efficient data loading
- **Transforms**: 
  - Resize to 256×256
  - Normalize with ImageNet statistics
  - Convert to tensors

```python
train_ds = DatasetProcessor(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    transform=train_transform
)
```

### 2. `deeplabv3_plus.py`

Core architecture implementation:

- **ResNet_50**: Pretrained backbone for feature extraction
- **Atrous_Convolution**: Dilated convolution with BatchNorm and ReLU
- **Atrous_Spatial_Pyramid_Pooling (ASPP)**: Multi-scale feature aggregation
- **Deeplabv3Plus**: Main model combining all components

```python
model = Deeplabv3Plus(num_classes=1)  # 1 for binary segmentation
```

### 3. `loss.py`

Custom loss functions for medical image segmentation:

**DiceBCELoss**: Combines Dice loss (good for boundary detection) with BCE loss
```python
loss = DiceBCELoss()
```

**IOULoss**: Intersection over Union metric
```python
iou = IOULoss()
```

## Training

### Training Configuration

```python
# Hyperparameters
BATCH_SIZE = 16              # Adjust based on GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 22
DEVICE = "cuda"              # or "cpu"

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = DiceBCELoss()
iou_fn = IOULoss()

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
```

### Training Loop

Each epoch:
1. **Forward Pass**: Input image → model → logits
2. **Loss Computation**: Combined Dice-BCE loss
3. **Backward Pass**: Compute gradients
4. **Optimization**: Update weights via Adam optimizer
5. **Validation**: Evaluate on test set using Dice and IoU metrics

### Training Tips

- **Batch Size**: Start with 16 for 256×256 images; reduce if out of memory
- **Learning Rate**: 1e-4 works well; try 1e-5 for fine-tuning
- **Epochs**: 20-50 depending on dataset size
- **Checkpointing**: Save model at best validation performance
- **Early Stopping**: Monitor validation metrics to prevent overfitting

## Evaluation

### Metrics

1. **Dice Coefficient**: $\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$
   - Range: [0, 1], where 1 is perfect segmentation
   - Emphasizes overlap between predicted and ground truth

2. **Intersection over Union (IoU)**: $\text{IoU} = \frac{|X \cap Y|}{|X \cup Y|}$
   - Range: [0, 1]
   - Penalizes false positives more than Dice

3. **Pixel Accuracy**: $\text{Accuracy} = \frac{\text{Correct Pixels}}{\text{Total Pixels}}$
   - Can be misleading for imbalanced datasets

### Evaluation Code

```python
def evaluate(model, val_loader, device):
    model.eval()
    dice_score = 0
    iou_score = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds_binary = (preds > 0.5).float()
            
            dice_score += (2 * (preds_binary * y).sum()) / ((preds_binary + y).sum() + 1e-8)
            # IoU computation...
    
    return dice_score / len(val_loader)
```

## Inference & Visualization

### Making Predictions

```python
model.eval()
with torch.no_grad():
    predictions = model(images)  # Logits
    probabilities = torch.sigmoid(predictions)  # Probabilities [0, 1]
    binary_masks = (probabilities > 0.5).float()  # Binary masks
```

### Visualization

The `visualize_predictions.py` script displays:
- Original medical image
- Ground truth segmentation mask
- Model predicted mask
- Overlay of prediction on original image with contour

```python
from visualize_predictions import visualize_predictions
visualize_predictions(model, val_loader, DEVICE, num_samples=100)
```

## Results

### Expected Performance

On medical imaging datasets (organ/lesion segmentation):
- **Dice Coefficient**: 0.85-0.95
- **IoU Score**: 0.75-0.90
- **Pixel Accuracy**: 95%+

### Training Dynamics

Typical training progression:
- **Epoch 1-5**: Sharp improvement in metrics
- **Epoch 5-15**: Steady refinement, potential plateau
- **Epoch 15+**: Fine-tuning for marginal gains

## Advanced Usage

### Transfer Learning

Fine-tune on a new medical imaging dataset:

```python
# Load pretrained weights
model = Deeplabv3Plus(num_classes=1)
model.load_state_dict(torch.load('pretrained_model.pth'))

# Freeze backbone for initial epochs
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze after warm-up
for param in model.parameters():
    param.requires_grad = True
```

### Hyperparameter Tuning

Experiment with:
- Learning rates: [1e-5, 1e-4, 1e-3]
- Batch sizes: [8, 16, 32]
- Different loss combinations: [Dice, Dice-BCE, Dice-Focal]

### Data Augmentation

Extend augmentations in `dataset_processor.py`:

```python
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.2),
    # Add more augmentations...
    A.Normalize(...),
    ToTensorV2()
])
```

## References

- Chen, L. C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.
- Medical image segmentation best practices
- PyTorch documentation

## License

Include your license information here.

## Contributing

Contributions are welcome! Please follow standard guidelines for pull requests. 
