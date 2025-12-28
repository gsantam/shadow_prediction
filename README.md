# Shadow Prediction with Deep Learning

A deep learning project that predicts 3D light source positions from 2D shadow images using synthetic training data and ResNet-18.

## Overview

This project generates synthetic 3D scenes with geometric shapes (cylinders and cones), renders them as 2D grayscale images with realistic shadows, and trains a neural network to predict the 3D position of the light source from the shadow patterns.

## Features

- **Synthetic Data Generation**: On-the-fly generation of training data with randomized:
  - 3D object parameters (size, position)
  - Camera viewing angles (azimuth, elevation)
  - Light source positions
  - Ray-traced shadows on ground plane

- **Deep Learning Model**: ResNet-18 architecture modified for:
  - Grayscale single-channel input (224x224)
  - 3D coordinate regression output (x, y, z)
  - GPU acceleration support (CUDA/MPS)

- **Complete Training Pipeline**: Modular codebase with:
  - PyTorch Dataset with on-the-fly generation
  - Training/validation loops with progress tracking
  - Model checkpointing and loss visualization

## Project Structure

```
shadow_prediction/
├── synthetic_data.py           # Core data generation module
├── shadow_prediction/          # Training package
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset
│   └── model.py               # ResNet-18 model
├── scripts/
│   └── train.py               # Training script
└── requirements.txt           # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/shadow_prediction.git
cd shadow_prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```python
from synthetic_data import generate_synthetic_scene, visualize_synthetic_scene

# Generate training image (fast, no visualization)
image = generate_synthetic_scene(
    random_projection=True,
    randomize_objects=True,
    add_shadows=True,
    img_width=224,
    img_height=224
)

# Generate and visualize (shows 3D scene, projection, and binary image)
visualize_synthetic_scene(
    random_projection=True,
    randomize_objects=True,
    add_shadows=True
)
```

### Train the Model

```bash
python scripts/train.py
```

Or customize training parameters:

```python
from scripts.train import train

model, train_losses, val_losses = train(
    num_epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    train_size=10000,
    val_size=2000
)
```

## Technical Details

### Data Generation

- **3D Shapes**: Parametric meshes for cylinders and cones
- **Shadow Computation**: Ray tracing from light source through objects to ground plane (z=0)
- **Projection**: Orthographic projection with random viewing angles
- **Rasterization**: ConvexHull-based polygon filling

### Model Architecture

- Base: ResNet-18 (pretrained optional)
- Input: Single-channel 224x224 grayscale images
- Output: 3D coordinates (x, y, z) of light source
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

### Image Encoding

- `0.0` - Objects (black)
- `0.5` - Shadows (gray)
- `1.0` - Background (white)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib
- SciPy
- tqdm

## Hardware Support

- **CPU**: Supported on all platforms
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2/M3) GPU acceleration

## Examples

### Sample Generated Scene

The generator creates diverse training data with:
- Random object sizes and positions
- Variable camera angles (azimuth 0-360°, elevation 10-80°)
- Random light positions on hemisphere (elevation 20-80°)

### Training Performance

Typical training configuration:
- 200,000 training samples
- 10,000 validation samples
- Batch size: 64
- ~5 epochs to convergence

## License

MIT License - see LICENSE file for details

## Author

Built with assistance from GitHub Copilot

## Acknowledgments

- PyTorch for deep learning framework
- SciPy for computational geometry (ConvexHull)
- ResNet architecture from torchvision.models
