# How MNIST Works

## What is MNIST?

MNIST (Modified National Institute of Standards and Technology) is a dataset of handwritten digits commonly used for machine learning tasks. It contains 70,000 images of digits (0-9), each 28×28 pixels in grayscale.

## The Dataset

- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Size**: 28×28 pixels
- **Format**: Grayscale (values 0-255)

## How the Drawing Canvas Works

The canvas on the main page lets you draw digits by hand. Here's what happens:

1. **Pixel Capture**: Your drawing is captured as a 500×500 pixel canvas and scaled down to 28×28 pixels
2. **Normalization**: Pixel values are converted to grayscale (0-255 scale)
3. **Model Input**: The normalized image is fed into a neural network
4. **Prediction**: The model outputs a probability for each digit (0-9)

## Machine Learning Model

The prediction uses a neural network trained on the MNIST dataset:

- **Input layer**: 784 neurons (28×28 flattened)
- **Hidden layers**: Process and extract features
- **Output layer**: 10 neurons (one per digit 0-9)
- **Training**: Uses supervised learning with labeled examples

## Why MNIST?

MNIST is ideal for learning because:

- Simple problem (10 possible outputs)
- Well-studied benchmark for model performance
- Small enough to train quickly on any hardware
- Large enough to train meaningful models

## Drawing Tips

- Draw digits in the center of the canvas
- Use clear strokes, similar to how you'd write on paper
- The model works best with digits sized consistently
- Click "Clear" to reset and try again

## What's Next?

Future improvements could include:

- Real-time prediction as you draw
- Confidence scores for predictions
- Multiple model architectures for comparison
- Training your own model on custom data
