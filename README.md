# OPTIMIZING THE TRAINING OF NEURAL NETWORKS FOR MUSIC GENERATION
course work
# Music Generation with Neural Networks

## Project Description

This project focuses on the task of music generation using neural networks. The model learns to predict the next state of a music matrix (piano-roll) based on previous time steps. The task represents a multidimensional multi-label binary classification problem, where the goal is to determine which notes will be active at the next time step.

## Key Features

- **Musical Sequence Generation**: The model generates polyphonic music in MIDI format
- **Architecture**: Combined neural network architecture based on convolutional (Conv1D) and recurrent layers (GRU)
- **Training Optimization**: The main focus of this work is investigating different optimizers and loss functions to overcome data sparsity issues

## Data Structure

### Piano-roll Format
- **Dimensions**: T × N (T = 128 time steps, N = 84 notes)
- **Note Range**: MIDI notes from 24 to 108
- **Data Type**: Binary matrix (0/1, where 1 indicates an active note)
- **Sparsity**: Less than 2% active notes per time step

### Dataset
- **Size**: 292 MIDI files containing approximately 705,000 notes
- **Preprocessing**:
  1. MIDI file parsing using `pretty_midi`
  2. Conversion to fixed sampling rate
  3. Extraction of binary note activation matrices
  4. Truncation/padding to fixed length (128 steps)
  5. Normalization and batching for training

## Experiments

### Investigated Approaches

#### Optimizers:
- **Adam** - standard adaptive optimization method
- **AdamW** - Adam with weight decay for improved regularization
- **RMSprop** - adaptive method considering gradient history

#### Loss Functions:
- **Binary Cross-Entropy (BCE)** - standard function for binary classification
- **Focal Loss** - BCE modification focusing on hard examples, helpful for class imbalance

### Experiment Goals
Comparison of different optimizer and loss function combinations for:
1. Improving training stability
2. Accelerating model convergence
3. Enhancing music generation quality
4. Overcoming data sparsity issues

## Technical Requirements

- Python 3.7+
- Libraries: pretty_midi, PyTorch/TensorFlow (depending on implementation), numpy, pandas


