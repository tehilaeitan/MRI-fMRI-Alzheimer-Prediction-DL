# MRI-fMRI-Alzheimer-Prediction-DL

This project is focused on developing advanced neural network architectures for early and efficient prediction of Alzheimer’s Disease (AD), Mild Cognitive Impairments (MCI) and Cognitive Normal (CN). By integrating functional connectivity analysis in fMRI data through specific Regions of Interest (ROIs), this approach provides a robust framework for understanding brain function, which enhances the diagnosis and treatment of neurological and psychiatric disorders.

The model is designed to work with both 3D Convolutional Neural Networks (CNNs) and Vision Transformers. 3D CNNs are powerful for analyzing volumetric data, capturing local spatial features, and providing a comprehensive view of the brain by learning spatial hierarchies in three dimensions. Vision Transformers, on the other hand, utilize self-attention mechanisms to capture global dependencies in the data. When combined with convolutional layers, they can effectively analyze complex spatiotemporal patterns in fMRI data.

The project includes the development and application of these neural network models to achieve comprehensive analysis of brain activity and connectivity. By leveraging the strengths of both 3D CNNs and Vision Transformers, the project aims to provide very accurate classifications of AD and MCI cases. This sets the stage for future research, where further improvements in model architectures and data processing techniques can lead to even more accurate and insightful findings in neuroimaging and brain research.

# Installation
Before running the code, ensure that you have Python 3.8+ installed.

You will need to install the required Python packages. You can install them using pip:

pip install torch torchmetrics matplotlib seaborn scikit-learn tqdm wandb pandas numpy

# Required Packages:

Required Packages:

os: Interacts with the operating system for file and directory operations.

scipy: Provides tools for scientific and technical computing.

nibabel: Reads and writes neuroimaging data formats (e.g., NIfTI files).

nilearn: Facilitates statistical learning on neuroimaging data.

numpy: Fundamental package for numerical computations with arrays and matrices.

pandas: Data manipulation and analysis library.

sklearn: Machine learning tools, including cross-validation and metrics.

matplotlib: Plotting library for creating visualizations.

tensorflow: End-to-end platform for building and training machine learning models.

seaborn: Statistical data visualization library built on Matplotlib.

# Hyperparameters

The main script allows for several hyperparameters to be configured. Below is a description of the key hyperparameters:

Convolutional Filters: Number of filters used in the convolutional layers (e.g., 64, 32).

Kernel Size: Size of the convolutional kernel (e.g., 3x3x3).

Batch Size: Number of samples processed in each batch (e.g., 16).

Dropout Rate: Dropout rate applied to prevent overfitting (e.g., 0).

Pool Size: Size of the pooling window (e.g., 2x2x2).

Epochs: Number of complete passes through the training dataset (e.g., 200, with training stopped at 60 epochs).

Learning Rate: Learning rate for the optimizer (e.g., 0.001).

Loss Function: Loss function used during training (e.g., Categorical Crossentropy).

Optimizer: Optimization algorithm used (e.g., Adam).

Early Stopping Patience: Number of epochs with no improvement to wait before stopping training early (e.g., 30).
