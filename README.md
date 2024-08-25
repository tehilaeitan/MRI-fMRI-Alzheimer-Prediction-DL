# MRI-fMRI-Memory-Prediction-DL
Deep learning models for predicting Alzheimer's Disease stages using MRI and fMRI data. Models include 3D-CNN, 2D-CNN, and Vision Transformer, achieving up to 95.6% accuracy in distinguishing Alzheimer's, MCI, and Healthy Controls for early diagnosis.

# Installation
Before running the code, ensure that you have Python 3.8+ installed.

You will need to install the required Python packages. You can install them using pip:

pip install torch torchmetrics matplotlib seaborn scikit-learn tqdm wandb pandas numpy

# Required Packages:

os: Provides a way to interact with the operating system. Used for file and directory operations.

scipy: A library for scientific and technical computing.

scipy.ndimage.zoom: For resizing images.

nibabel: A library to read and write neuroimaging data formats (e.g., NIfTI files).

nilearn: A Python library for fast and easy statistical learning on NeuroImaging data.

nilearn.plotting: Used for visualizing brain imaging data.

numpy: A fundamental package for scientific computing with Python. Provides support for arrays and matrices.

pandas: Data analysis and manipulation library. Useful for handling structured data.

sklearn: Machine learning library for Python.

sklearn.metrics: Metrics for evaluating model performance (e.g., classification report, confusion matrix).

sklearn.model_selection: Functions for splitting data and cross-validation (e.g., train_test_split, KFold).

matplotlib.pyplot: A plotting library for creating static, animated, and interactive visualizations.

tensorflow: An end-to-end open-source platform for machine learning.

tensorflow.keras: High-level API for building and training deep learning models.

layers, models, Conv3D, MaxPooling3D, Flatten, Dense, GlobalAveragePooling2D, Dropout: Various layers and utilities for building neural networks.

CategoricalCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy: Loss functions for training models.

Adam, SGD: Optimizers for training models.

Accuracy, CategoricalAccuracy, SparseCategoricalAccuracy: Metrics for evaluating models.

Model, Sequential: Types of models in Keras.

tensorflow.keras.preprocessing.image: Functions for loading and preprocessing images.

tensorflow.keras.callbacks: Callbacks for model training (e.g., EarlyStopping).

seaborn: Statistical data visualization library built on top of Matplotlib.
