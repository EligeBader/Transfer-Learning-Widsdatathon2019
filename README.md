# Transfer Learning Project: Widsdatathon

## Image Classification Using Convolutional Neural Networks and Transfer Learning

### Introduction
**Project Objective:**
- Experimenting with different deep learning models for image classification tasks to achieve high Area Under the Curve (AUC) scores.

**Scope:**
- Utilizing various sampling techniques and tuning parameters with models such as MobileNet, ResNet50, ResNet101, and EfficientNet.

**Overview:**
- Convolutional neural networks (CNNs) and transfer learning were employed to enhance the performance of image classification.

### Dataset Analysis
**Number of Features and Type:**
- Train folder: 11,000 Images with a size of 256x256 in color.
- Test folder: 9,013 Images with size 256x256 in color.
- Leaderboard folder: 2,178 Images with size 256x256 in color.
- Train labels: 15,244 rows (image labels for training), 3 columns.
- Test labels: 4,356 rows (image labels for testing), 3 columns.
- Holdout labels: 2,178 rows (image labels for testing), 3 columns.

**Target Class Balance:**
- We are dealing with very imbalanced data.
- Class distribution: {0: 8968, 1: 322}.
    - Class 0: doesn't have oilpalm.
    - Class 1: has oilpalm.

**Data Transformation:**
- Resizing: Initially, images were resized to 224x224x3.
- Normalization: The images were normalized by dividing by 255.
- Augmentation: Data augmentation techniques were applied using ImageDataGenerator with parameters like rescale, rotation, width shift, height shift, shear, zoom, horizontal flip, and fill mode.

**Summary and Assumptions:**
- Class Imbalance Handling: Techniques such as oversampling the minority class or undersampling the majority class were employed.
- Transfer Learning: Utilizing pre-trained models with resized images to 224x224x3.
- Model Performance Metrics: Focus on achieving high AUC scores.
- Data Augmentation: Applied to increase the diversity of the training data.

### Methodology
**Model Selection:**
- MobileNet: Efficient in terms of computation.
- ResNet50: Balanced approach to accuracy and computational load.
- ResNet101: Higher accuracy at the cost of more resources.
- EfficientNet: Uses a compound scaling method.

**Fine-Tuning:**
- Added Flatten Layer, Dense Layer (512 Units, 'relu'), and Output Layer (2 Units, 'softmax').

### Implementation
**Tools and Libraries:**
- TensorFlow and Keras for building and training models.
- NumPy and pandas for data manipulation and preprocessing.

**Data Preprocessing:**
- Undersampling and Data Augmentation.
- Undersampling then Oversampling using SMOTE.

**Training Process:**
- Initial Training: 3 Epochs, Adam optimizer with a learning rate of 0.001.
- Experimental Phase: 5 Epochs, Adam optimizer with learning rates of 0.001 and 0.0001.
- Extended Training with Callback: 10 Epochs, learning rate 0.001.

**Validation and Testing:**
- Combined test and holdout datasets.
- Evaluation Metrics: AUC.

### Analysis
**Performance Metrics:**
- MobileNet: Original 83%, Undersampled & Augmented 80%, Undersampled to Oversampled 75%.
- ResNet50: Original 57%, Undersampled & Augmented 73%, Undersampled to Oversampled 58%.
- ResNet101: Original 53%, Undersampled & Augmented 57%, Undersampled to Oversampled 56%.
- EfficientNet: Original 50%, Undersampled & Augmented 85%, Undersampled to Oversampled 47%.

**Reshaping Image Dimensions:**
- MobileNet: AUC dropped from 83% to 67%.
- ResNet50: AUC dropped from 73% to 41%.
- EfficientNet: AUC dropped from 85% to 73%.

**Parameter Tuning:**
- Increased Epochs and adjusted learning rates for EfficientNet.

### Summary
- MobileNet showed strong initial performance but declined with undersampling and augmentation.
- ResNet50 benefitted significantly from undersampling and augmentation.
- ResNet101 showed modest improvements.
- EfficientNet was the standout performer with undersampling and augmentation.

### Recommendation for Further Improvement
- Hyperparameter Optimization: Grid Search or Random Search.
- Transfer Learning: Explore other pre-trained models.
- Ensemble Learning: Combine predictions from multiple models.
- Regularization Techniques: Dropout, L2 regularization, batch normalization.
- Advanced Augmentation: Use generative models like GANs.
- Learning Rate Schedulers: Implement adaptive learning rate schedulers.
