# 🌍 WiDS Datathon Image Classification Project

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-blue)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-green)
![Image Classification](https://img.shields.io/badge/Image%20Classification-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle-yellow)
![Data Science](https://img.shields.io/badge/Data%20Science-red)

---

### 📄 Description
In advance of the March 4, 2019, Global WiDS Conference, the Global WiDS team, the West Big Data Innovation Hub, and the WiDS Datathon Committee collaborated with Planet and Figure Eight to bring a dataset of high-resolution satellite imagery to participants. The aim is to build awareness about deforestation and oil palm plantations through this predictive analytics challenge focused on social impact.

---

### 🌴 Why Oil Palm?
Deforestation due to the growth of oil palm plantations has significant economic and environmental impacts. While oil palm is present in many everyday products, its cultivation has led to deforestation, increased carbon emissions, and biodiversity loss. However, it also provides many valuable jobs. This challenge aims to develop affordable, timely, and scalable methods to address the expansion and management of oil palm plantations globally.

---

### 📊 Evaluation
Submissions are evaluated on the Area under the ROC curve (AUC) between the predicted probability and the observed target (has_oilpalm).

---

### 📁 Dataset Description
- **train_images/image_[9-digit number].jpg:** Training images from Planet.
- **traininglabels.csv:** Crowdsourced annotations/labels of the presence or absence of oil palm in the training data.
- **leaderboard_test_data/image_[9-digit number].jpg:** Test images from Planet.
- **SampleSubmission.csv:** Sample submission file in the correct format.

**Data Fields:**
- **image_id:** An anonymous ID unique to a given image.
- **has_oilpalm:** Annotation or label for a given image (0 for no oil palm, 1 for presence of oil palm plantations).
- **score:** Confidence score based on the aggregated results from crowdsourcing the annotations.

---

### 🛠 Tools & Technologies
- **Python 3.8+:** Primary programming language
- **TensorFlow & Keras:** For building and training deep learning models
- **Pandas:** For data manipulation and analysis
- **OpenCV:** For image processing
- **Scikit-learn:** For machine learning tools and evaluation metrics
- **Imbalanced-learn:** For handling imbalanced datasets

---

### 🔄 Workflow

1. **Libraries and Data Loading:**
    - Imported the necessary libraries for data manipulation, image processing, and deep learning.
    - Loaded the dataset using functions to read images and CSV files.

2. **Data Preprocessing:**
    - **Image Preprocessing:** Resized and normalized the images to prepare them for model training.
    - **Label Encoding:** Transformed categorical labels into numerical format.
    - **Handling Duplicates and Missing Values:** Cleaned the data by handling duplicates and missing values in the annotations.

3. **Data Splitting:**
    - Split the data into training, validation, and test sets to build and evaluate models. Used stratified sampling to ensure balanced classes in each set.

4. **Resampling Techniques:**
    - **Undersampling and SMOTE:** Used RandomUnderSampler and SMOTE to handle imbalanced data by undersampling the majority class and oversampling the minority class.

5. **Data Augmentation:**
    - Applied data augmentation techniques to increase the diversity of the training data and prevent overfitting. Techniques included rotation, width and height shifts, shear transformation, zoom, and horizontal flip.

6. **Model Training:**
    - **Transfer Learning:** Used pre-trained models like MobileNet, ResNet50, ResNet101, and EfficientNetB0 to leverage existing knowledge for feature extraction.
    - **Fine-Tuning:** Fine-tuned the pre-trained models on the satellite imagery dataset to improve performance.

7. **Prediction:**
    - Made predictions on the test set using the trained models. Generated probabilities for the target variable (has_oilpalm).

8. **Evaluation:**
    - Evaluated the models using metrics like ROC-AUC, accuracy, precision, recall, and F1-score to determine their performance.

---

### 📂 Project Structure

```
- WiDS_Datathon_2019_Image_Classification
  - data/
    - leaderboard_holdout_data/
    - leaderboard_test_data/
    - train_images/
    - holdout.csv
    - SampleSubmission.csv
    - testlabels.csv
    - traininglabels.csv
  - models/
    - mobilnet_model_resample.pickle
    - resnet50_model_resample.pickle
    - resnet101_model_resample.pickle
    - efficientnet_model_resample.pickle
    - mobilnet_model_undersampled.pickle
    - resnet50_model_undersampled.pickle
    - resnet101_model_undersampled.pickle
    - efficientnet_model_args.pickle
    - efficientnet_model_args1.pickle
  - notebooks/
    - report.ipynb
    - train_resample.ipynb
    - train_undersampled&Aug.ipynb
    - Transfer_Learn_test.ipynb
    - Transfer_Learn_train.ipynb
  - scripts/
    - helper_functions.py
  - submission/
    - predictions_mobilnet.csv
    - predictions_resnet50.csv
    - predictions_resnet101.csv
    - predictions_efficientnet.csv
  - README.md
```

---

### 🧩 Models and Results

- **MobileNet:**
    - Leveraged the MobileNet architecture for efficient feature extraction.
    - Applied undersampling and data augmentation techniques.
    - **ROC-AUC:**
      - Original: 83%
      - Undersampled & Augmented: 80%
      - Undersampled to Oversampled: 75%

- **ResNet50:**
    - Used the ResNet50 model to capture deeper features and enhance classification accuracy.
    - Applied undersampling and data augmentation techniques.
    - **ROC-AUC:**
      - Original: 57%
      - Undersampled & Augmented: 73%
      - Undersampled to Oversampled: 58%

- **ResNet101:**
    - Applied the ResNet101 model for more advanced feature extraction and robust performance.
    - Applied undersampling and data augmentation techniques.
    - **ROC-AUC:**
      - Original: 53%
      - Undersampled & Augmented: 57%
      - Undersampled to Oversampled: 56%

- **EfficientNet:**
    - Utilized the EfficientNetB0 model for state-of-the-art image classification.
    - Applied undersampling and data augmentation techniques.
    - **ROC-AUC:**
      - Original: 50%
      - Undersampled & Augmented: 85%
      - Undersampled to Oversampled: 47%

---

### 📊 Impact of Reshaping Image Dimensions

Experiments were conducted to reshape the images to a larger dimension (256x256) for the three best-performing models:

- **MobileNet:**
  - Original: 83% dropped to 67% after reshaping.
- **ResNet50:**
  - Undersampled & Augmented: 73% dropped to 41% after reshaping.
- **EfficientNet:**
  - Undersampled & Augmented: 85% dropped to 73% after reshaping.

[ROC-AUC Scores](Wids Datathon graph.jpg)

---

### 🌟 Best Model and Parameter Tuning

- **Best Model: EfficientNet (Undersampled & Augmented):**
  - Achieved the highest AUC score of 85%.

Further parameter tuning efforts included:
- Increased Epochs to 5 & reduced learning rate to 0.0001: AUC Score = 82%
- Increased Epochs to 5 & learning rate to 0.001: AUC Score = 77%
- Added Callback and Early Stopping with epoch=10 and learning rate=0.001: AUC Score = 54%

---

### 🚀 Recommendations for Further Improvement

1. **Hyperparameter Optimization:**
    - Utilize techniques like Grid Search or Random Search to systematically explore a wider range of hyperparameter combinations.
2. **Transfer Learning:**
    - Explore other pre-trained models or fine-tune more layers within EfficientNet to leverage their learned features more effectively.
3. **Ensemble Learning:**
    - Combine predictions from multiple models (e.g., MobileNet, ResNet50, EfficientNet) using techniques like voting or stacking to potentially improve overall performance.
4. **Regularization Techniques:**
    - Experiment with regularization methods such as dropout, L2 regularization, or batch normalization to reduce overfitting and enhance model robustness.
5. **Advanced Augmentation:**
    - Use more sophisticated data augmentation techniques, possibly including generative models like GANs (Generative Adversarial Networks) to create additional synthetic training data.
6. **Learning Rate Schedulers:**
    - Implement adaptive learning rate schedulers that adjust the learning rate dynamically during training, potentially leading to better convergence.
