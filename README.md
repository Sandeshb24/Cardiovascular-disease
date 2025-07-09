## Cardiovascular Disease Prediction

This repository contains a machine learning project aimed at predicting cardiovascular disease (CVD) based on an individual's health data. The project utilizes a Gradient Boosting Classifier to build a predictive model.

### Project Overview

Cardiovascular diseases are a major health concern worldwide. Early prediction can significantly aid in prevention and timely medical intervention. This project addresses this by developing a classification model that can identify individuals at risk of CVD based on various health metrics.

### Exploratory Data Analysis (EDA) and Feature Engineering

An initial Exploratory Data Analysis (EDA) was performed to understand the dataset's characteristics, identify potential issues, and prepare the data for modeling. As part of this process, feature engineering steps were undertaken:

  * **Categorical Feature Encoding**: The `cholesterol`, `smoke`, `alco`, `active`, and `gluc` features, which are categorical, were converted into a numerical format suitable for machine learning algorithms using one-hot encoding via `pd.get_dummies`. This transforms each unique category into a new binary column.

### Dataset

The project uses a dataset named `health_data.csv` (presumably containing anonymized health records). The dataset includes features such as:

  * `id`: Patient ID
  * `age`: Age in days
  * `gender`: Gender (binary encoded, likely 0 and 1)
  * `height`: Height in cm
  * `weight`: Weight in kg
  * `ap_hi`: Systolic blood pressure
  * `ap_lo`: Diastolic blood pressure
  * `cholesterol`: Cholesterol levels (categorized: 0, 1, 2)
  * `gluc`: Glucose levels (categorized: 0, 1, 2)
  * `smoke`: Smoking status (binary: 0 for non-smoker, 1 for smoker)
  * `alco`: Alcohol intake (binary: 0 for no, 1 for yes)
  * `active`: Physical activity (binary: 0 for no, 1 for yes)
  * `cardio`: Presence of cardiovascular disease (target variable: 0 for no CVD, 1 for CVD)

### Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Inspection**: The `health_data.csv` file is loaded into a Pandas DataFrame. The first few rows are displayed to understand the data structure.
2.  **Data Preprocessing**:
      * The `Unnamed: 0` column, which appears to be an artifact of data export, is dropped as it's not relevant for the model.
      * Categorical features (`cholesterol`, `smoke`, `alco`, `active`, `gluc`) are one-hot encoded using `pd.get_dummies` to convert them into a numerical format suitable for machine learning algorithms.
3.  **Feature and Target Separation**: The dataset is split into features (`X`) and the target variable (`y`), where `y` is the `cardio` column.
4.  **Data Splitting**: The data is divided into training and testing sets using `train_test_split` with a test size of 20%, ensuring the model is evaluated on unseen data.
5.  **Model Training**: A `GradientBoostingClassifier` is initialized with specific hyperparameters (`n_estimators = 200`, `learning_rate = 0.01`, `max_depth = 5`, `random_state = 42`) and trained on the training data.
6.  **Model Evaluation**: The trained model's performance is evaluated on the test set using:
      * **Accuracy Score**: The proportion of correctly predicted instances.
      * **Classification Report**: Provides precision, recall, and F1-score for each class (CVD and no CVD).
      * **Confusion Matrix**: A table showing the true positive, true negative, false positive, and false negative predictions.

### Results

The Gradient Boosting Classifier achieved an accuracy of approximately **72.77%** on the test set.

The classification report provides a more detailed breakdown of the model's performance:

| Class | Precision | Recall | F1-Score | Support |
| :---- | :-------- | :----- | :------- | :------ |
| 0 | 0.71 | 0.77 | 0.74 | 7006 |
| 1 | 0.75 | 0.68 | 0.72 | 6994 |
| **Accuracy** | | | **0.73** | **14000** |
| **Macro Avg** | 0.73 | 0.73 | 0.73 | 14000 |
| **Weighted Avg** | 0.73 | 0.73 | 0.73 | 14000 |

The confusion matrix shows:

```
[[5401 1605]
 [2207 4787]]
```

  * **True Negatives (0,0)**: 5401 instances correctly predicted as not having CVD.
  * **False Positives (0,1)**: 1605 instances incorrectly predicted as having CVD (Type I error).
  * **False Negatives (1,0)**: 2207 instances incorrectly predicted as not having CVD (Type II error).
  * **True Positives (1,1)**: 4787 instances correctly predicted as having CVD.

