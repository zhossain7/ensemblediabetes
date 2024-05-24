# Detecting Diabetes in the Elderly Using Ensemble Methods



## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Project Overview](#project-overview)
    * [Establish a Dataset](#establish-a-dataset)
    * [Machine Learning Models](#machine-learning-models)
        - [Logistic Regression](#logistic-regression)
        - [SVM Model](#svm-model)
        - [XGBoost](#xgboost)
    * [Ensemble Methods](#ensemble-methods)
        - [Voting (Soft & Hard)](#voting-soft--hard)
        - [AdaBoost](#adaboost)
        - [Stacking](#stacking)
3. [Data Analysis](#data-analysis)
    * [Data Preprocessing](#data-preprocessing)
        - [Synthetic Minority Over-sampling Technique (SMOTE)](#synthetic-minority-over-sampling-technique-smote)
        - [Visualisations](#visualisations)
        - [Exploratory Data Analysis](#exploratory-data-analysis)
        - [Feature Importance](#feature-importance)
4. [Methodology](#methodology)
5. [Model Evaluation](#model-evaluation)
6. [Final Results](#final-results)
7. [Conclusion](#conclusion)


## Project Introduction
This project was created because I was doing a subject in university which was discussing the potential benefits of artificial intelligence (AI) if introduced to the medical field. 
I conducted some research, and while artificial intelligence still needs time before it can be fully implemented into the medical field, it has the potential to be very important in the future.
AI has the potential to revolutionise healthcare by providing tools that can assist in early diagnosis, treatment planning, and patient management. 

This project serves as a step towards integrating AI into healthcare.
It inspired me to do reserach and see how exactly AI can help, which led me to this project which encompasses AI and data analysis, both a core part of my degree and major.

	
## Project Overview

### Establish a dataset
As it mentioned previously, the main aim of this project is to use ensemble methods to predict diabetes in elderly care. First the most important thing to do is find areliable and valid dataset that can be used in my project, this includes preorcessing and analysing the data to ensure it
is suited for my scenario. Data analysis will be spoken about in more depth later on.

### Machine Learning Models
It will be important to find individual machine learning models that will be incroproated into the final ensemble methods. The three models I have utilised are as follows.

#### Logistic Regression 
* Logistic Regression is a statistical method used for binary classification that models the probability of a binary outcome based on one or more predictor variables.
* This model is useful for understanding the influence of various factors on the likelihood of diabetes, as it provides coefficients that can be interpreted in terms of odds ratios.

#### SVM MODEL
* Support Vector Machine (SVM) is a supervised learning algorithm that finds the hyperplane that best separates data into different classes, often used for classification tasks.
* The SVM model can effectively handle high-dimensional data and identify non-linear relationships between features, which can improve the accuracy of diabetes prediction.

#### XGBOOST
* XGBoost (Extreme Gradient Boosting) is a powerful ensemble learning method that builds multiple decision trees in a sequential manner, optimising for the best performance.
* XGBoost is known for its high performance and efficiency in handling large datasets with many features, making it well-suited for predicting diabetes with high accuracy by capturing complex interactions between variables.


### Ensemble Methods
These individual models, each with their unique strengths, are combined in the ensemble method to leverage their collective predictive power, aiming for improved overall performance and robustness in diabetes prediction.

#### Voting (Soft & Hard)
* The Voting ensemble method combines multiple machine learning models by averaging their predictions (soft voting) or using a majority rule (hard voting) to make a final prediction.
Analysis: By combining the strengths of Logistic Regression, SVM, and XGBoost, the Voting method can balance their individual biases and variances, potentially increasing the accuracy and reliability of diabetes predictions.

#### ADABoost
* ADABoost (Adaptive Boosting) is an ensemble technique that creates a strong classifier by combining multiple weak classifiers, each trained on the data with adjusted weights to focus on previously misclassified instances.
Analysis: ADABoost can significantly improve model performance by concentrating on difficult cases, enhancing the model's ability to correctly classify individuals at risk of diabetes even when individual models struggle.

#### Stacking
* Stacking involves training a meta-model (stacker) to combine the predictions of several base models. The base models' outputs are used as inputs for the meta-model, which then makes the final prediction.
* Stacking leverages the unique strengths of each base model by allowing the meta-model to learn how to best combine their predictions, often leading to superior performance compared to individual models or simpler ensemble methods. This approach is particularly effective for capturing complex patterns in the diabetes dataset.



## Data Analysis
The dataset useed for this project  is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It can be availible in Kaggle [right here.](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).

![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/bfe27470-2e66-4146-80bd-e6b76df509a0)
![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/d4dd3612-710b-460b-804c-564de8db06fa)

The dataset has 100,000 individuals, however after extracing indviduals aged 65 or above, I was left with 18,568. This shorter dataset is what was used to train my models and then predict on.

### Data Preprocessing
Data preprocessing is an important step when using datasets for AI models. I did extensive research on possible preprocessing techniques and identified the most effective ones for my scenario and specific dataset. The following preprocessing steps were applied:

1. **Loading the Dataset**:
    - The dataset is loaded from a CSV file using pandas.

    ```python
    import pandas as pd
    file_path = 'diabetes_65.csv'
    diabetes_data = pd.read_csv(file_path)
    ```

2. **Identifying Column Types**:																		
    - Columns are classified into categorical and numerical types for appropriate preprocessing.

    ```python
    categorical_cols = diabetes_data.select_dtypes(include=['object']).columns
    numerical_cols = diabetes_data.select_dtypes(include=['float64', 'int64']).columns.drop('diabetes')
    ```

3. **Defining Preprocessing Steps**:
    - For categorical data, OneHotEncoder is used to handle categorical variables.
    - For numerical data, StandardScaler is used to standardize the features.

    ```python
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    ```

4. **Handling Class Imbalance**:
    - SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the classes in the dataset.

    ```python
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImblearnPipeline

    pipeline = ImblearnPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42))
    ])
    ```

5. **Preparing Features and Target Variable**:
    - The features (X) and target (y) are separated for model training.

    ```python
    X = diabetes_data.drop('diabetes', axis=1)  # Features
    y = diabetes_data['diabetes']  # Target variable
    ```

6. **Applying Preprocessing and SMOTE**:
    - The preprocessing and SMOTE transformations are applied to the features. SMOTE was an important technqiue, an explanation will be provided later on.

    ```python
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    ```

7. **Splitting the Dataset**:
    - The dataset is split into training and testing sets to evaluate model performance.

    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    ```

8. **Saving the Preprocessed Data**:
    - The preprocessed training and testing data are saved to CSV files for further use.

    ```python
    train_data = pd.DataFrame(X_train, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
    train_data['diabetes'] = y_train.values

    test_data = pd.DataFrame(X_test, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
    test_data['diabetes'] = y_test.values

    train_data.to_csv('preprocessed_diabetes_train.csv', index=False)
    test_data.to_csv('preprocessed_diabetes_test.csv', index=False)
    ```

By applying these preprocessing steps, the dataset is prepared for training machine learning models with improved accuracy and reliability. This structured approach ensures that the data is clean, balanced, and ready for effective model training and evaluation.


### Exploratory Data Analysis

#### Synthetic Minority Over-sampling Technique (SMOTE)

The application of SMOTE (Synthetic Minority Over-sampling Technique) significantly improved the recall for predicting diabetes (Class 1) using both Hard Voting and Soft Voting ensemble methods.

**Analysis**

**1. Before Applying SMOTE:**

Both Hard Voting and Soft Voting methods exhibited high recall for non-diabetic cases (Class 0) but struggled with recall for diabetic cases (Class 1). This indicates that the models were more adept at correctly identifying non-diabetic individuals but often missed diabetic cases, which is problematic for medical diagnosis.

**2. After Applying SMOTE:**
The recall for diabetic cases (Class 1) improved notably for both Hard Voting and Soft Voting methods. This enhancement suggests that SMOTE effectively balanced the dataset by addressing class imbalances, leading to better model performance in identifying actual diabetes cases. Consequently, the models became more reliable for predicting diabetes, ensuring fewer false negatives and enhancing the overall effectiveness of the diagnostic tool.

In summary, the use of SMOTE improved the model's ability to detect diabetic cases without compromising the identification of non-diabetic cases, thereby providing a more balanced and effective predictive performance.

#### Visualisations 
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of age distribution
plt.figure(figsize=(10, 6))
plt.hist(diabetes_data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot of BMI by diabetes status
plt.figure(figsize=(10, 6))
sns.boxplot(x='diabetes', y='bmi', data=diabetes_data)
plt.title('BMI by Diabetes Status')
plt.xlabel('Diabetes Status')
plt.ylabel('BMI')
plt.show()

# Scatter plot of BMI vs. Blood Glucose Level
plt.figure(figsize=(10, 6))
plt.scatter(diabetes_data['bmi'], diabetes_data['blood_glucose_level'], alpha=0.6, c='darkcyan')
plt.title('BMI vs. Blood Glucose Level')
plt.xlabel('BMI')
plt.ylabel('Blood Glucose Level')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = diabetes_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![1be549d8-20d6-4c20-9614-f72a0e524d4b](https://github.com/zhossain7/ensemblediabetes/assets/100549035/268d2999-841d-4f71-8883-d5c001e7f1d1)
* Explanation: The bar chart illustrates the age distribution of elderly individuals in a diabetes care dataset, with the x-axis showing age and the y-axis representing the frequency of individuals at each age.
* Analysis: The chart reveals a notable peak at age 80, indicating a higher frequency of having diabetes at this age.
  
![a0bc1042-65be-4b20-b739-55e672283e32](https://github.com/zhossain7/ensemblediabetes/assets/100549035/52e24626-8edd-40bb-b47b-354b1d10047f)
* Explanation: This box plot compares the BMI distributions between individuals with and without diabetes, highlighting any differences in their BMI values.
* Analysis: The plot reveals that individuals with diabetes tend to have a slightly higher median BMI compared to those without diabetes, suggesting a potential link between higher BMI and the likelihood of having diabetes.

![download](https://github.com/zhossain7/ensemblediabetes/assets/100549035/e9204adc-6393-4d89-9bec-7c08227f142a)
* Explanation: This scatter plot visualises the relationship between BMI and blood glucose levels, allowing for an examination of how these two variables interact.
* Analysis: The plot shows a positive trend, indicating that as BMI increases, blood glucose levels tend to increase as well, suggesting a possible correlation between higher BMI and higher blood glucose levels.

![9b5bd86d-c96b-4b05-b39c-a90176f928a5](https://github.com/zhossain7/ensemblediabetes/assets/100549035/86f2ba84-0a7e-4235-a3da-ae2b5a8808bc)
* Explanation: The heatmap displays the correlation coefficients between different numeric variables in the dataset, showing the strength and direction of their relationships.
* Analysis: The matrix highlights that blood glucose level has a strong positive correlation with diabetes status, suggesting it is a significant predictor for diabetes in the dataset.

#### Feature Importance
![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/a8daebe5-c40b-449e-beb2-a1c930dd1391)

Feature importance helps us understand which features contribute the most to the prediction of diabetes. Here, I use the XGBoost model to determine the importance of various features in our dataset. 
The feature importance plot provides insights into which features have the most significant impact on the model's predictions.


**Explanation:**
* The bar chart visualizes the importance of each feature used by the XGBoost model to predict diabetes.
* The x-axis represents the different features in the dataset, and the y-axis shows their respective importance scores.
* 
**Analysis:**

* **num_HbA1c_level:** This feature has the highest importance score, indicating that HbA1c levels are the most influential in predicting diabetes. HbA1c is a measure of blood sugar levels over the past three months, making it a critical indicator of diabetes.
* **num_blood_glucose_level:** The second most important feature is the blood glucose level. This feature is also directly related to diabetes, as high blood glucose levels are a primary symptom of the disease.
* *8num_bmi:** BMI (Body Mass Index) is another significant feature. Higher BMI is associated with an increased risk of diabetes, reflecting its importance in the model.
* **num_hypertension:** Hypertension (high blood pressure) also shows some importance, indicating its role as a potential risk factor for diabetes.

The feature importance plot highlights that HbA1c levels and blood glucose levels are the most critical predictors of diabetes in this dataset. Other features like BMI, hypertension, and smoking history also play significant roles. 
Understanding these importance scores helps in focusing on the most relevant features for diabetes prediction and provides insights into the underlying factors influencing the model's decisions.


#### Insights
* The majority of the dataset consists of females, which is reflected in the distribution of the gender feature.
* The age distribution shows that the majority of individuals are between 65 and 75 years old, which is expected given the dataset's focus on elderly individuals.
* The bmi distribution indicates a significant variance, with some individuals having a BMI above 30, indicating obesity, which is a known risk factor for diabetes.
* The blood_glucose_level feature also shows a wide range, with some individuals having significantly higher levels, which is a direct indicator of diabetes.
* A correlation heatmap reveals that HbA1c_level and blood_glucose_level are strongly correlated, as expected since both are indicators of blood sugar levels.
* There is a noticeable correlation between age and bmi, suggesting that older individuals in this dataset tend to have higher BMI


## Methodology

### Individual Models
### Logistic Regression
Logistic Regression is a simple yet powerful model for binary classification. It was used with a grid search for hyperparameter tuning, 
exploring different values of the regularisation parameter 𝐶 and types of regularisation (L1 and L2).
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define parameter grid for hyperparameter tuning
param_grid = {
    'C': np.logspace(-4, 4, 4),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Create Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Logistic Regression model
best_log_reg = grid_search.best_estimator_
```
### SVM
Support Vector Machine (SVM) is effective in high-dimensional spaces. I used a grid search to find the best values for the regularisation parameter 
𝐶 and the kernel coefficient 𝛾.
```python
from sklearn.svm import SVC

# Define parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

# Create SVM model
svm_model = SVC(random_state=42, probability=True)

# Setup GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best SVM model
best_svm = grid_search.best_estimator_
```
### XGBOOST
XGBoost is an efficient and scalable implementation of gradient boosting. I tuned parameters such as the learning rate, maximum depth of trees, number of estimators, and subsample ratio.
```python
import xgboost as xgb

# Define parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

# Create XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best XGBoost model
best_xgboost = grid_search.best_estimator_
```

## Ensemble Methods

### Voting (Soft and Hard)
The Voting ensemble method combines multiple machine learning models. In soft voting, the predicted probabilities are averaged, while in hard voting, the class labels are averaged.
```python
from sklearn.ensemble import VotingClassifier

# Create VotingClassifier for soft voting
voting_clf_soft = VotingClassifier(
    estimators=[('svm', best_svm), ('xgb', best_xgboost), ('log_reg', best_log_reg)],
    voting='soft'
)

# Fit the ensemble classifier
voting_clf_soft.fit(X_train, y_train)
```
### ADABoost
ADABoost combines multiple weak classifiers to create a strong classifier. I used decision trees as the base estimator and adjusted the number of estimators and learning rate.
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Create base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)

# Create ADABoost classifier
ada_boost_clf = AdaBoostClassifier(
    base_estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Fit ADABoost
ada_boost_clf.fit(X_train, y_train)
```
### Stacking
```python
from sklearn.ensemble import StackingClassifier

# Define the base models
base_models = [
    ('svm', best_svm),
    ('xgb', best_xgboost),
    ('log_reg', best_log_reg)
]

# Define meta-learner
meta_learner = LogisticRegression()

# Create StackingClassifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Fit the stacking classifier
stacking_clf.fit(X_train, y_train)
```
By integrating these individual models and ensemble methods, I aim to develop a robust predictive model that enhances the accuracy and reliability of diabetes prediction. 
The combination of diverse models helps mitigate the weaknesses of individual models, resulting in improved overall performance.

## Model Evaluation
In this section, I evaluate the performance of our trained models using a variety of metrics. The primary metrics I focus on is precision, recall, F1-score, and accuracy. These metrics provide a comprehensive understanding of each model's strengths and weaknesses. 

1. Precision: Precision is the ratio of true positive predictions to the total number of positive predictions made by the model. High precision indicates a low false positive rate.
* Class 0 (No Diabetes): Precision indicates how many of the individuals predicted to not have diabetes actually do not have it.
* Class 1 (Diabetes): Precision indicates how many of the individuals predicted to have diabetes actually have it.

 2. Recall: Recall is the ratio of true positive predictions to the total number of actual positives. High recall indicates a low false negative rate.
* Class 0 (No Diabetes): Recall indicates how many of the actual non-diabetic individuals were correctly identified.
* Class 1 (Diabetes): Recall indicates how many of the actual diabetic individuals were correctly identified.

3. F1-Score: The F1-Score is the harmonic mean of precision and recall. It provides a balance between precision and recall, especially useful when the class distribution is imbalanced.
* Class 0 (No Diabetes): F1-Score balances the precision and recall for non-diabetic individuals.
* Class 1 (Diabetes): F1-Score balances the precision and recall for diabetic individuals.

4. Accuracy: Accuracy is the ratio of correctly predicted instances to the total instances. It is a measure of the overall effectiveness of the model.

### Individual Models
![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/3a162ca1-0fa7-43bf-b4d0-71607983ecb7)

#### Evaluation Analysis
* Logistic Regression: The model shows a balanced performance between precision and recall for both classes, with an accuracy of 84%. The F1-scores are also fairly balanced between the two classes, indicating the model's effectiveness in predicting both diabetic and non-diabetic individuals.

* SVM: The SVM model exhibits a balanced precision and recall for both classes, with an accuracy of 87%. This indicates that the SVM model is effective in correctly identifying both diabetic and non-diabetic individuals with equal emphasis.

* XGBoost: XGBoost demonstrates the highest performance among the models with an accuracy of 95%. It shows very high precision and recall for both classes, with F1-scores of 0.95 for both classes. This suggests that XGBoost is highly effective in correctly identifying both diabetic and non-diabetic individuals, making it the most robust model in this evaluation.


### Ensemble Models


#### Evaluation Analysis
* ADABoost: The AdaBoost classifier demonstrates a balanced performance with high precision and recall for both classes. The overall accuracy of 92% indicates that the model is effective in distinguishing between diabetic and non-diabetic individuals.
  The balanced F1-scores (0.92) for both classes further confirm the model's robustness in handling class imbalances and maintaining consistent predictive performance.
  
* Stacking: The Stacking Ensemble model performs exceptionally well, achieving an accuracy of 95%. The precision and recall for both classes are high, with F1-scores of 0.95, indicating that the model effectively combines the strengths of its base models. This results in superior predictive performance, making it highly reliable for predicting diabetes.
  
* Soft Voting:The Soft Voting classifier shows a balanced precision and recall for both classes, with an overall accuracy of 89%. The F1-scores are also balanced at 0.90 and 0.89 for classes 0 and 1, respectively. This indicates that the model effectively balances the predictions of its base models to achieve reliable performance, though it is slightly less accurate compared to AdaBoost and Stacking Ensemble.

These evaluation metrics provide a comprehensive understanding of how each ensemble method performs. The Stacking Ensemble model stands out as the best performing method, followed by AdaBoost and Soft Voting. The Stacking Ensemble's ability to leverage multiple base models results in superior performance, while AdaBoost and Soft Voting also demonstrate strong predictive capabilities.

## Final Results

### Summary of Model Performance
![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/14093b41-b5b8-41f4-8cc0-81cf8493972e)

### Analysis of Results

**AdaBoost:**
* **Performance:** The AdaBoost classifier achieves a high accuracy of 92%, with balanced precision and recall for both diabetic and non-diabetic classes. The F1-scores of 0.92 indicate the model's effectiveness in handling class imbalances and maintaining consistent predictive performance.
* **Significance:** The balanced performance of AdaBoost makes it a reliable choice for predicting diabetes, providing both high precision (reducing false positives) and high recall (reducing false negatives).
  
**Stacking Ensemble:**
* Performance:** The Stacking Ensemble model demonstrates the highest accuracy at 95%, with very high precision and recall for both classes. The F1-scores of 0.95 reflect the model's robustness and ability to leverage multiple base models for superior predictive performance.
* **Significance:** The exceptional performance of the Stacking Ensemble model highlights its potential as the best approach for predicting diabetes. It combines the strengths of individual models to achieve high accuracy and reliability, making it a powerful tool for medical diagnosis.
  
**Soft Voting:**

* **Performance:** The Soft Voting classifier shows balanced precision and recall for both classes, with an overall accuracy of 89%. The F1-scores of 0.90 and 0.89 for classes 0 and 1, respectively, indicate a reliable performance, though slightly lower than AdaBoost and Stacking Ensemble.
* **Significance:** While the Soft Voting classifier is slightly less accurate, its balanced performance makes it a viable option for diabetes prediction. It effectively combines the predictions of its base models, providing a robust and reliable approach.

The final results demonstrate that ensemble methods significantly enhance the predictive performance of machine learning models for diabetes detection. Among the evaluated methods, the Stacking Ensemble model stands out with the highest accuracy and balanced performance across all metrics. AdaBoost also shows strong results, making it a reliable alternative. Soft Voting, while slightly less accurate, still provides a robust approach for predicting diabetes.

These findings underscore the importance of leveraging ensemble methods in medical diagnosis tasks, where high accuracy and reliability are crucial. The results also highlight the potential of machine learning models to aid in early detection and management of diabetes, ultimately contributing to better healthcare outcomes for the elderly population.

