# Detecting Diabetes in the Elderly Using Ensemble Methods



## Table of contents
1. Project Introduction
2. Project Overview
3. Data Analysis
4. Methodology
5. Model Evaluation
6. Final Results
7. Conclusion

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
Support Vector Machine (SVM) is effective in high-dimensional spaces. We used a grid search to find the best values for the regularisation parameter 
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
XGBoost is an efficient and scalable implementation of gradient boosting. We tuned parameters such as the learning rate, maximum depth of trees, number of estimators, and subsample ratio.
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
ADABoost combines multiple weak classifiers to create a strong classifier. We used decision trees as the base estimator and adjusted the number of estimators and learning rate.
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

