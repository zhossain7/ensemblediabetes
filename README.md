# Detecting Diabetes in the Elderly Using Ensemble Methods



## Table of contents
1. Project Introduction
2. Project Overview
3. Data Analysis

## Project Introduction
This project was created because I was doing a subject in university which was discussing the potential benefits of artificial intelligence (AI) if introduced to the medical field. 
I conducted some research, and while artificial intelligence still needs time before it can be fully implemented into the medical field, it has the potential to be very important in the future.
AI has the potential to revolutionise healthcare by providing tools that can assist in early diagnosis, treatment planning, and patient management. 

This project serves as a step towards integrating AI into healthcare.
It inspired me to do reserach and see how exactly AI can help, which led me to this project which encompasses AI and data analysis, both a core part of my degree and major.

	
## Project Overview

	
## Data Analysis
The dataset useed for this project  is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It can be availible in Kaggle [right here.](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).

![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/bfe27470-2e66-4146-80bd-e6b76df509a0)
![image](https://github.com/zhossain7/ensemblediabetes/assets/100549035/d4dd3612-710b-460b-804c-564de8db06fa)

The dataset has 100,000 individuals, however after extracing indviduals aged 65 or above, I was left with 18,568. This shorter dataset is what was used to train my models and then predict on.\

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




