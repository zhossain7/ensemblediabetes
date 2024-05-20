# Detecting Diabetes in the Elderly Using Ensemble Methods



## Table of contents
1. Project Introduction
2. Project Overview
3. [Setup](#setup)

## Project Introduction
This project was created because I was doing a subject in university which was discussing the potential benefits of artificial intelligence (AI) if introduced to the medical field. 
I conducted some research, and while artificial intelligence still needs time before it can be fully implemented into the medical field, it has the potential to be very important in the future.
AI has the potential to revolutionise healthcare by providing tools that can assist in early diagnosis, treatment planning, and patient management. 

This project serves as a step towards integrating AI into healthcare.
It inspired me to do reserach and see how exactly AI can help, which led me to this project which encompasses AI and data analysis, both a core part of my degree and major.

	
## Project Overview

	
## Data and Analysis
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



