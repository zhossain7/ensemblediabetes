import pandas as pd
from sklearn.metrics import roc_curve, auc, classification_report, f1_score, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
from xgboost import XGBClassifier

file_path = 'locationhere'
diabetes_data = pd.read_csv(file_path)

# Identifying the types of columns in the dataset to apply appropriate preprocessing.
categorical_cols = diabetes_data.select_dtypes(include=['object']).columns
numerical_cols = diabetes_data.select_dtypes(include=['float64', 'int64']).columns.drop('diabetes')

# Define preprocessing steps for categorical data.
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define preprocessing steps for numerical data.
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine the transformations for both numerical and categorical data into a single ColumnTransformer.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define a pipeline that includes preprocessing and the SMOTE technique.
pipeline = ImblearnPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42))
])

# Preparing the features (X) and the target (y) for model training.
X = diabetes_data.drop('diabetes', axis=1)  # Features
y = diabetes_data['diabetes']  # Target variable

# Apply the preprocessing and SMOTE transformations to the features.
X_resampled, y_resampled = pipeline.fit_resample(X, y)

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Outputting the sizes of the transformed and split dataset to ensure everything is as expected.
print(f'Training set size: {X_train.shape}')  # Reflects balanced dataset size
print(f'Test set size: {X_test.shape}')

# Combine features and target variables back into DataFrames for training and testing sets.
train_data = pd.DataFrame(X_train, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
train_data['diabetes'] = y_train.values

test_data = pd.DataFrame(X_test, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
test_data['diabetes'] = y_test.values

### SVM MODEl ###

# Assuming X_train, X_test, y_train, y_test are already defined from the previous steps
# Define a parameter grid to search through for hyperparameter tuning.
# 'C' is the regularisation parameter. It controls the trade-off between smooth decision boundary and classifying training points correctly.
# A smaller value of 'C' creates a smooth decision boundary, while a larger value aims to classify more training points correctly.
param_grid = {
    'C': [0.1, 1, 10],  # Different values of 'C' to explore.
    # 'gamma' defines the influence of a single training example. High values lead to tight fit (potential overfitting).
    # 'scale' and 'auto' are choices for gamma. 'scale' uses 1 / (n_features * X.var()) as value of gamma,
    # 'auto' uses 1 / n_features.
    'gamma': ['scale', 'auto'],
    # 'kernel' specifies the kernel type to be used in the algorithm. It can transform the input space to a higher dimensional space.
    # 'rbf' and 'linear' are two types of kernels. 'rbf' is useful for non-linear hyperplane. 'linear' is for linear hyperplane.
    'kernel': ['rbf', 'linear']
}

# Initialise the SVM model with specific settings.
# 'random_state' ensures reproducibility of your results by setting a seed for random number generation.
# 'probability=True' enables probability estimates, which are necessary for calculating ROC curves and probabilities.
svm_model = SVC(random_state=42, probability=True)

# Setup GridSearchCV to perform an exhaustive search over specified parameter values for the SVM model.
# 'cv=5' specifies 5-fold cross-validation.
# 'scoring='accuracy'' means the evaluation of the models is based on how accurately they classify the test data.
# 'n_jobs=-1' allows the process to use all CPU cores to speed up the operation.
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model on the training data. This step finds the best combination of parameters defined in 'param_grid'.
grid_search.fit(X_train, y_train)

# Retrieve the best SVM model found by GridSearchCV.
best_svm = grid_search.best_estimator_

# Use the best model to make predictions on the test dataset.
predictions = best_svm.predict(X_test)

# Calculate probability estimates for each class. This is useful for metrics like the ROC curve.
probabilities = best_svm.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification report:")
print(classification_report(y_test, predictions))

# Calculating F1 score
f1 = f1_score(y_test, predictions)
print(f"F1 Score: {f1}")

# Precision-Recall curve and average precision
precision, recall, _ = precision_recall_curve(y_test, probabilities)
average_precision = average_precision_score(y_test, probabilities)

# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(15, 5))

# Plotting Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.step(recall, precision, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')

# Plotting ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")




plt.tight_layout()
plt.show()





# Assuming X_train, X_test, y_train, y_test are already defined

# Define a parameter grid to search through for hyperparameter tuning.
param_grid = {
    'C': np.logspace(-4, 4, 4),
    'penalty': ['l1', 'l2'],  # l1 and l2 regularization
    'solver': ['liblinear']  # liblinear is good for small datasets and supports l1
}

# Create an instance of the LogisticRegression model.
log_reg = LogisticRegression(random_state=42)

# Set up GridSearchCV with a focus on recall
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='recall', n_jobs=-1)  # Focus on recall

# Train the model using the grid search
grid_search.fit(X_train, y_train)

# Use the best estimator to make predictions
best_log_reg = grid_search.best_estimator_

# Predictions on the test set
predictions = best_log_reg.predict(X_test)
probabilities = best_log_reg.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification report:")
print(classification_report(y_test, predictions))

# Calculating F1 score
f1 = f1_score(y_test, predictions)
print(f"F1 Score: {f1}")

# Precision-Recall curve and average precision
precision, recall, _ = precision_recall_curve(y_test, probabilities)
average_precision = average_precision_score(y_test, probabilities)

# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(15, 5))

# Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.step(recall, precision, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')

# ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()





# Assuming X_train, X_test, y_train, y_test are already defined

# Define a parameter grid to explore during hyperparameter tuning.
# 'learning_rate' (or 'eta') controls how quickly the model fits the residual error using additional base learners.
# A low learning rate will require more boosting rounds to achieve the same reduction in residual error as a model with a high learning rate.
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],  # Values to try for 'learning_rate'.
    'max_depth': [3, 5, 7],  # Maximum depth of a tree. Increasing this value will make the model more complex and likely to overfit.
    'n_estimators': [100, 200, 300],  # Number of trees. More trees can lead to better performance but can also lead to overfitting.
    'subsample': [0.8, 0.9, 1.0]  # Subsample ratio of the training instances. A lower value can lead to more variance in the sampling process.
}

# Initialise the XGBoost classifier model with specific settings to avoid future warnings or errors.
# 'use_label_encoder=False' avoids using the deprecated label encoder and instead relies on ordinal encoding.
# 'eval_metric='logloss'' is chosen to evaluate the performance of the model during the training process.
# 'random_state=42' ensures that the same sequence of random numbers is generated each time the code is run, making results reproducible.
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Setup GridSearchCV for exhaustive search over specified parameter values for the XGBoost model.
# 'cv=5' means using 5-fold cross-validation to ensure that each fold serves as the test set exactly once.
# 'scoring='accuracy'' specifies that the evaluation metric for selecting the best model is accuracy.
# 'n_jobs=-1' uses all available computing power to speed up the search.
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the training data to find the best parameters.
grid_search.fit(X_train, y_train)

# Retrieve the best XGBoost model identified by the grid search.
best_xgboost = grid_search.best_estimator_

# Use the best model to make predictions on the test set.
predictions = best_xgboost.predict(X_test)

# Obtain the probability of the positive class for each instance in the test set, useful for calculating ROC curve and other metrics.
probabilities = best_xgboost.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification report:")
print(classification_report(y_test, predictions))

# Calculating F1 score
f1 = f1_score(y_test, predictions)
print(f"F1 Score: {f1}")

# Precision-Recall curve and average precision
precision, recall, _ = precision_recall_curve(y_test, probabilities)
average_precision = average_precision_score(y_test, probabilities)

# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plotting (reuse the previous plotting code)
plt.figure(figsize=(15, 5))

# Plotting Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.step(recall, precision, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')



plt.tight_layout()
plt.show()










# Assuming best_svm, best_xgboost, and best_log_reg are already defined from your GridSearchCV results
# For the sake of this example, I will use placeholders. Replace these with your actual model instances.
# Example placeholders (remove these and use your actual models):


# Placeholder models, replace with your actual trained models
best_svm = SVC(probability=True)  # Replace with your actual trained model
best_xgboost = XGBClassifier()  # Replace with your actual trained model
best_log_reg = LogisticRegression()  # Replace with your actual trained model

# Fit these models to your data (this is just for placeholder purposes, use your actual fitted models)
best_svm.fit(X_train, y_train)
best_xgboost.fit(X_train, y_train)
best_log_reg.fit(X_train, y_train)

# Instantiate the VotingClassifier for hard voting
voting_clf_hard = VotingClassifier(
    estimators=[('svm', best_svm), ('xgb', best_xgboost), ('log_reg', best_log_reg)],
    voting='hard'
)

# Fit the ensemble classifier to the training data
voting_clf_hard.fit(X_train, y_train)

# Save the VotingClassifier to a .pkl file
joblib.dump(voting_clf_hard, 'voting_clf_hard.pkl')

# Predictions for evaluation metrics
predictions_hard = voting_clf_hard.predict(X_test)

# Evaluation Metrics
print("Classification report for hard voting:")
print(classification_report(y_test, predictions_hard))

f1_hard = f1_score(y_test, predictions_hard)
print(f"F1 Score for hard voting: {f1_hard}")

precision_hard, recall_hard, _ = precision_recall_curve(y_test, predictions_hard)
average_precision_hard = average_precision_score(y_test, predictions_hard)
fpr_hard, tpr_hard, _ = roc_curve(y_test, predictions_hard)
roc_auc_hard = auc(fpr_hard, tpr_hard)



# Plot the ROC curve for hard voting
plt.figure()
plt.plot(fpr_hard, tpr_hard, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_hard)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Hard Voting)')
plt.legend(loc="lower right")
plt.show()

# Instantiate the VotingClassifier for soft voting
voting_clf_soft = VotingClassifier(
    estimators=[('svm', best_svm), ('xgb', best_xgboost), ('log_reg', best_log_reg)],
    voting='soft'
)

# Fit the ensemble classifier to the training data
voting_clf_soft.fit(X_train, y_train)

# Save the VotingClassifier to a .pkl file
joblib.dump(voting_clf_soft, 'voting_clf_soft.pkl')

# Predictions and probabilities for ROC and precision-recall curves
predictions_soft = voting_clf_soft.predict(X_test)
probabilities_soft = voting_clf_soft.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("Classification report for soft voting:")
print(classification_report(y_test, predictions_soft))

f1_soft = f1_score(y_test, predictions_soft)
print(f"F1 Score for soft voting: {f1_soft}")

precision_soft, recall_soft, _ = precision_recall_curve(y_test, probabilities_soft)
average_precision_soft = average_precision_score(y_test, probabilities_soft)
fpr_soft, tpr_soft, _ = roc_curve(y_test, probabilities_soft)
roc_auc_soft = auc(fpr_soft, tpr_soft)



# Plot the ROC curve for soft voting
plt.figure()
plt.plot(fpr_soft, tpr_soft, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_soft)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Soft Voting)')
plt.legend(loc="lower right")
plt.show()
