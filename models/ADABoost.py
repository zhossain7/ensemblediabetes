import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt


# Load the dataset from a CSV file.
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


# Create the base model for AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=1)

ada_boost_clf = AdaBoostClassifier(
    estimator=base_estimator,  # Updated argument name
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Assuming X_train, y_train, X_test, and y_test are already defined and preprocessed

# Fit AdaBoost on the training data
ada_boost_clf.fit(X_train, y_train)

# Make predictions
probabilities = ada_boost_clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Save the trained AdaBoost model as a .pkl file
joblib.dump(ada_boost_clf, 'ada_boost_clf.pkl')

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
