import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load models (example paths, replace with actual paths)
models = {
    'Soft Voting': '/voting_clf_soft.pkl',
    'ADABoost': '/ada_boost_clf.pkl',
    'Stacking Classifier': '/stacking_clf.pkl'
}

# Define the same preprocessing steps used in the original CSV preprocessing
categorical_cols = ['gender', 'smoking_history']
numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

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

# Preprocessing function
def preprocess_input(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data], columns=numerical_cols + categorical_cols)
    # Apply preprocessing
    preprocessed_data = preprocessor.transform(input_df)
    return preprocessed_data

# Prediction function
def predict():
    try:
        # Collect user input
        gender = gender_var.get()
        age = int(age_entry.get())
        if age < 65 or age > 80:
            raise ValueError("Age must be between 65 and 80.")
        hypertension = int(hypertension_var.get())
        heart_disease = int(heart_disease_var.get())
        smoking_history = smoking_history_var.get()
        bmi = float(bmi_entry.get())
        hba1c_level = float(hba1c_entry.get())
        blood_glucose_level = float(blood_glucose_entry.get())

        # Preprocess the input data
        input_data = preprocess_input([
            age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level, gender, smoking_history
        ])

        # Load the selected model
        model_name = model_var.get()
        model = joblib.load(models[model_name])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]  # Get the probability for the positive class

        # Display the result
        result_text.set(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}\nProbability: {probability * 100:.2f}%")

    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Diabetes Prediction")

# Apply a dark theme if available
theme_path = "C:/Users/zoobe/Documents/projects/diabetesensemble/ensemblediabetes/azure-dark.tcl"
if os.path.exists(theme_path):
    style = ttk.Style()
    root.tk.call("source", theme_path)
    style.theme_use("azure-dark")

# Configure the main window to resize properly
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Define input variables
gender_var = tk.StringVar()
age_var = tk.StringVar()
hypertension_var = tk.StringVar()
heart_disease_var = tk.StringVar()
smoking_history_var = tk.StringVar()
bmi_var = tk.StringVar()
hba1c_var = tk.StringVar()
blood_glucose_var = tk.StringVar()
model_var = tk.StringVar()

# Create input fields
fields = {
    'Gender': (gender_var, ['Female', 'Male']),
    'Age': age_var,
    'Hypertension': (hypertension_var, [0, 1]),
    'Heart Disease': (heart_disease_var, [0, 1]),
    'Smoking History': (smoking_history_var, ['never', 'current', 'former', 'not current', 'No Info']),
    'BMI': bmi_var,
    'HbA1c Level': hba1c_var,
    'Blood Glucose Level': blood_glucose_var
}

# Place input fields
entries = {}
for idx, (label, var) in enumerate(fields.items()):
    ttk.Label(mainframe, text=label).grid(row=idx, column=0, padx=10, pady=5, sticky=tk.W)
    if isinstance(var, tuple):
        entry = ttk.Combobox(mainframe, textvariable=var[0], values=var[1])
    else:
        entry = ttk.Entry(mainframe, textvariable=var)
    entry.grid(row=idx, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
    entries[label] = entry

age_entry = entries['Age']
bmi_entry = entries['BMI']
hba1c_entry = entries['HbA1c Level']
blood_glucose_entry = entries['Blood Glucose Level']

# Make the input fields expand with window resizing
for entry in entries.values():
    entry.grid(sticky=(tk.W, tk.E))

# Model selection
ttk.Label(mainframe, text="Select Model").grid(row=len(fields), column=0, padx=10, pady=5, sticky=tk.W)
model_dropdown = ttk.Combobox(mainframe, textvariable=model_var, values=list(models.keys()))
model_dropdown.grid(row=len(fields), column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

# Prediction button
predict_button = ttk.Button(mainframe, text="Predict", command=predict)
predict_button.grid(row=len(fields)+1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

# Result display
result_text = tk.StringVar()
result_label = ttk.Label(mainframe, textvariable=result_text)
result_label.grid(row=len(fields)+2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

# Make the mainframe expand with window resizing
for i in range(len(fields) + 3):
    mainframe.rowconfigure(i, weight=1)
mainframe.columnconfigure(0, weight=1)
mainframe.columnconfigure(1, weight=1)

# Fit the preprocessor to the entire dataset to ensure proper transformation
file_path = 'C:/Users/zoobe/Documents/projects/diabetesensemble/ensemblediabetes/data/diabetes_65.csv'
diabetes_data = pd.read_csv(file_path)
X = diabetes_data.drop('diabetes', axis=1)  # Features
preprocessor.fit(X)

# Start the GUI event loop
root.mainloop()
