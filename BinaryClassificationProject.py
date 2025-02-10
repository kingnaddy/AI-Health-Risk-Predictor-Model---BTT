# %%
#STEP 1
#Import neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns

#Display settings, to ensure all columns are shown
pd.set_option("display.max_columns", None)


# %%
#STEP 2
#Loading testing and training datasets
train = pd.read_csv("/kaggle/input/btt-dataspark-2025/train.csv")
test = pd.read_csv("/kaggle/input/btt-dataspark-2025/test.csv")

print("Train Dataset sample: ")
print(train.head())

print("\nTrain Dataset info: ")
print(train.info())

print("\nTrain Dataset summary: ")
print(train.describe())

print("\nTest Dataset sample: ")
print(test.head())

print("\nTest Dataset info: ")
print(test.info())

print("\nTest Dataset summary: ")
print(test.describe())

# %%
#STEP 3
#Checking for missing values in the dataset

# Check for missing values in the training data
print("\nMissing Values in Training Data:")
print(train.isnull().sum())

# Check for missing values in the test data
print("\nMissing Values in Test Data:")
print(test.isnull().sum())


# %%
#STEP 4
#Check for outliers
numerical_columns = train.select_dtypes(include=["int64", "float64"]).columns

# Plot boxplots for numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=train[numerical_columns])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# %%
# Plot the distribution of the smoking column
sns.countplot(x="smoking", data=train)
plt.title("Distribution of Smoking Status")
plt.xlabel("Smoking Status (1 = Smoker, 0 = Non-Smoker)")
plt.ylabel("Count")
plt.show()

# Calculate the ratio of non-smokers to smokers
non_smokers = train["smoking"].value_counts()[0]
smokers = train["smoking"].value_counts()[1]
print(f"Non-smokers: {non_smokers}, Smokers: {smokers}, Ratio: {non_smokers/smokers:.2f}:1")

# %%
#Using a histogram to check the distribution of the age against the smoking column
#Set the style for the plot
sns.set(style="whitegrid")

# Create overlapping histograms for age by smoking status
plt.figure(figsize=(10, 6))
sns.histplot(train[train["smoking"] == 0]["age"], color="blue", label="Non-Smokers", kde=True, bins=30)
sns.histplot(train[train["smoking"] == 1]["age"], color="red", label="Smokers", kde=True, bins=30)
plt.title("Age Distribution by Smoking Status")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend(title="Smoking Status")
plt.show()

# %%
#Haemoglobin versus smoking 
plt.figure(figsize=(6, 5))
sns.boxplot(x=train["smoking"], y=train["hemoglobin"], palette="coolwarm")
plt.title("Hemoglobin Levels vs Smoking")
plt.xlabel("Smoking (0 = Non-smoker, 1 = Smoker)")
plt.ylabel("Hemoglobin Level")
plt.show()


# %%
# Correlation between hemoglobin and smoking
correlation = train["hemoglobin"].corr(train["smoking"])
print(f"Correlation between Hemoglobin and Smoking: {correlation:.4f}")


# %%
#Scatterplot of Hemoglobin vs. Age
plt.figure(figsize=(8, 5))
sns.scatterplot(x=train["age"], y=train["hemoglobin"], hue=train["smoking"], palette="coolwarm", alpha=0.7)
plt.title("Hemoglobin vs Age (Colored by Smoking)")
plt.xlabel("Age")
plt.ylabel("Hemoglobin Level")
plt.legend(title="Smoking", labels=["Non-Smoker", "Smoker"])
plt.show()

# %%
# Correlation between Cholesterol and Smoking
cholesterol_smoking_correlation = train["Cholesterol"].corr(train["smoking"])
print(f"Correlation between Cholesterol and Smoking: {cholesterol_smoking_correlation:.4f}")


# %%
# Compute correlation with smoking inorder to check for strong predictor variable
correlation_matrix = train.corr()
smoking_correlation = correlation_matrix["smoking"].sort_values(ascending=False)

# Display top correlated features
print(smoking_correlation)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Select features and target
X = train[["height(cm)", "hemoglobin"]]  # Predictor variables
y = train["smoking"]  # Target variable

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

# %%
#Model Improvement
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the new class distribution
print("Resampled Class Distribution:")
print(y_resampled.value_counts())

# %%
#Retrain the Model
from sklearn.linear_model import LogisticRegression

# Train a logistic regression model on the resampled data
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)

# %%
#Evaluate the model
from sklearn.metrics import classification_report, roc_auc_score

# Evaluate the model
y_pred = model.predict(X_val)
print("Classification Report with SMOTE:")
print(classification_report(y_val, y_pred))
print("AUC-ROC Score with SMOTE:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

# %%
#Using the trained model to predict probabilities for the test dataset.
# Select the same features used for training
X_test = test[["height(cm)", "hemoglobin"]]

# Predict probabilities for the test dataset
y_test_probs = model.predict_proba(X_test)[:, 1]

# Create a submission file
submission = pd.DataFrame({
    "id": test["id"],  # Use the ID column from the test dataset
    "smoking": y_test_probs  # Use predicted probabilities
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

# Display the first few rows of the submission file
print(submission.head)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Select features and target
X = train[["Gtp", "hemoglobin","serum creatinine", "weight(kg)","triglyceride", "height(cm)", "waist(cm)", "age"]]  # Predictor variable (BMI)
y = train["smoking"]  # Target variable (smoking)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
y_probs = model.predict_proba(X_val)[:, 1]  # Probabilities for the positive class (smoking)

print("Classification Report:")
print(classification_report(y_val, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_val, y_probs))


# %%
#using the trained model to predict probablities for the test dataset
# Select the same features used for training
features = ["Gtp", "hemoglobin","serum creatinine", "weight(kg)","triglyceride", "height(cm)", "waist(cm)", "age"]  # Replace with your selected features
X_test = test[features]

# Make predictions on the test dataset
# Predict probabilities for the test dataset
y_test_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (smoking)

# Create a submission file
submission = pd.DataFrame({
    "id": test["id"],  # Use the ID column from the test dataset
    "smoking": y_test_probs  # Use predicted probabilities
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

# Display the first few rows of the submission file
print("Submission File Preview:")
print(submission.head())


