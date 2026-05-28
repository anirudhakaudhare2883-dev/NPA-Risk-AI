"""
==========================================================
 NPA Prediction System — Model Training Script
==========================================================
 This script trains a Random Forest classifier on the
 NPA dataset and saves the model + encoders to disk.

 Usage:
   python train_model.py
==========================================================
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
print("📂 Loading dataset...")
df = pd.read_csv("dataset/npa_dataset.csv")
print(f"   ✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 2. Select features and target
# ----------------------------------------------------------
# Features used for prediction
feature_columns = [
    "Loan_Amount",
    "Loan_Type",
    "Credit_Score",
    "Repayment_History",
    "Collateral_Value",
    "Loan_Tenure"
]

target_column = "Default_Status"

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"\n📊 Feature columns: {feature_columns}")
print(f"   Target column: {target_column}")
print(f"   Default distribution:\n{y.value_counts().to_string()}")

# ----------------------------------------------------------
# 3. Encode categorical variables
# ----------------------------------------------------------
# Loan_Type is the only categorical feature — encode it
print("\n🔄 Encoding categorical variables...")

loan_type_encoder = LabelEncoder()
X["Loan_Type"] = loan_type_encoder.fit_transform(X["Loan_Type"])

print(f"   Loan_Type classes: {list(loan_type_encoder.classes_)}")
print(f"   Encoded values:   {list(range(len(loan_type_encoder.classes_)))}")

# Store all encoders in a dictionary for easy access
encoders = {
    "Loan_Type": loan_type_encoder
}

# ----------------------------------------------------------
# 4. Split data into training and testing sets
# ----------------------------------------------------------
print("\n📐 Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# ----------------------------------------------------------
# 5. Train the Random Forest model
# ----------------------------------------------------------
print("\n🌲 Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,       # Number of trees in the forest
    max_depth=15,           # Maximum depth of each tree
    min_samples_split=5,    # Minimum samples needed to split a node
    min_samples_leaf=2,     # Minimum samples in a leaf node
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores for speed
)

model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# ----------------------------------------------------------
# 6. Evaluate the model
# ----------------------------------------------------------
print("\n📈 Evaluating model performance...")

# Training accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"   Training Accuracy: {train_accuracy * 100:.2f}%")

# Testing accuracy
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"   Testing Accuracy:  {test_accuracy * 100:.2f}%")

# Detailed classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# Feature importances
print("🔑 Feature Importances:")
for feature, importance in sorted(
    zip(feature_columns, model.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    print(f"   {feature:25s} → {importance:.4f}")

# ----------------------------------------------------------
# 7. Save the model and encoders
# ----------------------------------------------------------
print("\n💾 Saving model and encoders...")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
with open("model/npa_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("   ✅ Model saved to: model/npa_model.pkl")

# Save the encoders dictionary
with open("model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("   ✅ Encoders saved to: model/encoders.pkl")

# ----------------------------------------------------------
# Done!
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("🎉 Model training complete!")
print(f"   Accuracy: {test_accuracy * 100:.2f}%")
print("   Files saved in 'model/' directory")
print("   You can now run: python app.py")
print("=" * 50)
