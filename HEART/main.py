# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 2: Load the dataset
df = pd.read_csv("heart.csv")

# Step 3: Check for missing values (there shouldn't be any in this dataset)
print("Missing values:\n", df.isnull().sum())

# Step 4: Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 5: Define features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Step 6: Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save the model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("\n💾 Model saved as 'heart_disease_model.pkl'")