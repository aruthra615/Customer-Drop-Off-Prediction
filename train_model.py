import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("dataset/telco_churn.csv")

# Remove ID column
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert TotalCharges safely
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df = df.fillna(0)

# Convert categorical columns automatically
df = pd.get_dummies(df)

# Separate features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")