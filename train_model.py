import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("telco_churn.csv")

# Remove customerID column
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df = df.fillna(0)

# Convert categorical columns to numbers
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = label_encoder.fit_transform(df[column])

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train machine learning model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")
