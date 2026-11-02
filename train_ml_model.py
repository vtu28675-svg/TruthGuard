
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed dataset
data = pd.read_csv("dataset/processed/news_dataset.csv")

# Remove rows with missing title or text
data = data.dropna(subset=["title", "text"])

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# Input and output
X = data["content"]
y = data["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        max_features=50000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
print("Training TruthGuard ML Model...")
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL TRAINING COMPLETED!")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "Real News"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fake_news_model.pkl")

print("\nModel saved successfully!")
print("Location: models/fake_news_model.pkl")
