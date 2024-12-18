import json
from preprocess import preprocess_patterns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load intents.json
with open("intents.json", "r") as file:
    data = json.load(file)

# Preprocess data
patterns, tags = preprocess_patterns(data)
print("Data preprocessing complete.")

# Convert patterns to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Train the model
model = LogisticRegression()
model.fit(X, y)
print("Model training complete.")

# Evaluate the model
y_pred = model.predict(X)  # Predictions on training data

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

# Save model and preprocessing objects
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and preprocessing objects saved successfully.")
