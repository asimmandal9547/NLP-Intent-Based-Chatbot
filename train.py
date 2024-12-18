from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib  # To save the trained model

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model, vectorizer, label_encoder):
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Model saved successfully!")
