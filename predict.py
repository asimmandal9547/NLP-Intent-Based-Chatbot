import json
import joblib
import nltk
from nltk.stem import PorterStemmer
import random

# Download punkt resource
nltk.download('punkt_tab')

# Initialize the stemmer
stemmer = PorterStemmer()

# Load model and preprocessing objects
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Function to predict intent and generate response
def predict_intent(user_input):
    words = nltk.word_tokenize(user_input.lower())  # Tokenize input
    words = [stemmer.stem(w) for w in words if w.isalnum()]  # Stem and clean words
    processed_input = " ".join(words)

    vectorized_input = vectorizer.transform([processed_input]).toarray()
    tag_index = model.predict(vectorized_input)[0]
    predicted_tag = label_encoder.inverse_transform([tag_index])[0]

    # Retrieve a random response for the predicted tag
    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Chat loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = predict_intent(user_input)
    print(f"Bot: {response}")
