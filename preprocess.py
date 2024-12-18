import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')

stemmer = PorterStemmer()

def preprocess_patterns(intents_data):
    """ Tokenize, stem, and clean the patterns from intents data """
    patterns = []
    tags = []

    for intent in intents_data:
        for pattern in intent['patterns']:
            words = nltk.word_tokenize(pattern.lower())  # Tokenize
            words = [stemmer.stem(w) for w in words if w.isalnum()]  # Stem and remove non-alphanumerics
            patterns.append(" ".join(words))
            tags.append(intent['tag'])
    return patterns, tags
