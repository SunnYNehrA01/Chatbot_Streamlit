import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load intents JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Prepare data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Vectorize input patterns
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags

# Train the classifier
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(x, y)

# Save the vectorizer and classifier
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("classifier.pkl", "wb") as clf_file:
    pickle.dump(clf, clf_file)

print("Chatbot model and vectorizer saved successfully!")