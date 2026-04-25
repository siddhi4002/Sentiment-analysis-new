import json
from sentiment_analysis_system import SentimentSystem

print("Training model...")
system = SentimentSystem()
system.train_and_evaluate(verbose=False)

model = system.best_model
pipeline = model.pipeline
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['clf']

# Extract parameters
vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
idf = vectorizer.idf_.tolist()
coef = classifier.coef_.tolist() # shape (n_classes, n_features)
intercept = classifier.intercept_.tolist() # shape (n_classes,)
classes = classifier.classes_.tolist()

model_data = {
    "vocabulary": vocab,
    "idf": idf,
    "coef": coef,
    "intercept": intercept,
    "classes": classes
}

with open("model.json", "w") as f:
    json.dump(model_data, f)
print("Saved to model.json")
