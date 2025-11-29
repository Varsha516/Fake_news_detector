import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Step 1 — Load Dataset
# We'll use a public Fake/True news dataset
# You can download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

try:
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
except Exception as e:
    print("Error loading datasets:", e)
    exit()

df_fake["label"] = 0  # Fake news
df_true["label"] = 1  # Real news

df_fake["text"] = df_fake["title"] + " " + df_fake["text"]
df_true["text"] = df_true["title"] + " " + df_true["text"]

df = pd.concat([df_fake[["text", "label"]], df_true[["text", "label"]]])
df = df.sample(frac=1, random_state=42)  # Shuffle dataset

print(f"Dataset loaded: {len(df)} samples")

# Step 2 — Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Step 3 — Build Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", max_df=0.8)),
    ('clf', LogisticRegression(solver="liblinear", max_iter=1000))
])

# Step 4 — Train Model
model.fit(X_train, y_train)

# Step 5 — Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")

# Step 6 — Save Model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
