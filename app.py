from flask import Flask, render_template, request
import joblib, requests, os
from dotenv import load_dotenv
from urllib.parse import quote
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load ML model
model = joblib.load("model.pkl")

# API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY not found")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found")

if not GOOGLE_CSE_ID:
    print("Warning: GOOGLE_CSE_ID not found")


def google_search(query, api_key, cse_id, num=5):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num).execute()
        return res.get("items", [])
    except Exception as e:
        print("Google Search error:", e)
        return []


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    sources = []

    if request.method == "POST":
        user_input = request.form["news"].strip()

        if user_input:
            # ML Model Prediction
            pred = model.predict([user_input])[0]
            ml_conf = float(max(model.predict_proba([user_input])[0])) * 100

            # NewsAPI
            try:
                query = " ".join(user_input.split()[:6])
                url = f"https://newsapi.org/v2/everything?q={quote(query)}&apiKey={NEWS_API_KEY}"
                response = requests.get(url, timeout=5).json()
                articles = response.get("articles", [])

                if articles:
                    for article in articles[:5]:
                        sources.append({
                            "title": article.get("title"),
                            "source": article.get("source", {}).get("name"),
                            "url": article.get("url")
                        })
            except:
                pass

            # Google Search Fallback
            if not sources:
                gs_results = google_search(user_input, GOOGLE_API_KEY, GOOGLE_CSE_ID, num=5)
                for item in gs_results:
                    sources.append({
                        "title": item.get("title"),
                        "source": item.get("displayLink"),
                        "url": item.get("link")
                    })

            # Final result
            result = "Real" if pred == 1 else "Fake"
            confidence = round((ml_conf * 0.5) + 50, 2)

        else:
            result = "Please enter a news statement."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        sources=sources
    )


if __name__ == "__main__":
    app.run()
