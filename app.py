
from flask import Flask, render_template, request
import joblib
import os
import csv
from datetime import datetime

app = Flask(__name__)

# File paths
MODEL_PATH = "models/fake_news_model.pkl"
HISTORY_FILE = "history.csv"


# Load trained ML model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found! Please run train_ml_model.py first."
    )

model = joblib.load(MODEL_PATH)


# ----------------------------------
# 1. SAVE PREDICTION HISTORY
# ----------------------------------

def save_history(prediction, confidence):

    file_exists = os.path.exists(HISTORY_FILE)

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date and Time",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            prediction,
            str(confidence) + "%"
        ])


# ----------------------------------
# 2. GET RECENT PREDICTION HISTORY
# ----------------------------------

def get_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    with open(
        HISTORY_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)
        history = list(reader)

    # Return latest five predictions
    return history[-5:][::-1]


# ----------------------------------
# 3. CALCULATE DASHBOARD STATISTICS
# ----------------------------------

def get_statistics():

    all_history = []

    if os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)
            all_history = list(reader)

    total_news = len(all_history)

    fake_count = sum(
        1 for item in all_history
        if item["Prediction"] == "Fake News"
    )

    real_count = sum(
        1 for item in all_history
        if item["Prediction"] == "Real News"
    )

    fake_percentage = (
        round((fake_count / total_news) * 100, 2)
        if total_news > 0
        else 0
    )

    return (
        total_news,
        fake_count,
        real_count,
        fake_percentage
    )


# ----------------------------------
# 4. HOME PAGE AND NEWS PREDICTION
# ----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    news_text = ""
    error = None

    if request.method == "POST":

        news_text = request.form.get(
            "news", ""
        ).strip()

        if not news_text:

            error = "Please enter a news article."

        else:

            try:

                # Predict news category
                result = model.predict([news_text])[0]

                # Calculate model confidence
                probabilities = model.predict_proba(
                    [news_text]
                )[0]

                confidence = round(
                    float(max(probabilities)) * 100,
                    2
                )

                # Convert prediction label
                if result == 0:
                    prediction = "Fake News"
                else:
                    prediction = "Real News"

                # Save result in CSV
                save_history(
                    prediction,
                    confidence
                )

                print("Prediction:", prediction)
                print("Confidence:", confidence, "%")

            except Exception as e:

                error = "Unable to analyze the news. Please try again."

                print("Prediction Error:", e)

    # Get latest history
    history = get_history()

    # Get dashboard statistics
    (
        total_news,
        fake_count,
        real_count,
        fake_percentage
    ) = get_statistics()

    return render_template(
        "index.html",

        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        error=error,

        history=history,

        total_news=total_news,
        fake_count=fake_count,
        real_count=real_count,
        fake_percentage=fake_percentage
    )


# ----------------------------------
# 5. RUN FLASK APPLICATION
# ----------------------------------

if __name__ == "__main__":
    app.run(debug=True)
