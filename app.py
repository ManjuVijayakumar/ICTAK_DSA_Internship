from flask import Flask, render_template, request
import pickle
import pandas as pd

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

##@app.route("/")
##def home():
##    return render_template("index.html")

# -----------------------------
# Load Random Forest Model
# -----------------------------
with open("rf_course_success_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# -----------------------------
# Top 10 Courses (Cleaned)
# -----------------------------
TOP_10_COURSES = [
    "AI Engineer with Microsoft Azure",
    "Antimicrobial Stewardship for Africa",
    "Aquatic Mammals",
    "Become a Successful Baker – BBC Good Food",
    "Beginning Your Digital Marketing Journey",
    "Brand Personality – Jonathan Wilson",
    "The Highland Clans",
    "The Networks Connecting People",
    "The NHS Explained",
    "The Role of Semiconductor Innovation in Shaping Geopolitical Dynamics"
]

# -----------------------------
# Feature Engineering Helpers
# (Must match training logic exactly)
# -----------------------------

def get_duration_bucket(hours):
    if hours <= 10:
        return "Very Short"
    elif hours <= 30:
        return "Short"
    elif hours <= 60:
        return "Medium"
    elif hours <= 100:
        return "Long"
    else:
        return "Very Long"


def get_title_word_count(title):
    return len(title.split())


def get_title_length_bucket(word_count):
    if word_count <= 3:
        return "Very Short"
    elif word_count <= 6:
        return "Short"
    elif word_count <= 10:
        return "Medium"
    else:
        return "Long"

# --------------------------------
# Probability based GenAI Advisor
# (Course Design Advisor)
# --------------------------------

def course_design_advisor(predicted_class, confidence, duration, difficulty, title_words):
    suggestions = []

    # Tier + confidence based advice
    if predicted_class == "High":
        if confidence > 0.8:
            suggestions.append("The course has very strong success potential.")
        else:
            suggestions.append("The course is predicted to be successful, but small refinements could improve ratings.")

    elif predicted_class == "Medium":
        if confidence > 0.7:
            suggestions.append("The course is close to high success. Minor improvements could significantly boost outcomes.")
        else:
            suggestions.append("The course has moderate success potential. Several improvements are recommended.")

    else:  # Low
        suggestions.append("The course is predicted to have low success. Major changes are recommended.")

    # Duration-based advice
    if duration < 20:
        suggestions.append("Increasing course duration may improve learner satisfaction and perceived value.")
    elif duration > 80:
        suggestions.append("The course is quite long; consider splitting it into smaller modules.")

    # Difficulty-based advice
    if difficulty == "Beginner":
        suggestions.append("Ensure step-by-step explanations and beginner-friendly examples.")
    elif difficulty == "Advanced":
        suggestions.append("Include advanced projects and real-world case studies to justify the difficulty level.")

    # Title-based advice
    if title_words < 4:
        suggestions.append("Expanding the course title could better communicate learning outcomes.")
    elif title_words > 10:
        suggestions.append("Shortening the course title may improve clarity and engagement.")

    suggestions.append(f"Model confidence in this prediction is {confidence:.0%}.")

    return suggestions

# -----------------------------
# Flask Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def predict_course_success():

    if request.method == "POST":

        # User-friendly inputs
        # course_title = request.form["course_title"]
        selected_title = request.form.get("existing_course")
        custom_title = request.form.get("course_title")

        # Priority: dropdown > manual input
        course_title = selected_title if selected_title else custom_title
        
        difficulty_level = request.form["difficulty_level"]
        duration_hours = float(request.form["duration_hours"])

        # Internal feature engineering
        title_word_count = get_title_word_count(course_title)
        title_length_bucket = get_title_length_bucket(title_word_count)
        duration_bucket = get_duration_bucket(duration_hours)

        # Prepare model input
        input_df = pd.DataFrame([{
            "difficulty_level": difficulty_level,
            "duration_hours": duration_hours,
            "duration_bucket": duration_bucket,
            "title_word_count": title_word_count,
            "title_length_bucket": title_length_bucket
        }])

        # Prediction
##        prediction = rf_model.predict(input_df)[0]
        prediction = rf_model.predict(input_df)

        # Map predicted class
        success_map = {0: "Low", 1: "Medium", 2: "High"}
##        predicted_class = success_map[prediction]
        predicted_class = prediction[0]
        probabilities = rf_model.predict_proba(input_df)[0]

        # Extract confidence
        class_index = list(rf_model.classes_).index(prediction)
        confidence = probabilities[class_index]

        # Generate GenAI-style suggestions
        ai_suggestions = course_design_advisor(
            predicted_class,
            confidence,
            duration_hours,
            difficulty_level,
            title_word_count
        )

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=f"{confidence:.0%}",
            suggestions=ai_suggestions,
            course_title=course_title
        )

    return render_template("index.html", courses=TOP_10_COURSES)

# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=False)


