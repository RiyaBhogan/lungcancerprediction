from flask import Flask, request, render_template
import pandas as pd
from pickle import load

# Load model
with open("ld.pkl", "rb") as f:
    model = load(f)

# Load feature column order
with open("ld_cols.pkl", "rb") as f:
    cols = load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    lung_cancer = None
    if request.method == "POST":
        gender = request.form.get("gender")
        age = int(request.form.get("age"))

        smoking = int(request.form.get("smoking"))
        yellow_fingers = int(request.form.get("yellow_fingers"))
        anxiety = int(request.form.get("anxiety"))
        peer_pressure = int(request.form.get("peer_pressure"))
        chronic_disease = int(request.form.get("chronic_disease"))
        fatigue = int(request.form.get("fatigue"))
        allergy = int(request.form.get("allergy"))
        wheezing = int(request.form.get("wheezing"))
        alcohol = int(request.form.get("alcohol"))
        coughing = int(request.form.get("coughing"))
        shortness_of_breath = int(request.form.get("shortness_of_breath"))
        swallowing_difficulty = int(request.form.get("swallowing_difficulty"))
        chest_pain = int(request.form.get("chest_pain"))

        # ------- Create dataframe ------
        cancer = {
            "GENDER": gender,  # stays as M/F string
            "AGE": age,
            "SMOKING": smoking,
            "YELLOW_FINGERS": yellow_fingers,
            "ANXIETY": anxiety,
            "PEER_PRESSURE": peer_pressure,
            "CHRONIC DISEASE": chronic_disease,
            "FATIGUE": fatigue,
            "ALLERGY": allergy,
            "WHEEZING": wheezing,
            "ALCOHOL": alcohol,
            "COUGHING": coughing,
            "SHORTNESS OF BREATH": shortness_of_breath,
            "SWALLOWING DIFFICULTY": swallowing_difficulty,
            "CHEST PAIN": chest_pain
        }

        cancer_df = pd.DataFrame([cancer])

        # Encode categorical variables same as training
        cancer_df = pd.get_dummies(cancer_df)
        cancer_df = cancer_df.reindex(columns=cols, fill_value=0)

        # ----- Predict ------
        lung_cancer = model.predict(cancer_df)[0]  # "YES" or "NO"

    return render_template("home.html", lung_cancer=lung_cancer)

if __name__ == "__main__":
    app.run(debug=True)
