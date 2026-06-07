

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# Load Models
# =========================
diabetes_model = joblib.load("diabetes.pkl")
heart_model = joblib.load("heart.pkl")
kidney_model = joblib.load("kidney.pkl")
liver_model = joblib.load("liver.pkl")


# =========================
# Home Page
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        diabetes_result=None,
        heart_result=None,
        kidney_result=None,
        liver_result=None,
        active_tab="diabetes"
    )


# =========================
# Diabetes Prediction
# =========================
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        input_data = pd.DataFrame([{
            "Pregnancies": float(request.form["Pregnancies"]),
            "Glucose": float(request.form["Glucose"]),
            "BloodPressure": float(request.form["BloodPressure"]),
            "SkinThickness": float(request.form["SkinThickness"]),
            "Insulin": float(request.form["Insulin"]),
            "BMI": float(request.form["BMI"]),
            "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
            "Age": float(request.form["Age"])
        }])

        prediction = diabetes_model.predict(input_data)

        result = (
            "Diabetic"
            if prediction[0] == 1
            else "Non-Diabetic"
        )

        return render_template(
    "index.html",
    diabetes_result=result,
    active_tab="diabetes"
)

    except Exception as e:
       return render_template(
        "index.html",
        diabetes_result=f"Error: {str(e)}",
        active_tab="diabetes"
    )

# =========================
# Heart Disease Prediction
# =========================
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        input_data = pd.DataFrame([{
            "Age": float(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Blood Pressure": float(request.form["BloodPressure"]),
            "Cholesterol Level": float(request.form["CholesterolLevel"]),
            "Exercise Habits": request.form["ExerciseHabits"],
            "Smoking": request.form["Smoking"],
            "Family Heart Disease": request.form["FamilyHeartDisease"],
            "Diabetes": request.form["Diabetes"],
            "BMI": float(request.form["BMI"]),
            "High Blood Pressure": request.form["HighBloodPressure"],
            "Low HDL Cholesterol": request.form["LowHDLCholesterol"],
            "High LDL Cholesterol": request.form["HighLDLCholesterol"],
            "Alcohol Consumption": request.form["AlcoholConsumption"],
            "Stress Level": float(request.form["StressLevel"]),
            "Sleep Hours": float(request.form["SleepHours"]),
            "Sugar Consumption": float(request.form["SugarConsumption"]),
            "Triglyceride Level": float(request.form["TriglycerideLevel"]),
            "Fasting Blood Sugar": float(request.form["FastingBloodSugar"]),
            "CRP Level": float(request.form["CRPLevel"]),
            "Homocysteine Level": float(request.form["HomocysteineLevel"])
        }])

        prediction = heart_model.predict(input_data)

        result = (
            "Heart Disease Detected"
            if prediction[0] == 1
            else "No Heart Disease"
        )

        return render_template(
    "index.html",
    heart_result=result,
    active_tab="heart"
)

    except Exception as e:
       return render_template(
        "index.html",
        heart_result=f"Error: {str(e)}",
        active_tab="heart"
    )


# =========================
# Kidney Disease Prediction
# =========================
@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    try:
        input_data = pd.DataFrame([{
            "age": float(request.form["age"]),
            "bp": float(request.form["bp"]),
            "sg": float(request.form["sg"]),
            "al": float(request.form["al"]),
            "su": float(request.form["su"]),
            "rbc": request.form["rbc"],
            "pc": request.form["pc"],
            "pcc": request.form["pcc"],
            "ba": request.form["ba"],
            "bgr": float(request.form["bgr"]),
            "bu": float(request.form["bu"]),
            "sc": float(request.form["sc"]),
            "sod": float(request.form["sod"]),
            "pot": float(request.form["pot"]),
            "hemo": float(request.form["hemo"]),
            "pcv": float(request.form["pcv"]),
            "wc": float(request.form["wc"]),
            "rc": float(request.form["rc"]),
            "htn": request.form["htn"],
            "dm": request.form["dm"],
            "cad": request.form["cad"],
            "appet": request.form["appet"],
            "pe": request.form["pe"],
            "ane": request.form["ane"]
        }])

        prediction = kidney_model.predict(input_data)

        result = (
            "Chronic Kidney Disease Detected"
            if prediction[0] == 1
            else "No Kidney Disease"
        )

        return render_template(
    "index.html",
    kidney_result=result,
    active_tab="kidney"
)

    except Exception as e:
       return render_template(
        "index.html",
        kidney_result=f"Error: {str(e)}",
        active_tab="kidney"
    )


# =========================
# Liver Disease Prediction
# =========================
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        input_data = pd.DataFrame([{
            "Age": float(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Total_Bilirubin": float(request.form["Total_Bilirubin"]),
            "Direct_Bilirubin": float(request.form["Direct_Bilirubin"]),
            "Alkaline_Phosphotase": float(request.form["Alkaline_Phosphotase"]),
            "Alamine_Aminotransferase": float(request.form["ALT"]),
            "Aspartate_Aminotransferase": float(request.form["AST"]),
            "Total_Protiens": float(request.form["Total_Proteins"]),
            "Albumin": float(request.form["Albumin"]),
            "Albumin_and_Globulin_Ratio": float(request.form["AG_Ratio"])
        }])

        prediction = liver_model.predict(input_data)

        result = (
            "Liver Disease Detected"
            if prediction[0] == 1
            else "No Liver Disease"
        )

        return render_template(
    "index.html",
    liver_result=result,
    active_tab="liver"
)
        

    except Exception as e:
       return render_template(
        "index.html",
        liver_result=f"Error: {str(e)}",
        active_tab="liver"
    )


# =========================
# Run Flask App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
