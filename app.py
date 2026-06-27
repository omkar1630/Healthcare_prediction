import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load your uploaded XGBoost model
MODEL_PATH = "xgboost.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Mappings to convert categorical interface values to numerical inputs for your XGBoost model
GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
BLOOD_TYPE_MAP = {"A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7}
MEDICAL_CONDITION_MAP = {"Checkup": 0, "Diabetes": 1, "Hypertension": 2, "Asthma": 3, "Other": 4}
INSURANCE_MAP = {"None": 0, "Medicare": 1, "Medicaid": 2, "Private": 3}
ADMISSION_MAP = {"Elective": 0, "Emergency": 1, "Urgent": 2}
MEDICATION_MAP = {"None": 0, "Ibuprofen": 1, "Paracetamol": 2, "Penicillin": 3, "Other": 4}
HOSPITAL_MAP = {"General Hospital": 0, "City Clinic": 1, "St. Jude": 2, "Mayo Clinic": 3, "Other": 4}

# HTML Template layout written directly inside the Python string
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XGBoost Medical Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 font-sans min-h-screen flex items-center justify-center py-12 px-4">

    <div class="max-w-4xl w-full bg-white rounded-2xl shadow-xl overflow-hidden flex flex-col md:flex-row">
        
        <div class="md:w-1/3 bg-blue-600 text-white p-8 flex flex-col justify-between">
            <div>
                <h2 class="text-3xl font-extrabold tracking-tight mb-4">Analytics Engine</h2>
                <p class="text-blue-100 text-sm leading-relaxed">
                    Provide the medical patient parameters to evaluate class predictions instantaneously powered by your trained XGBoost Classifier architecture.
                </p>
            </div>
            <div class="text-xs text-blue-200 mt-6 md:mt-0">
                &copy; 2026 Production Deployment Portal
            </div>
        </div>

        <div class="md:w-2/3 p-8">
            <h1 class="text-2xl font-bold text-slate-800 mb-6">Patient Diagnostics Input</h1>
            
            <form action="/" method="POST" class="space-y-5">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Age</label>
                        <input type="number" name="Age" min="0" max="120" value="45" required
                               class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Billing Amount ($)</label>
                        <input type="number" step="0.01" name="Billing_Amount" value="1500.00" required
                               class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Gender</label>
                        <select name="Gender" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>Male</option>
                            <option>Female</option>
                            <option>Other</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Blood Type</label>
                        <select name="Blood_Type" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>O+</option><option>O-</option><option>A+</option><option>A-</option>
                            <option>B+</option><option>B-</option><option>AB+</option><option>AB-</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Medical Condition</label>
                        <select name="Medical_Condition" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>Checkup</option>
                            <option>Diabetes</option>
                            <option>Hypertension</option>
                            <option>Asthma</option>
                            <option>Other</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Admission Type</label>
                        <select name="Admission_Type" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>Elective</option>
                            <option>Emergency</option>
                            <option>Urgent</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Insurance Provider</label>
                        <select name="Insurance_Provider" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>None</option>
                            <option>Medicare</option>
                            <option>Medicaid</option>
                            <option>Private</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Medication</label>
                        <select name="Medication" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                            <option>None</option>
                            <option>Ibuprofen</option>
                            <option>Paracetamol</option>
                            <option>Penicillin</option>
                            <option>Other</option>
                        </select>
                    </div>

                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Hospital Site</label>
                    <select name="Hospital" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-700">
                        <option>General Hospital</option>
                        <option>City Clinic</option>
                        <option>St. Jude</option>
                        <option>Mayo Clinic</option>
                        <option>Other</option>
                    </select>
                </div>

                <button type="submit" class="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition duration-200 mt-2 cursor-pointer shadow-md">
                    Generate Prediction Class
                </button>
            </form>

            {% if prediction_text %}
            <div class="mt-6 p-4 bg-slate-100 border-l-4 border-emerald-500 rounded-r-lg">
                <span class="block text-xs font-bold text-slate-500 uppercase tracking-wider">Engine Result</span>
                <p class="text-slate-800 font-medium text-lg mt-1">{{ prediction_text }}</p>
            </div>
            {% endif %}

        </div>
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    if request.method == "POST":
        try:
            # 1. Fetch raw variables from form submission
            age = int(request.form.get("Age", 30))
            gender_str = request.form.get("Gender")
            blood_str = request.form.get("Blood_Type")
            condition_str = request.form.get("Medical_Condition")
            hospital_str = request.form.get("Hospital")
            insurance_str = request.form.get("Insurance_Provider")
            billing = float(request.form.get("Billing_Amount", 0.0))
            admission_str = request.form.get("Admission_Type")
            medication_str = request.form.get("Medication")

            # 2. Safety conversion from beautiful labels back to model numerical features
            gender = GENDER_MAP.get(gender_str, 0)
            blood_type = BLOOD_TYPE_MAP.get(blood_str, 0)
            medical_condition = MEDICAL_CONDITION_MAP.get(condition_str, 0)
            hospital = HOSPITAL_MAP.get(hospital_str, 0)
            insurance = INSURANCE_MAP.get(insurance_str, 0)
            admission_type = ADMISSION_MAP.get(admission_str, 0)
            medication = MEDICATION_MAP.get(medication_str, 0)

            # 3. Create structural configuration array matching your model's 9 features
            features = np.array([[
                age, gender, blood_type, medical_condition, 
                hospital, insurance, billing, admission_type, medication
            ]])

            # 4. Generate the prediction class output
            pred = model.predict(features)[0]
            prediction_text = f"Predicted Health Outcome/Class: {pred}"

        except Exception as e:
            prediction_text = f"Error processing prediction: {str(e)}"

    return render_template_string(HTML_LAYOUT, prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
