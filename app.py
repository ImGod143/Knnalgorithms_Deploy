import pickle
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained KNN model
with open("knnmodel.pkl", "rb") as file:
    model = pickle.load(file)

# Exact feature order extracted from knnmodel.pkl
FEATURE_NAMES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioPredict | Heart Disease Risk Assessment</title>
    <style>
        :root {
            --bg-page: #F1F5F9;
            --bg-card: #FFFFFF;
            --primary-navy: #0F172A;
            --accent-teal: #0D9488;
            --accent-teal-hover: #0F766E;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --border-color: #CBD5E1;
            --danger-bg: #FEF2F2;
            --danger-border: #EF4444;
            --danger-text: #991B1B;
            --success-bg: #ECFDF5;
            --success-border: #10B981;
            --success-text: #065F46;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-page);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 820px;
            background: var(--bg-card);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08);
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }

        .header {
            background: linear-gradient(135deg, var(--primary-navy) 0%, #1E293B 100%);
            color: #FFFFFF;
            padding: 2rem;
            border-bottom: 4px solid var(--accent-teal);
        }

        .header h1 {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }

        .header p {
            color: #94A3B8;
            font-size: 0.95rem;
        }

        .content {
            padding: 2rem;
        }

        .result-box {
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.75rem;
            border-left: 5px solid;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .result-high {
            background-color: var(--danger-bg);
            border-color: var(--danger-border);
            color: var(--danger-text);
        }

        .result-low {
            background-color: var(--success-bg);
            border-color: var(--success-border);
            color: var(--success-text);
        }

        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .result-badge {
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.7);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary-navy);
            margin-bottom: 0.4rem;
        }

        .hint {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 4px;
        }

        input, select {
            padding: 0.65rem 0.85rem;
            border: 1.5px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            color: var(--text-main);
            background-color: #F8FAFC;
            transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-teal);
            background-color: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15);
        }

        .actions {
            margin-top: 2rem;
            display: flex;
            gap: 1rem;
        }

        button {
            flex: 1;
            background-color: var(--accent-teal);
            color: #FFFFFF;
            border: none;
            padding: 0.9rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--accent-teal-hover);
        }

        button:active {
            transform: scale(0.99);
        }

        .footer-note {
            margin-top: 1.5rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>CardioPredict KNN Diagnostic</h1>
        <p>Enter patient clinical metrics to evaluate heart disease risk using K-Nearest Neighbors.</p>
    </div>

    <div class="content">
        {% if error %}
            <div class="result-box result-high">
                <div class="result-title">{{ error }}</div>
            </div>
        {% endif %}

        {% if prediction is not none %}
            {% if prediction == 1 %}
                <div class="result-box result-high">
                    <div>
                        <div class="result-title">High Risk: Heart Disease Detected (Class 1)</div>
                        <div style="font-size: 0.88rem; margin-top: 4px;">Clinical follow-up and diagnostic testing recommended.</div>
                    </div>
                    <div class="result-badge">Model Confidence: {{ probability }}%</div>
                </div>
            {% else %}
                <div class="result-box result-low">
                    <div>
                        <div class="result-title">Low Risk: Normal / No Heart Disease (Class 0)</div>
                        <div style="font-size: 0.88rem; margin-top: 4px;">Metrics align with normal cardiovascular profile.</div>
                    </div>
                    <div class="result-badge">Model Confidence: {{ probability }}%</div>
                </div>
            {% endif %}
        {% endif %}

        <form method="POST" action="/">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Age">Age <span class="hint">(years)</span></label>
                    <input type="number" id="Age" name="Age" min="1" max="120" required
                           value="{{ form_data.get('Age', '54') }}">
                </div>

                <div class="form-group">
                    <label for="Sex">Sex</label>
                    <select id="Sex" name="Sex" required>
                        <option value="1" {% if form_data.get('Sex', '1') == '1' %}selected{% endif %}>Male (1)</option>
                        <option value="0" {% if form_data.get('Sex') == '0' %}selected{% endif %}>Female (0)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="ChestPainType">Chest Pain Type</label>
                    <select id="ChestPainType" name="ChestPainType" required>
                        <option value="0" {% if form_data.get('ChestPainType', '0') == '0' %}selected{% endif %}>ASY - Asymptomatic (0)</option>
                        <option value="1" {% if form_data.get('ChestPainType') == '1' %}selected{% endif %}>ATA - Atypical Angina (1)</option>
                        <option value="2" {% if form_data.get('ChestPainType') == '2' %}selected{% endif %}>NAP - Non-Anginal Pain (2)</option>
                        <option value="3" {% if form_data.get('ChestPainType') == '3' %}selected{% endif %}>TA - Typical Angina (3)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="RestingBP">Resting BP <span class="hint">(mm Hg)</span></label>
                    <input type="number" id="RestingBP" name="RestingBP" min="0" max="300" required
                           value="{{ form_data.get('RestingBP', '130') }}">
                </div>

                <div class="form-group">
                    <label for="Cholesterol">Cholesterol <span class="hint">(mg/dl)</span></label>
                    <input type="number" id="Cholesterol" name="Cholesterol" min="0" max="700" required
                           value="{{ form_data.get('Cholesterol', '240') }}">
                </div>

                <div class="form-group">
                    <label for="FastingBS">Fasting Blood Sugar <span class="hint">(&gt; 120 mg/dl)</span></label>
                    <select id="FastingBS" name="FastingBS" required>
                        <option value="0" {% if form_data.get('FastingBS', '0') == '0' %}selected{% endif %}>No / False (0)</option>
                        <option value="1" {% if form_data.get('FastingBS') == '1' %}selected{% endif %}>Yes / True (1)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="RestingECG">Resting ECG</label>
                    <select id="RestingECG" name="RestingECG" required>
                        <option value="1" {% if form_data.get('RestingECG', '1') == '1' %}selected{% endif %}>Normal (1)</option>
                        <option value="2" {% if form_data.get('RestingECG') == '2' %}selected{% endif %}>ST - ST-T Wave Abnormality (2)</option>
                        <option value="0" {% if form_data.get('RestingECG') == '0' %}selected{% endif %}>LVH - Left Ventricular Hypertrophy (0)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="MaxHR">Max Heart Rate <span class="hint">(60–220)</span></label>
                    <input type="number" id="MaxHR" name="MaxHR" min="40" max="250" required
                           value="{{ form_data.get('MaxHR', '150') }}">
                </div>

                <div class="form-group">
                    <label for="ExerciseAngina">Exercise-Induced Angina</label>
                    <select id="ExerciseAngina" name="ExerciseAngina" required>
                        <option value="0" {% if form_data.get('ExerciseAngina', '0') == '0' %}selected{% endif %}>No (0)</option>
                        <option value="1" {% if form_data.get('ExerciseAngina') == '1' %}selected{% endif %}>Yes (1)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Oldpeak">Oldpeak <span class="hint">(ST depression)</span></label>
                    <input type="number" step="0.1" id="Oldpeak" name="Oldpeak" min="-5" max="10" required
                           value="{{ form_data.get('Oldpeak', '1.0') }}">
                </div>

                <div class="form-group">
                    <label for="ST_Slope">ST Slope</label>
                    <select id="ST_Slope" name="ST_Slope" required>
                        <option value="2" {% if form_data.get('ST_Slope', '2') == '2' %}selected{% endif %}>Up - Upsloping (2)</option>
                        <option value="1" {% if form_data.get('ST_Slope') == '1' %}selected{% endif %}>Flat (1)</option>
                        <option value="0" {% if form_data.get('ST_Slope') == '0' %}selected{% endif %}>Down - Downsloping (0)</option>
                    </select>
                </div>
            </div>

            <div class="actions">
                <button type="submit">Run Risk Prediction</button>
            </div>
        </form>

        <div class="footer-note">
            Note: Verify that the categorical dropdown integer mappings (0, 1, 2, 3) match your notebook's <code>LabelEncoder</code> order.
        </div>
    </div>
</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            input_values = [
                float(request.form["Age"]),
                float(request.form["Sex"]),
                float(request.form["ChestPainType"]),
                float(request.form["RestingBP"]),
                float(request.form["Cholesterol"]),
                float(request.form["FastingBS"]),
                float(request.form["RestingECG"]),
                float(request.form["MaxHR"]),
                float(request.form["ExerciseAngina"]),
                float(request.form["Oldpeak"]),
                float(request.form["ST_Slope"]),
            ]

            input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

            pred_class = int(model.predict(input_df)[0])
            pred_proba = model.predict_proba(input_df)[0]

            prediction = pred_class
            probability = round(float(pred_proba[pred_class]) * 100, 1)

        except Exception as e:
            error = f"Invalid input or prediction error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=probability,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
