import joblib
import re

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
class MessageRequest(BaseModel):
    message: str

app = FastAPI(title="UPIscam message detection using machine learning")

model =joblib.load("scam_model.pkl")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>UPI Scam Message Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            color: #1e293b;
        }

        header {
            background: #101d3d;
            color: white;
            padding: 25px 8%;
        }

        header h1 {
            margin: 0;
            font-size: 27px;
        }

        header p {
            color: #cbd5e1;
            margin-bottom: 0;
        }

        .container {
            max-width: 1000px;
            margin: 35px auto;
            padding: 0 20px;
        }

        .welcome {
            margin-bottom: 25px;
        }

        .welcome h2 {
            color: #172554;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 4px 18px #dbe2ea;
            margin-bottom: 22px;
        }

        textarea {
            width: 100%;
            height: 140px;
            padding: 15px;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            font-size: 16px;
            resize: vertical;
        }

        button {
            background: #3157d5;
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
        }

        button:hover {
            background: #1e40af;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 18px;
        }

        .stat {
            background: white;
            padding: 22px;
            border-radius: 12px;
            box-shadow: 0 3px 12px #dbe2ea;
        }

        .stat h3 {
            color: #64748b;
            font-size: 14px;
        }

        .stat p {
            font-size: 19px;
            font-weight: bold;
        }

        #result {
            display: none;
            margin-top: 22px;
            padding: 20px;
            border-radius: 10px;
            line-height: 1.8;
            overflow-wrap: anywhere;
        }

        footer {
            text-align: center;
            padding: 25px;
            color: #64748b;
        }
    </style>
</head>

<body>

<header>
    <h1>🛡️ UPI Scam Message Detection</h1>
    <p>Using Machine Learning</p>
</header>

<div class="container">

    <div class="welcome">
        <h2>Welcome to Scam Detection Dashboard</h2>
        <p>
            Analyze suspicious messages and identify
            possible UPI fraud.
        </p>
    </div>

    <div class="card">

        <h2>📩 Message Analyzer</h2>

        <p>Enter your UPI message below:</p>

        <textarea id="message"
        placeholder="Paste your UPI message here..."></textarea>

        <br>

        <button onclick="checkMessage()">
            Analyze Message
        </button>

        <div id="result"></div>

    </div>

    <div class="stats">

        <div class="stat">
            <h3>Machine Learning</h3>
            <p>Logistic Regression</p>
        </div>

        <div class="stat">
            <h3>Text Processing</h3>
            <p>TF-IDF</p>
        </div>

        <div class="stat">
            <h3>Model Accuracy</h3>
            <p>97.14%</p>
        </div>

    </div>

    <div class="card">

        <h2>About This Project</h2>

        <p>
            This project uses Machine Learning to classify
            messages as Potential Scam or Likely Legitimate.
            It also displays the risk level and scam probability.
        </p>

        <p>
            <b>Technologies:</b>
            Python, Pandas, Scikit-learn, FastAPI,
            HTML, CSS and JavaScript.
        </p>

    </div>

</div>

<footer>
    UPI Scam Message Detection Using Machine Learning
</footer>

<script>

async function checkMessage() {

    const message = document
        .getElementById("message")
        .value.trim();

    const result = document.getElementById("result");

    result.style.display = "block";

    if (!message) {
        result.textContent = "Please enter a message.";
        result.style.background = "#fee2e2";
        result.style.color = "#991b1b";
        return;
    }

    result.textContent = "Analyzing message...";

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            result.textContent =
                data.error || "Analysis failed.";
            return;
        }

        const isScam =
            data.prediction === "Potential Scam";

        result.style.background =
            isScam ? "#fee2e2" : "#dcfce7";

        result.style.color =
            isScam ? "#991b1b" : "#166534";

        result.innerHTML = `

            <h2>
                ${isScam
                    ? "⚠️ Scam Detected"
                    : "✅ Likely Genuine"}
            </h2>

            <p>
                <b>Prediction:</b>
                ${data.prediction}
            </p>

            <p>
                <b>Risk Level:</b>
                ${data.risk_level}
            </p>

            <p>
                <b>Scam Score:</b>
                ${data.spam_score}%
            </p>

            <p>
                <b>Detected URLs:</b>
                ${data.urls.length
                    ? data.urls.join(", ")
                    : "None"}
            </p>

            <p>
                <b>Detected Phone Numbers:</b>
                ${data.phone_numbers.length
                    ? data.phone_numbers.join(", ")
                    : "None"}
            </p>

        `;

    } catch (error) {

        result.textContent =
            "Something went wrong. Please try again.";

    }

}

</script>

</body>
</html>
"""

@app.post("/analyze")
def analyze_message(request: MessageRequest):

    message = request.message.strip()

    if not message:
        return {"error": "Please enter a message"}

    probabilities = model.predict_proba([message])[0]
    scam_index = list(model.classes_).index(1)

    spam_score = round(
        float(probabilities[scam_index]) * 100, 2
    )

    if spam_score >= 75:
        risk_level = "High"
    elif spam_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        message,
        flags=re.IGNORECASE
    )

    phones = re.findall(
        r"(?<!\w)(?:\+91[-\s]?)?[6-9]\d{9}(?!\w)",
        message
    )

    return {
        "message": message,
        "spam_score": spam_score,
        "fraud_score": spam_score,
        "risk_level": risk_level,
        "prediction": (
            "Potential Scam"
            if spam_score >= 50
            else "Likely Legitimate"
        ),
        "urls": urls,
        "phone_numbers": phones
    }