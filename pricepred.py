from fastapi import FastAPI, Request, Form, Depends, Header, HTTPException, status
import os
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1️⃣ Initialize app & templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/live")
def home():
    return {"message": "Render FastAPI is live!"}


# 2️⃣ Load trained model
model = joblib.load("models/best_model_RandomForest.joblib")

# Optional API Key security for n8n calls
apikey = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != apikey:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return x_api_key


# 3️⃣ Homepage route (form UI for humans)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})




# 4️⃣ Prediction route (form submission → for humans in browser)
@app.post("/predict_ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):
    n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
    n8n_user = os.getenv("N8N_USERNAME")
    n8n_pass = os.getenv("N8N_PASSWORD")

    payload = {
        "features": [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    }

    # --- Safe retry wrapper for n8n call ---
    def safe_n8n_call(data, retries=3, delay=6):
        """Retry call to n8n if it's waking up or returns empty response."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    n8n_webhook_url,
                    json=data,
                    auth=HTTPBasicAuth(n8n_user, n8n_pass),
                    timeout=20
                )
                response.raise_for_status()

                try:
                    result = response.json()
                    logging.info("✅ n8n responded successfully.")
                    return result
                except ValueError:
                    logging.info(f"⚠️ Attempt {attempt+1}: empty/non-JSON reply (n8n waking). Retrying in {delay}s...")
                    time.sleep(delay)

            except requests.exceptions.RequestException as e:
                logging.warning(f"⚠️ Attempt {attempt+1}: n8n unreachable ({e}). Retrying in {delay}s...")
                time.sleep(delay)

        return {"error": "n8n still starting up. Try again later."}

    # --- Safe call to n8n ---
    try:
        result = safe_n8n_call(payload)

        if "error" in result:
            raise ValueError(result["error"])

        # Safely extract prediction
        predicted_value = result.get("Raw Value") or result.get("prediction")

        if predicted_value is None:
            raise ValueError("No prediction returned from n8n. Please ensure your n8n workflow is running.")

        return templates.TemplateResponse(
            "home.html",
            {"request": request, "prediction": f"Predicted Price: ${predicted_value:,.2f}"}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "home.html",
            {"request": request, "error": f"Something went wrong: {str(e)}"}
        )


# 5️⃣ JSON-based prediction API (for n8n and programmatic access)
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.post("/predict_api", dependencies=[Depends(verify_api_key)])
def predict_api(features: HouseFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                      features.AveBedrms, features.Population, features.AveOccup,
                      features.Latitude, features.Longitude]])

    prediction = model.predict(data)[0]
    prediction = round(float(prediction), 2)

    return JSONResponse(content={
        "Estimated Price": f"${prediction:,.2f}",
        "Raw Value": prediction
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/health")
def health():
    return {"status": "ok"}
