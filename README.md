# 🏡 AI – House Price Prediction Automation (FastAPI + n8n + Render)

This project integrates a **Machine Learning model** deployed via **FastAPI** with an **n8n automation workflow** that handles API orchestration and response management.  
The goal is to automate the prediction of house prices using a fully cloud-deployed setup — no manual intervention, no local dependencies.

---

## 🚀 Project Overview

The **AI – House Price Prediction** system demonstrates how to:
- Build an ML regression model with Scikit-learn  
- Deploy it using **FastAPI** on **Render Cloud**  
- Automate workflow execution and prediction handling through **n8n Cloud**

This project combines **Machine Learning + MLOps + Automation**, and it is production-ready.

---

## 🧩 Architecture


flowchart TD
    A[User Input / API Call] --> B[FastAPI App on Render]
    B --> C[n8n Webhook (Cloud)]
    C --> D[ML Model Prediction Script]
    D --> E[n8n Webhook Response Node]
    E --> F[JSON Output → Render → User]


### Workflow
1. User submits housing data to `/predict` endpoint on FastAPI.  
2. FastAPI sends the data to an **n8n Webhook**.  
3. n8n runs the prediction logic (via Set + HTTP Request nodes).  
4. The ML model predicts the price.  
5. n8n sends the response back as JSON, and FastAPI displays it to the user.

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest (Scikit-learn)
- **Dataset:** California Housing Dataset  
- **Features:**
  - `MedInc` – Median Income  
  - `HouseAge` – Median house age  
  - `AveRooms` – Average rooms per household  
  - `AveBedrms` – Average bedrooms per household  
  - `Population` – Population of the block  
  - `AveOccup` – Average occupancy  

The model was trained using Scikit-learn, serialized via **joblib**, and stored as `model.joblib`.

---

## 🧰 Tech Stack

| Component | Tool / Framework |
|------------|------------------|
| API | FastAPI |
| ML | Scikit-learn, Pandas, NumPy, mlflow |
| Automation | n8n Cloud |
| Deployment | Render Cloud |
| Language | Python 3.10+ |
| Model Storage | joblib |

---

## 🔗 Live Demo

### 🟢 FastAPI App on Render
👉 **[https://house-price-api.onrender.com/docs](https://house-price-api.onrender.com/docs)**

**Sample Input:**
```json
{
  "MedInc": 5,
  "HouseAge": 20,
  "AveRooms": 5,
  "AveBedrms": 1,
  "Population": 1000,
  "AveOccup": 3
}
```

**Expected Output:**
```json
{
  "prediction": 4.23
}
```

---

## ⚙️ n8n Workflow Setup

**n8n Nodes Used:**
1. 🟢 **Webhook Node** – Receives FastAPI request  
2. 🔁 **HTTP Request Node** – Runs prediction logic or external call  
3. ⚙️ **Set Node** – Maps and structures incoming JSON  
4. 🟢 **Webhook Response Node** – Returns JSON with predicted price  

### Production Webhook:
```
https://tabjulnagayasaswini.app.n8n.cloud/webhook/house-price
```

Ensure your n8n workflow is **Active (green circle ON)** before calling from Render.

---

## 🗂️ Project Structure

```
house-price-app/
│
├── main.py                 # FastAPI application
├── hpMlmodel.py            # ML model training/prediction script
├── model.joblib            # Trained model
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Render Deployment Details

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Render automatically injects the `$PORT` variable, so no manual port setting is needed.

---

## 🧪 Run Locally

To test locally before deployment:

```bash
git clone https://github.com/<your-username>/AI-House-Price-Prediction.git
cd AI-House-Price-Prediction
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧭 Example Prediction Flow

| Step | Action | Tool |
|------|---------|------|
| 1️⃣ | User submits input | FastAPI |
| 2️⃣ | FastAPI sends JSON payload | HTTP POST |
| 3️⃣ | n8n receives and processes | Workflow |
| 4️⃣ | Model predicts | Scikit-learn |
| 5️⃣ | Response sent to user | JSON |

---

## 📈 Future Enhancements

- 🎨 Add a **Streamlit or Gradio** front-end for a user-friendly UI  
- 💾 Store predictions in a **PostgreSQL** database  
- 📊 Integrate **MLflow** for experiment tracking  
- 🔄 Automate model retraining using **n8n cron triggers**  
- ☁️ Deploy model artifacts to **AWS S3 or Hugging Face Spaces**

---

## 👩‍💻 Author

**Naga Yasaswini Tabjul**  
Associate Analyst @ Deloitte | Aspiring Machine Learning Engineer  

🌐 **Portfolio:** [tabjulnagayasaswini.app](https://tabjulnagayasaswini.app)  
💼 **LinkedIn:** [linkedin.com/in/nagayasaswini](https://www.linkedin.com/in/nagayasaswini)  
📘 **GitHub:** [github.com/<your-username>](https://github.com/<your-username>)  

---

## 🏁 Summary

✅ Built and deployed a real-world ML model using **FastAPI**  
✅ Automated prediction flow through **n8n Cloud**  
✅ Hosted API publicly via **Render Cloud**  
✅ Demonstrates practical MLOps deployment and cloud integration

> 💡 *"From model training to automated cloud prediction — this project unites Machine Learning, FastAPI, and n8n automation into a seamless AI service."*
