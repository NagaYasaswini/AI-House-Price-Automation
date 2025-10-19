# 🏡 AI House Price Prediction — Automated MLOps Pipeline

### 🚀 End-to-End ML Web App powered by **FastAPI**, **n8n**, **MLflow**, and **Render Cloud**

---

## 📘 Overview
**AI House Price Prediction** is a fully automated **end-to-end MLOps project** that predicts house prices based on key property features.

This system brings together:
- 🧠 **Machine Learning models**, tracked and validated with **MLflow**
- ⚙️ **FastAPI backend** to handle user inputs and serve predictions
- 🔄 **n8n workflow automation** for model orchestration and pipeline automation
- 🗄️ **PostgreSQL** for persistent workflow storage
- ☁️ **Render** for seamless deployment of all services (FastAPI, n8n, and PostgreSQL)

This project combines **Machine Learning + MLOps + Automation**, and it is production-ready.
> The project demonstrates the full lifecycle of an ML solution — from model experimentation (MLflow) → deployment (FastAPI + n8n) → automation (n8n + Render).

---

## 🧰 Tech Stack

| Component | Tool / Framework |
|------------|------------------|
| ML | Scikit-learn, Pandas, NumPy, mlflow |
| Backend API | **FastAPI** |
| Workflow Automation | **n8n** |
| Database | **PostgreSQL (Render)** |
| Deployment | **Render Cloud Platform** |
| Containerization | **Docker** |
| Language | **Python 3.11** |
| Model Storage | **joblib** |

---

## 🌐 Live Demo

| Service | Description | URL |
|----------|--------------|------|
| 🧠 FastAPI App | Main user interface (predicts house price) | **[https://ai-house-price-automation-1.onrender.com](https://ai-house-price-automation-1.onrender.com)** |
| ⚙️ n8n Workflow | Backend workflow automation (deployed via Docker) | **[https://ai-house-price-n8n.onrender.com](https://ai-house-price-n8n.onrender.com)** |
| 🗄️ PostgreSQL DB | Persistent data storage | Internal (Render) |

> 🟢 Visit the FastAPI link to test predictions — no manual n8n activation required.  
> The backend workflow is triggered automatically.

---

## 🧩 Architecture Flow

```text
          ┌──────────────────────┐
          │   MLflow Tracking    │
          │  (model experiments) │
          └──────────┬───────────┘
                     │
                     ▼
         ┌────────────────────────────┐
         │   Trained Model (.joblib)  │
         │   validated via MLflow     │
         └──────────┬─────────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │        FastAPI Backend       │
     │  /predict_ui | /predict_api  │
     └──────────┬───────────────────┘
                │
                ▼
https://ai-house-price-n8n.onrender.com/webhook/house-price
                │
                ▼
     ┌──────────────────────────────┐
     │          n8n Workflow        │
     │ (executes ML model + returns │
     │       prediction output)     │
     └──────────────────────────────┘
                │
                ▼
       ✅ Predicted Price Returned
```


---

## ⚙️ Project Structure

```bash
📦 ai-house-price-prediction
│
├── main.py                        # FastAPI main app
├── model/
│   ├── house_price_model.joblib   # Trained ML model
│   ├── mlruns/                    # MLflow tracking data
│
├── templates/
│   ├── home.html                  # Input page
│   └── result.html                # Output page
│
├── n8n/
│   ├── Dockerfile                 # n8n deployment setup
│   ├── House-price.json           # Workflow definition
│
├── requirements.txt               # Python dependencies
├── .dockerignore
├── README.md
└── render.yaml (optional)
```

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

---

## 🧠 MLflow Integration Highlights

🧩 **Model Tracking**
- Logged model performance metrics such as R², MAE, and RMSE.  
- Compared multiple regression models to select the best-performing one.

⚙️ **Experiment Management**
- Version-controlled all experiments with MLflow’s local tracking server.  
- Saved model artifacts for reproducibility.

✅ **Model Registry**
- Final selected model exported as a `.joblib` file for deployment.  
- Ensures model lineage and transparency during deployment.

---

## 🐳 Deployment Setup on Render

### 1️⃣ FastAPI App
- Runtime: **Python 3**
- Start Command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 10000
  ```

### 2️⃣ n8n Workflow
- Runtime: **Docker**
- Dockerfile:
  ```dockerfile
  FROM n8nio/n8n:latest
  WORKDIR /data
  COPY ./House-price.json /data/workflows/House-price.json
  ENV N8N_IMPORT_EXPORT_DIR=/data/workflows
  ENV N8N_IMPORT_EXPORT_OVERWRITE=true
  ENV N8N_AUTO_ACTIVATE_WORKFLOW=true
  ENV N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=false
  ENTRYPOINT ["/bin/sh", "-c", "n8n import:workflow --input=/data/workflows/House-price.json && n8n start"]
  ```

### 3️⃣ PostgreSQL Database
- Added via **Render → New → PostgreSQL**
- Environment variable in n8n service:
  ```
  DATABASE_URL = postgresql://<user>:<password>@<host>:5432/database_houseprice
  ```

---

## 🧩 Key Features
✅ ML model versioning and validation via MLflow  
✅ End-to-end automation with n8n (no manual runs)  
✅ Persistent workflow storage via PostgreSQL  
✅ Fully containerized deployment using Docker  
✅ Scalable cloud setup (FastAPI + n8n + DB)  

---

## 🧠 Example Use Case
**Input example:**
```
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

## 🔒 Environment Variables Summary

| Variable | Purpose |
|-----------|----------|
| `DATABASE_URL` | Connects n8n to PostgreSQL |
| `N8N_HOST` | n8n domain |
| `N8N_PROTOCOL` | Should be `https` |
| `N8N_TRUSTED_PROXIES` | Enables proxy trust on Render |
| `WEBHOOK_URL` | Base webhook URL |
| `N8N_BASIC_AUTH_ACTIVE` | Activates n8n UI auth |
| `N8N_BASIC_AUTH_USER` | n8n username |
| `N8N_BASIC_AUTH_PASSWORD` | n8n password |
| `MLFLOW_TRACKING_URI` | Local MLflow tracking URI (optional) |

---

## 🎯 Future Improvements
- Deploy MLflow tracking server on Render or AWS  
- Integrate model re-training workflow inside n8n  
- Add real-time monitoring dashboards  
- Create a React-based interactive frontend  

---

## 🧑‍💻 Author
**Naga Yasaswini Tabjul**  
Associate Analyst @ Deloitte | Aspiring Machine Learning Engineer  

📫 **Connect with me:**  
- [LinkedIn](https://www.linkedin.com/in/naga-yasaswini-tabjul)  
- [GitHub](https://github.com/NagaYasaswini)

---

## 🌟 Summary

This project showcases:
- **Practical MLOps** using MLflow for model tracking  
- **Automation-first approach** with n8n  
- **Cloud-native deployment** using Render  
- **FastAPI-based prediction service** integrated with workflow orchestration  

> 🏁 Try it live:  
> 👉 **[AI House Price Prediction App](https://ai-house-price-automation-1.onrender.com/)**    
> 👉 **[n8n actiavtion](https://ai-house-price-n8n.onrender.com)**  
