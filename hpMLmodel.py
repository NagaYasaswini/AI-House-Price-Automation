import os
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score


# ------------- Load the Boston Housing dataset -------------------
housing = fetch_california_housing()

# Create a DataFrame
hpdata = pd.DataFrame(housing.data, columns=housing.feature_names)
hpdata['PRICE'] = housing.target

# Save to CSV
hpdata.to_csv("data/hpdata.csv", index=False)


# ----------- EDA : Exploratory Data Analysis ------------------
os.makedirs('eda_outputs', exist_ok=True)
print("Shape of Dataset\n", hpdata.shape)
print("\nMissing values sum\n",hpdata.isnull().sum())
print("\nInfo: ")
print(hpdata.info())

hpdata.describe().T.to_csv(os.path.join('eda_outputs', "summary_stats.csv"))


# ----------- Target distribution ----------------
sns.histplot(hpdata['PRICE'], bins=40, kde=True)
plt.show()
plt.savefig(os.path.join('eda_outputs', "target_distribution.png"))


# ---------- Feature distribution-------------

for col in hpdata.drop(['PRICE'], axis=1):
   sns.histplot(hpdata[col], kde=True, bins=20)
   plt.figure(figsize=(12,8))
   plt.title(f"Distribution of {col}")
   plt.xlabel= col
   plt.savefig(os.path.join('eda_outputs', f"dist_{col}.png"))
   plt.close()

# ------- Correlation details -----
corr = hpdata.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr)
plt.title("Correlation heatmap")
plt.savefig(os.path.join('eda_outputs', "Corr_Heatmap.png"))

# ---- List of variables highly correlated in order ----
corr_target = corr["PRICE"].sort_values(ascending=False)


# Feature vs Target scatter plots
# -------------------------
top_corr_features = corr["PRICE"].sort_values(ascending=False).index[1:4]  # top 3 features
for col in top_corr_features:
   plt.figure(figsize=(8,6))
   sns.scatterplot(x=hpdata[col], y=hpdata["PRICE"], alpha=0.5)
   plt.title(f"{col} vs PRICE")
   plt.xlabel=col
   plt.ylabel("Price")
   plt.show()
   plt.savefig(os.path.join('eda_outputs', f"{col} Vs Target distribution.png"))

# 6. Outlier detection (boxplots)
   # -------------------------
   for col in hpdata.columns:
      plt.figure(figsize=(8, 6))
      sns.boxplot(x=hpdata[col], color="orange")
      plt.title(f"Boxplot of {col}")
      plt.savefig(os.path.join('eda_outputs', f"box_{col}.png"))
      plt.close()



print(f"\n✅ EDA completed. Plots & stats saved in eda_outputs")
print("""
Observations: 
1. Price distribution is right-skewed.
2. Median Income (MedInc) has the strongest correlation with Price.
3. HouseAge and AveRooms also show positive correlations.
4. Some features (e.g., AveOccup) may have outliers, as shown in boxplots.
5. Correlation heatmap reveals multicollinearity among some features.
""")



# --------------- Train, Test, Validation split ---------------

X = hpdata.drop(columns=["PRICE"])
y = hpdata["PRICE"]

# Train/Validation/Test split (already done, but showing for clarity)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape, "Validation:", X_valid.shape, "Test:", X_test.shape)

# --------------- Checking for multicollinearity -------------


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X):
    X_const = sm.add_constant(X)
    vif = pd.Series([variance_inflation_factor(X_const.values, i+1)  # skip const index 0
                     for i in range(X.shape[1])], index=X.columns)
    return vif.sort_values(ascending=False)

vif_train = compute_vif(X_train)
print(vif_train)


# ----- Observations
print('Latitude, Longitude, AveBedrms, AveRooms : These 4 variablaes are correlated')        


# --- Model evaluation ----
def evaluate_and_log(model, model_name, X_train, y_train, X_val, y_val):
    with mlflow.start_run(run_name=model_name):
        # Fit
        model.fit(X_train, y_train)

        # Predict & evaluate
        preds = model.predict(X_val) 
        rmse = mean_squared_error(y_val, preds) ** 0.5
        r2 = r2_score(y_val, preds)

        # Log params, metrics
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name}: RMSE={rmse:.4f}, R2={r2:.4f}")

        # ✅ Return model info so it can be stored in results
        return {"name": model_name, "rmse": rmse, "r2": r2, "model": model}


def main(X_train, y_train, X_val, y_val):
    mlflow.set_experiment("house_price_comparison")
    results = []

    # 1. Random Forest (raw features)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    results.append(evaluate_and_log(rf, "RandomForest", X_train, y_train, X_val, y_val))

    # 2. SVM (with RobustScaler)
    svm_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVR(C=10, epsilon=0.2, kernel="rbf"))
    ])
    results.append(evaluate_and_log(svm_pipeline, "SVM_RobustScaled", X_train, y_train, X_val, y_val))

    # 3. RidgeCV with RobustScaler
    ridge_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5))
    ])
    results.append(evaluate_and_log(ridge_pipeline, "RidgeCV_RobustScaled", X_train, y_train, X_val, y_val))


    # -------------------------
    # Pick Best Model
    # -------------------------
    best_run = min(results, key=lambda x: x["rmse"])
    print(f"\n✅ Best Model: {best_run['name']} with RMSE={best_run['rmse']:.4f}")

    # Save best model as joblib
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/best_model_{best_run['name']}.joblib"
    joblib.dump(best_run["model"], best_model_path)
    print(f"💾 Saved best model to {best_model_path}")

    return best_run


best_run = main(X_train, y_train, X_valid, y_valid)



